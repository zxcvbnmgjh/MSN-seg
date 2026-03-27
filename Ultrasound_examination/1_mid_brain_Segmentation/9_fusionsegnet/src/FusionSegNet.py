import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class PointwiseConv(nn.Module): #1x1 卷积
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DepthwiseSeparableConvBN(nn.Module): # 深度可分离卷积
    # 先用深度卷积在每个通道内部提取空间特征，再用 1×1 点卷积做通道混合，前后各配一个 BN 和 ReLU
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation_rate: int = 1, # 空洞卷积的膨胀率
        bias: bool = False,
    ):
        super().__init__()
        padding = dilation_rate * (kernel_size // 2)
        self.depthwise = nn.Conv2d( #Depthwise Conv（深度卷积），每个输入通道单独做卷积，只提取空间信息
            in_channels, 
            in_channels, # 深度卷积阶段只是在每个通道上各自处理空间，不改变通道数
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation_rate,
            groups=in_channels,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias) # Pointwise Conv（点卷积，1×1卷积），把各个通道再线性组合，完成通道信息融合
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act(x)
        return x


class ECALayer(nn.Module):
    # 通道注意力模块：根据每个通道的重要性，给每个通道分配一个权重，然后用这个权重去重新缩放原始特征图。
    # 先把每个通道压缩成一个全局描述值，再让这些通道描述值彼此“交流”一下，得到每个通道的权重，最后把这个权重乘回原特征图。
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t = int(abs((torch.log2(torch.tensor(float(channels))) + b) / gamma)) # 根据通道数 channels，动态算出一个中间值 t
        k_size = max(1, t if t % 2 else t + 1) # 得到最终的 1D 卷积核大小 k_size (奇数，奇数卷积核配这种 padding，才能保证卷积前后长度不变，并且“中心对齐”。)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 每个通道只保留一个数，表示这个通道在整张特征图上的平均响应
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)  # 对“通道描述序列”做一维卷积
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)  # [B,C,H,W]->[B,C,1,1]->[B,C,1]->[B,1,C]
        y = self.conv1d(y).transpose(-1, -2).unsqueeze(-1)  # [B,1,C]->[B,1,C]->[B,C,1]->[B,C,1,1]
        scale = self.sigmoid(y)
        return x * scale 


class HMAA(nn.Module):
    # 先判断哪些通道重要，再判断高度方向哪些块重要、宽度方向哪些网格重要，最后再生成一张空间注意力图，把这些注意力逐层叠加到输入特征上。
    # X→通道注意力→高度块注意力→宽度网格注意力→空间注意力→最终融合
    """
    Paper-aligned HMAA:
      1) channel attention from GAP/GMP shared MLP,
      2) height-wise block attention,
      3) width-wise grid attention,
      4) spatial attention from avg/max-pooled maps,
      5) final multiplication with channel-refined features.
    """

    def __init__(self, channels: int, block_size: int = 7, grid_size: int = 7, reduction_ratio: int = 16):
        super().__init__()
        reduced_channels = max(1, channels // reduction_ratio) # 给通道注意力里的共享 MLP 计算中间隐藏维度。通道注意力是先压缩到更小味道再恢复
        self.block_size = block_size
        self.grid_size = grid_size

        # 通道注意力的共享两层全连接
        self.shared_dense_one = nn.Linear(channels, reduced_channels)
        self.shared_dense_two = nn.Linear(reduced_channels, channels)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1) # 全局平均池化，偏整体响应
        self.global_max_pool = nn.AdaptiveMaxPool2d(1) # 全局最大池化，偏最强激活

        # 高度/宽度注意力用的 LN 和全连接
        self.layer_norm_block_h = nn.LayerNorm(channels)
        self.layer_norm_grid_w = nn.LayerNorm(channels)
        self.dense_block_h = nn.Linear(channels, channels)
        self.dense_grid_w = nn.Linear(channels, channels)

        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True) # 空间注意力卷积
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def _height_attention(self, x: torch.Tensor) -> torch.Tensor:
        
        # 计算能分成多少个高度块，高度维最多能完整切成多少个 self.block_size 高的小块。
        b, c, h, w = x.shape
        block_h_size = h // self.block_size
        if block_h_size <= 0:
            return x

        # 截取可整除部分，多余的部分后续插值回去
        h_crop = block_h_size * self.block_size
        x_crop = x[:, :, :h_crop, :]

        block_h = x_crop.view(b, c, block_h_size, self.block_size, w) # reshape 成高度块结构，把高度维拆成“块编号 × 块内位置”两层。
        block_h = block_h.permute(0, 2, 4, 3, 1).contiguous().view(-1, self.block_size, c) # 调整维度顺序，把“每个宽度位置上的一个高度块”，都整理成一个长度为 7、特征维为 C 的小序列
        block_h = self.layer_norm_block_h(block_h)
        block_h_att = self.sigmoid(self.dense_block_h(block_h))
        block_h = block_h * block_h_att
        block_h = block_h.view(b, block_h_size, w, self.block_size, c) # 把块级注意力后的特征恢复成标准卷积特征图格式
        block_h = block_h.permute(0, 1, 3, 2, 4).contiguous().view(b, c, h_crop, w)

        # 如果裁掉过边缘，就插值回原高度
        if h_crop != h:
            block_h = F.interpolate(block_h, size=(h, w), mode="bilinear", align_corners=False)
        return block_h

    def _width_attention(self, x: torch.Tensor) -> torch.Tensor:
        
        # 计算能分成多少个宽度网格
        b, c, h, w = x.shape
        grid_w_size = w // self.grid_size
        if grid_w_size <= 0:
            return x

        w_crop = grid_w_size * self.grid_size
        x_crop = x[:, :, :, :w_crop]
        grid_w = x_crop.view(b, c, h, grid_w_size, self.grid_size) # reshape 成宽度网格结构
        grid_w = grid_w.permute(0, 3, 2, 4, 1).contiguous().view(-1, self.grid_size, c) # 调整顺序并拉平成一个个长度为 7、特征维为 C 的小序列，和高度块注意力的处理方式类似，只不过这次是针对宽度方向的网格做注意力
        grid_w = self.layer_norm_grid_w(grid_w)
        grid_w_att = self.sigmoid(self.dense_grid_w(grid_w))
        grid_w = grid_w * grid_w_att
        grid_w = grid_w.view(b, grid_w_size, h, self.grid_size, c)
        grid_w = grid_w.permute(0, 2, 1, 3, 4).contiguous().view(b, h, w_crop, c)
        grid_w = grid_w.permute(0, 3, 1, 2).contiguous()

        if w_crop != w:
            grid_w = F.interpolate(grid_w, size=(h, w), mode="bilinear", align_corners=False)
        return grid_w

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = inputs.shape

        local_att = self.global_avg_pool(inputs).view(b, c)
        local_att = self.relu(self.shared_dense_one(local_att))
        local_att = self.sigmoid(self.shared_dense_two(local_att)).view(b, c, 1, 1)

        global_att = self.global_max_pool(inputs).view(b, c)
        global_att = self.relu(self.shared_dense_one(global_att))
        global_att = self.sigmoid(self.shared_dense_two(global_att)).view(b, c, 1, 1)

        channel_att = self.sigmoid(local_att + global_att)
        channel_refined = inputs * channel_att

        block_h = self._height_attention(channel_refined)
        grid_w = self._width_attention(channel_refined)

        height_width_att = self.sigmoid(block_h + grid_w)
        height_width_refined = channel_refined * height_width_att

        avg_pool = torch.mean(height_width_refined, dim=1, keepdim=True)
        max_pool, _ = torch.max(height_width_refined, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = self.sigmoid(self.conv_spatial(spatial_input))

        spatial_refined = height_width_refined * spatial_att
        refined_output = spatial_refined * channel_refined
        return refined_output


class FuzzyMembershipFunction(nn.Module):

    # 模块的输入：卷积/注意力模块处理后的中间特征图 [B, C, H, W]
    # 模块的输出：模糊隶属函数映射并聚合后的“模糊表示” [B, C, H, W]
    """Implements Eq. (1): Gaussian-like fuzzy membership functions."""

    def __init__(self, channels: int, num_membership_functions: int):
        # 初试话定义参数：num_membership_functions = N（模糊函数个数） ；channels = C（通道数）
        super().__init__()
        self.channels = channels
        self.num_membership_functions = num_membership_functions
        self.mu = nn.Parameter(torch.zeros(num_membership_functions, channels, 1, 1)) # 每一个模糊隶属函数、每一个通道，都有一个可学习的中心值 μ，[N, C, 1, 1]
        self.log_sigma = nn.Parameter(torch.zeros(num_membership_functions, channels, 1, 1)) # 它不是直接学习 sigma，而是学习 log_sigma，然后在 forward 里通过 softplus 转成正数标准差。

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        
        x = inputs.unsqueeze(1)  
        # [B, N, C, H, W] after broadcasting with params 插入了一个新的维度，方便后面与 mu 和 sigma 做广播运算。在第 1 维插入一个维度，后面要把每个输入特征，同时和 N 个模糊隶属函数 比较
        
        mu = self.mu.unsqueeze(0) # 和 x.shape = [B, 1, C, H, W] 对齐
        sigma = F.softplus(self.log_sigma).unsqueeze(0) + 1e-6

        fuzzy_values = torch.exp(-((x - mu) ** 2) / (2.0 * sigma.pow(2))) 
        # (x - mu).shape = [B, N, C, H, W] ；对于每个 batch、每个 membership function、每个通道、每个空间位置 (h,w)，都计算一次高斯响应
        # 这个值当前这个输入特征值，与某个模糊集合中心 mu 有多接近
        
        fuzzy_representation = torch.prod(fuzzy_values, dim=1) # 现在对第 1 维，也就是 membership function 维度求乘积；一个特征位置要同时在多个模糊函数下都具有较高隶属度，最后乘出来才会高
        return fuzzy_representation 
        # fuzzy_representation：它表示这个特征在当前通道和空间位置上，对整组模糊原型的综合适配度。
        # 值大：说明这个特征在多组模糊约束下都比较“合理”、比较稳定
        # 值小：说明这个特征和这些模糊原型整体不一致，可能更不确定、噪声更大，或者处在过渡区域
    


class FuzzyLearningModule(nn.Module):
    # 先从模糊特征估计 uncertainty，再用 uncertainty 把输入软分成高置信和低置信两路，分别用轻/重两种卷积策略处理，最后融合并加残差得到更鲁棒的输出特征。
    # 基于模糊不确定性的特征重分配与重建模块
    # 超声图像里有些区域边界清晰、特征可靠；有些区域噪声大、边界模糊、不确定性高。所以不能把所有特征一视同仁地处理，而应该：
    # 1）对高置信区域：尽量保持稳定，轻量细化
    # 2）对低置信区域：做更强的上下文建模和修正
    # 最后再融合成一个更鲁棒的输出
    """
    FLM aligned to the paper:
    - fuzzy membership aggregation by product,
    - 1x1 PConv,
    - uncertainty map,
    - high/low confidence branches,
    - concatenation + 3x3 DConv + 1x1 PConv + BN,
    - residual addition.

    Note:
    The paper describes variance across the batch after MC dropout.
    This is implemented literally as a batch-wise channel-preserving uncertainty map.
    """

    def __init__(self, in_channels: int, num_membership_functions: int, uncertainty_drop_rate: float = 0.2, filters: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.filters = filters
        self.fuzzy_membership = FuzzyMembershipFunction(in_channels, num_membership_functions) # 把输入转成模糊表示

        self.fuzzy_pointwise = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False) # 1×1 卷积，对模糊表示做线性通道变换
        self.dropout = nn.Dropout2d(p=uncertainty_drop_rate) # 制造随机扰动，便于通过方差估计 uncertainty
        
        # 高置信分支 1×1 PW → 3×3 DW → 1×1 PW 。先通道混合 再深度卷积做局部空间提炼 再通道映射到 filters
        self.high_pw1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False) # 1×1 卷积，对高置信区域做线性通道变换   
        self.high_dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.high_pw2 = nn.Conv2d(in_channels, filters, kernel_size=1, bias=False)

        # 低置信分支 5×5 DW → 1×1 PW → BN → 3×3 DW → 1×1 PW
        self.low_dw1 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels, bias=False)
        self.low_pw1 = nn.Conv2d(in_channels, filters, kernel_size=1, bias=False)
        self.low_bn = nn.BatchNorm2d(filters)
        self.low_dw2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=filters, bias=False)
        self.low_pw2 = nn.Conv2d(filters, filters, kernel_size=1, bias=False)

        # 融合部分。高低两路拼接后，通道会变成 2*filters，把两路特征进一步融合压回 filters 通道
        self.combine_dw = nn.Conv2d(filters * 2, filters * 2, kernel_size=3, padding=1, groups=filters * 2, bias=False)
        self.combine_pw = nn.Conv2d(filters * 2, filters, kernel_size=1, bias=False)
        self.combine_bn = nn.BatchNorm2d(filters)

        # 残差部分：保留原始输入信息，防止不确定性建模把原特征破坏得太厉害
        self.residual_proj = nn.Identity() if in_channels == filters else nn.Conv2d(in_channels, filters, kernel_size=1, bias=False) # 如果 in_channels == filters，直接 identity；如果不等，就先用 1×1 conv 把输入通道变到 filters
        self.residual_bn = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fuzzy_features = self.fuzzy_membership(inputs) # 输入转换成模糊表示
        fuzzy_transformed = self.fuzzy_pointwise(fuzzy_features) # 1×1 卷积细化模糊特征
        mc_dropout = self.dropout(fuzzy_transformed) #dropout 制造随机扰动

        # Paper text states variance is computed across batch after MC dropout.根据 batch 方差生成 uncertainty map
        # 对固定的 (通道c, 位置h,w) 来说：同一个 batch 里，不同样本在这个位置上的激活波动大不大
        # 波动大，方差大，说明这里不稳定，不确定性高；波动小，方差小，说明这里更稳定，不确定性低
        uncertainty_map = torch.var(mc_dropout, dim=0, keepdim=True, unbiased=False) # [1, C, H, W]
        uncertainty_map = self.sigmoid(uncertainty_map) # [1, C, H, W] 数值大致被压到 (0,1) 范围
        # uncertainty_map[0, c, h, w] 表示的是： 第 c 个通道在位置 (h,w) 的不确定程度

        # 用 uncertainty_map 把输入拆成两部分: 输入 inputs.shape = [B, C, H, W] 、不确定图 uncertainty_map.shape = [1, C, H, W]
        high_conf_features = inputs * (1.0 - uncertainty_map) # 如果某位置不确定性低，u 小，那么 1-u 大，这个位置更多地流向高置信分支
        low_conf_features = inputs * uncertainty_map # 如果某位置不确定性高，u 大，那么它更多地流向低置信分支

        #高置信分支处理（高置信分支比较轻，因为高置信特征已经比较可靠了，不需要太激进地改动。所以这里只做较轻的稳定增强）
        high_conf_processed = self.relu(self.high_pw1(high_conf_features)) # 通道混合
        high_conf_processed = self.relu(self.high_dw(high_conf_processed)) # 每个通道单独做 3×3 空间提炼
        high_conf_processed = self.relu(self.high_pw2(high_conf_processed)) # 最后一步才投影到 filters

        # 低置信分支处理
        # 低置信分支更重:低置信区域通常对应模糊边界，噪声干扰，结构不明显的病灶区域，所以作者用更大的 5×5 感受野先抓上下文，再继续做一轮 3×3 精修。
        low_conf_processed = self.relu(self.low_dw1(low_conf_features)) # 5×5 depthwise conv，感受野更大
        low_conf_processed = self.relu(self.low_pw1(low_conf_processed))
        low_conf_processed = self.low_bn(low_conf_processed)
        low_conf_processed = self.relu(self.low_dw2(low_conf_processed))
        low_conf_processed = self.relu(self.low_pw2(low_conf_processed))

        # 高低两路融合 concat → 3×3 DConv → 1×1 PConv → BN
        combined_features = torch.cat([high_conf_processed, low_conf_processed], dim=1) # 通道维 dim=1 上拼
        combined_features = self.relu(self.combine_dw(combined_features)) # 深度卷积，对拼接后的各通道做空间融合，再激活
        combined_features = self.combine_pw(combined_features) # 把通道压回 F
        refined_features = self.combine_bn(combined_features) 

        # 残差分支
        residual = self.residual_bn(self.residual_proj(inputs)) 
        # 若in_channels == filters，那 residual_proj = Identity()，[B, C, H, W] -> [B, C, H, W]
        # 若in_channels != filters，那 residual_proj = 1×1 conv，[B, in_channels, H, W] -> [B, filters, H, W]   


        refined_output = residual + refined_features
        return refined_output, uncertainty_map
        # refined_output：FLM 处理后的主输出特征[B, filters, H, W] ； 
        # uncertainty_map：一个 batch 里的所有样本，共享同一张 uncertainty map [1, C, H, W]


class MCAUBranch(nn.Module):
    """Depthwise separable strip-convolution branch used in MCAU."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.dw_h = nn.Conv2d( # 二维卷积，卷积核为(kernel_size,1），纵向条带卷积，或者说沿高度方向拉长的卷积核，主要在垂直方向上聚合信息，对宽度方向几乎不扩张感受野。
            channels,
            channels,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
            groups=channels, # 深度卷积，每个输入通道单独卷积，不同通道之间这一步不混合
            bias=False,
        ) # 输入输出的形状一致[B,C,H,W]
        
        self.dw_w = nn.Conv2d( # 二维卷积，卷积核为(kernel_size,1）,横向条带卷积，主要在宽度方向提取上下文信息.
            channels,
            channels,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
            groups=channels,
            bias=False,
        ) # 输入输出的形状一致[B,C,H,W]

        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False) # 1×1 卷积,不同通道的信息线性组合起来(前面的两个 depthwise 条带卷积，只是在每个通道内部做空间提取，没有通道混合)
        self.norm = nn.LayerNorm(channels) # PyTorch 里的 LayerNorm(channels) 默认是对最后一个维度做归一化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_h(x)
        x = self.dw_w(x)
        x = self.pointwise(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


class MCAU(nn.Module):
    # 对输入特征图同时从小、中、大三个尺度去提取“横向和纵向”的上下文信息，再用通道注意力筛选重要信息，最后把这些注意力增强后的多尺度信息反过来作用到原输入特征上
    def __init__(self, channels: int):
        super().__init__()
        # 单个分支，负责一种尺度的条带卷积提特征
        self.branch3 = MCAUBranch(channels, kernel_size=3)
        self.branch5 = MCAUBranch(channels, kernel_size=5)
        self.branch11 = MCAUBranch(channels, kernel_size=11)


        self.eca = ECALayer(channels)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        multi_scale = self.branch3(input_tensor) + self.branch5(input_tensor) + self.branch11(input_tensor)
        attention_modulated = self.eca(multi_scale)
        output = input_tensor * attention_modulated
        return output


class FMSANStage(nn.Module):
    # 把输入特征先做一次前馈卷积变换，再做多尺度上下文注意力增强，再做模糊不确定性感知细化，最后降采样后输出给下一层 encoder stage
    # input_tensor∈B×Cin​×H×W,output∈R×F×H′×W′ 。输入一张特征图，输出一张“通道变成 F = num_filters、空间尺寸变小一半”的更深层特征图。
    def __init__(self, in_channels: int, num_filters: int, num_membership_functions: int, stage_index: int):
        super().__init__()
        self.stage_index = stage_index # 记录这是第几个 stage
        self.num_filters = num_filters # 当前 stage 想输出的通道数

        self.ffn_pointwise = nn.Conv2d(in_channels, num_filters, kernel_size=1, bias=False) # 1×1 conv 把输入通道映射到当前 stage 的通道宽度
        self.ffn_norm1 = nn.LayerNorm(num_filters)
        self.ffn_conv = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False) # 3×3 conv 做局部空间特征提炼
        self.ffn_norm2 = nn.LayerNorm(num_filters)

        self.input_align = nn.Identity() if in_channels == num_filters else nn.Conv2d(in_channels, num_filters, kernel_size=1, bias=False) # 最后和对齐后的输入做残差相加
        # 让残差分支的张量形状和主分支输出形状对齐，保证后面可以做逐元素相加
        # 通常用 1×1 Conv 来做这个对齐,不改变空间尺寸、可以灵活改变通道数、它计算量相对小。
        
        self.relu = nn.ReLU(inplace=True)
        self.mcau = MCAU(num_filters)
        self.mcau_norm = nn.LayerNorm(num_filters)

        self.fuzzy_learning = FuzzyLearningModule(
            in_channels=num_filters,
            num_membership_functions=num_membership_functions,
            uncertainty_drop_rate=0.2,
            filters=num_filters,
        )

        self.downsample = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=2, padding=1, bias=False)

    @staticmethod # 静态方法 #对卷积特征图按通道做 LayerNorm 的辅助函数
    def _apply_ln(x: torch.Tensor, layer_norm: nn.LayerNorm) -> torch.Tensor: 
        # [B,C,H,W]->[B,H,W,C]->[B,H,W,C]（做 LayerNorm）->[B,C,H,W]
        return layer_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        residual_input = self.input_align(input_tensor)

        ffn_output = self.ffn_pointwise(input_tensor)
        ffn_output = self._apply_ln(ffn_output, self.ffn_norm1)
        ffn_output = self.relu(ffn_output)
        ffn_output = self.ffn_conv(ffn_output)
        ffn_output = self._apply_ln(ffn_output, self.ffn_norm2)
        ffn_output = residual_input + ffn_output

        mcau_output = self.mcau(ffn_output)
        mcau_output = self._apply_ln(mcau_output, self.mcau_norm)

        fuzzy_output, _ = self.fuzzy_learning(mcau_output)

        output = ffn_output + fuzzy_output
        output = self.downsample(output)
        return output


class FMSAN(nn.Module):
    # 输入 [B,Cin​,H,W]
    # 输出 encoder_features: List[torch.Tensor]，列表里每个元素，都是一个 stage 的输出特征图
    def __init__(self, in_channels: int, num_filters_initial: int = 64, num_stages: int = 4, num_membership_functions: int = 5):
        super().__init__()
        self.num_stages = num_stages
        self.stages = nn.ModuleList() # 专门用来存子模块的列表

        current_channels = in_channels
        filters = num_filters_initial
        for stage in range(num_stages):
            self.stages.append(FMSANStage(current_channels, filters, num_membership_functions, stage_index=stage + 1))
            current_channels = filters
            filters *= 2

    def forward(self, input_tensor: torch.Tensor) -> List[torch.Tensor]:
        x = input_tensor
        encoder_features = []
        for stage in self.stages:
            # stage = self.stages[0]
            # stage = self.stages[1]
            # stage = self.stages[2]
            # stage = self.stages[3]
            x = stage(x)
            encoder_features.append(x)
        return encoder_features


class GMFAM(nn.Module):
    # 先把 f4、f3、f_pha 对齐后拼接，再做三尺度共享深度可分离卷积，生成一个控制张量；
    # 从这个控制张量里分出三张门控图分别筛选三路输入，同时保留一份共享上下文 n，最后把它们重新融合成一个与 f4 同尺度、同通道的高层语义特征。
    def __init__(self, f4_channels: int, f3_channels: int, f_pha_channels: int):
        super().__init__()

        # 先算整个模块内部的通道规划
        total_channels = f4_channels + f3_channels + f_pha_channels
        shared_out = total_channels // 2 # 每个多尺度共享卷积分支的输出通道数
        combined_channels = shared_out * 3  
        assert combined_channels % 4 == 0, "GMFAM channel split must be divisible by 4."
        split_channels = combined_channels // 4

        self.shared_conv1 = self._make_shared_conv(total_channels, shared_out, kernel_size=1)
        self.shared_conv3 = self._make_shared_conv(total_channels, shared_out, kernel_size=3)
        self.shared_conv5 = self._make_shared_conv(total_channels, shared_out, kernel_size=5)

        self.multi_scale_dwconv = nn.Conv2d(
            combined_channels,
            combined_channels,
            kernel_size=3,
            padding=1,
            groups=combined_channels,
            bias=False,
        )
        self.multi_scale_bn = nn.BatchNorm2d(combined_channels)

        self.g4_conv = nn.Conv2d(split_channels, f4_channels, kernel_size=1)
        self.g3_conv = nn.Conv2d(split_channels, f3_channels, kernel_size=1)
        self.g_pha_conv = nn.Conv2d(split_channels, f_pha_channels, kernel_size=1)
        self.r_conv = nn.Conv2d(split_channels + f4_channels + f3_channels + f_pha_channels, f4_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _make_shared_conv(in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=in_channels, # 深度卷积，每个通道单独卷积，负责多尺度空间上下文提取
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, f4: torch.Tensor, f3: torch.Tensor, f_pha: torch.Tensor) -> torch.Tensor:
        target_h, target_w = f4.shape[2:] # 取 f4 的空间尺寸做目标尺寸，整个 GMFAM 的输出空间尺度以 f4 为准
        f3_resized = F.interpolate(f3, size=(target_h, target_w), mode="bilinear", align_corners=False) # 把 f3 和 f_pha resize 到 f4 大小，f3 会从更高分辨率降到 f4 的大小
        f_pha_resized = F.interpolate(f_pha, size=(target_h, target_w), mode="bilinear", align_corners=False) # 把 f_pha resize 到 f4 大小，f_pha 在当前网络配置下通常本来就和 f4 同尺寸，但代码仍然统一写了 resize，增强通用性

        concatenated = torch.cat([f4, f3_resized, f_pha_resized], dim=1)
        conv1 = self.shared_conv1(concatenated)
        conv3 = self.shared_conv3(concatenated)
        conv5 = self.shared_conv5(concatenated)
        multi_scale = torch.cat([conv1, conv3, conv5], dim=1)

        q = self.multi_scale_bn(self.multi_scale_dwconv(multi_scale)) # 融合了三路输入、又包含多尺度信息的“共享控制特征
        split_channels = q.size(1) // 4  # 按通道切成 4 份，每一份都是：[B,split_channels,H4,W4]
        n = q[:, :split_channels] # 共享上下文补充项
        g4 = q[:, split_channels : 2 * split_channels] # 给 f4 生成门控
        g3 = q[:, 2 * split_channels : 3 * split_channels] # 给 f3 生成门控 
        g_pha = q[:, 3 * split_channels : 4 * split_channels] # 给 f3 生成门控

        g4 = self.sigmoid(self.g4_conv(g4))
        g3 = self.sigmoid(self.g3_conv(g3))
        g_pha = self.sigmoid(self.g_pha_conv(g_pha))

        f4_gated = f4 * g4
        f3_gated = f3_resized * g3
        f_pha_gated = f_pha_resized * g_pha

        gated_concat = torch.cat([n, f4_gated, f3_gated, f_pha_gated], dim=1)
        output = self.relu(self.r_conv(gated_concat))
        return output


class AAFM(nn.Module):
    """Atrous Attention Fusion Module. Output is 4C after concatenation, per paper text."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.out_channels = channels * 4
        self.f1 = DepthwiseSeparableConvBN(channels, channels, kernel_size=3, dilation_rate=1)
        self.f2 = DepthwiseSeparableConvBN(channels, channels, kernel_size=3, dilation_rate=2)
        self.f3 = DepthwiseSeparableConvBN(channels, channels, kernel_size=3, dilation_rate=4)
        self.f4 = DepthwiseSeparableConvBN(channels, channels, kernel_size=3, dilation_rate=6)
        self.eca = ECALayer(self.out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        f1 = inputs + self.f1(inputs)
        f2 = f1 + self.f2(f1)
        f3 = f2 + self.f3(f2)
        f4 = f3 + self.f4(f3)

        concat_features = torch.cat([f1, f2, f3, f4], dim=1)
        refined = self.eca(concat_features)
        return refined


class GMSFB(nn.Module):
    # 先用 decoder 特征生成一张门控图，去筛选 encoder 的跳连特征；再对筛选后的特征做 3 个不同膨胀率的深度可分离卷积；最后把多尺度结果融合，并加一个残差
    # 用 decoder 当前语义信息作为“筛子”，从 encoder 特征里挑出有用的高分辨率细节，再做多尺度卷积增强，把增强结果作为这一层 decoder 的输入
    def __init__(self, encoder_channels: int, decoder_channels: int, filters: int):
        super().__init__()
        self.encoder_align = nn.Identity() if encoder_channels == filters else nn.Conv2d(encoder_channels, filters, kernel_size=1, bias=False) # 很多时候通道数就是对齐的可以直接indentity
        self.gating_conv = nn.Conv2d(decoder_channels, filters, kernel_size=1, bias=True) # 用 decoder 特征生成门控图的“前身”
        self.f1 = DepthwiseSeparableConvBN(filters, filters, kernel_size=3, dilation_rate=1)
        self.f2 = DepthwiseSeparableConvBN(filters, filters, kernel_size=3, dilation_rate=2)
        self.f3 = DepthwiseSeparableConvBN(filters, filters, kernel_size=3, dilation_rate=4)
        self.fused_conv = nn.Conv2d(filters * 3, filters, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoder_features: torch.Tensor, decoder_features: torch.Tensor) -> torch.Tensor:
        target_h, target_w = decoder_features.shape[2:] # 取 decoder 当前空间尺寸作为目标尺寸，GMSFB 的输出空间大小以 decoder 当前特征为准
        encoder_features = F.interpolate(encoder_features, size=(target_h, target_w), mode="bilinear", align_corners=False) # 把 encoder 特征 resize 到 decoder 尺寸
        encoder_features = self.encoder_align(encoder_features)

        gating_map = self.sigmoid(self.gating_conv(decoder_features)) # 用 decoder 特征生成 gating map
        attention_features = encoder_features * gating_map # 筛选 encoder 特征

        f1 = self.f1(attention_features)
        f2 = self.f2(attention_features)
        f3 = self.f3(attention_features)
        concat_features = torch.cat([f1, f2, f3], dim=1)
        fused_features = self.fused_conv(concat_features)
        output = fused_features + attention_features
        return output


class WMFM(nn.Module):
    # 把 4 个编码器 stage 输出融合成一个高层统一特征，
    # 统一通道和尺寸后，给每层 encoder 特征一个学习到的重要性权重，再做逐元素乘法融合
    def __init__(self, encoder_channels_list: List[int]):
        super().__init__()
        self.target_channels = encoder_channels_list[-1] # 把最后一个 encoder stage 的通道数，作为所有特征统一后的目标通道数
        self.feature_transforms = nn.ModuleList([
            nn.Conv2d(ch, self.target_channels, kernel_size=1, bias=False) for ch in encoder_channels_list
        ]) # 为每一个 encoder stage准备一个 1×1 conv，用于把该 stage 的通道数变到统一的 target_channels。
        self.weights = nn.Parameter(torch.ones(len(encoder_channels_list))) # 引入一组可学习权重，再通过 softmax 归一化，学习每个 encoder stage 特征的重要性
    def forward(self, encoder_outputs: List[torch.Tensor]) -> torch.Tensor:
        target_h, target_w = encoder_outputs[-1].shape[2:] # WMFM 统一后的空间大小，以最后一个 encoder 输出为准

        # 逐个 stage 做通道统一和尺寸统一
        transformed_features = []
        for feat, transform in zip(encoder_outputs, self.feature_transforms):
            feat = transform(feat)
            feat = F.interpolate(feat, size=(target_h, target_w), mode="bilinear", align_corners=False)
            transformed_features.append(feat)

        normalized_weights = F.softmax(self.weights, dim=0) # 对可学习权重做 softmax 归一化
        weighted_features = [w * feat for w, feat in zip(normalized_weights, transformed_features)] # 把每个权重乘到对应特征上

        fused_feature = weighted_features[0]  # 逐元素乘法融合
        for feat in weighted_features[1:]:
            fused_feature = fused_feature * feat
        return fused_feature


class FusionSegNet(nn.Module):
    # FusionSegNet = 先用 FMSAN 提多层编码特征，再在 bottleneck 处做加权融合与高层特征聚合，然后用 4 级解码器逐步恢复分辨率，并通过 GMSFB + HMAA 融合跳连特征，最后输出分割 logits。
    def __init__(self, input_channels: int = 3, num_filters_initial: int = 64, num_stages: int = 4, num_membership_functions: int = 5, num_classes: int = 1):
        super().__init__()
        if num_stages != 4:
            raise ValueError("FusionSegNet paper configuration uses exactly 4 encoder/decoder stages.")

        self.encoder = FMSAN(
            in_channels=input_channels,
            num_filters_initial=num_filters_initial,
            num_stages=num_stages,
            num_membership_functions=num_membership_functions,
        )

        encoder_channels = [num_filters_initial * (2 ** i) for i in range(num_stages)]
        decoder_filters = [512, 256, 128, 64]

        self.wmfm = WMFM(encoder_channels)
        self.hmaa_bottleneck = HMAA(encoder_channels[-1])
        self.gmfam = GMFAM(encoder_channels[3], encoder_channels[2], encoder_channels[-1])
        self.aafm = AAFM(encoder_channels[3])

        self.decoder_stages = nn.ModuleList()
        decoder_in_channels = [self.aafm.out_channels] + decoder_filters[:-1]
        encoder_indices = [3, 2, 1, 0]

        for in_ch, enc_idx, filters in zip(decoder_in_channels, encoder_indices, decoder_filters):
            decoder_stage = nn.ModuleDict(
                {
                    "upsample": nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    "conv": nn.Conv2d(in_ch, filters, kernel_size=3, padding=1, bias=False),
                    "bn": nn.BatchNorm2d(filters),
                    "gmsfb": GMSFB(encoder_channels[enc_idx], filters, filters),
                    "hmaa": HMAA(filters),
                }
            )
            self.decoder_stages.append(decoder_stage)

        self.output_conv = nn.Conv2d(decoder_filters[-1], num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor | dict:
        input_h, input_w = x.shape[2:]
        encoder_outputs = self.encoder(x)

        fused_encoder_output = self.wmfm(encoder_outputs)
        enhanced_output = self.hmaa_bottleneck(fused_encoder_output)
        gmfa_output = self.gmfam(encoder_outputs[3], encoder_outputs[2], enhanced_output)
        decoder_input = self.aafm(gmfa_output)

        encoder_indices = [3, 2, 1, 0]
        decoder_outputs = []
        for i, stage in enumerate(self.decoder_stages):
            out = stage["upsample"](decoder_input)
            out = stage["conv"](out)
            out = stage["bn"](out)
            out = self.relu(out)
            out = stage["gmsfb"](encoder_outputs[encoder_indices[i]], out)
            out = stage["hmaa"](out)
            decoder_input = out
            decoder_outputs.append(out)

        logits = self.output_conv(decoder_outputs[-1])
        logits = F.interpolate(logits, size=(input_h, input_w), mode="bilinear", align_corners=False)
        output = self.sigmoid(logits)

        if return_features:
            return {
                "out": logits,
                "decoder_outputs": decoder_outputs,
                "encoder_outputs": encoder_outputs,
                "gmfa_output": gmfa_output,
            }
        return {"out": logits}
        #return output 


if __name__ == "__main__":
    # The self-check below uses a single CPU thread to avoid slowdowns from
    # excessive thread scheduling on some environments.
    torch.set_num_threads(1)
    model = FusionSegNet(input_channels=3, num_filters_initial=64, num_stages=4, num_membership_functions=5)
    x = torch.randn(8, 3, 512, 512)

    model.eval()
    with torch.no_grad():
        output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
