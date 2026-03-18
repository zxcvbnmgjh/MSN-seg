# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        # nn.Linear期望输入的第一个维度是批次大小，最后一个维度是特征维度
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        # x.flatten(2)：将输入的四维张量（batch_size, channels, height, width），转换为三维张量（batch_size, channels, height*width）
        # x.transpose(1, 2)：交换第二个和第三个维度，（batch_size, channels, height*width）转换为（batch_size, height*width，channels）
        x = self.proj(x)
        return x
    
class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        # num_classes: 分割任务中的类别数量
        # in_channels: 列表形式给出的各层输出通道数，对应于模型中不同深度的特征图
        # embedding_dim: 经过线性嵌入后的统一维度
        # dropout_ratio: Dropout 层的丢弃率，用于防止过拟合
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        # 各层transformer block输出的特征图转换到同一维度，便于之后拼接
        # MLP 层：对于每个输入特征图（c1到c4），使用一个线性嵌入层（MLP）将其投影到相同的维度embedding_dim，以便后续的拼接和融合
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        # 融合层：linear_fuse 使用一个卷积层将四个经过线性嵌入的特征图拼接后进行融合，减少通道数至embedding_dim
        self.linear_fuse = ConvModule(
            c1=embedding_dim*4,
            c2=embedding_dim,
            k=1,
        )

        # 预测层与Dropout：最后通过一个 nn.Conv2d 层对融合后的特征图进行分类预测，并在之前应用了一个 nn.Dropout2d 层来进一步防止过拟合
        self.linear_pred    = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout        = nn.Dropout2d(dropout_ratio)
    
    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        
        # 输入处理：首先，将输入的特征图分别通过各自的MLP层进行线性变换，调整为相同维度，并且重新排列尺寸以适应后续操作
        # 上采样：为了使所有特征图具有相同的尺寸，采用双线性插值方法对除最底层（c1）外的所有特征图进行上采样，使其与c1大小一致

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        # self.linear_c3(c3)：通过一个MLP（多层感知器）模块对输入c3进行线性变换，c3是一个形状为(n, c3_in_channels, h, w)的张量，通过MLP模块线性嵌入，c3的每个空间位置的特征向量从c3_in_channels维映射到embedding_dim维
        # .permute(0,2,1)：线性变换后，输出的形状变为(n, h*w, embedding_dim)。这里使用permute方法改变维度顺序，将第二个维度（即特征维度embedding_dim）与第三个维度（即高度乘以宽度h*w）交换位置，得到形状为(n, embedding_dim, h*w)的张量。这样做的目的是为了后续能够正确地重塑张量形状。 
        # .reshape(n, -1, c3.shape[2], c3.shape[3])：接着，将张量重塑回原始的空间尺寸。这里-1表示自动计算该维度的大小，使得总元素数量保持不变。由于之前我们已经将特征维度从c3_in_channels变为了embedding_dim，所以现在我们需要将特征图恢复为其原始的空间分辨率(h, w)，同时保证每个位置的特征向量维度为embedding_dim。因此，最终得到的张量形状为(n, embedding_dim, h, w)。
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
        # 通过双线性插值方法，调整特征图 _c3 的大小，使其与 c1 特征图的空间尺寸相匹配


        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        # 将 _c4, _c3, _c2, _c1 在dim=1，通道(channel)维度上进行拼接，之后进行变换到embedding_dim维

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x

class SegFormer(nn.Module):
    def __init__(self, num_classes = 21, phi = 'b0', pretrained = False):
        super(SegFormer, self).__init__()
        self.in_channels = { # 各层特征图通道数[C1,C2,C3,C4]
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone   = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)
        self.embedding_dim   = {   # 嵌入向量维度数
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]
        self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        
        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)
        
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return {"out": x}  # 返回字典
   
