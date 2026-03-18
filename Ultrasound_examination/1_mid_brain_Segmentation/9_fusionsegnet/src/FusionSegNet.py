import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class PointwiseConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DepthwiseSeparableConvBN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation_rate: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        padding = dilation_rate * (kernel_size // 2)
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation_rate,
            groups=in_channels,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
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
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t = int(abs((torch.log2(torch.tensor(float(channels))) + b) / gamma))
        k_size = max(1, t if t % 2 else t + 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv1d(y).transpose(-1, -2).unsqueeze(-1)
        scale = self.sigmoid(y)
        return x * scale


class HMAA(nn.Module):
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
        reduced_channels = max(1, channels // reduction_ratio)
        self.block_size = block_size
        self.grid_size = grid_size

        self.shared_dense_one = nn.Linear(channels, reduced_channels)
        self.shared_dense_two = nn.Linear(reduced_channels, channels)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        self.layer_norm_block_h = nn.LayerNorm(channels)
        self.layer_norm_grid_w = nn.LayerNorm(channels)
        self.dense_block_h = nn.Linear(channels, channels)
        self.dense_grid_w = nn.Linear(channels, channels)

        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def _height_attention(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        block_h_size = h // self.block_size
        if block_h_size <= 0:
            return x

        h_crop = block_h_size * self.block_size
        x_crop = x[:, :, :h_crop, :]
        block_h = x_crop.view(b, c, block_h_size, self.block_size, w)
        block_h = block_h.permute(0, 2, 4, 3, 1).contiguous().view(-1, self.block_size, c)
        block_h = self.layer_norm_block_h(block_h)
        block_h_att = self.sigmoid(self.dense_block_h(block_h))
        block_h = block_h * block_h_att
        block_h = block_h.view(b, block_h_size, w, self.block_size, c)
        block_h = block_h.permute(0, 1, 3, 2, 4).contiguous().view(b, c, h_crop, w)

        if h_crop != h:
            block_h = F.interpolate(block_h, size=(h, w), mode="bilinear", align_corners=False)
        return block_h

    def _width_attention(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        grid_w_size = w // self.grid_size
        if grid_w_size <= 0:
            return x

        w_crop = grid_w_size * self.grid_size
        x_crop = x[:, :, :, :w_crop]
        grid_w = x_crop.view(b, c, h, grid_w_size, self.grid_size)
        grid_w = grid_w.permute(0, 3, 2, 4, 1).contiguous().view(-1, self.grid_size, c)
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
        height_width_refined = height_width_att

        avg_pool = torch.mean(height_width_refined, dim=1, keepdim=True)
        max_pool, _ = torch.max(height_width_refined, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = self.sigmoid(self.conv_spatial(spatial_input))

        spatial_refined = height_width_refined * spatial_att
        refined_output = spatial_refined * channel_refined
        return refined_output


class FuzzyMembershipFunction(nn.Module):
    """Implements Eq. (1): Gaussian-like fuzzy membership functions."""

    def __init__(self, channels: int, num_membership_functions: int):
        super().__init__()
        self.channels = channels
        self.num_membership_functions = num_membership_functions
        self.mu = nn.Parameter(torch.zeros(num_membership_functions, channels, 1, 1))
        self.log_sigma = nn.Parameter(torch.zeros(num_membership_functions, channels, 1, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.unsqueeze(1)  # [B, N, C, H, W] after broadcasting with params
        mu = self.mu.unsqueeze(0)
        sigma = F.softplus(self.log_sigma).unsqueeze(0) + 1e-6
        fuzzy_values = torch.exp(-((x - mu) ** 2) / (2.0 * sigma.pow(2)))
        fuzzy_representation = torch.prod(fuzzy_values, dim=1)
        return fuzzy_representation


class FuzzyLearningModule(nn.Module):
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
        self.fuzzy_membership = FuzzyMembershipFunction(in_channels, num_membership_functions)

        self.fuzzy_pointwise = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout2d(p=uncertainty_drop_rate)

        self.high_pw1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.high_dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.high_pw2 = nn.Conv2d(in_channels, filters, kernel_size=1, bias=False)

        self.low_dw1 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels, bias=False)
        self.low_pw1 = nn.Conv2d(in_channels, filters, kernel_size=1, bias=False)
        self.low_bn = nn.BatchNorm2d(filters)
        self.low_dw2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=filters, bias=False)
        self.low_pw2 = nn.Conv2d(filters, filters, kernel_size=1, bias=False)

        self.combine_dw = nn.Conv2d(filters * 2, filters * 2, kernel_size=3, padding=1, groups=filters * 2, bias=False)
        self.combine_pw = nn.Conv2d(filters * 2, filters, kernel_size=1, bias=False)
        self.combine_bn = nn.BatchNorm2d(filters)

        self.residual_proj = nn.Identity() if in_channels == filters else nn.Conv2d(in_channels, filters, kernel_size=1, bias=False)
        self.residual_bn = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fuzzy_features = self.fuzzy_membership(inputs)
        fuzzy_transformed = self.fuzzy_pointwise(fuzzy_features)
        mc_dropout = self.dropout(fuzzy_transformed)

        # Paper text states variance is computed across batch after MC dropout.
        uncertainty_map = torch.var(mc_dropout, dim=0, keepdim=True, unbiased=False)
        uncertainty_map = self.sigmoid(uncertainty_map)

        high_conf_features = inputs * (1.0 - uncertainty_map)
        low_conf_features = inputs * uncertainty_map

        high_conf_processed = self.relu(self.high_pw1(high_conf_features))
        high_conf_processed = self.relu(self.high_dw(high_conf_processed))
        high_conf_processed = self.relu(self.high_pw2(high_conf_processed))

        low_conf_processed = self.relu(self.low_dw1(low_conf_features))
        low_conf_processed = self.relu(self.low_pw1(low_conf_processed))
        low_conf_processed = self.low_bn(low_conf_processed)
        low_conf_processed = self.relu(self.low_dw2(low_conf_processed))
        low_conf_processed = self.relu(self.low_pw2(low_conf_processed))

        combined_features = torch.cat([high_conf_processed, low_conf_processed], dim=1)
        combined_features = self.relu(self.combine_dw(combined_features))
        combined_features = self.combine_pw(combined_features)
        refined_features = self.combine_bn(combined_features)

        residual = self.residual_bn(self.residual_proj(inputs))
        refined_output = residual + refined_features
        return refined_output, uncertainty_map


class MCAUBranch(nn.Module):
    """Depthwise separable strip-convolution branch used in MCAU."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.dw_h = nn.Conv2d(
            channels,
            channels,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
            groups=channels,
            bias=False,
        )
        self.dw_w = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_h(x)
        x = self.dw_w(x)
        x = self.pointwise(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


class MCAU(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
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
    def __init__(self, in_channels: int, num_filters: int, num_membership_functions: int, stage_index: int):
        super().__init__()
        self.stage_index = stage_index
        self.num_filters = num_filters

        self.ffn_pointwise = nn.Conv2d(in_channels, num_filters, kernel_size=1, bias=False)
        self.ffn_norm1 = nn.LayerNorm(num_filters)
        self.ffn_conv = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.ffn_norm2 = nn.LayerNorm(num_filters)
        self.input_align = nn.Identity() if in_channels == num_filters else nn.Conv2d(in_channels, num_filters, kernel_size=1, bias=False)

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

    @staticmethod
    def _apply_ln(x: torch.Tensor, layer_norm: nn.LayerNorm) -> torch.Tensor:
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
    def __init__(self, in_channels: int, num_filters_initial: int = 64, num_stages: int = 4, num_membership_functions: int = 5):
        super().__init__()
        self.num_stages = num_stages
        self.stages = nn.ModuleList()

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
            x = stage(x)
            encoder_features.append(x)
        return encoder_features


class GMFAM(nn.Module):
    def __init__(self, f4_channels: int, f3_channels: int, f_pha_channels: int):
        super().__init__()
        total_channels = f4_channels + f3_channels + f_pha_channels
        shared_out = total_channels // 2
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
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, f4: torch.Tensor, f3: torch.Tensor, f_pha: torch.Tensor) -> torch.Tensor:
        target_h, target_w = f4.shape[2:]
        f3_resized = F.interpolate(f3, size=(target_h, target_w), mode="bilinear", align_corners=False)
        f_pha_resized = F.interpolate(f_pha, size=(target_h, target_w), mode="bilinear", align_corners=False)

        concatenated = torch.cat([f4, f3_resized, f_pha_resized], dim=1)
        conv1 = self.shared_conv1(concatenated)
        conv3 = self.shared_conv3(concatenated)
        conv5 = self.shared_conv5(concatenated)
        multi_scale = torch.cat([conv1, conv3, conv5], dim=1)

        q = self.multi_scale_bn(self.multi_scale_dwconv(multi_scale))
        split_channels = q.size(1) // 4
        n = q[:, :split_channels]
        g4 = q[:, split_channels : 2 * split_channels]
        g3 = q[:, 2 * split_channels : 3 * split_channels]
        g_pha = q[:, 3 * split_channels : 4 * split_channels]

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
    def __init__(self, encoder_channels: int, decoder_channels: int, filters: int):
        super().__init__()
        self.encoder_align = nn.Identity() if encoder_channels == filters else nn.Conv2d(encoder_channels, filters, kernel_size=1, bias=False)
        self.gating_conv = nn.Conv2d(decoder_channels, filters, kernel_size=1, bias=True)
        self.f1 = DepthwiseSeparableConvBN(filters, filters, kernel_size=3, dilation_rate=1)
        self.f2 = DepthwiseSeparableConvBN(filters, filters, kernel_size=3, dilation_rate=2)
        self.f3 = DepthwiseSeparableConvBN(filters, filters, kernel_size=3, dilation_rate=4)
        self.fused_conv = nn.Conv2d(filters * 3, filters, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoder_features: torch.Tensor, decoder_features: torch.Tensor) -> torch.Tensor:
        target_h, target_w = decoder_features.shape[2:]
        encoder_features = F.interpolate(encoder_features, size=(target_h, target_w), mode="bilinear", align_corners=False)
        encoder_features = self.encoder_align(encoder_features)

        gating_map = self.sigmoid(self.gating_conv(decoder_features))
        attention_features = encoder_features * gating_map

        f1 = self.f1(attention_features)
        f2 = self.f2(attention_features)
        f3 = self.f3(attention_features)
        concat_features = torch.cat([f1, f2, f3], dim=1)
        fused_features = self.fused_conv(concat_features)
        output = fused_features + attention_features
        return output


class WMFM(nn.Module):
    def __init__(self, encoder_channels_list: List[int]):
        super().__init__()
        self.target_channels = encoder_channels_list[-1]
        self.feature_transforms = nn.ModuleList([
            nn.Conv2d(ch, self.target_channels, kernel_size=1, bias=False) for ch in encoder_channels_list
        ])
        self.weights = nn.Parameter(torch.ones(len(encoder_channels_list)))

    def forward(self, encoder_outputs: List[torch.Tensor]) -> torch.Tensor:
        target_h, target_w = encoder_outputs[-1].shape[2:]
        transformed_features = []
        for feat, transform in zip(encoder_outputs, self.feature_transforms):
            feat = transform(feat)
            feat = F.interpolate(feat, size=(target_h, target_w), mode="bilinear", align_corners=False)
            transformed_features.append(feat)

        normalized_weights = F.softmax(self.weights, dim=0)
        weighted_features = [w * feat for w, feat in zip(normalized_weights, transformed_features)]

        fused_feature = weighted_features[0]
        for feat in weighted_features[1:]:
            fused_feature = fused_feature * feat
        return fused_feature


class FusionSegNet(nn.Module):
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
