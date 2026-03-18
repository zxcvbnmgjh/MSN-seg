import math

from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def np2th(weights, conv=False):
    # weights (numpy.ndarray): 输入的权重，格式为NumPy数组
    # conv (bool): 是否是卷积层的权重，如果为True，则将权重从HWIO格式转换为OIHW格式
    """
    Possibly convert HWIO to OIHW.
    HWIO格式表示高度、宽度、输入通道数和输出通道数。
    OIHW格式表示输出通道数、输入通道数、高度和宽度。
    这种转换通常用于卷积层权重的格式调整
    """
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights) # torch.Tensor: 转换后的权重张量


class StdConv2d(nn.Conv2d):
    # 在执行卷积操作之前对卷积核的权重进行标准化处理，确保每个输出通道的权重具有相似的分布，从而提高模型训练的稳定性和效率
    def forward(self, x):
        w = self.weight # 获取当前卷积层的权重。
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False) # 计算权重在每个输出通道上的均值和方差
        w = (w - m) / torch.sqrt(v + 1e-5) #对权重进行标准化处理
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups) # 使用标准化后的权重 w 对输入张量 x 进行卷积操作，并返回卷积结果


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.预激活版本的瓶颈残差块
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False) # 一个1x1卷积层，用于下采样和调整通道数
            self.gn_proj = nn.GroupNorm(cout, cout)  # 组归一化层，对下采样后的特征图进行归一化处理

    def forward(self, x):

        # Residual branch
        residual = x # 保存输入特征图作为残差分支
        if hasattr(self, 'downsample'):
            residual = self.downsample(x) # 进行下采样和通道数调整
            residual = self.gn_proj(residual) # 进行归一化处理

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y) # 残差连接
        return y
    """
    def load_from(self, weights, n_block, n_unit): # 从给定的权重加载器中加载预训练的权重
        
        # weights: 包含预训练权重的字典或数据结构。
        # n_block: 当前残差块的名称或索引。
        # n_unit: 当前残差单元的名称或索引。
        

        # 通过 np2th 函数将NumPy格式的权重转换为PyTorch张量，并根据需要调整卷积层权重的格式

        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)  
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])


        # 将这些权重张量复制到 PreActBottleneck 类的各个层中

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))
    """
    

class ResNetV2(nn.Module): 
    """Implementation of Pre-activation (v2) ResNet mode.预激活版本的 ResNet 模型实现"""

    def __init__(self, block_units, width_factor):
        # block_units 是一个列表，表示每个残差块中的残差单元数量
        # width_factor 表示网络的宽度因子，用于调整网络的宽度
        # width 变量根据宽度因子计算得到，作为模型宽度的基础
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        #预激活版本的瓶颈残差块的预激活部分
        self.root = nn.Sequential(OrderedDict([  # 一个顺序容器 nn.Sequential，包含一系列按顺序排列的层。
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))
        #预激活版本的瓶颈残差块的残差块部分
        self.body = nn.Sequential(OrderedDict([ # 一个顺序容器，包含三个残差块（block1, block2, block3）
            ('block1', nn.Sequential(OrderedDict(
                # 定义第一个单元，输入通道数为width，输出通道数为width*4，中间通道数为width
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                # 依次定义剩余的单元，每个单元的输入和输出通道数为width*4，中间通道数为width
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = [] # 存储每个残差块的输出特征图，用于之后的跳跃连接
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x) # 使用最大池化层对特征图进行下采样
        for i in range(len(self.body)-1): # 对于 self.body 中除了最后一个残差块以外的所有残差块，执行以下操作
            x = self.body[i](x) # 传递特征图给当前残差块进行处理
            right_size = int(in_size / 4 / (i+1)) # 计算当前残差块输出特征图的预期高度和宽度 right_size
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]  # 如果特征图的实际高度和宽度与预期值不同，则创建一个尺寸合适的零张量，并将特征图复制到该张量中，以确保特征图的尺寸一致。
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:] 
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x) # 将特征图传递给最后一个残差块进行处理 
        return x, features[::-1] # 返回最后一个残差块的输出特征图和 features 列表的逆序
