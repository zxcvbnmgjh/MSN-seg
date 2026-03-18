import torch
import torch.nn as nn

def build_target(target: torch.Tensor, num_classes: int = 2): #将某张groundtruth转化为针对每个类别的groundtruth
    """build target for dice coefficient"""
    # 将形状为[N(批量大小)，H(高度)，W(宽度)]的真实groundtruth图像集合转化成形状为[N(批量大小)，C(类别个数)，H(高度)，W(宽度)]的one-hot编码格式
    dice_target = target.clone() #target包含前景1、背景0
    """
    if ignore_index >= 0:
        ignore_mask = torch.eq(target, ignore_index) # 寻找在target中像素值为ignore_index的位置
        dice_target[ignore_mask] = 0 # 将dice_target中像素为ignore_target的位置设置成0
        # [N, H, W] -> [N, H, W, C]
        dice_target = nn.functional.one_hot(dice_target, num_classes).float() #对dice_target进行one-hot编码（将输入的原始groundtruth转化为针对每一个类别的groundtruth）
        dice_target[ignore_mask] = ignore_index  #将像素为255的位置填充回255、此时的dice_target是针对每个类别的groundtruth，含对于每个类别而言的前景、背景、不感兴趣区域
    else:
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()
    """
    dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    return dice_target.permute(0, 3, 1, 2)


def dice_coeff(x: torch.Tensor, target: torch.Tensor, epsilon=1e-6):
    # x：针对某一类别的预测概率矩阵；target：该类别的groundtruth；
    # Average of Dice coefficient for all batches, or for a single mask
    # 计算一个batch中所有图片某个类别的dice_coefficient
    d = 0.
    batch_size = x.shape[0]
    for i in range(batch_size):
        x_i = x[i].reshape(-1) # x[i]:当前batch中第i张图片对应某一类别的预测概率矩阵；然后转化为向量 x_i
        t_i = target[i].reshape(-1) # target[i]:当前batch中第i张图片对应某一类别的真实groundtruth矩阵；然后转化为向量target[i]
        """
        if ignore_index >= 0:
            # 找出mask中不为ignore_index的区域
            roi_mask = torch.ne(t_i, ignore_index) #找到真正感兴趣的区域，在target中去掉ignore_index
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]
        """
        inter = torch.dot(x_i, t_i) # 向量内积
        sets_sum = torch.sum(x_i) + torch.sum(t_i) #两个向量各自求元素之和
        if sets_sum == 0: # 此时x_i与t_i均为0，预测正确
            sets_sum = 2 * inter

        d += (2 * inter + epsilon) / (sets_sum + epsilon) #epsilon防止分母为0
    return d / batch_size


def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, epsilon=1e-6):
    # x: 模型的预测输出，形状为 [N, C, H, W];target: 真实标签，形状为 [N, C, H, W]（通常是 one-hot 编码）
    """Average of Dice coefficient for all classes"""
    dice = 0.
    for channel in range(x.shape[1]):
        dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], epsilon) #遍历每一个类别的预测值与target的dice_codff并相加

    return dice / x.shape[1] #得到所有类别的dice_codff的均值


def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    x = nn.functional.softmax(x, dim=1) #对x在channel方向进行softmax处理，得到每个像素属于每个类别的概率
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target, ) # 得到预测分割图与原图groundtruth之间的dice_loss





