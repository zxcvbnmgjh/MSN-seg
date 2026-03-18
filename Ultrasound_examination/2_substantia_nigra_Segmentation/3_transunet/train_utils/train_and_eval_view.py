import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
from torch import nn
import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random


def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True):
    losses = {}
     # 检查 inputs 是否为张量
    if isinstance(inputs, torch.Tensor):
        # 如果是张量，则直接将其包装为字典
        inputs = {"out": inputs}
    for name, x in inputs.items():

        target = torch.where(target == 255, torch.tensor(0, device=target.device), target)

        unique_values = torch.unique(target)
        # print(f"Unique values in target: {unique_values}")

        # 断言确保没有非法标签值
        assert all((unique_values >= 0) & (unique_values < num_classes)), \
        f"Label contains invalid value(s): {unique_values}"

        loss = nn.functional.cross_entropy(x, target, weight=loss_weight,ignore_index=255) # 计算交叉熵损失
        if dice is True:
            dice_target = build_target(target, num_classes) #此时的dice_target是针对每个类别的groundtruth，含对于每个类别而言的前景、背景、不感兴趣区域
            loss += dice_loss(x, dice_target, multiclass=True) #网络预测的结果与dice_target计算损失
        losses[name] = loss
    if len(losses) == 1:
        return losses['out']
    return losses['out'] + 0.5 * losses['aux']

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def train_one_epoch_view(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if num_classes == 2:
        loss_weight = torch.as_tensor([1.0, 4.0], device=device)
    else:
        loss_weight = None

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            output = model(image)
            loss = criterion(output, target, loss_weight, num_classes=num_classes)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        metric_logger.update(loss=loss.item(), lr=lr)

        # 可视化当前批次的第一张图像、标签和预测图
        visualize_batch(image[0], target[0], output['out'][0], epoch, 8, "train")

    return metric_logger.meters['loss'].global_avg, lr


def visualize_batch(image, target, output, epoch, batch_idx, phase):
    """
    可视化当前批次的第一张图像、标签和预测图。
    :param image: 输入图像 [C, H, W]
    :param target: 真实标签图 [H, W]
    :param output: 模型输出 [C, H, W] 或 [H, W]（argmax 后）
    :param epoch: 当前 epoch
    :param batch_idx: 当前批次索引
    :param phase: 'train' 或 'val'
    """
    # 创建保存目录
    save_dir = f"./visualizations/{phase}/epoch_{epoch}"
    os.makedirs(save_dir, exist_ok=True)

    # 归一化图像
    image = (image.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)  # 假设均值和标准差是 [0.5, 0.5]

    # 获取预测图
    pred_mask = torch.argmax(output, dim=0).cpu().numpy() if output.dim() == 3 else output.cpu().numpy()

    # 绘制图像、标签和预测图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(target.cpu().numpy(), cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"batch_{batch_idx}.png"))
    plt.close()


def evaluate_view(model, data_loader, device, num_classes, epoch):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'val:'

    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']
            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)
            visualize_batch(image[0], target[0], output[0], epoch,  8, "val")

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()

    return confmat, dice.value.item()