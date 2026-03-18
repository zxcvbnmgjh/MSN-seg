import torch
from torch import nn
import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target


def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True):
    # 计算Dice_loss + CE_loss
    losses = {}
    for name, x in inputs.items():

        unique_values = torch.unique(target)
        # print(f"Unique values in target: {unique_values}")

        target = torch.where(target == 255, torch.tensor(0, device=target.device), target)

        unique_values = torch.unique(target)
        # print(f"Unique values in target: {unique_values}")

        # 断言确保没有非法标签值
        assert all((unique_values >= 0) & (unique_values < num_classes)), \
        f"Label contains invalid value(s): {unique_values}"

        loss = nn.functional.cross_entropy(x, target, weight=loss_weight) # 计算交叉熵损失
        if dice is True:
            dice_target = build_target(target, num_classes) #此时的dice_target是针对每个类别的groundtruth，含对于每个类别而言的前景、背景、不感兴趣区域
            loss += dice_loss(x, dice_target, multiclass=True) #网络预测的结果与dice_target计算损失
            # dice_loss()：调用之前定义的 Dice Loss ; multiclass=True：表示当前任务是多分类任务(包括二分类)
        losses[name] = loss
    if len(losses) == 1:
        return losses['out']
    return losses['out'] + 0.5 * losses['aux']



def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes) #  混淆矩阵对象，用于记录预测结果与真实标签之间的匹配情况
    dice = utils.DiceCoefficient(num_classes=num_classes) # Dice 系数对象，用于计算平均 Dice 系数
    metric_logger = utils.MetricLogger(delimiter="  ")  #日志工具，用于记录和显示评估过程中的信息
    header = 'val:'
    if num_classes == 2: # 如果是二分类任务（num_classes == 2），定义一个权重张量 [1.0, 2.0]，其中背景类权重为 1.0，前景类权重为 2.0。
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        loss_weight = torch.as_tensor([1.0, 5.0], device=device)
    else: # 对于多分类任务，不使用自定义权重（loss_weight = None）
        loss_weight = None
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            # val_loss = criterion(output, target, loss_weight, num_classes=num_classes)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())  
            dice.update(output, target)

        confmat.reduce_from_all_processes()   # 混淆矩阵对象
        dice.reduce_from_all_processes()  # 计算得到的平均 Dice 系数

    return confmat, dice.value.item()


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None): # print_freq：打印日志的频率，每print_freq个批次打印一次日志，批次总数*batch_size=图像总数
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ") # 用于记录和显示训练过程中的指标（如损失、学习率）。
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if num_classes == 2: # 如果是二分类任务（num_classes == 2），定义一个权重张量 [1.0, 2.0]，其中背景类权重为 1.0，前景类权重为 2.0。
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else: # 对于多分类任务，不使用自定义权重（loss_weight = None）
        loss_weight = None

    for image, target in metric_logger.log_every(data_loader, print_freq, header): # 遍历数据加载器 data_loader，每次处理一个批次的数据
        image, target = image.to(device), target.to(device)
        with torch.amp.autocast(device_type='cuda', enabled=scaler is not None):
            output = model(image) # 前向传播，计算模型输出。
            loss = criterion(output, target, loss_weight, num_classes=num_classes) #  计算损失（ 计算的是当前批次的平均损失），结合交叉熵损失和 Dice 损失
            # x 和 dice_target 的形状均为 [N, C, H, W]，x 是模型的预测值，dice_target 是真实标签（one-hot 编码）

        optimizer.zero_grad()

        # 如果启用混合精度训练，使用 scaler.scale(loss).backward() 进行反向传播，并通过 scaler.step(optimizer) 更新参数；否则，直接调用 loss.backward() 和 optimizer.step()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()  # 在每个批次后调用一次 step() 方法，从而更新学习率

        lr = optimizer.param_groups[0]["lr"] # 获取的是当前优化器的学习率

        metric_logger.update(loss=loss.item(), lr=lr) #  将当前批次的损失值记录到 metric_logger 中，lr 被更新为最后一个批次完成后的学习率
 
    return metric_logger.meters["loss"].global_avg, lr
    # 返回的损失是整个训练数据集的全局平均损失（每个批次中图像的平均损失在整个数据集上的平均值），最终返回的学习率lr是该 epoch 的最后一个批次运行完成后更新的学习率。


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

def evaluate_loss(model, data_loader, device, num_classes):
    # 验证集损失为dice_loss + CE_loss , 五折交叉验证（训练损失和验证损失一致）
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes) #  混淆矩阵对象，用于记录预测结果与真实标签之间的匹配情况
    dice = utils.DiceCoefficient(num_classes=num_classes) # Dice 系数对象，用于计算平均 Dice 系数
    metric_logger = utils.MetricLogger(delimiter="  ")  #日志工具，用于记录和显示评估过程中的信息
    header = 'val:'
    if num_classes == 2: # 如果是二分类任务（num_classes == 2），定义一个权重张量 [1.0, 2.0]，其中背景类权重为 1.0，前景类权重为 2.0。
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        loss_weight = torch.as_tensor([1.0, 5.0], device=device)
    else: # 对于多分类任务，不使用自定义权重（loss_weight = None）
        loss_weight = None
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = criterion(output, target, loss_weight, num_classes=num_classes)
            # val_loss = criterion(output, target, loss_weight, num_classes=num_classes)
            output = output['out']
            confmat.update(target.flatten(), output.argmax(1).flatten())  
            dice.update(output, target)

        metric_logger.update(loss=loss.item())
        confmat.reduce_from_all_processes()   # 混淆矩阵对象
        dice.reduce_from_all_processes()  # 计算得到的平均 Dice 系数

    return metric_logger.meters["loss"].global_avg , confmat , dice.value.item()
