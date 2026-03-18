import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
import utils
import MedViT 

def get_args_parser():
    parser = argparse.ArgumentParser('MedViT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=4, type=int) 
    parser.add_argument('--epochs', default=500, type=int)

    # Model parameters
    parser.add_argument('--model', default='MedViT_small', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=256, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.1, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT', # 在训练过程中随机丢弃部分路径
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--flops', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    # Optimizer parameters 优化器参数配置
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', #  优化器类型，默认为adamw
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', # 优化器Epsilon值，默认为1e-8
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA', #  优化器的Betas值
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=5, metavar='NORM', # 梯度裁剪的范数
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', # 设置动量值，仅适用于SGD优化器
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05, # 设置权重衰减值
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters 学习率调度参数
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER', # 学习率调度器类型
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR', # 初始学习率
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', # 学习率噪音参数，用于在特定的训练周期内添加学习率波动
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', # 学习率噪音的波动百分比
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', # 学习率噪音的标准差
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', # 预热阶段的学习率
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', # 设置学习率的最低值
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N', # 学习率衰减的周期间隔
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=20, metavar='N', # 设置学习率预热的周期数
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', # 学习率冷却的周期数
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N', # 设置Plateau学习率调度器的耐心周期
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', # 学习率衰减率
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters 数据增强参数
    parser.add_argument('--color-jitter', type=float, default=0.5, metavar='PCT', # 图像颜色抖动因子
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME', # AutoAugment策略
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)') # 标签平滑参数
    parser.add_argument('--train-interpolation', type=str, default='bicubic',  # 训练时的图像插值方法
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true') # 是否启用重复的数据增强
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=False)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT', # 随机擦除的概率
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel', # 随机擦除的模式
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, # 随机擦除的次数
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False, # 是否在第一次增强时不进行随机擦除
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.5, # Mixup参数值，大于0时启用
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0, #  Cutmix参数值
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None, # 设置Cutmix的最小和最大比例，用于覆盖alpha值
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0, # 执行Mixup或Cutmix的概率
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5, # 在同时启用Mixup和Cutmix时，切换到Cutmix的概率
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch', # 设置Mixup或Cutmix的执行模式
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', action='store_true', help='Perform finetune.') # 是否进行微调训练

    # Dataset parameters
    parser.add_argument('--data-path', default='/data2/gaojiahao/SN+project/MedViT-main/TCSdatatset', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19', 'image_folder'],
                        type=str, help='CIFAR dataset path')
    parser.add_argument('--use-mcloader', action='store_true', default=False, help='Use mcloader') # 是否使用特定的数据加载器
    parser.add_argument('--inat-category', default='name', # 设置在INAT数据集下的分类条件
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity') 

    parser.add_argument('--output-dir', default='/data2/gaojiahao/SN+project/MedViT-main/Results',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint') # 从指定的检查点恢复训练
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', # 设置训练起始的周期数
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only') # 是否仅进行评估
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation') #是否启用分布式评估 
    parser.add_argument('--num_workers', default=10, type=int) # 设置数据加载时使用的进程数
    parser.add_argument('--pin-mem', action='store_true', # 是否固定CPU内存
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', #
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,  # 分布式进程的数量
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training') # 设置用于初始化分布式训练的URL

    # test throught
    parser.add_argument('--throughout', action='store_true', help='Perform throughout only') # 是否仅进行吞吐量测试
    return parser


# 计算和记录模型的吞吐量（在特定条件下的每秒处理样本数量）
@torch.no_grad() # 禁用梯度计算（评估或测试阶段）
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader): # 使用for循环遍历数据加载器中提供的数据批次。images代表批次中的图像数据，_表示不使用标签数据
        images = images.cuda(non_blocking=True) # 将图像数据移动到GPU中以加速计算
        batch_size = images.shape[0] # 获得批次大小
        for i in range(50): # 先进行50次前向传播，以确保GPU已经“热身” ，避免在计算吞吐量时因初始阶段的性能波动而得出不准确的结果
            model(images) 
        torch.cuda.synchronize() # 等待GPU计算完成
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time() # 记录当前时间，作为30次前向传播的开始时间
        for i in range(30): # 进行30次前向传播
            model(images)
        torch.cuda.synchronize() # 再次同步GPU操作，确保所有计算都已完成
        tic2 = time.time() # 记录当前时间，作为30次前向传播的结束时间。
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}") # 计算吞吐量，即每秒处理的样本数，并使用日志记录器记录下来
        return # 函数执行完毕后，立即返回。这意味着只计算并记录第一个批次的吞吐量。

def main(args):
    utils.init_distributed_mode(args) # 调用函数来初始化分布式训练模式
    # print(args)

    device = torch.device(args.device) # 设置训练设备

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank() # 确保在分布式环境下，每个进程的随机种子是唯一的，从而保证各个进程的随机操作是独立的
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True # 用于加速卷积神经网络的运行，但会占用更多的显存，并且在输入尺寸和通道数固定的情况下，性能会更好。

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args) # 创建训练数据集
    dataset_val, _ = build_dataset(is_train=False, args=args) # 创建验证数据集


    # 采样器设置部分
    if args.distributed: # 检查命令行参数 args.distributed 来判断是否启用了分布式训练模式
        num_tasks = utils.get_world_size()      # 获取分布式训练中的进程总数（即分布式训练中参与训练的GPU数量）
        global_rank = utils.get_rank()           # 获取当前进程的排名
        if args.repeated_aug: # 根据是否启用了重复数据增强，选择不同的采样器来处理训练数据集
            sampler_train = RASampler( # 使用RASampler来处理训练数据集，该采样器在每个周期内重复数据增强的采样策略，确保每个GPU能够观察到所有的样本并进行重复增强
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler( #  使用DistributedSampler来处理训练数据集，分布式训练：将数据分片给不同 rank 的进程，每个进程只处理分片的数据，从而实现并行训练。
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval: # 是否启用了分布式评估（args.dist_eval），代码会选择不同的采样器来处理验证数据集。
            if len(dataset_val) % num_tasks != 0:
                
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train) # 随机采样（决定“每个 batch 从数据集中取出哪些样本索引）
        sampler_val = torch.utils.data.SequentialSampler(dataset_val) # 按序采样



    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=250,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
 

    #  Mixup 和 Cutmix 是数据增强技术，可以增加模型的泛化能力。通过混合不同样本的数据，模型在训练时会遇到更多的样本组合，有助于减少过拟合
    mixup_fn = None # 判断是否启用了 Mixup 或 Cutmix
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup( # 创建一个 Mixup 对象 mixup_fn
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)


    print(f"Creating model: {args.model}")
    model = create_model( # 创建指定的深度学习模
        args.model,
        num_classes=args.nb_classes,
    )

    if not args.distributed or args.rank == 0: # 判断是否为分布式训练模式
        print(model)  # 仅在非分布式训练模式或主进程中执行模型信息打印和 FLOPS 计算
        input_tensor = torch.zeros((1, 3, 224, 224), dtype=torch.float32) # 模拟输入的张量，用于计算模型在特定输入下的 FLOPS
        model.eval()
        utils.cal_flops_params_with_fvcore(model, input_tensor) # 自定义函数，用于计算模型的 FLOPS 和参数量

    model.to(device) # 将模型移动到指定的设备（如 GPU）上。
    model_ema = None # 模型的指数加权平均，对模型参数（权重）进行平滑更新



    if args.distributed: # 启用分布式训练,模型可以在多个GPU之间进行并行训练
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False) # 模型包装在DistributedDataParallel中
        model_without_ddp = model.module # 在保存模型时获取不包含DistributedDataParallel包装的原始模型
    else:
        model_without_ddp = model

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0 # 根据批量大小（args.batch_size）和世界大小（即参与训练的GPU数量，utils.get_world_size()）来线性缩放初始学习率（args.lr

    args.lr = linear_scaled_lr # 更新学习率
    optimizer = create_optimizer(args, model_without_ddp) # 创建优化器

    loss_scaler = NativeScaler() # 在混合精度训练中自动缩放损失值

    lr_scheduler, _ = create_scheduler(args, optimizer) # 创建学习率调度器

    criterion = LabelSmoothingCrossEntropy() # 设置为LabelSmoothingCrossEntropy 标签平滑的交叉熵损失函数

    if args.mixup > 0.: # 启用了Mixup
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy() # 损失函数更改为SoftTargetCrossEntropy
    elif args.smoothing: # 启用了标签平滑
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing) # 损失函数更改为LabelSmoothingCrossEntropy
    else:
        criterion = torch.nn.CrossEntropyLoss() 
    
    criterion = DistillationLoss(
        criterion, None, 'none', 0, 0
    ) # 将损失函数包装在DistillationLoss中

    if not args.output_dir:
        args.output_dir = args.model
        if utils.is_main_process(): # 检查当前进程是否为主进程（在分布式训练中，通常只有主进程进行日志或文件的写入操作）
            import os
            if not os.path.exists(args.model):
                os.mkdir(args.model)

    output_dir = Path(args.output_dir)
    if args.resume: # 恢复训练的检查点路径
        if args.resume.startswith('https'): # 根据路径是URL还是本地文件路径来加载检查点
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        if 'model' in checkpoint: # 判断这个 checkpoint 是“完整训练检查点”，还是“只有模型权重的文件”
            #model_without_ddp.load_state_dict(checkpoint['model'])
            checkpoint_model = checkpoint['model'] # 从检查点中提取模型状态字典
        else:
            #model_without_ddp.load_state_dict(checkpoint)
            checkpoint_model = checkpoint
        """
        if args.data_set != 'IMNET':    
            # 只有当你的数据集类别数不等于 ImageNet 的 1000 类时，预训练权重里的分类头（proj_head）参数形状才会不匹配；
            # 需要删掉这些 key，才能把 backbone 权重加载进来并重新初始化你自己的分类头。
            state_dict = model_without_ddp.state_dict()
            for k in ['proj_head.0.weight', 'proj_head.0.bias']: # 检查模型状态字典中特定键的形状是否与检查点中的对应键的形状相匹配
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k] 
        """
        state_dict = model_without_ddp.state_dict()
        for k in ['proj_head.0.weight', 'proj_head.0.bias']: # 检查模型状态字典中特定键的形状是否与检查点中的对应键的形状相匹配
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        utils.load_state_dict(model_without_ddp, checkpoint_model) # 加载模型状态字典

        
        if not args.finetune: # 仅在非微调模式且非评估模式下
            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler']) # 从检查点中恢复优化器状态 (optimizer)、学习率调度器状态 (lr_scheduler) 以及开始的训练周期
                if not args.finetune:
                    args.start_epoch = checkpoint['epoch'] + 1
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                    
    
    if args.eval: # 若是评估模式，则不进行模型训练，直接进行模型评估，跳出并终止主函数
        if hasattr(model.module, "merge_bn"): # 检查模型是否存在merge_bn方法，用于合并BN层，合并BatchNorm层可以加速推理过程
            print("Merge pre bn to speedup inference.")
            model.module.merge_bn()
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    if args.throughout:
        from logger import create_logger
        logger = create_logger(output_dir=output_dir, dist_rank=utils.get_rank(), name=args.model)
        throughput(data_loader_val, model, logger)
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=True,
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        if test_stats["acc1"] > max_accuracy:
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint_best.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MedViT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
