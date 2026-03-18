import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from train_utils.train_and_eval import train_one_epoch, evaluate_loss, create_lr_scheduler
from torch.utils.data import Subset
from src.FusionSegNet import FusionSegNet
from my_dataset import Midbrain_Full
import transforms as T
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
import faulthandler
faulthandler.enable()

# 模型实现可运行，但当前环境下 cuDNN 不稳定，需要关闭 cuDNN
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False


class SegmentationPresetTrain:
    def __init__(self, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.16882014, 0.16886712, 0.16884266), std=(0.2274119, 0.22743388, 0.22739229)):
        trans = []
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, crop_size, mean=(0.16882014, 0.16886712, 0.16884266), std=(0.2274119, 0.22743388, 0.22739229)
                 ):
        self.transforms = T.Compose([
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


class EarlyStopping:
    def __init__(self, patience=50, min_delta=0.005, mode='max'):
        """
        Args:
            patience (int): 连续多少个 epoch 没有改善就停止（小样本推荐 20-30）
            min_delta (float): 判断"改善"的最小变化量（Dice 提升 >0.005 才算有效）
            mode (str): 'max' 表示越大越好（如 dice）
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric):
        if self.best_score is None:
            self.best_score = val_metric
        else:
            if self.mode == 'max':
                if val_metric > self.best_score + self.min_delta:
                    self.best_score = val_metric
                    self.counter = 0
                else:
                    self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)


def get_transform_train(crop_size=512, mean=(0.16882014, 0.16886712, 0.16884266),
                        std=(0.2274119, 0.22743388, 0.22739229)):
    return SegmentationPresetTrain(crop_size=crop_size, mean=mean, std=std)
def get_transform_val(crop_size=512, mean=(0.16882014, 0.16886712, 0.16884266),
                        std=(0.2274119, 0.22743388, 0.22739229)):
    return SegmentationPresetEval(crop_size=crop_size, mean=mean, std=std)


def create_model(num_classes):
    """创建 FusionSegNet 模型"""
    model = FusionSegNet(
        input_channels=3,
        num_filters_initial=64,
        num_stages=4,
        num_membership_functions=5,
        num_classes=num_classes
    )
    return model


def get_k_fold_indices(dataset_size):
    fold_ranges = [
        (0, 139),    # Fold 0
        (140, 279),  # Fold 1
        (280, 417), # Fold 2
        (418, 558), # Fold 3
        (559, 699)  # Fold 4
    ]

    if dataset_size < 700:
        raise ValueError(f"数据集大小 ({dataset_size}) 小于所需的最大索引 (393)。")

    folds = []
    for start_idx, end_idx in fold_ranges:
        end_idx = min(end_idx, dataset_size - 1)
        fold_indices = list(range(start_idx, end_idx + 1))
        folds.append(np.array(fold_indices))

    return folds


def save_fold_filenames(train_names, val_names, test_names, fold_num, save_dir):
    """保存文件名（而非索引）"""
    os.makedirs(save_dir, exist_ok=True)

    def write_list_to_file(path, data):
        with open(path, 'w') as f:
            for item in data:
                f.write(f"{item}\n")

    train_file = os.path.join(save_dir, f"fold_{fold_num}_train.txt")
    val_file = os.path.join(save_dir, f"fold_{fold_num}_val.txt")
    test_file = os.path.join(save_dir, f"fold_{fold_num}_test.txt")

    write_list_to_file(train_file, train_names)
    write_list_to_file(val_file, val_names)
    write_list_to_file(test_file, test_names)

    print(f"Fold {fold_num} 文件名已保存至 {save_dir}")


def main(args, fold_idx, all_indices, full_dataset, rank, world_size):
    set_seed(args.seed + rank)  # 每个进程使用不同的随机种子
    device = torch.device(f"cuda:{rank}")
    batch_size = args.batch_size
    num_classes = args.num_classes + 1

    mean=(0.16882014, 0.16886712, 0.16884266)
    std=(0.2274119, 0.22743388, 0.22739229)

    # 获取文件名列表
    try:
        all_filenames = full_dataset.img_names
    except AttributeError:
        all_filenames = [f"image_{i:04d}.png" for i in range(len(full_dataset))]

    # 当前 fold 的划分（索引）
    test_idx = all_indices[fold_idx]
    val_idx = all_indices[(fold_idx + 1) % len(all_indices)]
    train_idx = np.concatenate([all_indices[i] for i in range(5) if i != fold_idx and i != (fold_idx + 1) % 5])

    # 转为文件名
    test_names = [all_filenames[i] for i in test_idx]
    val_names = [all_filenames[i] for i in val_idx]
    train_names = [all_filenames[i] for i in train_idx]

    # 只在 rank 0 打印文件名
    if rank == 0:
        print("\n" + "="*50)
        print(f"📌 FOLD {fold_idx} 划分:")
        print(f"Train ({len(train_names)}): {train_names[:3]}{'...' if len(train_names)>3 else ''}")
        print(f"Val   ({len(val_names)}): {val_names[:3]}{'...' if len(val_names)>3 else ''}")
        print(f"Test  ({len(test_names)}): {test_names[:3]}{'...' if len(test_names)>3 else ''}")
        print("="*50)

        # 保存文件名到文件
        save_idx_dir = "./5fold_save_result_GPUs/fold_indices"
        save_fold_filenames(train_names, val_names, test_names, fold_num=fold_idx, save_dir=save_idx_dir)

    # 数据集初始化
    original_dataset = Midbrain_Full(args.data_path, transforms=None)
    train_subset = Subset(original_dataset, train_idx)
    val_subset = Subset(original_dataset, val_idx)

    class TransformedSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x, y = self.transform(x, y)
            return x, y

        def __len__(self):
            return len(self.subset)

    train_dataset = TransformedSubset(train_subset, get_transform_train(mean=mean, std=std))
    val_dataset = TransformedSubset(val_subset, get_transform_val(mean=mean, std=std))

    # 分布式采样器
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    collate_fn = original_dataset.collate_fn

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, sampler=val_sampler,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

    model = create_model(num_classes=num_classes)

    # 将模型移动到当前设备
    model.to(device)
    # 使用 DistributedDataParallel 包装模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)

    # 只对模型参数进行优化（排除 DDP 添加的参数）
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if args.amp else None
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    best_dice = 0.
    best_epoch = 0

    # 只在 rank 0 保存结果
    results_dir = f"./5fold_save_result_GPUs/results_fold{fold_idx}"
    if rank == 0:
        os.makedirs(results_dir, exist_ok=True)
    dist.barrier()  # 等待所有进程创建目录

    results_dir_path = os.path.join(results_dir, "train_metrics.txt")

    save_dir = f"./save_weights_GPUs/5fold/fold_{fold_idx}"
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier()

    train_losses = []
    val_dices = []
    val_losses = []

    early_stopping = EarlyStopping(patience=50, min_delta=0.005, mode='max')

    for epoch in range(args.start_epoch, args.epochs):
        # 设置采样器 epoch
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        mean_loss, lr = train_one_epoch(
            model, optimizer, train_loader, device, epoch, num_classes,
            lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        val_loss, confmat, dice = evaluate_loss(
            model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)

        train_losses.append(mean_loss)
        val_dices.append(dice)
        val_losses.append(val_loss)

        # 只在 rank 0 打印和保存
        if rank == 0:
            print(f"[Fold {fold_idx}, Epoch {epoch}] Dice: {dice:.4f}")
            with open(results_dir_path, "a") as f:
                f.write(f"[Epoch {epoch}]\n"
                        f"train_loss: {mean_loss:.4f}\n"
                        f"lr: {lr:.6f}\n"
                        f"val dice coefficient: {dice:.4f}\n"
                        f"val_loss: {val_loss:.4f}\n"
                        f"{val_info}\n")

            if args.save_best:
                if dice > best_dice:
                    best_dice = dice
                    best_epoch = epoch
                    save_file = {
                        "model": model.module.state_dict(),  # 保存原始模型状态
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args
                    }
                    if args.amp:
                        save_file["scaler"] = scaler.state_dict()
                    torch.save(save_file, os.path.join(save_dir, "best_model.pth"))

            early_stopping(dice)
            if early_stopping.early_stop:
                msg = f"Early stopping at epoch {epoch}. Best Dice: {best_dice:.4f} at epoch {best_epoch}"
                print(msg)
                with open(results_dir_path, "a") as f:
                    f.write(f"\n{msg}\n")
                break

    # 只在 rank 0 保存最终结果
    if rank == 0:
        with open(results_dir_path, "a") as f:
            f.write(f"\nBest Dice Coefficient: {best_dice:.3f}, Achieved at Epoch: {best_epoch}\n")

        # 绘制损失曲线
        plt.figure(figsize=(15, 8))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_dices, label='Validation Dice')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.title(f'Fold {fold_idx + 1} Training and Validation Metrics')
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"loss_curve_fold{fold_idx}.png"))
        plt.close()

    # 同步所有进程
    dist.barrier()

    # 返回 best_dice 用于汇总
    return {"best_val_dice": best_dice, "best_epoch": best_epoch}


def parse_args():
    parser = argparse.ArgumentParser(description="pytorch unet training with multi-gpu (no test evaluation)")
    parser.add_argument("--data-path", default="../", help="dataset root")
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=2, type=int, help="batch size per gpu")
    parser.add_argument("--epochs", default=300, type=int, metavar="N")
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--save-best', default=True, type=bool)
    parser.add_argument("--amp", default=False, type=bool)
    parser.add_argument("--gpus", default="0,1", type=str, help="GPU IDs to use, e.g., '0,1'")
    args = parser.parse_args()
    return args


def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()


def run_fold_training(args, fold, indices, full_dataset, rank, world_size):
    """在指定 GPU 上运行 fold 训练"""
    setup(rank, world_size)

    print(f"\n{'=' * 20} Fold {fold + 1} / {5} {'=' * 20}")
    fold_info = main(args, fold, indices, full_dataset, rank, world_size)

    cleanup()
    return fold_info


if __name__ == '__main__':
    args = parse_args()
    total_folds = 5

    # 设置可见的 GPU
    gpu_ids = [int(x) for x in args.gpus.split(',')]
    world_size = len(gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    print(f"Using GPUs: {args.gpus}")
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"Effective batch size: {args.batch_size * world_size}")

    # 加载完整数据集以获取文件名
    full_dataset = Midbrain_Full(args.data_path, transforms=None)
    indices = get_k_fold_indices(len(full_dataset))

    all_fold_info = []

    # 顺序训练每个 fold（每个 fold 使用多卡）
    for fold in range(total_folds):
        # 使用 torch.multiprocessing.spawn 启动多个进程
        mp.spawn(
            run_fold_training,
            args=(fold, indices, full_dataset, world_size),
            nprocs=world_size,
            join=True
        )

        # 由于每个 fold 的训练在独立的进程组中运行，需要从文件中读取结果
        results_dir = f"./5fold_save_result_GPUs/results_fold{fold}"
        results_file = os.path.join(results_dir, "train_metrics.txt")

        best_dice = 0.
        best_epoch = 0
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                for line in f:
                    if "Best Dice Coefficient:" in line:
                        parts = line.split(":")
                        if len(parts) >= 2:
                            dice_part = parts[1].strip().split(",")[0]
                            best_dice = float(dice_part)
                    if "Achieved at Epoch:" in line:
                        parts = line.split(":")
                        if len(parts) >= 2:
                            best_epoch = int(parts[1].strip())

        all_fold_info.append({"best_val_dice": best_dice, "best_epoch": best_epoch})

    # 打印每轮最佳验证 Dice
    print("\n" + "="*50)
    print("BEST VALIDATION DICE FOR EACH FOLD:")
    for i, info in enumerate(all_fold_info):
        print(f"Fold {i}: {info['best_val_dice']:.4f} @ epoch {info['best_epoch']}")

    print("\nTraining completed. Models saved. No test evaluation performed.")
