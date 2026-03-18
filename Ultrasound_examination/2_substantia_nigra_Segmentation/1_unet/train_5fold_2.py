import os
import time
import torch
from train_utils.train_and_eval import train_one_epoch, evaluate_loss, create_lr_scheduler
from torch.utils.data import Subset
from src import UNet
from my_dataset import SubstantiaNigra_Full
import transforms as T
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse


class SegmentationPresetTrain:
    def __init__(self, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.2767337, 0.27674654, 0.27666409),
                std=(0.25553482, 0.25551227, 0.25543953)):
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
    def __init__(self, crop_size, mean=(0.2767337, 0.27674654, 0.27666409),
                 std=(0.25553482, 0.25551227, 0.25543953)):
        self.transforms = T.Compose([
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


class EarlyStopping:
    def __init__(self, patience=50, min_delta=0.01, mode='max'):
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
            elif self.mode == 'min':
                if val_metric < self.best_score - self.min_delta:
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


def get_transform_train(crop_size=512, mean=(0.2767337, 0.27674654, 0.27666409),
                        std=(0.25553482, 0.25551227, 0.25543953)):
    return SegmentationPresetTrain(crop_size=crop_size, mean=mean, std=std)


def get_transform_val(crop_size=512, mean=(0.2767337, 0.27674654, 0.27666409),
                      std=(0.25553482, 0.25551227, 0.25543953)):
    return SegmentationPresetEval(crop_size=crop_size, mean=mean, std=std)


def create_model(num_classes):
    model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    return model

"""
def save_fold_indices(train_idx, val_idx, test_idx, fold_num, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    def write_list_to_file(path, data):
        with open(path, 'w') as f:
            for item in data:
                f.write(f"{item}\n")

    train_file = os.path.join(save_dir, f"fold_{fold_num}_train.txt")
    val_file = os.path.join(save_dir, f"fold_{fold_num}_val.txt")
    test_file = os.path.join(save_dir, f"fold_{fold_num}_test.txt")

    write_list_to_file(train_file, train_idx)
    write_list_to_file(val_file, val_idx)
    write_list_to_file(test_file, test_idx)

    print(f"Fold {fold_num} indices saved to {save_dir}")

"""
def save_fold_indices_by_filename(train_names, val_names, test_names, fold_num, save_dir):
    """
    保存训练/验证/测试集的**文件名**（而非索引）到文本文件。
    """
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

    print(f"Fold {fold_num} filenames saved to {save_dir}")

def get_fixed_dev_test_split(full_dataset, dev_folds_indices, test_indices):
    try:
        all_filenames = full_dataset.img_names
    except AttributeError:
        all_filenames = [f"image_{i:04d}.png" for i in range(len(full_dataset))]

    print("\n" + "=" * 60)
    print("📌 TEST SET FILENAMES:")
    for idx in test_indices:
        print(f"  {all_filenames[idx]}")
    print("=" * 60)

    dev_folds = []
    for i, fold_idx in enumerate(dev_folds_indices):
        fold_idx = np.array(fold_idx)
        dev_folds.append(fold_idx)
        print(f"\n📌 DEV FOLD {i} FILENAMES (n={len(fold_idx)}):")
        for idx in fold_idx:
            print(f"  {all_filenames[idx]}")
    print("=" * 60)

    return dev_folds, np.array(test_indices)


def main_with_fixed_split(args, fold_idx, train_idx, val_idx, test_idx):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_classes = args.num_classes + 1

    mean = (0.2767337, 0.27674654, 0.27666409)
    std = (0.25553482, 0.25551227, 0.25543953)

    original_dataset = SubstantiaNigra_Full(args.data_path, transforms=None)

    save_idx_dir = "/data2/gaojiahao/Ultrasound_examination/2_substantia_nigra_Segmentation/1_unet/5fold_save_result_2/fold_indices"
    # save_fold_indices(train_idx, val_idx, test_idx, fold_num=fold_idx, save_dir=save_idx_dir)

    # 获取文件名
    all_filenames = original_dataset.img_names  # 确保已定义
    train_names = [all_filenames[i] for i in train_idx]
    val_names = [all_filenames[i] for i in val_idx]
    test_names = [all_filenames[i] for i in test_idx]

    save_fold_indices_by_filename(train_names, val_names, test_names, fold_num=fold_idx, save_dir=save_idx_dir)

    # Create subsets
    train_subset = Subset(original_dataset, train_idx)
    val_subset = Subset(original_dataset, val_idx)


    # Transforms
    train_dataset = TransformedSubset(train_subset, get_transform_train(mean=mean, std=std))
    val_dataset = TransformedSubset(val_subset, get_transform_val(mean=mean, std=std))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    collate_fn = original_dataset.collate_fn

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=num_workers,
        pin_memory=True, collate_fn=collate_fn)


    model = create_model(num_classes=num_classes)
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if args.amp else None
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    best_dice = 0.
    best_epoch = 0

    results_dir = f"/data2/gaojiahao/Ultrasound_examination/2_substantia_nigra_Segmentation/1_unet/5fold_save_result_2/results_fold{fold_idx}"
    os.makedirs(results_dir, exist_ok=True)
    results_dir_path = os.path.join(results_dir, "train_metrics.txt")

    save_dir = f"/data2/gaojiahao/Ultrasound_examination/2_substantia_nigra_Segmentation/1_unet/save_weights_2/5fold/fold_{fold_idx}"
    os.makedirs(save_dir, exist_ok=True)

    train_losses = []
    val_dices = []
    val_losses = []

    early_stopping = EarlyStopping(patience=50, min_delta=0.002, mode='max')

    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(
            model, optimizer, train_loader, device, epoch, num_classes,
            lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        val_loss, confmat, dice = evaluate_loss(
            model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)

        train_losses.append(mean_loss)
        val_dices.append(dice)
        val_losses.append(val_loss)

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
                    "model": model.state_dict(),
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

    with open(results_dir_path, "a") as f:
        f.write(f"\nBest Dice Coefficient: {best_dice:.3f}, Achieved at Epoch: {best_epoch}\n")

    # ✅ 移除测试评估部分（不再加载 best_model，不再运行 test）

    # Plot
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

    # 不再返回测试指标，返回 None 或验证集最佳指标（可选）
    return {"best_val_dice": best_dice, "best_epoch": best_epoch}


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


def main():
    parser = argparse.ArgumentParser(description="UNet training with fixed dev/test split (no test evaluation)")
    parser.add_argument("--data-path", default="/data2/gaojiahao/Ultrasound_examination/2_substantia_nigra_Segmentation", help="dataset root")
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument("--device", default="cuda:3", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--save-best', default=True, type=bool)
    parser.add_argument("--amp", default=False, type=bool)
    args = parser.parse_args()

    # 🔴 定义你的固定划分（根据实际数据修改）
    dev_folds_indices = [
        list(range(70, 132)),    # fold0
        list(range(132, 196)),  # fold1
        list(range(196, 261)), # fold2
        list(range(261, 329)), # fold3
        list(range(329, 393))  # fold4
    ]
    test_indices = list(range(0, 70))  # 测试集索引（仅用于打印，不参与训练/验证）

    # Load dataset to get filenames
    full_dataset = SubstantiaNigra_Full(args.data_path, transforms=None)
    dev_folds, test_idx = get_fixed_dev_test_split(full_dataset, dev_folds_indices, test_indices)

    """
    # Save indices to files
    save_idx_dir = "/data2/gaojiahao/Ultrasound_examination/2_substantia_nigra_Segmentation/1_unet/5fold_save_result_2/fold_indices"
    os.makedirs(save_idx_dir, exist_ok=True)
    with open(os.path.join(save_idx_dir, "test_set.txt"), "w") as f:
        for idx in test_idx:
            f.write(f"{idx}\n")
    for i, fold in enumerate(dev_folds):
        with open(os.path.join(save_idx_dir, f"dev_fold_{i}.txt"), "w") as f:
            for idx in fold:
                f.write(f"{idx}\n")
    """

    # Save filenames (not indices)
    save_idx_dir = "/data2/gaojiahao/Ultrasound_examination/2_substantia_nigra_Segmentation/1_unet/5fold_save_result_2/fold_indices"
    os.makedirs(save_idx_dir, exist_ok=True)

    # 获取文件名
    all_filenames = full_dataset.img_names

    test_names = [all_filenames[i] for i in test_idx]
    with open(os.path.join(save_idx_dir, "test_set.txt"), "w") as f:
        for name in test_names:
            f.write(f"{name}\n")

    for i, fold in enumerate(dev_folds):
        fold_names = [all_filenames[idx] for idx in fold]
        with open(os.path.join(save_idx_dir, f"dev_fold_{i}.txt"), "w") as f:
            for name in fold_names:
                f.write(f"{name}\n")

    # 5-fold training loop (only train + val)
    all_fold_info = []
    for fold in range(5):
        print(f"\n{'=' * 20} ROUND {fold + 1}/5: VALIDATE ON DEV FOLD {fold} {'=' * 20}")
        val_idx = dev_folds[fold]
        train_idx = np.concatenate([dev_folds[i] for i in range(5) if i != fold])
        current_test_idx = test_idx  # 仅用于保存索引文件，不用于训练

        print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test (held-out): {len(current_test_idx)}")
        fold_info = main_with_fixed_split(args, fold, train_idx, val_idx, current_test_idx)
        all_fold_info.append(fold_info)

    # 打印每轮最佳验证 Dice
    print("\n" + "="*50)
    print("BEST VALIDATION DICE FOR EACH FOLD:")
    for i, info in enumerate(all_fold_info):
        print(f"Fold {i}: {info['best_val_dice']:.4f} @ epoch {info['best_epoch']}")

    print("\nTraining completed. Models saved. No test evaluation performed.")


if __name__ == '__main__':
    main()