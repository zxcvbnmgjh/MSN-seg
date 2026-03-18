# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
+ ✅ Added validation loop
+ ✅ Save best model based on validation loss
+ ✅ Added Early Stopping: stop if val_dice improves <= 0.01 in 100 epochs
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "6"


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )  # (1024, 1024, 3)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )  # multiple labels [0, 1,4,5...], (256,256)
        assert img_name == os.path.basename(self.gt_path_files[index]), (
            "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )
        label_ids = np.unique(gt)[1:]
        if len(label_ids) == 0:
            gt2D = np.zeros_like(gt, dtype=np.uint8)
        else:
            gt2D = np.uint8(
                gt == random.choice(label_ids.tolist())
            )  # only one label, (256, 256)
        assert np.max(gt2D) == 1 or np.max(gt2D) == 0, "ground truth should be 0 or 1"
        if np.max(gt2D) == 0:
            H, W = gt2D.shape
            bboxes = np.array([0, 0, W, H])
        else:
            y_indices, x_indices = np.where(gt2D > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            H, W = gt2D.shape
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(W, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(H, y_max + random.randint(0, self.bbox_shift))
            if x_min >= x_max:
                x_min, x_max = 0, W
            if y_min >= y_max:
                y_min, y_max = 0, H
            bboxes = np.array([x_min, y_min, x_max, y_max])
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )


# %% sanity test of dataset class
tr_dataset = NpyDataset("data/npy/CT_Abd")
tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True)
for step, (image, gt, bboxes, names_temp) in enumerate(tr_dataloader):
    print(image.shape, gt.shape, bboxes.shape)
    _, axs = plt.subplots(1, 2, figsize=(25, 25))
    idx = random.randint(0, 7)
    axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    show_mask(gt[idx].cpu().numpy(), axs[0])
    show_box(bboxes[idx].numpy(), axs[0])
    axs[0].axis("off")
    axs[0].set_title(names_temp[idx])
    idx = random.randint(0, 7)
    axs[1].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    show_mask(gt[idx].cpu().numpy(), axs[1])
    show_box(bboxes[idx].numpy(), axs[1])
    axs[1].axis("off")
    axs[1].set_title(names_temp[idx])
    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.savefig("./data_sanitycheck.png", bbox_inches="tight", dpi=300)
    plt.close()
    break

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--tr_npy_path",
    type=str,
    default="data/npy/CT_Abd",
    help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument(
    "--val_npy_path",
    type=str,
    default="data/npy/CT_Abd_val",
    help="path to validation npy files; two subfolders: gts and imgs",
)
parser.add_argument("-task_name", type=str, default="MedSAM-ViT-B")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument(
    "-checkpoint", type=str, default="work_dir/SAM/sam_vit_b_01ec64.pth"
)
parser.add_argument("-work_dir", type=str, default="./work_dir")
parser.add_argument("-num_epochs", type=int, default=1000)
parser.add_argument("-batch_size", type=int, default=2)
parser.add_argument("-num_workers", type=int, default=4)
parser.add_argument("-weight_decay", type=float, default=0.01)
parser.add_argument("-lr", type=float, default=0.0001)
parser.add_argument("-use_wandb", type=bool, default=False)
parser.add_argument("-use_amp", action="store_true", default=False)
parser.add_argument("--resume", type=str, default="")
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

if args.use_wandb:
    import wandb
    wandb.login()
    wandb.init(
        project=args.task_name,
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "data_path": args.tr_npy_path,
            "val_path": args.val_npy_path,
            "model_type": args.model_type,
        },
    )

# %% set up model for training
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)

class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks


def validate(medsam_model, val_dataloader, seg_loss, ce_loss, device):
    """验证函数：返回平均 loss 和 平均 Dice"""
    medsam_model.eval()
    val_loss = 0
    dice_metric = monai.metrics.DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    post_pred = monai.transforms.AsDiscrete(argmax=False, threshold=0.5)
    post_gt = monai.transforms.AsDiscrete(to_onehot=None)

    with torch.no_grad():
        for step, (image, gt2D, boxes, _) in enumerate(val_dataloader):
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)
            medsam_pred = medsam_model(image, boxes_np)
            loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
            val_loss += loss.item()

            # 计算 Dice
            medsam_pred_sigmoid = torch.sigmoid(medsam_pred)
            medsam_pred_discrete = post_pred(medsam_pred_sigmoid)
            gt_discrete = post_gt(gt2D)
            dice_metric(y_pred=medsam_pred_discrete, y=gt_discrete)

    avg_val_loss = val_loss / len(val_dataloader)
    avg_val_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    medsam_model.train()
    return avg_val_loss, avg_val_dice


def check_early_stopping(val_dice_history, patience=100, min_delta=0.01):
    """检查是否满足早停条件"""
    if len(val_dice_history) < patience:
        return False
    
    # 取最近 patience 个 epoch
    recent_dices = val_dice_history[-patience:]
    max_dice = max(recent_dices)
    min_dice = min(recent_dices)
    improvement = max_dice - min_dice
    
    return improvement <= min_delta


def main():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    medsam_model.train()

    print("Number of total parameters: ", sum(p.numel() for p in medsam_model.parameters()))
    print("Number of trainable parameters: ", sum(p.numel() for p in medsam_model.parameters() if p.requires_grad))

    img_mask_encdec_params = list(medsam_model.image_encoder.parameters()) + list(medsam_model.mask_decoder.parameters())
    optimizer = torch.optim.AdamW(img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay)
    print("Number of image encoder and mask decoder parameters: ", sum(p.numel() for p in img_mask_encdec_params if p.requires_grad))

    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    # %% 加载训练和验证数据集
    train_dataset = NpyDataset(args.tr_npy_path)
    val_dataset = NpyDataset(args.val_npy_path)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 初始化
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_dices = []  # 👈 新增：记录验证集 Dice

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            medsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(f"Loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # %% train loop
    for epoch in range(start_epoch, args.num_epochs):
        medsam_model.train()
        epoch_loss = 0

        # 训练阶段
        for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")):
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)

            if args.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred = medsam_model(image, boxes_np)
                    loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                medsam_pred = medsam_model(image, boxes_np)
                loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # 验证阶段
        avg_val_loss, avg_val_dice = validate(medsam_model, val_dataloader, seg_loss, ce_loss, device)
        val_losses.append(avg_val_loss)
        val_dices.append(avg_val_dice)  # 👈 记录 Dice

        # 打印日志
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        print(f'Time: {current_time}, Epoch: {epoch+1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Val Dice: {avg_val_dice:.4f}')

        if args.use_wandb:
            wandb.log({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_dice": avg_val_dice,
                "epoch": epoch+1
            })

        # 保存最新模型
        checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_dice": avg_val_dice,
        }
        torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))

        # 保存最佳模型（基于验证损失）
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_best = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_dice": avg_val_dice,
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint_best, join(model_save_path, "medsam_model_best.pth"))
            print(f"🎉 New best validation loss: {best_val_loss:.6f} - Model saved!")

        # 👇👇👇 早停检查 👇👇👇
        if check_early_stopping(val_dices, patience=100, min_delta=0.01):
            print(f"🛑 Early stopping triggered at epoch {epoch+1}!")
            print(f"   Validation Dice improved by <= 0.01 in the last 100 epochs.")
            break

        # 绘制损失和 Dice 曲线
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(train_losses, label='Train Loss', color='red', marker='o', alpha=0.7)
        ax1.plot(val_losses, label='Val Loss', color='darkred', marker='s', alpha=0.7)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Dice', color=color)
        ax2.plot(val_dices, label='Val Dice', color='blue', marker='^')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')
        
        plt.title("Training and Validation Metrics")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(join(model_save_path, f"{args.task_name}_metrics_curve.png"), dpi=200, bbox_inches='tight')
        plt.close()

    print("✅ Training completed!")
    if epoch < args.num_epochs - 1:
        print("⏹️  Training stopped early due to no significant improvement in validation Dice.")
    print(f"🏆 Best validation loss: {best_val_loss:.6f}")
    if len(val_dices) > 0:
        print(f"🎯 Best validation Dice: {max(val_dices):.4f}")


if __name__ == "__main__":
    main()


"""
python train_with_early_stop.py \
  --tr_npy_path data/npy/train \
  --val_npy_path data/npy/val \
  --task_name MedSAM-Midbrain \
  --checkpoint sam_vit_b_01ec64.pth \
  --batch_size 4 \
  --num_epochs 1000 \  # 即使设1000，也会早停
  --device cuda:0
"""
