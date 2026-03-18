import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import transforms as T
from nets.segformer import SegFormer
from my_dataset import Midbrain_Test
import argparse
from pathlib import Path
import csv


# ================== 配置 ==================
# 与 SegFormer 训练完全一致的归一化参数
MEAN = (0.16882014, 0.16886712, 0.16884266)
STD = (0.2274119, 0.22743388, 0.22739229)
CROP_SIZE = 512

# 输出掩码后缀（与指标脚本一致）
PRED_SUFFIX = "_image_predict.png"
GT_SUFFIX = "_mask.png"

# 空间间距（若已知物理尺寸，可设为 (sx, sy)）
SPACING = None
# =========================================


class TestDataset(Dataset):
    """用于加载测试集图像"""
    def __init__(self, img_paths, transforms=None):
        self.img_paths = img_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        # 创建 dummy mask（仅用于 transform 兼容）
        dummy_mask = Image.fromarray(np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8))
        if self.transforms:
            img, dummy_mask = self.transforms(img, dummy_mask)
        return img, str(img_path)


def get_test_transform():
    """与训练验证阶段完全一致的预处理"""
    return T.Compose([
        # T.CenterCrop(CROP_SIZE),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


def load_model(model_path, num_classes=2, phi="b3", device="cuda:0"):
    """加载训练好的 SegFormer 模型"""
    model = SegFormer(num_classes=num_classes, phi=phi, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    # 兼容不同保存格式
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def single_model_predict(model, dataloader, device, save_dir):
    """
    SegFormer 单模型推理并保存掩码
    输出格式: {case_id}_image_predict.png
    """
    os.makedirs(save_dir, exist_ok=True)
    for images, img_paths in dataloader:
        images = images.to(device, non_blocking=True)  # [B, 3, H, W]
        B = images.shape[0]

        # 前向推理 (SegFormer 输出为 dict)
        output_dict = model(images)  # {'out': [B, 2, H, W]}
        
        # 提取 logits
        if isinstance(output_dict, dict):
            if "out" in output_dict:
                logits = output_dict["out"]
            else:
                # 取第一个 tensor
                for v in output_dict.values():
                    if torch.is_tensor(v):
                        logits = v
                        break
                else:
                    raise TypeError(f"Unexpected SegFormer output: {output_dict.keys()}")
        else:
            logits = output_dict

        # 转为离散掩码 [B, H, W] (0/1)
        pred_masks = torch.argmax(logits, dim=1).cpu().numpy()

        for i in range(B):
            pred_mask = pred_masks[i]
            orig_stem = Path(img_paths[i]).stem
            
            # 按你的命名规则：移除 "_image" 得到 case_id
            case_id = orig_stem.replace("_image", "")
            save_name = case_id + PRED_SUFFIX
            save_path = os.path.join(save_dir, save_name)
            
            # 保存为 0/255 的 PNG
            Image.fromarray((pred_mask * 255).astype(np.uint8), mode='L').save(save_path)
            print(f"[Saved] {save_name}")


# ============================= 分割指标函数（完全保留） =============================
from sklearn.metrics import precision_score, recall_score

def _binary_erosion_3x3(mask_bool: np.ndarray) -> np.ndarray:
    if mask_bool.ndim != 2:
        raise ValueError(f"Only 2D supported, got {mask_bool.ndim}D")
    padded = np.pad(mask_bool, ((1, 1), (1, 1)), mode="constant", constant_values=False)
    eroded = np.ones_like(mask_bool, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            eroded &= padded[1 + dy:1 + dy + mask_bool.shape[0],
                             1 + dx:1 + dx + mask_bool.shape[1]]
    return eroded

def _surface_points(mask_01: np.ndarray) -> np.ndarray:
    m = (mask_01 > 0).astype(bool)
    if m.sum() == 0:
        return np.zeros((0, 2), dtype=np.int32)
    eroded = _binary_erosion_3x3(m)
    surface = m ^ eroded
    return np.argwhere(surface).astype(np.int32)

def _min_distances(A: np.ndarray, B: np.ndarray, chunk: int = 4096, spacing=None) -> np.ndarray:
    if len(A) == 0 or len(B) == 0:
        return np.array([], dtype=np.float32)
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    if spacing is not None:
        sx, sy = float(spacing[0]), float(spacing[1])
        A_scaled = A.copy()
        B_scaled = B.copy()
        A_scaled[:, 0] *= sy
        A_scaled[:, 1] *= sx
        B_scaled[:, 0] *= sy
        B_scaled[:, 1] *= sx
    else:
        A_scaled = A
        B_scaled = B
    out = np.empty((A.shape[0],), dtype=np.float32)
    for i in range(0, A_scaled.shape[0], chunk):
        a = A_scaled[i:i + chunk]
        diff = a[:, None, :] - B_scaled[None, :, :]
        d2 = (diff ** 2).sum(axis=2)
        out[i:i + chunk] = np.sqrt(d2.min(axis=1)).astype(np.float32)
    return out

def assd_surface(pred_01: np.ndarray, gt_01: np.ndarray, spacing=None) -> float:
    sp = _surface_points(pred_01)
    sg = _surface_points(gt_01)
    if len(sp) == 0 and len(sg) == 0:
        return 0.0
    if len(sp) == 0 or len(sg) == 0:
        return float("inf")
    d_sp_to_sg = _min_distances(sp, sg, spacing=spacing)
    d_sg_to_sp = _min_distances(sg, sp, spacing=spacing)
    return float((d_sp_to_sg.mean() + d_sg_to_sp.mean()) / 2.0)

def hd95_surface(pred_01: np.ndarray, gt_01: np.ndarray, spacing=None) -> float:
    sp = _surface_points(pred_01)
    sg = _surface_points(gt_01)
    if len(sp) == 0 and len(sg) == 0:
        return 0.0
    if len(sp) == 0 or len(sg) == 0:
        return float("inf")
    d_sp_to_sg = _min_distances(sp, sg, spacing=spacing)
    d_sg_to_sp = _min_distances(sg, sp, spacing=spacing)
    return float(max(np.percentile(d_sp_to_sg, 95), np.percentile(d_sg_to_sp, 95)))

def _dice_iou(pred_01: np.ndarray, gt_01: np.ndarray):
    intersection = np.sum(pred_01 * gt_01)
    dice = (2.0 * intersection) / (np.sum(pred_01) + np.sum(gt_01) + 1e-8)
    union = np.sum(pred_01) + np.sum(gt_01) - intersection
    iou = intersection / (union + 1e-8)
    return float(dice), float(iou)

def calculate_metrics_from_binary(pred_01: np.ndarray, gt_01: np.ndarray, spacing=None):
    dice, iou = _dice_iou(pred_01, gt_01)
    precision = precision_score(gt_01.flatten(), pred_01.flatten(), zero_division=0)
    recall = recall_score(gt_01.flatten(), pred_01.flatten(), zero_division=0)
    assd = assd_surface(pred_01, gt_01, spacing=spacing)
    hd95 = hd95_surface(pred_01, gt_01, spacing=spacing)
    return {
        "Dice": float(dice),
        "IoU": float(iou),
        "Precision": float(precision),
        "Recall": float(recall),
        "ASSD": float(assd),
        "Hausdorff_Distance_95%": float(hd95),
    }

def _summary_stats(values: list[float]):
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return dict(mean=np.nan, std=np.nan, median=np.nan, q1=np.nan, q3=np.nan)
    return dict(
        mean=float(np.mean(v)),
        std=float(np.std(v, ddof=1)) if v.size >= 2 else 0.0,
        median=float(np.median(v)),
        q1=float(np.percentile(v, 25)),
        q3=float(np.percentile(v, 75)),
    )

def _ensure_output_file(output_file: str) -> str:
    if os.path.isdir(output_file):
        return os.path.join(output_file, "metrics_report.txt")
    return output_file

def evaluate_png_folders(pred_folder: str, label_folder: str, output_file: str, spacing=None):
    output_file = _ensure_output_file(output_file)
    out_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else "."
    os.makedirs(out_dir, exist_ok=True)

    pred_map = {}
    for f in os.listdir(pred_folder):
        if f.endswith(PRED_SUFFIX):
            key = f[:-len(PRED_SUFFIX)]
            pred_map[key] = f

    gt_map = {}
    for f in os.listdir(label_folder):
        if f.endswith(GT_SUFFIX):
            key = f[:-len(GT_SUFFIX)]
            gt_map[key] = f

    common_keys = sorted(set(pred_map.keys()) & set(gt_map.keys()))
    if len(common_keys) == 0:
        raise RuntimeError("未找到可匹配的预测/标签对。")

    print(f"✅ 找到 {len(common_keys)} 对可匹配样本")

    per_case_rows = []
    metrics_all = []

    for k in common_keys:
        pred_path = os.path.join(pred_folder, pred_map[k])
        gt_path = os.path.join(label_folder, gt_map[k])

        try:
            pred = np.array(Image.open(pred_path).convert("L"), dtype=np.uint8)
            gt = np.array(Image.open(gt_path).convert("L"), dtype=np.uint8)

            if pred.shape != gt.shape:
                print(f"⚠️ 尺寸不匹配，跳过 {k}")
                continue

            pred_01 = (pred > 127).astype(np.uint8)
            gt_01 = (gt > 127).astype(np.uint8)

            m = calculate_metrics_from_binary(pred_01, gt_01, spacing=spacing)
            metrics_all.append(m)

            row = {"CaseID": k, "PredPath": pred_map[k], "GtPath": gt_map[k]}
            row.update(m)
            per_case_rows.append(row)

            print(f"✓ {k}")

        except Exception as e:
            print(f"❌ 处理 {k} 出错: {e}")
            continue

    if len(metrics_all) == 0:
        raise RuntimeError("没有任何样本成功计算指标。")

    def finite_filter(vals):
        return [v for v in vals if np.isfinite(v)]

    report = {}
    for metric_name in metrics_all[0].keys():
        vals = [m[metric_name] for m in metrics_all]
        if metric_name in ("ASSD", "Hausdorff_Distance_95%"):
            vals_used = finite_filter(vals)
        else:
            vals_used = [v for v in vals if not np.isnan(v)]
        report[metric_name] = _summary_stats(vals_used)

    per_case_csv = os.path.join(out_dir, "per_case_metrics.csv")
    fieldnames = ["CaseID", "PredPath", "GtPath", "Dice", "IoU", "Precision", "Recall", "ASSD", "Hausdorff_Distance_95%"]
    with open(per_case_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_case_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    unit = "mm" if spacing is not None else "pixel"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("SegFormer Model Segmentation Metrics Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Pred folder: {pred_folder}\n")
        f.write(f"GT folder  : {label_folder}\n")
        f.write(f"Matched pairs: {len(common_keys)}\n")
        f.write(f"Spacing: {spacing}\n")
        f.write("\nSummary (Mean ± Std; Median [Q1, Q3])\n")
        f.write("-" * 80 + "\n")
        for metric_name, s in report.items():
            f.write(
                f"{metric_name}: "
                f"{s['mean']:.4f} ± {s['std']:.4f}; "
                f"{s['median']:.4f} [{s['q1']:.4f}, {s['q3']:.4f}]\n"
            )
        f.write(f"\nPer-case CSV saved to: {per_case_csv}\n")

    print(f"\n✅ 汇总报告: {output_file}")
    print(f"✅ 每例指标: {per_case_csv}")


# ============================= 主函数 =============================
def main():
    parser = argparse.ArgumentParser(description="SegFormer inference for substantia nigra segmentation")
    parser.add_argument("--data-path", type=str, 
                        help="Root path of dataset (same as training, e.g., '../')",
                        default="/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation")
    parser.add_argument("--test-names-file", type=str,
                        help="Path to test_set.txt containing TEST IMAGE FILENAMES (e.g., 'substantia-nigra001_image.png')",
                        default="/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/2_segformer/5fold_save_result/fold_indices/fold_4_test.txt")
    parser.add_argument("--weights", type=str,
                        help="Path to SegFormer model checkpoint (.pth)",
                        default="/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/2_segformer/save_weights/5fold/fold_4/best_model.pth")
    parser.add_argument("--gt-mask-dir", type=str, 
                        help="Directory of ground truth masks (for evaluation)",
                        default="/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/0_midbrain_data/Test(512*512)groundtruth/fold4/masks")
    parser.add_argument("--phi", type=str, default="b3",
                        help="SegFormer variant (b0, b1, b2, b3, b4, b5)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (e.g., cuda:0, cpu)")
    parser.add_argument("--pred-save-dir", type=str, 
                        help="Directory to save predicted masks",
                        default="/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/2_segformer/predict_results/predicts/iteration5")
    parser.add_argument("--metrics-output", type=str,
                        help="Path to save metrics report (e.g., ./metrics.txt)",
                        default="/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/2_segformer/predict_results/test_metrics/iteration5")
    parser.add_argument("--spacing", nargs=2, type=float, default=None,
                        help="Pixel spacing (sx sy) for physical distance metrics")
    args = parser.parse_args()

    # Step 1: Load test image filenames from test_set.txt
    with open(args.test_names_file, "r") as f:
        test_names = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(test_names)} test images from {args.test_names_file}")

    # Step 2: Get full image paths using the same dataset class as training
    full_dataset = Midbrain_Test(args.data_path, transforms=None)
    name_to_path = {os.path.basename(p): p for p in full_dataset.img_list}
    
    # Validate all test images exist
    missing = [name for name in test_names if name not in name_to_path]
    if missing:
        raise FileNotFoundError(f"Missing test images in dataset: {missing[:5]}")

    test_img_paths = [name_to_path[name] for name in test_names]

    # Step 3: Create test dataloader
    test_transform = get_test_transform()
    test_dataset = TestDataset(test_img_paths, transforms=test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=min(4, os.cpu_count()), pin_memory=True
    )

    # Step 4: Load trained SegFormer model
    model = load_model(
        args.weights, 
        num_classes=2, 
        phi=args.phi, 
        device=args.device
    )
    print(f"Loaded SegFormer-{args.phi} model from {args.weights}")

    # Step 5: Run inference and save predictions
    single_model_predict(model, test_loader, args.device, args.pred_save_dir)
    print(f"Predictions saved to: {args.pred_save_dir}")

    # Step 6: Evaluate against ground truth
    evaluate_png_folders(
        pred_folder=args.pred_save_dir,
        label_folder=args.gt_mask_dir,
        output_file=args.metrics_output,
        spacing=args.spacing
    )


if __name__ == "__main__":
    main()