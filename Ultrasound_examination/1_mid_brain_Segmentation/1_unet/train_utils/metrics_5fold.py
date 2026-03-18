import torch
import numpy as np
from PIL import Image
import os
from sklearn.metrics import precision_score, recall_score


# ----------------------------- Surface-based metrics (no SciPy) -----------------------------
def _binary_erosion_3x3(mask_bool: np.ndarray) -> np.ndarray:
    """
    2D 二值mask 的 3x3 腐蚀（不依赖 scipy）。
    mask_bool: bool array, shape (H, W)
    return: eroded bool array, shape (H, W)
    """
    if mask_bool.ndim != 2:
        raise ValueError(f"Only 2D supported, got {mask_bool.ndim}D")

    padded = np.pad(mask_bool, ((1, 1), (1, 1)), mode="constant", constant_values=False)
    eroded = np.ones_like(mask_bool, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            eroded &= padded[1 + dy:1 + dy + mask_bool.shape[0], 1 + dx:1 + dx + mask_bool.shape[1]]
    return eroded


def _surface_points(mask_01: np.ndarray) -> np.ndarray:
    """
    提取二值mask的边界点（surface points）坐标。
    mask_01: 2D {0,1}
    return: N x 2 int32
    """
    m = (mask_01 > 0).astype(bool)
    if m.sum() == 0:
        return np.zeros((0, 2), dtype=np.int32)
    eroded = _binary_erosion_3x3(m)
    surface = m ^ eroded
    return np.argwhere(surface).astype(np.int32)


def _min_distances(A: np.ndarray, B: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """
    对 A 中每个点，计算到 B 的最小欧氏距离（分块，避免内存爆）。
    A: (Na, 2), B: (Nb, 2)
    return: (Na,)
    """
    if len(A) == 0 or len(B) == 0:
        return np.array([], dtype=np.float32)

    A = A.astype(np.float32)
    B = B.astype(np.float32)
    out = np.empty((A.shape[0],), dtype=np.float32)

    for i in range(0, A.shape[0], chunk):
        a = A[i:i + chunk]
        diff = a[:, None, :] - B[None, :, :]
        d2 = (diff ** 2).sum(axis=2)
        out[i:i + chunk] = np.sqrt(d2.min(axis=1)).astype(np.float32)
    return out


def calculate_assd_surface(pred_01: np.ndarray, gt_01: np.ndarray) -> float:
    """
    标准 ASSD：基于边界点 surface-to-surface 的 Average Symmetric Surface Distance
    """
    sp = _surface_points(pred_01)
    sg = _surface_points(gt_01)
    if len(sp) == 0 or len(sg) == 0:
        return float("inf")
    d_sp_to_sg = _min_distances(sp, sg)
    d_sg_to_sp = _min_distances(sg, sp)
    return float((d_sp_to_sg.mean() + d_sg_to_sp.mean()) / 2.0)


def calculate_hd95_surface(pred_01: np.ndarray, gt_01: np.ndarray) -> float:
    """
    标准 HD95：基于边界点 surface-to-surface 的 95% Hausdorff Distance
    """
    sp = _surface_points(pred_01)
    sg = _surface_points(gt_01)
    if len(sp) == 0 or len(sg) == 0:
        return float("inf")
    d_sp_to_sg = _min_distances(sp, sg)
    d_sg_to_sp = _min_distances(sg, sp)
    return float(max(np.percentile(d_sp_to_sg, 95), np.percentile(d_sg_to_sp, 95)))


# ----------------------------- Main metrics (online eval) -----------------------------
def calculate_metrics(pred_mask, true_mask):
    """
    计算分割指标（Dice, IoU, Precision, Recall, ASSD(surface), HD95(surface)）
    pred_mask / true_mask:
        - 可以是概率图/灰度图（0~1 或 0~255），内部会二值化
    """
    # 二值化为 0/1
    pred_01 = (pred_mask > 0.5).astype(np.uint8)
    gt_01 = (true_mask > 0.5).astype(np.uint8)

    # Dice
    intersection = np.sum(pred_01 * gt_01)
    dice = (2.0 * intersection) / (np.sum(pred_01) + np.sum(gt_01) + 1e-8)

    # IoU
    union = np.sum(pred_01) + np.sum(gt_01) - intersection
    iou = intersection / (union + 1e-8)

    # Precision / Recall
    precision = precision_score(gt_01.flatten(), pred_01.flatten(), zero_division=0)
    recall = recall_score(gt_01.flatten(), pred_01.flatten(), zero_division=0)

    # ASSD / HD95（surface-based）
    assd = calculate_assd_surface(pred_01, gt_01)
    hd95 = calculate_hd95_surface(pred_01, gt_01)

    return {
        "Dice": float(dice),
        "IoU": float(iou),
        "Precision": float(precision),
        "Recall": float(recall),
        "ASSD": float(assd),
        "Hausdorff_Distance_95%": float(hd95),
    }


def calculate_test_metrics(model, data_loader, device, save_dir=None):
    """
    在线评估：从 dataloader 读取 images/targets，跑 model 得到 pred，然后逐样本算指标并取平均。
    注意：ASSD/HD95 的平均会自动剔除 inf（空mask/无边界导致）。
    """
    model.eval()
    metrics_list = []
    image_idx = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)

            # targets 形状可能是 [B,H,W] 或 [B,1,H,W] 或 one-hot 等；这里只按二分类mask来处理
            targets_np = targets.detach().cpu().numpy()

            outputs = model(images)

            # 兼容 torchvision segmentation 风格：outputs['out'] = [B, C, H, W]
            if isinstance(outputs, dict) and "out" in outputs:
                logits = outputs["out"]
            else:
                logits = outputs

            # 二分类：取前景通道概率
            probs_fg = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()  # [B,H,W]

            # 对齐 target 维度：确保是 [B,H,W]
            if targets_np.ndim == 4 and targets_np.shape[1] == 1:
                targets_np = targets_np[:, 0]
            elif targets_np.ndim == 4 and targets_np.shape[1] > 1:
                # 如果是 one-hot / 多类，这里默认前景在 channel=1（你可按实际改）
                targets_np = targets_np[:, 1]

            for pred, gt in zip(probs_fg, targets_np):
                m = calculate_metrics(pred.squeeze(), gt.squeeze())
                metrics_list.append(m)

                # 可选：保存二值mask（保持你原来结构，默认注释/不启用）
                if save_dir is not None:
                    os.makedirs(save_dir, exist_ok=True)
                    pred_path = os.path.join(save_dir, f"pred_{image_idx}.png")
                    gt_path = os.path.join(save_dir, f"label_{image_idx}.png")
                    save_prediction_mask(pred, pred_path)
                    save_prediction_mask(gt, gt_path)
                    image_idx += 1

    if len(metrics_list) == 0:
        raise RuntimeError("No samples were evaluated. Please check your dataloader/model output.")

    # 计算平均指标：ASSD/HD95 剔除 inf；其余剔除 nan
    avg_metrics = {}
    keys = metrics_list[0].keys()
    for k in keys:
        vals = [m[k] for m in metrics_list]
        if k in ("ASSD", "Hausdorff_Distance_95%"):
            valid = [v for v in vals if np.isfinite(v)]
        else:
            valid = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        avg_metrics[k] = float(np.mean(valid)) if len(valid) > 0 else float("nan")

    return avg_metrics


def save_prediction_mask(mask, path):
    """
    保存单张 mask 为 PNG（前景=255, 背景=0）
    mask: 可以是概率图或二值图
    """
    mask_01 = (mask > 0.5).astype(np.uint8)
    img = Image.fromarray(mask_01 * 255)
    img.save(path)
