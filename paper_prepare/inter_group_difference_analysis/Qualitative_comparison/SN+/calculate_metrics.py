import os
import csv
import numpy as np
from PIL import Image
from sklearn.metrics import precision_score, recall_score


# ============================= 论文级配置（按需修改） =============================
THRESHOLD = 127                     # 灰度二值化阈值：>127 视为前景
SPACING = None
# 若你想把 ASSD/HD95 以 mm 输出，而不是像素，请设置：
# SPACING = (sx, sy)  # sx: x方向像素间距(mm/px), sy: y方向像素间距(mm/px)
# 注意：图像坐标为 (row, col) = (y, x)，所以 row 对应 sy，col 对应 sx。


# ============================= Surface-based metrics (no SciPy) =============================
def _binary_erosion_3x3(mask_bool: np.ndarray) -> np.ndarray:
    """
    2D 二值mask 的 3x3 腐蚀（8邻域结构元），不依赖 SciPy。
    mask_bool: bool, shape (H, W)
    return: bool, shape (H, W)
    """
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
    """
    提取二值mask的边界点坐标（surface points）
    使用：surface = mask XOR erosion(mask)
    mask_01: {0,1}, shape(H,W)
    return: int32, shape(N,2)  (row, col)
    """
    m = (mask_01 > 0).astype(bool)
    if m.sum() == 0:
        return np.zeros((0, 2), dtype=np.int32)
    eroded = _binary_erosion_3x3(m)
    surface = m ^ eroded
    return np.argwhere(surface).astype(np.int32)


def _min_distances(A: np.ndarray, B: np.ndarray, chunk: int = 4096, spacing=None) -> np.ndarray:
    """
    对 A 中每个点，计算到 B 的最小欧氏距离（分块，避免内存爆）。
    A: (Na,2)  (row,col)
    B: (Nb,2)
    spacing: None 或 (sx, sy)；若提供则输出为物理距离（mm）
    """
    if len(A) == 0 or len(B) == 0:
        return np.array([], dtype=np.float32)

    A = A.astype(np.float32)
    B = B.astype(np.float32)

    # 坐标缩放：row->sy, col->sx
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
        a = A_scaled[i:i + chunk]                 # (c,2)
        diff = a[:, None, :] - B_scaled[None, :, :]  # (c,Nb,2)
        d2 = (diff ** 2).sum(axis=2)              # (c,Nb)
        out[i:i + chunk] = np.sqrt(d2.min(axis=1)).astype(np.float32)

    return out


def assd_surface(pred_01: np.ndarray, gt_01: np.ndarray, spacing=None) -> float:
    """
    标准 ASSD（surface-to-surface, symmetric average）
    空集约定（论文更规范）：
      - pred空 且 gt空 -> 0
      - 仅一方空 -> inf
    """
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
    """
    标准 HD95（surface-to-surface, symmetric 95th percentile）
    空集约定同上：
      - pred空 且 gt空 -> 0
      - 仅一方空 -> inf
    """
    sp = _surface_points(pred_01)
    sg = _surface_points(gt_01)

    if len(sp) == 0 and len(sg) == 0:
        return 0.0
    if len(sp) == 0 or len(sg) == 0:
        return float("inf")

    d_sp_to_sg = _min_distances(sp, sg, spacing=spacing)
    d_sg_to_sp = _min_distances(sg, sp, spacing=spacing)
    return float(max(np.percentile(d_sp_to_sg, 95), np.percentile(d_sg_to_sp, 95)))


# ============================= Pixel metrics =============================
def _dice_iou(pred_01: np.ndarray, gt_01: np.ndarray):
    intersection = np.sum(pred_01 * gt_01)
    dice = (2.0 * intersection) / (np.sum(pred_01) + np.sum(gt_01) + 1e-8)
    union = np.sum(pred_01) + np.sum(gt_01) - intersection
    iou = intersection / (union + 1e-8)
    return float(dice), float(iou)


def calculate_metrics_from_binary(pred_01: np.ndarray, gt_01: np.ndarray, spacing=None):
    """
    pred_01/gt_01: {0,1}, shape(H,W)
    """
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


# ============================= Statistics helpers =============================
def _summary_stats(values: list[float]):
    """
    返回 mean, std, median, q1, q3 （忽略 nan）
    """
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
    """
    防止用户把 output_file 传成目录导致 IsADirectoryError。
    若是目录，则自动写入该目录下 metrics_report.txt
    """
    if os.path.isdir(output_file):
        return os.path.join(output_file, "metrics_report.txt")
    return output_file


# ============================= Folder evaluation =============================
def evaluate_png_folders(pred_folder: str, label_folder: str, output_file: str):
    output_file = _ensure_output_file(output_file)
    out_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else "."
    os.makedirs(out_dir, exist_ok=True)

    # ---- 修改：直接匹配同名 PNG 文件 ----
    pred_files = {f for f in os.listdir(pred_folder) if f.lower().endswith('.png')}
    gt_files = {f for f in os.listdir(label_folder) if f.lower().endswith('.png')}
    common_files = sorted(pred_files & gt_files)

    if len(common_files) == 0:
        raise RuntimeError(
            f"未找到可匹配的预测/标签对。\n"
            f"请确保两个文件夹中有同名的 .png 文件。\n"
            f"pred_folder={pred_folder}\n"
            f"label_folder={label_folder}"
        )

    print(f"✅ 找到 {len(common_files)} 对同名 PNG 文件")

    per_case_rows = []
    mismatch_cases = []

    both_empty = []
    pred_empty_only = []
    gt_empty_only = []
    metrics_all = []

    for filename in common_files:
        pred_path = os.path.join(pred_folder, filename)
        gt_path = os.path.join(label_folder, filename)

        try:
            pred = np.array(Image.open(pred_path).convert("L"), dtype=np.uint8)
            gt = np.array(Image.open(gt_path).convert("L"), dtype=np.uint8)

            if pred.shape != gt.shape:
                mismatch_cases.append(filename)
                print(f"⚠️ 尺寸不匹配，跳过 {filename}: pred={pred.shape}, gt={gt.shape}")
                continue

            pred_01 = (pred > THRESHOLD).astype(np.uint8)
            gt_01 = (gt > THRESHOLD).astype(np.uint8)

            p_sum = int(pred_01.sum())
            g_sum = int(gt_01.sum())
            if p_sum == 0 and g_sum == 0:
                both_empty.append(filename)
            elif p_sum == 0 and g_sum > 0:
                pred_empty_only.append(filename)
            elif p_sum > 0 and g_sum == 0:
                gt_empty_only.append(filename)

            m = calculate_metrics_from_binary(pred_01, gt_01, spacing=SPACING)
            metrics_all.append(m)

            row = {"CaseID": filename, "PredPath": filename, "GtPath": filename}
            row.update(m)
            per_case_rows.append(row)

            print(f"✓ {filename}")

        except Exception as e:
            print(f"❌ 处理 {filename} 出错: {e}")
            continue

    if len(metrics_all) == 0:
        raise RuntimeError("没有任何样本成功计算指标，请检查输入数据。")

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

    # ---- 保存 per-case CSV ----
    per_case_csv = os.path.join(out_dir, "per_case_metrics.csv")
    fieldnames = ["CaseID", "PredPath", "GtPath", "Dice", "IoU", "Precision", "Recall", "ASSD", "Hausdorff_Distance_95%"]
    with open(per_case_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_case_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # ---- 保存汇总报告 txt ----
    unit = "mm" if SPACING is not None else "pixel"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Segmentation Metrics Report (paper-grade)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Pred folder: {pred_folder}\n")
        f.write(f"GT folder  : {label_folder}\n")
        f.write(f"Matched pairs: {len(common_files)}\n")
        f.write(f"Evaluated cases (after shape check): {len(metrics_all)}\n")
        if mismatch_cases:
            f.write(f"Shape-mismatch skipped: {len(mismatch_cases)}\n")
        f.write("\n")

        f.write("Binarization:\n")
        f.write(f"  threshold: > {THRESHOLD} => foreground=1\n")
        f.write("\n")

        f.write("Surface distance settings:\n")
        f.write("  surface extraction: surface = mask XOR erosion(mask), 3x3 (8-connectivity)\n")
        if SPACING is None:
            f.write(f"  distance unit: {unit} (no spacing provided)\n")
        else:
            f.write(f"  distance unit: {unit} (spacing used)\n")
            f.write(f"  spacing: sx={SPACING[0]}, sy={SPACING[1]}\n")
        f.write("\n")

        f.write("Empty-mask policy (paper-grade):\n")
        f.write("  pred empty & gt empty => ASSD=0, HD95=0\n")
        f.write("  only one empty        => ASSD=inf, HD95=inf (excluded from mean/std/median stats)\n")
        f.write("\n")
        f.write(f"Empty cases:\n")
        f.write(f"  both empty: {len(both_empty)}\n")
        f.write(f"  pred empty only: {len(pred_empty_only)}\n")
        f.write(f"  gt empty only: {len(gt_empty_only)}\n")
        f.write("\n")

        f.write("Summary (Mean ± Std; Median [Q1, Q3])\n")
        f.write("-" * 80 + "\n")
        for metric_name, s in report.items():
            f.write(
                f"{metric_name}: "
                f"{s['mean']:.4f} ± {s['std']:.4f}; "
                f"{s['median']:.4f} [{s['q1']:.4f}, {s['q3']:.4f}]\n"
            )

        f.write("\n")
        f.write(f"Per-case CSV saved to: {per_case_csv}\n")

    print(f"\n✅ 完成！汇总报告: {output_file}")
    print(f"✅ 每例指标CSV: {per_case_csv}")
    if pred_empty_only or gt_empty_only:
        print(f"⚠️ 注意：存在“仅一方为空”的样本，ASSD/HD95 为 inf，已在统计时剔除。")


if __name__ == "__main__":
    pred_folder = "/data2/gaojiahao/paper_prepare/inter_group_difference_analysis/Qualitative_comparison/SN+/masks_1"
    label_folder = "/data2/gaojiahao/paper_prepare/inter_group_difference_analysis/Qualitative_comparison/SN+/masks_2"
    output_file = "/data2/gaojiahao/paper_prepare/inter_group_difference_analysis/Qualitative_comparison/SN+/intergroup_differences/metrics.txt"

    evaluate_png_folders(pred_folder, label_folder, output_file)