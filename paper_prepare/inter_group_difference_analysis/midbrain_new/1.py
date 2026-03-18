import os
import csv
import numpy as np
from PIL import Image
from sklearn.metrics import precision_score, recall_score


# ============================= 论文级配置（按需修改） =============================
THRESHOLD = 127
SPACING = None
# SPACING = (sx, sy)  # 如果要输出 mm，请设置像素间距；否则为 None 输出 pixel


# ============================= Surface-based metrics (no SciPy) =============================
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

    # row->sy, col->sx
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

    out = np.empty((A_scaled.shape[0],), dtype=np.float32)

    for i in range(0, A_scaled.shape[0], chunk):
        a = A_scaled[i:i + chunk]
        diff = a[:, None, :] - B_scaled[None, :, :]
        d2 = (diff ** 2).sum(axis=2)
        out[i:i + chunk] = np.sqrt(d2.min(axis=1)).astype(np.float32)

    return out


def assd_surface(a_01: np.ndarray, b_01: np.ndarray, spacing=None) -> float:
    sa = _surface_points(a_01)
    sb = _surface_points(b_01)

    # 论文级空集约定
    if len(sa) == 0 and len(sb) == 0:
        return 0.0
    if len(sa) == 0 or len(sb) == 0:
        return float("inf")

    d_a_to_b = _min_distances(sa, sb, spacing=spacing)
    d_b_to_a = _min_distances(sb, sa, spacing=spacing)
    return float((d_a_to_b.mean() + d_b_to_a.mean()) / 2.0)


def hd95_surface(a_01: np.ndarray, b_01: np.ndarray, spacing=None) -> float:
    sa = _surface_points(a_01)
    sb = _surface_points(b_01)

    if len(sa) == 0 and len(sb) == 0:
        return 0.0
    if len(sa) == 0 or len(sb) == 0:
        return float("inf")

    d_a_to_b = _min_distances(sa, sb, spacing=spacing)
    d_b_to_a = _min_distances(sb, sa, spacing=spacing)
    return float(max(np.percentile(d_a_to_b, 95), np.percentile(d_b_to_a, 95)))


# ============================= Pixel metrics =============================
def _dice_iou(a_01: np.ndarray, b_01: np.ndarray):
    intersection = np.sum(a_01 * b_01)
    dice = (2.0 * intersection) / (np.sum(a_01) + np.sum(b_01) + 1e-8)
    union = np.sum(a_01) + np.sum(b_01) - intersection
    iou = intersection / (union + 1e-8)
    return float(dice), float(iou)


def calculate_metrics_between_two_masks(a_01: np.ndarray, b_01: np.ndarray, spacing=None):
    """
    两位标注者一致性指标。
    Precision/Recall 在观察者间不是核心（方向性），但可保留作补充。
    """
    dice, iou = _dice_iou(a_01, b_01)

    # 这里把 A 当作“pred”，B 当作“ref”
    precision = precision_score(b_01.flatten(), a_01.flatten(), zero_division=0)
    recall = recall_score(b_01.flatten(), a_01.flatten(), zero_division=0)

    assd = assd_surface(a_01, b_01, spacing=spacing)
    hd95 = hd95_surface(a_01, b_01, spacing=spacing)

    return {
        "Dice": float(dice),
        "IoU": float(iou),
        "Precision(A_vs_B)": float(precision),
        "Recall(A_vs_B)": float(recall),
        "ASSD": float(assd),
        "Hausdorff_Distance_95%": float(hd95),
    }


# ============================= Statistics helpers =============================
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


# ============================= Folder evaluation (Annotator A vs B) =============================
def evaluate_two_label_folders(folder_a: str, folder_b: str, output_file: str):
    """
    两个文件夹内：同一病例/图像的标签文件名完全一致（如 midbrain_001.png）
    """
    output_file = _ensure_output_file(output_file)
    out_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else "."
    os.makedirs(out_dir, exist_ok=True)

    # 只统计 png（如你还有 tif，可自行加）
    a_files = {f for f in os.listdir(folder_a) if f.lower().endswith(".png")}
    b_files = {f for f in os.listdir(folder_b) if f.lower().endswith(".png")}

    common = sorted(a_files & b_files)
    only_a = sorted(a_files - b_files)
    only_b = sorted(b_files - a_files)

    if len(common) == 0:
        raise RuntimeError(
            f"两个文件夹中未找到同名 PNG 文件。\nfolder_a={folder_a}\nfolder_b={folder_b}"
        )

    print(f"✅ A文件数: {len(a_files)} | B文件数: {len(b_files)} | 同名匹配: {len(common)}")

    per_case_rows = []
    mismatch_cases = []

    both_empty = []
    a_empty_only = []
    b_empty_only = []

    metrics_all = []

    for fname in common:
        a_path = os.path.join(folder_a, fname)
        b_path = os.path.join(folder_b, fname)

        try:
            a = np.array(Image.open(a_path).convert("L"), dtype=np.uint8)
            b = np.array(Image.open(b_path).convert("L"), dtype=np.uint8)

            if a.shape != b.shape:
                mismatch_cases.append(fname)
                print(f"⚠️ 尺寸不匹配，跳过 {fname}: A={a.shape}, B={b.shape}")
                continue

            a_01 = (a > THRESHOLD).astype(np.uint8)
            b_01 = (b > THRESHOLD).astype(np.uint8)

            a_sum = int(a_01.sum())
            b_sum = int(b_01.sum())
            if a_sum == 0 and b_sum == 0:
                both_empty.append(fname)
            elif a_sum == 0 and b_sum > 0:
                a_empty_only.append(fname)
            elif a_sum > 0 and b_sum == 0:
                b_empty_only.append(fname)

            m = calculate_metrics_between_two_masks(a_01, b_01, spacing=SPACING)
            metrics_all.append(m)

            row = {"FileName": fname}
            row.update(m)
            per_case_rows.append(row)

        except Exception as e:
            print(f"❌ 处理 {fname} 出错: {e}")
            continue

    if len(metrics_all) == 0:
        raise RuntimeError("没有任何样本成功计算指标，请检查输入数据。")

    def finite_filter(vals):
        return [v for v in vals if np.isfinite(v)]

    report = {}
    for metric_name in metrics_all[0].keys():
        vals = [m[metric_name] for m in metrics_all]
        if metric_name in ("ASSD", "Hausdorff_Distance_95%"):
            vals_used = finite_filter(vals)  # 剔除 inf
        else:
            vals_used = [v for v in vals if not np.isnan(v)]
        report[metric_name] = _summary_stats(vals_used)

    # ---- per-case CSV ----
    per_case_csv = os.path.join(out_dir, "per_case_interobserver_metrics.csv")
    fieldnames = ["FileName", "Dice", "IoU", "Precision(A_vs_B)", "Recall(A_vs_B)", "ASSD", "Hausdorff_Distance_95%"]
    with open(per_case_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_case_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # ---- report txt ----
    unit = "mm" if SPACING is not None else "pixel"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Inter-Observer Segmentation Agreement Report (paper-grade)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Annotator A folder: {folder_a}\n")
        f.write(f"Annotator B folder: {folder_b}\n\n")

        f.write("File matching:\n")
        f.write(f"  A files: {len(a_files)}\n")
        f.write(f"  B files: {len(b_files)}\n")
        f.write(f"  matched (same filename): {len(common)}\n")
        f.write(f"  only in A: {len(only_a)}\n")
        f.write(f"  only in B: {len(only_b)}\n")
        f.write(f"  evaluated (after shape check): {len(metrics_all)}\n")
        if mismatch_cases:
            f.write(f"  shape-mismatch skipped: {len(mismatch_cases)}\n")
        f.write("\n")

        f.write("Binarization:\n")
        f.write(f"  threshold: > {THRESHOLD} => foreground=1\n\n")

        f.write("Surface distance settings:\n")
        f.write("  surface extraction: surface = mask XOR erosion(mask), 3x3 (8-connectivity)\n")
        f.write(f"  distance unit: {unit}\n")
        if SPACING is not None:
            f.write(f"  spacing: sx={SPACING[0]}, sy={SPACING[1]}\n")
        f.write("\n")

        f.write("Empty-mask policy (paper-grade):\n")
        f.write("  A empty & B empty => ASSD=0, HD95=0\n")
        f.write("  only one empty    => ASSD=inf, HD95=inf (excluded from mean/std/median stats)\n\n")

        f.write("Empty cases:\n")
        f.write(f"  both empty: {len(both_empty)}\n")
        f.write(f"  A empty only: {len(a_empty_only)}\n")
        f.write(f"  B empty only: {len(b_empty_only)}\n\n")

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

        # 如需要，把 only_a / only_b 的文件名也写进报告（可选）
        if only_a:
            f.write("\nFiles only in A (first 50 shown):\n")
            for n in only_a[:50]:
                f.write(f"  {n}\n")
        if only_b:
            f.write("\nFiles only in B (first 50 shown):\n")
            for n in only_b[:50]:
                f.write(f"  {n}\n")

    print(f"\n✅ 完成！汇总报告: {output_file}")
    print(f"✅ 每例指标CSV: {per_case_csv}")
    if a_empty_only or b_empty_only:
        print("⚠️ 注意：存在“仅一方为空”的样本，ASSD/HD95 为 inf，已在统计时剔除。")


if __name__ == "__main__":
    # ========= 在此处指定两位标注者的标签文件夹 =========
    folder_a = "/data2/gaojiahao/paper_prepare/inter_group_difference_analysis/midbrain/stu_masks"
    folder_b = "/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/0_midbrain_data/raw_data/masks"

    # 输出文件（或输出目录也行）
    output_file = "/data2/gaojiahao/paper_prepare/inter_group_difference_analysis/midbrain_new/interobserver_metrics.txt"

    evaluate_two_label_folders(folder_a, folder_b, output_file)
