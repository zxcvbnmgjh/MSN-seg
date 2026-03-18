import os
import cv2
import numpy as np

# ---------------------------
# 配置：把这4个路径改成你的

# ---------------------------
FOLDER1_IMAGES = "/data2/gaojiahao/paper_prepare/inter_group_difference_analysis/Qualitative_comparison/SN+/images"   # 文件夹1：前缀_image.png
FOLDER2_GT        = "/data2/gaojiahao/paper_prepare/inter_group_difference_analysis/Qualitative_comparison/SN+/masks_1"       # 文件夹2：前缀_mask_A.png（真值）
FOLDER3_PRED      = "/data2/gaojiahao/paper_prepare/inter_group_difference_analysis/Qualitative_comparison/SN+/masks_2"     # 文件夹3：前缀_mask_B.png（预测）
FOLDER4_OUT       = "/data2/gaojiahao/paper_prepare/inter_group_difference_analysis/Qualitative_comparison/SN+/overlaps"      # 文件夹4：输出前缀_label_comparision.png

IMG_SUFFIX  = "_image.png"
GT_SUFFIX   = "_mask_A.png"
PRED_SUFFIX = "_mask_B.png"
OUT_SUFFIX  = "_label_comparison.png"

# =========================
# 颜色与样式（OpenCV 是 BGR）
# =========================
GT_COLOR   = (0, 0, 255)  # 黄色 Yellow
PRED_COLOR = (0, 255, 0)  # 青色 Cyan

THICKNESS_GT = 2
THICKNESS_PD = 2

ALPHA_GT = 0.9   # 轮廓透明度（只作用在轮廓像素）
ALPHA_PD = 0.9


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def extract_external_contours_from_mask(mask_gray: np.ndarray):
    """
    从二值灰度mask中提取外轮廓（前景255/背景0）。
    只返回轮廓，不做填充。
    """
    if mask_gray is None:
        return []

    if mask_gray.ndim == 3:
        mask_gray = cv2.cvtColor(mask_gray, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def blend_contours_only(base_bgr: np.ndarray,
                        contours,
                        color_bgr,
                        thickness: int = 2,
                        alpha: float = 0.85):
    """
    关键函数：只在“轮廓像素位置”进行 alpha 融合，其他像素完全不变。
    base_bgr: 原始输入图像（不会整体变亮）
    """
    if not contours:
        return base_bgr

    h, w = base_bgr.shape[:2]

    # 1) 做一张“轮廓掩码”（单通道），轮廓位置为255，其余为0
    contour_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(contour_mask, contours, -1, 255, thickness=thickness)

    # 2) 做一张“纯色层”（只用于轮廓像素）
    color_layer = np.zeros_like(base_bgr, dtype=np.uint8)
    color_layer[:] = color_bgr

    # 3) 只在 mask>0 的位置做 alpha blend
    out = base_bgr.copy()
    idx = contour_mask > 0

    # out = (1-alpha)*base + alpha*color  （仅轮廓处）
    out[idx] = ((1.0 - alpha) * out[idx] + alpha * color_layer[idx]).astype(np.uint8)

    return out


def process_one(prefix: str,
                img_path: str,
                gt_path: str,
                pred_path: str,
                out_path: str):
    # 读输入图（保持原样：不整体改变）
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"❌ 无法读取输入图像: {img_path}")
        return False

    # 读GT/Pred（灰度二值）
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        print(f"⚠️ 缺少真值: {gt_path}")
        return False
    if pred is None:
        print(f"⚠️ 缺少预测: {pred_path}")
        return False

    # 尺寸必须一致（否则轮廓位置会错）
    if gt.shape[:2] != img.shape[:2]:
        print(f"⚠️ 尺寸不一致: {prefix} | img={img.shape[:2]} gt={gt.shape[:2]}")
        return False
    if pred.shape[:2] != img.shape[:2]:
        print(f"⚠️ 尺寸不一致: {prefix} | img={img.shape[:2]} pred={pred.shape[:2]}")
        return False

    # 提取轮廓（只前景，且只轮廓，不要填充）
    contours_gt = extract_external_contours_from_mask(gt)
    contours_pd = extract_external_contours_from_mask(pred)

    # 叠加：先叠 GT（黄），再叠 Pred（青）
    out = img.copy()
    out = blend_contours_only(out, contours_gt, GT_COLOR, thickness=THICKNESS_GT, alpha=ALPHA_GT)
    out = blend_contours_only(out, contours_pd, PRED_COLOR, thickness=THICKNESS_PD, alpha=ALPHA_PD)

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, out)
    return True


def batch_overlay(folder1_images, folder2_gt, folder3_pred, folder4_out):
    ensure_dir(folder4_out)

    img_files = [f for f in os.listdir(folder1_images) if f.lower().endswith(IMG_SUFFIX)]
    if not img_files:
        print(f"⚠️ 未找到任何 *{IMG_SUFFIX} 于: {folder1_images}")
        return

    ok = 0
    total = len(img_files)

    for f in sorted(img_files):
        prefix = f[:-len(IMG_SUFFIX)]

        img_path  = os.path.join(folder1_images, f"{prefix}{IMG_SUFFIX}")
        gt_path   = os.path.join(folder2_gt,     f"{prefix}{GT_SUFFIX}")
        pred_path = os.path.join(folder3_pred,   f"{prefix}{PRED_SUFFIX}")
        out_path  = os.path.join(folder4_out,    f"{prefix}{OUT_SUFFIX}")

        if process_one(prefix, img_path, gt_path, pred_path, out_path):
            ok += 1

    print(f"✅ 完成：输入 {total} 张，成功输出 {ok} 张到：{folder4_out}")


if __name__ == "__main__":
    batch_overlay(FOLDER1_IMAGES, FOLDER2_GT, FOLDER3_PRED, FOLDER4_OUT)