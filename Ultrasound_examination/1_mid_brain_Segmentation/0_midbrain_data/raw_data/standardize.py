# -*- coding: utf-8 -*-
"""
批量亮度/对比度标准化（仅改 L 通道），适配命名：
  原图: xxx_image.png  (在 IMG_DIR_IN 目录)
  标签: xxx_mask.png   (在 MSK_DIR_IN 目录)
输出到 OUT_IMG_DIR / OUT_MSK_DIR；不改变图像其它属性。
"""

from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# ========== 1) 路径设置（改这里） ==========
IMG_DIR_IN  = Path(r"/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/0_midbrain_data/raw_data/images")   # 仅放 *_image.png
MSK_DIR_IN  = Path(r"/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/0_midbrain_data/raw_data/masks")    # 仅放 *_mask.png
OUT_IMG_DIR = Path(r"/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/0_midbrain_data/raw_data/standardize_output/images_norm")
OUT_MSK_DIR = Path(r"/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/0_midbrain_data/raw_data/standardize_output/labels")

# 标签是否做严格二值化（>0 -> 255）；不需要则保持 False
BINARIZE_MASK = False

# CLAHE 参数（亮度/对比度标准化）
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID  = (8, 8)

# ========== 2) 功能函数 ==========
def normalize_brightness_contrast_rgb_uint8(bgr: np.ndarray) -> np.ndarray:
    """
    对 uint8 BGR 图像做亮度/对比度标准化：
    - 转 LAB，仅在 L 通道做 CLAHE；A/B 色度不变
    - 返回 uint8 BGR；尺寸/通道/位深不变
    """
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("需要 3 通道 RGB/BGR PNG 图像。")
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

def read_mask_gray(path: Path, binarize: bool=False) -> np.ndarray:
    """
    读取灰度标签；保证单通道 uint8；可选严格二值化。
    """
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(str(path))
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    m = m.astype(np.uint8)
    if binarize:
        _, m = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY)
    return m

# ========== 3) 主流程 ==========
def main():
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MSK_DIR.mkdir(parents=True, exist_ok=True)

    image_list = sorted(IMG_DIR_IN.glob("*_image.png"))
    if not image_list:
        print(f"[警告] 在 {IMG_DIR_IN} 未找到 *_image.png")
        return

    print(f"[信息] 发现 {len(image_list)} 张原图，开始处理…")
    for img_path in tqdm(image_list):
        # 对应标签文件在“另一个文件夹”，同前缀
        stem = img_path.name[:-10]  # 去掉 "_image.png"
        msk_path = MSK_DIR_IN / f"{stem}_mask.png"

        # --- 处理原图 ---
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)  # BGR, uint8
        if img is None:
            print(f"[跳过] 无法读取：{img_path}")
            continue
        img_eq = normalize_brightness_contrast_rgb_uint8(img)
        out_img = OUT_IMG_DIR / img_path.name
        if not cv2.imwrite(str(out_img), img_eq):
            print(f"[错误] 保存失败：{out_img}")

        # --- 处理标签（保持灰度PNG；可选二值化） ---
        if msk_path.exists():
            try:
                msk = read_mask_gray(msk_path, binarize=BINARIZE_MASK)
                out_msk = OUT_MSK_DIR / msk_path.name
                if not cv2.imwrite(str(out_msk), msk):
                    print(f"[错误] 保存标签失败：{out_msk}")
            except Exception as e:
                print(f"[标签跳过] {msk_path.name}: {e}")
        else:
            print(f"[提示] 无对应标签：{msk_path.name}")

    print("[完成] 全部文件已处理。")
    print(f"图像输出目录：{OUT_IMG_DIR}")
    print(f"标签输出目录：{OUT_MSK_DIR}")

if __name__ == "__main__":
    main()
