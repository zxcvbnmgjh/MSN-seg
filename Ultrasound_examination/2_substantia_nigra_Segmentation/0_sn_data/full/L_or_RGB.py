import os
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd


def classify_png(path: Path, check_gray_like_rgb: bool = True):
    """
    返回: (mode, kind)
    kind: Gray / RGB / RGBA / Palette(P) / Other / RGB(gray_like)
    """
    with Image.open(path) as im:
        mode = im.mode  # 例如 L, RGB, RGBA, P, I;16 ...
        # 先按mode快速分类
        if mode in ("L", "I", "F", "I;16", "I;16B", "I;16L"):
            return mode, "Gray"
        if mode == "RGB":
            if not check_gray_like_rgb:
                return mode, "RGB"
            # 检查是否“灰度伪装成RGB”（三通道完全相同）
            arr = np.array(im)  # HxWx3
            if arr.ndim == 3 and arr.shape[2] == 3:
                if np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2]):
                    return mode, "RGB(gray_like)"
            return mode, "RGB"
        if mode == "RGBA":
            return mode, "RGBA"
        if mode == "P":
            # 调色板图：可以进一步看转RGB后的灰度性
            if not check_gray_like_rgb:
                return mode, "Palette(P)"
            rgb = im.convert("RGB")
            arr = np.array(rgb)
            if np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2]):
                return mode, "Palette(P)->RGB(gray_like)"
            return mode, "Palette(P)->RGB"
        # 其他模式兜底：转数组看通道
        arr = np.array(im)
        if arr.ndim == 2:
            return mode, "Gray"
        if arr.ndim == 3 and arr.shape[2] == 3:
            if check_gray_like_rgb and np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2]):
                return mode, "RGB(gray_like)"
            return mode, "RGB"
        if arr.ndim == 3 and arr.shape[2] == 4:
            return mode, "RGBA"
        return mode, f"Other({arr.shape})"


def scan_folder(folder: str, recursive: bool = True, check_gray_like_rgb: bool = True):
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder}")

    pattern = "**/*.png" if recursive else "*.png"
    png_files = sorted(folder_path.glob(pattern))

    rows = []
    for p in png_files:
        try:
            mode, kind = classify_png(p, check_gray_like_rgb=check_gray_like_rgb)
            rows.append({
                "file": str(p),
                "mode": mode,
                "type": kind
            })
        except Exception as e:
            rows.append({
                "file": str(p),
                "mode": None,
                "type": f"ERROR: {e}"
            })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    # ======== 改成你的文件夹路径 ========
    folder = r"/data2/gaojiahao/Ultrasound_examination/2_substantia_nigra_Segmentation/0_sn_data/full/masks_512"

    # 是否递归扫描子文件夹
    recursive = True

    # 是否把“RGB但三通道相同”的图也标为 gray_like
    check_gray_like_rgb = True

    df = scan_folder(folder, recursive=recursive, check_gray_like_rgb=check_gray_like_rgb)

    # 控制台输出统计
    print(df["type"].value_counts(dropna=False))
    print(df.head(10))

    # 保存结果
    out_csv = "masks_color_mode_report.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n已保存报告: {out_csv}")
