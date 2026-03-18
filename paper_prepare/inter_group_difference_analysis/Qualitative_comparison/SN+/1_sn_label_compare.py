# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle, ConnectionPatch

# ===========================
# 参数设置
# ===========================
image_dir = "/data2/gaojiahao/paper_prepare/inter_group_difference_analysis/Qualitative_comparison/SN+/compare_data"
image_ids = [f"{i:03d}" for i in range(1, 6)]  # 001 到 005

# 列标题
model_names = [
    "Image",
    "ROI (Zoomed)",
    "Annotator_A",
    "Annotator_B",
    "Label_comparison",
]
suffix = ".png"

# 可视化风格
ROI_COLOR = "red"
ROI_LW = 2.5
LINK_LW = 2.5
TITLE_FONTSIZE = 28
SUPTITLE_FONTSIZE = 36

# ===========================
# 加载 ROI 坐标
# ===========================
roi_file = "/data2/gaojiahao/paper_prepare/inter_group_difference_analysis/Qualitative_comparison/SN+/rois_new.json"
if not os.path.exists(roi_file):
    raise FileNotFoundError(f"ROI file {roi_file} not found.")

with open(roi_file, 'r') as f:
    roi_coords = json.load(f)

missing_ids = [id_ for id_ in image_ids if id_ not in roi_coords]
if missing_ids:
    raise ValueError(f"Missing ROI coordinates for images: {missing_ids}")

# ===========================
# 图像读取函数
# ===========================
def load_image(path):
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return None
    img = Image.open(path).convert("L")  # 灰度图；若对比图是彩色，后续会特殊处理
    return np.array(img)

def load_comparison_image(path):
    """对比图可能是彩色的，保留 RGB"""
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return None
    img = Image.open(path)
    return np.array(img)

# ===========================
# 主程序
# ===========================
n_rows, n_cols = len(image_ids), len(model_names)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(19, 20), dpi=150)

# 兼容单行/单列
if n_rows == 1:
    axes = np.expand_dims(axes, axis=0)
if n_cols == 1:
    axes = np.expand_dims(axes, axis=1)

# 设置列标题
for j, name in enumerate(model_names):
    axes[0, j].set_title(name, fontsize=TITLE_FONTSIZE, pad=10)
    axes[0, j].axis('off')

# 遍历每张图像
for i, img_id in enumerate(image_ids):
    roi_x, roi_y, roi_w, roi_h = roi_coords[img_id]

    # 加载原图
    image_path = os.path.join(image_dir, f"{img_id}_image{suffix}")
    input_img = load_image(image_path)
    if input_img is None:
        for j in range(n_cols):
            axes[i, j].text(0.5, 0.5, "N/A", ha='center', va='center', color='red', fontsize=10)
            axes[i, j].axis('off')
        continue

    # 加载各组件
    mask_a_path = os.path.join(image_dir, f"{img_id}_mask_A{suffix}")
    mask_b_path = os.path.join(image_dir, f"{img_id}_mask_B{suffix}")
    comp_path  = os.path.join(image_dir, f"{img_id}_label_comparison{suffix}")

    mask_a = load_image(mask_a_path)
    mask_b = load_image(mask_b_path)
    comp_img = load_comparison_image(comp_path)  # 保留彩色

    # -------- 第0列：原图 + ROI框 --------
    axes[i, 0].imshow(input_img, cmap='gray')
    axes[i, 0].axis('off')
    rect = Rectangle((roi_x, roi_y), roi_w, roi_h,
                     linewidth=ROI_LW, edgecolor=ROI_COLOR,
                     facecolor='none')
    axes[i, 0].add_patch(rect)

    # -------- 第1列：ROI 放大（来自原图）--------
    axes[i, 1].axis('off')
    try:
        roi_input = input_img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        axes[i, 1].imshow(roi_input, cmap='gray')
        for sp in axes[i, 1].spines.values():
            sp.set_edgecolor(ROI_COLOR)
            sp.set_linewidth(1.5)
    except Exception as e:
        print(f"Error cropping ROI input for {img_id}: {e}")
        axes[i, 1].text(0.5, 0.5, "N/A", ha='center', va='center', color='red', fontsize=10)

    # -------- 添加连接线 --------
    try:
        con1 = ConnectionPatch(
            xyA=(roi_x + roi_w, roi_y),
            coordsA=axes[i, 0].transData,
            xyB=(0.01, 0.99),
            coordsB=axes[i, 1].transAxes,
            arrowstyle='-|>',
            mutation_scale=14, lw=LINK_LW,
            color=ROI_COLOR
        )
        fig.add_artist(con1)

        con2 = ConnectionPatch(
            xyA=(roi_x + roi_w, roi_y + roi_h),
            coordsA=axes[i, 0].transData,
            xyB=(0.01, 0.01),
            coordsB=axes[i, 1].transAxes,
            arrowstyle='-|>',
            mutation_scale=14, lw=LINK_LW,
            color=ROI_COLOR
        )
        fig.add_artist(con2)
    except Exception as e:
        print(f"Error adding connections for {img_id}: {e}")

    # -------- 第2列：标注者A 的 ROI --------
    axes[i, 2].axis('off')
    if mask_a is not None:
        try:
            roi_a = mask_a[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            axes[i, 2].imshow(roi_a, cmap='gray')
        except Exception as e:
            print(f"Error loading Annotator A ROI for {img_id}: {e}")
            axes[i, 2].text(0.5, 0.5, "N/A", ha='center', va='center', color='red', fontsize=10)
    else:
        axes[i, 2].text(0.5, 0.5, "N/A", ha='center', va='center', color='red', fontsize=10)

    # -------- 第3列：标注者B 的 ROI --------
    axes[i, 3].axis('off')
    if mask_b is not None:
        try:
            roi_b = mask_b[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            axes[i, 3].imshow(roi_b, cmap='gray')
        except Exception as e:
            print(f"Error loading Annotator B ROI for {img_id}: {e}")
            axes[i, 3].text(0.5, 0.5, "N/A", ha='center', va='center', color='red', fontsize=10)
    else:
        axes[i, 3].text(0.5, 0.5, "N/A", ha='center', va='center', color='red', fontsize=10)

    # -------- 第4列：预生成的对比图 ROI --------
    axes[i, 4].axis('off')
    if comp_img is not None:
        try:
            # 自动判断是灰度还是 RGB
            if comp_img.ndim == 3:
                roi_comp = comp_img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w, :]
            else:
                roi_comp = comp_img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            axes[i, 4].imshow(roi_comp)
        except Exception as e:
            print(f"Error loading comparison ROI for {img_id}: {e}")
            axes[i, 4].text(0.5, 0.5, "N/A", ha='center', va='center', color='red', fontsize=10)
    else:
        axes[i, 4].text(0.5, 0.5, "N/A", ha='center', va='center', color='red', fontsize=10)

# 布局 & 保存
plt.tight_layout()
plt.subplots_adjust(top=0.93, bottom=0.05, left=0.05, right=0.98, hspace=0.04, wspace=0.04)

output_path = "/data2/gaojiahao/paper_prepare/inter_group_difference_analysis/Qualitative_comparison/SN+/SN+-label_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Figure saved to {output_path}")