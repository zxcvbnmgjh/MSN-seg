# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle, ConnectionPatch  # 新增 ConnectionPatch

# ===========================
# 参数设置
# ===========================
image_dir = "/data2/gaojiahao/paper_prepare/qualitative_comparison_table/substantia-nigra/data_together_2"
image_ids = [f"{i:03d}" for i in range(1, 6)]  # 001 到 005

# 列标题
model_names = [
    "Image",
    "ROI (Zoomed)",
    "Ground Truth",
    "UNet",
    "Segformer",
    "TransUNet",
    "DeepLabV3",
    "nnU-Net",
    "U-Mamba_Bot",
    "U-Mamba_Enc",
]
suffix = ".png"

# 可视化风格
ROI_COLOR = "red"   # ROI 框和连线颜色
ROI_LW = 2.5        # ROI 框线宽
LINK_LW = 2.5      # 连线线宽
TITLE_FONTSIZE = 28
SUPTITLE_FONTSIZE = 36

# ===========================
# 从 JSON 文件加载 ROI 坐标
# ===========================
roi_file = "/data2/gaojiahao/paper_prepare/qualitative_comparison_table/substantia-nigra/rois.json"  # 确保与脚本同目录或指定路径
if not os.path.exists(roi_file):
    raise FileNotFoundError(f"ROI file {roi_file} not found. Please create it.")

with open(roi_file, 'r') as f:
    roi_coords = json.load(f)

# 验证所有 image_id 是否都有 ROI
missing_ids = [id_ for id_ in image_ids if id_ not in roi_coords]
if missing_ids:
    raise ValueError(f"Missing ROI coordinates for images: {missing_ids}")

# ===========================
# 图像读取函数
# ===========================
def load_image(path, mode="RGB"):
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return None
    img = Image.open(path).convert(mode)  # mode="RGB" 或 "L"
    return np.array(img)

# ===========================
# 主程序
# ===========================
n_rows, n_cols = len(image_ids), len(model_names)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(34, 18), dpi=150)
"""
fig.suptitle("Midbrain Segmentation Results Comparison",
             fontsize=SUPTITLE_FONTSIZE, fontweight='bold')
"""


# 兼容 axes 维度
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
    roi_x, roi_y, roi_w, roi_h = roi_coords[img_id]  # ROI: x, y, w, h

    # 读取输入图像
    input_path = os.path.join(image_dir, f"{img_id}_input{suffix}")
    input_img = load_image(input_path, mode="RGB")

    if input_img is None:
        # 整行标注 N/A
        for j in range(n_cols):
            axes[i, j].text(0.5, 0.5, "N/A", ha='center', va='center',
                            color='red', fontsize=10)
            axes[i, j].axis('off')
        continue

    # 读取 GT
    gt_path = os.path.join(image_dir, f"{img_id}_gt{suffix}")
    gt_img = load_image(gt_path)

    # 读取各模型输出
    model_preds = []
    for model_name in model_names[3:]:  # 跳过 Image, ROI, GT
        model_file = f"{img_id}_{model_name.replace(' ', '').lower()}{suffix}"
        pred_path = os.path.join(image_dir, model_file)
        pred = load_image(pred_path)
        model_preds.append(pred)

    # -------- 第一列：原图 + ROI矩形 --------
    axes[i, 0].imshow(input_img)
    axes[i, 0].axis('off')

    rect = Rectangle(
        (roi_x, roi_y), roi_w, roi_h,
        linewidth=ROI_LW, edgecolor=ROI_COLOR,
        facecolor='none', linestyle='-'    # 虚线框
    )
    axes[i, 0].add_patch(rect)
    """
    # ROI 标签（可选）
    axes[i, 0].text(roi_x, max(0, roi_y-6), "ROI", color=ROI_COLOR,
                    fontsize=10, weight="bold", va="bottom")

    """

    # -------- 第二列：ROI 放大图 --------
    axes[i, 1].axis('off')
    try:
        roi_input = input_img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        axes[i, 1].imshow(roi_input)
        # 给放大图加上同色边框，方便视觉对应
        for sp in axes[i, 1].spines.values():
            sp.set_edgecolor(ROI_COLOR)
            sp.set_linewidth(1.5)
    except Exception as e:
        print(f"Error cropping ROI for {img_id}: {e}")
        axes[i, 1].text(0.5, 0.5, "N/A", ha='center', va='center',
                        color='red', fontsize=10)

    # -------- 关键：从第1列 ROI 到第2列放大图画“虚线箭头连线” --------
    try:
        # 线1：ROI 右上角 -> 第2列左上角
        con1 = ConnectionPatch(
            xyA=(roi_x + roi_w, roi_y),          # 源：右上角（数据坐标）
            coordsA=axes[i, 0].transData,
            xyB=(0.01, 0.99),                    # 目标：左上角（轴坐标，稍内缩）
            coordsB=axes[i, 1].transAxes,
            arrowstyle='-|>',
            mutation_scale=14, lw=LINK_LW,
            color=ROI_COLOR, linestyle='-'
        )
        fig.add_artist(con1)

        # 线2：ROI 右下角 -> 第2列左下角
        con2 = ConnectionPatch(
            xyA=(roi_x + roi_w, roi_y + roi_h),  # 源：右下角（数据坐标）
            coordsA=axes[i, 0].transData,
            xyB=(0.01, 0.01),                    # 目标：左下角（轴坐标，稍内缩）
            coordsB=axes[i, 1].transAxes,
            arrowstyle='-|>',
            mutation_scale=14, lw=LINK_LW,
            color=ROI_COLOR, linestyle='-'
        )
        fig.add_artist(con2)
    except Exception as e:
        print(f"Error adding connection for {img_id}: {e}")


    # -------- 第三列：GT 在 ROI 上的裁剪 --------
    axes[i, 2].axis('off')
    if gt_img is not None:
        try:
            roi_gt = gt_img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            axes[i, 2].imshow(roi_gt, cmap='gray')
        except Exception as e:
            print(f"Error cropping GT for {img_id}: {e}")
            axes[i, 2].text(0.5, 0.5, "N/A", ha='center', va='center',
                            color='red', fontsize=10)
    else:
        axes[i, 2].text(0.5, 0.5, "N/A", ha='center', va='center',
                        color='red', fontsize=10)

    # -------- 第四列及以后：各模型在 ROI 上的结果 --------
    for j, pred in enumerate(model_preds):
        axes[i, j+3].axis('off')
        if pred is not None:
            try:
                roi_pred = pred[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                axes[i, j+3].imshow(roi_pred, cmap='gray')
            except Exception as e:
                print(f"Error cropping {model_names[j+3]} for {img_id}: {e}")
                axes[i, j+3].text(0.5, 0.5, "N/A", ha='center', va='center',
                                  color='red', fontsize=10)
        else:
            axes[i, j+3].text(0.5, 0.5, "N/A", ha='center', va='center',
                              color='red', fontsize=10)

# 布局 & 保存
plt.tight_layout()
# 右边留一点边距，避免箭头被裁切
plt.subplots_adjust(top=0.93, bottom=0.05, left=0.05, right=0.98, hspace=0.04, wspace=0.04)

output_path = "/data2/gaojiahao/paper_prepare/qualitative_comparison_table/substantia-nigra/2_sn-segmentation_comparison_new_roi_view_connection.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Comparison figure saved to {output_path}")

# plt.show()  # 如需窗口预览，解开注释
