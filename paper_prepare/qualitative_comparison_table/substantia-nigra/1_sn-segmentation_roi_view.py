import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json
from matplotlib.patches import Rectangle  # 用于绘制矩形框

# ===========================
# 参数设置
# ===========================
image_dir = "/data2/gaojiahao/paper_prepare/qualitative_comparison_table/substantia-nigra/data"
image_ids = [f"{i:03d}" for i in range(1, 6)]  # 001 到 005

# 模型名称列表
model_names = [
    "Image",
    "ROI (Zoomed)",
    "Ground Truth",
    "unet",
    "Segformer",
    "TransUNet",
    "DeepLab3",
    "nnunet",
    "U-Mamba_Bot",
    "U-Mamba_Enc",
]

suffix = ".png"

# ===========================
# 从 JSON 文件加载 ROI 坐标
# ===========================
roi_file = "rois.json"  # 确保与脚本同目录或指定路径
if not os.path.exists(roi_file):
    raise FileNotFoundError(f"ROI file {roi_file} not found. Please create it.")

with open(roi_file, 'r') as f:
    roi_coords = json.load(f)

# 验证所有 image_id 是否都有 ROI
missing_ids = [id for id in image_ids if id not in roi_coords]
if missing_ids:
    raise ValueError(f"Missing ROI coordinates for images: {missing_ids}")

# ===========================
# 图像读取函数
# ===========================
def load_image(path):
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return None
    img = Image.open(path).convert("L")  # 转灰度图
    return np.array(img)

# ===========================
# 主程序
# ===========================
fig, axes = plt.subplots(len(image_ids), len(model_names), figsize=(32, 17))
fig.suptitle("Substantia-nigra Segmentation Results Comparison", fontsize=21, fontweight='bold')

# 设置列标题
for j, name in enumerate(model_names):
    axes[0, j].set_title(name, fontsize=18, pad=10)
    axes[0, j].axis('off')

# 遍历每张图像
for i, img_id in enumerate(image_ids):
    roi_x, roi_y, roi_w, roi_h = roi_coords[img_id]  # 获取当前图像的 ROI

    # 读取输入图像
    input_path = os.path.join(image_dir, f"{img_id}_input{suffix}")
    input_img = load_image(input_path)
    if input_img is None:
        continue

    # 读取 ground truth
    gt_path = os.path.join(image_dir, f"{img_id}_gt{suffix}")
    gt_img = load_image(gt_path)

    # 读取各模型输出
    model_preds = []
    for model_name in model_names[3:]:  # 跳过 Image, ROI, GT
        model_file = f"{img_id}_{model_name.replace(' ', '').lower()}{suffix}"
        pred_path = os.path.join(image_dir, model_file)
        pred = load_image(pred_path)
        model_preds.append(pred)

    # 第一列：原始图像 + 用红色虚线矩形框标出 ROI 区域
    axes[i, 0].imshow(input_img, cmap='gray')
    # 添加红色虚线矩形框
    rect = Rectangle(
        (roi_x, roi_y), roi_w, roi_h,
        linewidth=2,
        edgecolor='red',
        facecolor='none',
        linestyle='--'  # 设置为虚线
    )
    axes[i, 0].add_patch(rect)
    axes[i, 0].axis('off')

    # 第二列：ROI 放大图（来自输入图像）
    try:
        roi_input = input_img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        axes[i, 1].imshow(roi_input, cmap='gray')
    except Exception as e:
        print(f"Error cropping ROI for {img_id}: {e}")
        axes[i, 1].text(0.5, 0.5, "N/A", ha='center', va='center', color='red', fontsize=8)
    axes[i, 1].axis('off')

    # 第三列：GT 在 ROI 上的裁剪
    if gt_img is not None:
        try:
            roi_gt = gt_img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            axes[i, 2].imshow(roi_gt, cmap='gray')
        except Exception as e:
            print(f"Error cropping GT for {img_id}: {e}")
            axes[i, 2].text(0.5, 0.5, "N/A", ha='center', va='center', color='red', fontsize=8)
    else:
        axes[i, 2].text(0.5, 0.5, "N/A", ha='center', va='center', color='red', fontsize=8)
    axes[i, 2].axis('off')

    # 第四列及以后：各模型在 ROI 上的结果
    for j, pred in enumerate(model_preds):
        if pred is not None:
            try:
                roi_pred = pred[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                axes[i, j+3].imshow(roi_pred, cmap='gray')
            except Exception as e:
                print(f"Error cropping {model_names[j+3]} for {img_id}: {e}")
                axes[i, j+3].text(0.5, 0.5, "N/A", ha='center', va='center', color='red', fontsize=8)
        else:
            axes[i, j+3].text(0.5, 0.5, "N/A", ha='center', va='center', color='red', fontsize=8)
        axes[i, j+3].axis('off')

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.93, bottom=0.05, left=0.05, right=0.95, hspace=0.05, wspace=0.05)

# 保存图片
output_path = "sn_segmentation_comparison_roi_view.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Comparison figure saved to {output_path}")

# 显示图像（可选）
plt.show()