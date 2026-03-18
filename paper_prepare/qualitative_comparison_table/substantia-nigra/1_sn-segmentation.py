import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# ===========================
# 参数设置
# ===========================
image_dir = "/data2/gaojiahao/paper_prepare/qualitative_comparison_table/substantia-nigra/data"  # 替换为你的图像路径
image_ids = [f"{i:03d}" for i in range(1, 6)]  # 例如：001, 002, ..., 005

# 模型名称列表（对应列名）
model_names = [
    "Image",
    "Ground Truth",
    "unet",
    "Segformer",
    "TransUNet",
    "DeepLab3",
    "nnunet",
    "U-Mamba_Bot",
    "U-Mamba_Enc",
]

# 文件后缀（可修改）
suffix = ".png"

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
fig, axes = plt.subplots(len(image_ids), len(model_names), figsize=(24, 14))
fig.suptitle(" Substantia-nigra Segmentation Results Comparison", fontsize=18, fontweight='bold')

# 设置列标题
for j, name in enumerate(model_names):
    axes[0, j].set_title(name, fontsize=16, pad=10)
    axes[0, j].axis('off')  # 不显示坐标轴

# 遍历每张图像
for i, img_id in enumerate(image_ids):
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
    for model_name in model_names[2:]:  # 跳过 Image 和 GT
        model_file = f"{img_id}_{model_name.replace(' ', '').lower()}{suffix}"
        pred_path = os.path.join(image_dir, model_file)
        pred = load_image(pred_path)
        model_preds.append(pred)

    # 显示图像
    axes[i, 0].imshow(input_img, cmap='gray')
    axes[i, 0].axis('off')

    axes[i, 1].imshow(gt_img, cmap='gray')
    axes[i, 1].axis('off')

    for j, pred in enumerate(model_preds):
        if pred is not None:
            axes[i, j+2].imshow(pred, cmap='gray')
        else:
            axes[i, j+2].text(0.5, 0.5, "N/A", ha='center', va='center', color='red', fontsize=8)
        axes[i, j+2].axis('off')

# 调整子图间距
plt.tight_layout()
plt.subplots_adjust(top=0.93, bottom=0.05, left=0.05, right=0.95, hspace=0.05, wspace=0.05)

# 保存图片
output_path = "segmentation_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Comparison figure saved to {output_path}")

# 显示图像（可选）
plt.show()