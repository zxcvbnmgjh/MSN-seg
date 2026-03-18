import os
from PIL import Image
from torchvision import transforms
import torch

# 设置输入和输出路径
input_folder = "/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/0_midbrain_data/full/masks"   # ← 修改为你的输入文件夹路径
output_folder = "/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/0_midbrain_data/full/masks_512" # ← 修改为你的输出文件夹路径

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 定义中心裁剪变换
transform = transforms.Compose([
    transforms.CenterCrop(512)  # 裁剪为 512x512
])

# 遍历输入文件夹
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".png"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            # 使用 PIL 打开图像
            image = Image.open(input_path).convert("RGB")  # 转为 RGB，避免透明通道问题

            # 检查图像尺寸是否足够大
            width, height = image.size
            if width < 512 or height < 512:
                print(f"跳过 {filename}：图像尺寸 ({width}x{height}) 小于 512x512")
                continue

            # 应用中心裁剪
            cropped_image = transform(image)

            # 保存裁剪后的图像
            cropped_image.save(output_path, "PNG")
            print(f"已裁剪并保存: {filename}")

        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")

print("所有图像处理完成！")