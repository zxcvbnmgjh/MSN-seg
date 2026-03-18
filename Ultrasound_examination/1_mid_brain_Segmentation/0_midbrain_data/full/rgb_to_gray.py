import os
from PIL import Image
import numpy as np

def convert_rgb_graylike_to_grayscale(folder_path, overwrite=True, output_folder=None):
    """
    将指定文件夹中所有 RGB(gray_like) 的 PNG 图像转换为真正的灰度图（单通道）。
    
    参数:
        folder_path (str): 输入文件夹路径。
        overwrite (bool): 是否覆盖原文件。若为 False 且 output_folder 未指定，则保存到新子文件夹。
        output_folder (str or None): 输出文件夹路径。若为 None 且 overwrite=False，则自动创建 'grayscale_output'。
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"输入路径不是有效文件夹: {folder_path}")
    
    # 设置输出目录
    if not overwrite:
        if output_folder is None:
            output_folder = os.path.join(folder_path, "grayscale_output")
        os.makedirs(output_folder, exist_ok=True)
    
    png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    
    converted_count = 0
    skipped_count = 0

    for filename in png_files:
        input_path = os.path.join(folder_path, filename)
        try:
            with Image.open(input_path) as img:
                # 只处理 RGB 模式的图像
                if img.mode != 'RGB':
                    print(f"跳过（非 RGB 模式）: {filename} (模式: {img.mode})")
                    skipped_count += 1
                    continue

                arr = np.array(img)  # shape: (H, W, 3)

                # 检查是否 R == G == B（gray_like）
                if np.array_equal(arr[:, :, 0], arr[:, :, 1]) and np.array_equal(arr[:, :, 1], arr[:, :, 2]):
                    # 无损提取灰度：取任一通道
                    gray_arr = arr[:, :, 0]
                    gray_img = Image.fromarray(gray_arr, mode='L')
                    
                    # 确定输出路径
                    if overwrite:
                        output_path = input_path
                    else:
                        output_path = os.path.join(output_folder, filename)
                    
                    gray_img.save(output_path)
                    print(f"已转换: {filename}")
                    converted_count += 1
                else:
                    print(f"跳过（非 gray_like）: {filename}")
                    skipped_count += 1

        except Exception as e:
            print(f"处理失败: {filename}, 错误: {e}")
            skipped_count += 1

    print(f"\n完成！共处理 {len(png_files)} 个 PNG 文件。")
    print(f"成功转换: {converted_count} 个")
    print(f"跳过/失败: {skipped_count} 个")

# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    # 替换为你自己的文件夹路径
    input_folder = "/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/0_midbrain_data/full/masks_512"

    # 方式1：覆盖原文件（谨慎使用！建议先备份）
    convert_rgb_graylike_to_grayscale(input_folder, overwrite=True)
"""
    # 方式2：保存到新文件夹（推荐）
    convert_rgb_graylike_to_grayscale(
        input_folder,
        overwrite=False,
        output_folder="/path/to/output/grayscale"  # 可选，若不指定则自动建子文件夹
    )
"""
