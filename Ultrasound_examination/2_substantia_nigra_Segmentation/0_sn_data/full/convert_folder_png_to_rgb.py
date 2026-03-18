import os
from PIL import Image
import sys

def convert_to_rgb_if_needed(image_path):
    """
    读取图像，若不是 RGB 模式，则无损转换为 RGB 并保存（覆盖原文件）。
    支持的输入模式：'L' (灰度), 'LA', 'P' (调色板), 'RGBA', 'RGBX' 等。
    转换策略：
      - 'L' / 'LA' → 转为 RGB / RGBA 再转 RGB（丢弃 alpha）
      - 'P' → 先转换为 RGBA 或 RGB（保留透明度信息，但最终转为 RGB）
      - 其他非 RGB 模式也尽量安全转换为 RGB
    """
    try:
        with Image.open(image_path) as img:
            original_mode = img.mode
            if original_mode == 'RGB':
                # 已是 RGB，无需处理
                return False, original_mode

            # 转换为 RGB 的安全方式
            if original_mode == 'L':
                # 灰度图：直接转换为 RGB（复制单通道到三通道）
                rgb_img = img.convert('RGB')
            elif original_mode == 'LA':
                # 带 alpha 的灰度：先转 RGBA 再转 RGB（丢 alpha）
                rgb_img = img.convert('RGBA').convert('RGB')
            elif original_mode == 'P':
                # 调色板模式：先转 RGBA 保留透明信息，再转 RGB
                rgb_img = img.convert('RGBA').convert('RGB')
            elif original_mode == 'RGBA':
                # 带 alpha 的 RGB：直接丢弃 alpha 转为 RGB
                rgb_img = img.convert('RGB')
            elif original_mode in ('RGBX', 'CMYK', 'YCbCr', 'LAB', 'HSV'):
                # 其他模式统一转为 RGB
                rgb_img = img.convert('RGB')
            else:
                # 兜底：尝试直接转 RGB
                rgb_img = img.convert('RGB')

            # 保存为 PNG，无损覆盖原文件
            rgb_img.save(image_path, format='PNG')
            return True, original_mode
    except Exception as e:
        print(f"处理文件时出错: {image_path}，错误: {e}")
        return False, None

def process_folder(folder_path):
    """遍历文件夹中所有 .png 文件并处理"""
    if not os.path.isdir(folder_path):
        print(f"错误：路径 '{folder_path}' 不是有效文件夹。")
        return

    png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    if not png_files:
        print("未找到 .png 文件。")
        return

    converted_count = 0
    for filename in png_files:
        full_path = os.path.join(folder_path, filename)
        changed, orig_mode = convert_to_rgb_if_needed(full_path)
        if changed:
            print(f"已转换: {filename} (原模式: {orig_mode}) → RGB")
            converted_count += 1
        else:
            if orig_mode is not None:
                print(f"跳过: {filename} (已是 RGB)")
            else:
                print(f"跳过/出错: {filename}")

    print(f"\n处理完成！共 {len(png_files)} 个 PNG 文件，其中 {converted_count} 个被转换为 RGB。")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python convert_png_to_rgb.py <文件夹路径>")
        print("示例: python convert_png_to_rgb.py ./images")
    else:
        folder = sys.argv[1]
        process_folder(folder)