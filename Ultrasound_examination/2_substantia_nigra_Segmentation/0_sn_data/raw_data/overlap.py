import os
import cv2
import numpy as np
from collections import defaultdict

def find_all_contours(image):
    """
    从二值图像中找到所有轮廓。
    
    :param image: 输入的二值图像 (numpy array)
    :return: 轮廓列表
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_contours_on_image(image, contours, color, thickness=2):
    """
    在图像上绘制轮廓。
    
    :param image: 输入的彩色图像 (BGR)
    :param contours: 轮廓列表
    :param color: 绘制颜色 (BGR格式，如绿色 (0, 255, 0))
    :param thickness: 线条粗细
    :return: 绘制了轮廓的图像
    """
    image_with_contours = image.copy()
    for contour in contours:
        if contour is not None and len(contour) >= 3:
            cv2.drawContours(image_with_contours, [contour], -1, color, thickness)
    return image_with_contours

def extract_prefix(filename):
    """
    从文件名中提取第一个下划线 "_" 之前的字符作为前缀。
    """
    return filename.split('_')[0]

def group_files_by_prefix(folder_path):
    """
    将文件夹中的图像文件按前缀分组。
    """
    file_dict = defaultdict(list)
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            prefix = extract_prefix(filename)
            file_dict[prefix].append(filename)
    return file_dict

def process_and_overlay_gt_contours(image_folder, mask_folder, output_folder):
    """
    将真实标签的轮廓（绿色）叠加到原始图像上。
    
    :param image_folder: 原始图像文件夹路径
    :param mask_folder: 真实标签（ground truth）文件夹路径
    :param output_folder: 输出文件夹路径
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 检查文件夹是否存在
    assert os.path.exists(image_folder), f"文件夹不存在: {image_folder}"
    assert os.path.exists(mask_folder), f"文件夹不存在: {mask_folder}"

    # 按前缀分组
    image_dict = group_files_by_prefix(image_folder)
    mask_dict = group_files_by_prefix(mask_folder)

    # 找出共同前缀
    common_prefixes = set(image_dict.keys()) & set(mask_dict.keys())
    if not common_prefixes:
        raise ValueError("未找到图像和标签之间的匹配文件。")

    print(f"找到 {len(common_prefixes)} 组匹配的图像-标签对。正在处理...")

    for prefix in sorted(common_prefixes):
        # 假设每个前缀只对应一个文件
        img_files = image_dict[prefix]
        mask_files = mask_dict[prefix]

        if len(img_files) != 1 or len(mask_files) != 1:
            print(f"⚠️ 前缀 '{prefix}' 匹配不唯一（image: {len(img_files)}, mask: {len(mask_files)}），跳过。")
            continue

        img_name = img_files[0]
        # mask_name = mask_files[0]  # 不需要用于输出名

        try:
            # 构建路径
            img_path = os.path.join(image_folder, img_name)
            mask_path = os.path.join(mask_folder, mask_files[0])
            
            # --- 修改这里：生成新的输出文件名 ---
            name_prefix = prefix  # 已经是下划线前的部分
            file_ext = os.path.splitext(img_name)[1]  # 获取原文件扩展名
            output_filename = f"{name_prefix}_label{file_ext}"
            output_path = os.path.join(output_folder, output_filename)
            # --------------------------------------------

            # 1. 读取原始图像
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"无法读取图像: {img_path}")
            if len(image.shape) == 2:  # 如果是灰度图，转为三通道
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # 2. 读取真实标签并提取轮廓
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"无法读取标签图像: {mask_path}")

            # 二值化：假设前景为白色（接近255）
            _, binary = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
            contours = find_all_contours(binary)

            # 3. 在原始图像上绘制绿色轮廓（真实标签）
            overlay = draw_contours_on_image(image, contours, color=(0, 255, 0), thickness=1)  # 绿色

            # 4. 保存结果
            cv2.imwrite(output_path, overlay)
            print(f"✅ 已处理: {img_name} -> 保存为 {output_filename}")

        except Exception as e:
            print(f"❌ 处理 {prefix} 时出错: {e}")

            
# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    # --- 请修改为你的实际路径 ---
    image_folder = '/data2/gaojiahao/Ultrasound_examination/2_substantia_nigra_Segmentation/0_sn_data/raw_data/images'
    mask_folder = '/data2/gaojiahao/Ultrasound_examination/2_substantia_nigra_Segmentation/0_sn_data/raw_data/masks' 
    output_folder = '/data2/gaojiahao/Ultrasound_examination/2_substantia_nigra_Segmentation/0_sn_data/raw_data/overlapped_labels'

    # 执行处理
    process_and_overlay_gt_contours(image_folder, mask_folder, output_folder)