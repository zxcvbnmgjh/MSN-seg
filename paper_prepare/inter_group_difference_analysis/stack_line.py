# 真实标签边界（设置为绿色），预测结果边界（设置为蓝色），叠加在原始图像上，对比分割效果
import os
import cv2
import numpy as np
from collections import defaultdict

def find_all_contours(image):
    """
    从二值图像中找到所有轮廓。
    
    :param image: 输入的二值图像 (numpy array)
    :return: 轮廓列表 (每个轮廓是一个 numpy array)
    """
    # 查找所有轮廓
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_contours_on_image(image, contours, color, thickness=2):
    """
    在图像上绘制轮廓。
    
    :param image: 输入的彩色图像 (BGR)
    :param contours: 轮廓列表 (每个轮廓是一个 numpy array)
    :param color: 绘制颜色 (BGR格式, e.g., (0, 255, 0) for green)
    :param thickness: 线条粗细
    :return: 绘制了轮廓的图像
    """
    image_with_contours = image.copy()
    for contour in contours:
        if contour is not None and len(contour) >= 3:  # 确保轮廓至少有3个点才能绘制
            cv2.drawContours(image_with_contours, [contour], -1, color, thickness)
    return image_with_contours

def extract_prefix(filename):
    """
    从文件名中提取第一个下划线 "_" 之前的字符作为前缀。
    
    :param filename: 文件名字符串
    :return: 前缀字符串
    """
    return filename.split('_')[0]

def group_files_by_prefix(folder_path):
    """
    将文件夹中的文件按前缀分组。
    
    :param folder_path: 文件夹路径
    :return: 一个字典，键为前缀，值为具有该前缀的文件名列表
    """
    file_dict = defaultdict(list)
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            prefix = extract_prefix(filename)
            file_dict[prefix].append(filename)
    return file_dict

def process_and_stack_images(folder1, folder2, folder3, output_folder):
    """
    处理三个文件夹中的图像，在第一个文件夹的图像上绘制第二和第三个文件夹图像的轮廓。
    
    :param folder1: 第一个文件夹路径（原始图像）
    :param folder2: 第二个文件夹路径（真实标签）
    :param folder3: 第三个文件夹路径（预测结果）
    :param output_folder: 输出文件夹路径
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 检查文件夹是否存在
    assert os.path.exists(folder1), f"文件夹 {folder1} 不存在。"
    assert os.path.exists(folder2), f"文件夹 {folder2} 不存在。"
    assert os.path.exists(folder3), f"文件夹 {folder3} 不存在。"

    # 按前缀分组文件
    files1_dict = group_files_by_prefix(folder1)
    files2_dict = group_files_by_prefix(folder2)
    files3_dict = group_files_by_prefix(folder3)

    # 获取所有共同的前缀
    common_prefixes = set(files1_dict.keys()) & set(files2_dict.keys()) & set(files3_dict.keys())
    if not common_prefixes:
        raise ValueError("三个文件夹中没有找到共同的文件名前缀。")

    print(f"找到 {len(common_prefixes)} 组匹配的图像。正在处理...")

    # 处理每组图像
    for prefix in common_prefixes:
        # 获取该前缀下的所有文件
        img1_names = files1_dict[prefix]
        img2_names = files2_dict[prefix]
        img3_names = files3_dict[prefix]

        # 这里假设每组前缀只对应一个文件，如果对应多个，可以添加循环
        if not (len(img1_names) == len(img2_names) == len(img3_names) == 1):
            print(f"警告: 前缀 '{prefix}' 在某个文件夹中不唯一或缺失，跳过。")
            continue

        img1_name = img1_names[0]
        img2_name = img2_names[0]
        img3_name = img3_names[0]

        try:
            # 构造完整路径
            img_path1 = os.path.join(folder1, img1_name)
            img_path2 = os.path.join(folder2, img2_name)
            img_path3 = os.path.join(folder3, img3_name)
            output_path = os.path.join(output_folder, img1_name)  # 输出文件名与原始图像一致

            # 1. 读取原始图像
            img1 = cv2.imread(img_path1)
            if img1 is None:
                raise ValueError(f"无法读取图像 {img_path1}")
            if len(img1.shape) == 2:
                img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

            # 2. 读取真实标签图像并找到所有轮廓
            img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
            if img2 is None:
                raise ValueError(f"无法读取图像 {img_path2}")
            # 二值化：假设白色是标签区域 (像素值接近255)
            _, binary_img2 = cv2.threshold(img2, 200, 255, cv2.THRESH_BINARY)
            contours_gt = find_all_contours(binary_img2)

            # 3. 读取预测结果图像并找到所有轮廓
            img3 = cv2.imread(img_path3, cv2.IMREAD_GRAYSCALE)
            if img3 is None:
                raise ValueError(f"无法读取图像 {img_path3}")
            # 二值化：假设白色是预测区域
            _, binary_img3 = cv2.threshold(img3, 200, 255, cv2.THRESH_BINARY)
            contours_pred = find_all_contours(binary_img3)

            # 4. 在原始图像上绘制轮廓
            # 绿色轮廓 (BGR: 0, 255, 0) 代表真实标签
            combined = draw_contours_on_image(img1, contours_gt, (0, 255, 0), thickness=2)
            # 蓝色轮廓 (BGR: 255, 0, 0) 代表预测结果
            combined = draw_contours_on_image(combined, contours_pred, (255, 0, 0), thickness=2)

            # 5. 保存结果
            cv2.imwrite(output_path, combined)
            print(f"已处理: {img1_name}")

        except Exception as e:
            print(f"处理前缀 '{prefix}' 时出错: {e}")

if __name__ == "__main__":
    # --- 请在此处修改为您自己的文件夹路径 ---
    folder1 = "/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/1_unet/predict_results/images"      # 原始图像文件夹
    folder2 = "/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/1_unet/predict_results/groundtruth"         # 真实标签文件夹 (白色区域)
    folder3 = "/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/1_unet/predict_results/predicts"          # 预测结果文件夹 (白色区域)
    output_folder = "/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/1_unet/predict_results/stack_contours"         # 输出文件夹
    # ------------------------------------------------

    # 调用函数
    process_and_stack_images(folder1, folder2, folder3, output_folder)