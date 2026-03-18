import cv2
import numpy as np

def calculate_dice_coefficient(image_path1, image_path2):
    """
    计算两张图像之间的 Dice 系数。
    
    参数:
        image_path1 (str): 第一张图像的路径。
        image_path2 (str): 第二张图像的路径。
    
    返回:
        float: Dice 系数。
    """
    # 读取图像并转换为灰度图
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise ValueError("无法读取图像，请检查路径是否正确！")
    
    # 确保图像大小一致
    if img1.shape != img2.shape:
        raise ValueError("两张图像的尺寸不一致！")

    # 将图像二值化 (0 或 1)
    _, img1_binary = cv2.threshold(img1, 127, 1, cv2.THRESH_BINARY)
    _, img2_binary = cv2.threshold(img2, 127, 1, cv2.THRESH_BINARY)

    # 计算交集和各自的像素总数
    intersection = np.sum(img1_binary * img2_binary)  # A ∩ B
    total_pixels_img1 = np.sum(img1_binary)          # |A|
    total_pixels_img2 = np.sum(img2_binary)          # |B|

    # 计算 Dice 系数
    dice = (2.0 * intersection) / (total_pixels_img1 + total_pixels_img2)

    return dice

# 示例用法
if __name__ == "__main__":
    # 替换为实际图像路径
    image_path1 = "/data2/gaojiahao/Ultrasound_examination/Segmentation/unet/midbrain_data/test/masks/lyj5_mask.png"
    image_path2 = "/data2/gaojiahao/Ultrasound_examination/Segmentation/unet/test_result.png"

    try:
        dice_score = calculate_dice_coefficient(image_path1, image_path2)
        print(f"Dice Coefficient: {dice_score:.4f}")
    except ValueError as e:
        print(e)