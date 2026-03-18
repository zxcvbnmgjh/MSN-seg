import os
from PIL import Image
import numpy as np


def main():
    img_channels = 3
    img_dir = "/data2/gaojiahao/Ultrasound_examination/2_substantia_nigra_Segmentation/0_sn_data/full/images_512"

    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."

  

    img_name_list = [i for i in os.listdir(img_dir) if i.endswith(".png")]
    pixel_sum = np.zeros(img_channels)  # 累计所有像素值
    pixel_squared_sum = np.zeros(img_channels)  # 累计所有像素值的平方
    num_pixels = 0  # 记录总像素数

    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        img = np.array(Image.open(img_path)) / 255.  # 归一化到 [0, 1]

        # 确保图像是 RGB 格式
        if img.ndim == 2:  # 灰度图像
            img = np.stack([img] * 3, axis=-1)  # 转换为三通道

        # 计算当前图像的像素数
        h, w, c = img.shape
        num_pixels += h * w

        # 累计像素值和像素值的平方
        pixel_sum += img.reshape(-1, c).sum(axis=0)
        pixel_squared_sum += (img ** 2).reshape(-1, c).sum(axis=0)

    # 计算全局均值和标准差
    mean = pixel_sum / num_pixels
    std = np.sqrt(pixel_squared_sum / num_pixels - mean ** 2)

    print(f"mean: {mean}")
    print(f"std: {std}")


if __name__ == '__main__':
    main()

