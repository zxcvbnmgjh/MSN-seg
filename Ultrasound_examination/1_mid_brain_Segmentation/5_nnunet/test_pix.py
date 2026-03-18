import os
from PIL import Image

def inspect_png_pixels(folder_path):
    # 获取文件夹下所有的文件名
    files = os.listdir(folder_path)
    
    # 过滤出png文件
    png_files = [f for f in files if f.lower().endswith('.png')]
    
    for png_file in png_files:
        full_path = os.path.join(folder_path, png_file)
        
        # 打开图像文件
        with Image.open(full_path) as img:
            # 将图像转换为RGB模式，以统一处理方式
            img = img.convert('RGB')
            # 获取图像的像素数据
            pixels = img.getdata()
            
            # 使用set来存储不同的像素值，因为set不允许重复元素
            unique_pixels = set(pixels)
            
            print(f"Image: {png_file}")
            print(f"Unique pixel values: {unique_pixels}\n")

# 指定要检查的文件夹路径
folder_path = '/data2/gaojiahao/nnU-net/nnUNet/DATASET/nnUNet_raw/Dataset134_MidbrainSegmentation/labelsTs'
inspect_png_pixels(folder_path)