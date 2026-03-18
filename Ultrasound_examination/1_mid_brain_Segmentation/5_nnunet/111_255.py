import os
from PIL import Image

def modify_and_save_pngs(source_folder, destination_folder):
    """
    遍历源文件夹中的所有PNG图像，
    将像素值为(0, 0, 0)的像素更改为(255, 255, 255)，
    并将修改后的图像保存到目标文件夹。
    
    :param source_folder: 源文件夹路径，包含待处理的PNG图像
    :param destination_folder: 目标文件夹路径，用于保存处理后的图像
    """
    # 如果目标文件夹不存在，则创建
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 获取源文件夹下所有的文件名
    files = os.listdir(source_folder)
    
    # 过滤出png文件
    png_files = [f for f in files if f.lower().endswith('.png')]
    
    for png_file in png_files:
        full_path = os.path.join(source_folder, png_file)
        # 打开图像文件
        with Image.open(full_path) as img:
            # 将图像转换为RGB模式，以统一处理方式
            img = img.convert('RGB')
            
            # 加载像素数据
            pixels = img.load()
            width, height = img.size
            
            # 遍历每个像素，检查是否为(0, 0, 0)，如果是则修改为(255, 255, 255)
            for x in range(width):
                for y in range(height):
                    if pixels[x, y] == (1, 1, 1):
                        pixels[x, y] = (255, 255, 255)
            
            # 构造目标文件的完整路径
            new_full_path = os.path.join(destination_folder, png_file)
            
            # 保存修改后的图像
            img.save(new_full_path)
            print(f"Processed and saved: {new_full_path}")

# 指定源文件夹路径和目标文件夹路径
source_folder_path = '/data2/gaojiahao/nnU-net/nnUNet/DATASET/Results/134midbrain_2D_results/midbrain_2d_predict_post'
destination_folder_path = '/data2/gaojiahao/nnU-net/nnUNet/DATASET/Results/134midbrain_2D_results/midbrain_2d_predict_post_255'

modify_and_save_pngs(source_folder_path, destination_folder_path)