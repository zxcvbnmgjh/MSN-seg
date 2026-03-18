import os
import shutil

def rename_png_files_by_order(source_folder, target_folder, prefix="midbrain", suffix="_mask"):
    """
    将源文件夹中所有 PNG 文件按排序顺序重命名为：
        {prefix}XXX{suffix}.png
    例如：midbrain001_image.png, midbrain002_image.png ...
    
    参数：
        source_folder: 源文件夹路径
        target_folder: 目标文件夹路径
        prefix: 前缀，默认 "midbrain"
        suffix: 后缀，默认 "_image"
    """
    
    # 创建目标文件夹
    os.makedirs(target_folder, exist_ok=True)
    
    # 获取所有 PNG 文件（不区分大小写）
    png_files = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith('.png')
    ]
    
    if not png_files:
        print("⚠️  源文件夹中没有找到 PNG 文件。")
        return
    
    # 按文件名排序（确保顺序稳定）
    png_files.sort()
    
    print(f"📁 找到 {len(png_files)} 个 PNG 文件，开始重命名并复制...")
    
    for index, filename in enumerate(png_files, start=1):
        # 格式化编号为三位数
        number_str = str(index).zfill(3)
        
        # 构建新文件名
        new_filename = f"{prefix}{number_str}{suffix}.png"
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(target_folder, new_filename)
        
        # 复制文件
        shutil.copy2(src_path, dst_path)
        
        print(f"✅ [{index:03d}] {filename} → {new_filename}")
    
    print(f"\n🎉 处理完成！共复制并重命名 {len(png_files)} 个文件到：{target_folder}")

# ========================masks
# 使用示例
# ========================
if __name__ == "__main__":
    # ⚠️ 请替换为你的实际路径
    SOURCE_DIR = r"/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/0_midbrain_data/raw_data/masks000"
    TARGET_DIR = r"/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/0_midbrain_data/full/masks"
    
    rename_png_files_by_order(SOURCE_DIR, TARGET_DIR)