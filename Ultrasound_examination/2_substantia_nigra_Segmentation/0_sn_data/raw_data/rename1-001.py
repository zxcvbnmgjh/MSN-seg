import os
import shutil
import re

def rename_and_copy_png_files(source_folder, target_folder):
    """
    将源文件夹中 PNG 文件重命名：
    提取第一个 "_" 前部分中的末尾数字，格式化为三位数，其余结构不变。
    例如：
        cq1_image.png     → cq001_image.png
        0411cq12_image.png → 0411cq012_image.png
    并复制到目标文件夹。
    """
    
    # 创建目标文件夹（如果不存在）
    os.makedirs(target_folder, exist_ok=True)
    
    # 获取所有 PNG 文件
    png_files = [f for f in os.listdir(source_folder) if f.lower().endswith('.png')]
    
    if not png_files:
        print("⚠️  源文件夹中没有找到 PNG 文件。")
        return
    
    renamed_count = 0
    unchanged_count = 0
    
    for filename in png_files:
        # 拆分：第一个 "_" 之前的部分 和 之后的部分
        parts = filename.split('_', 1)  # 只分割一次
        if len(parts) < 2:
            # 没有下划线，跳过重命名
            src = os.path.join(source_folder, filename)
            dst = os.path.join(target_folder, filename)
            shutil.copy2(src, dst)
            print(f"⚠️  无下划线，直接复制: {filename}")
            unchanged_count += 1
            continue
        
        prefix = parts[0]      # 下划线前部分，如 "cq1" 或 "0411cq12"
        suffix = parts[1]      # 下划线后部分，如 "image.png"
        
        # 使用正则从 prefix 末尾提取连续数字
        match = re.search(r'(\d+)$', prefix)
        if not match:
            # 下划线前无数字，直接复制
            src = os.path.join(source_folder, filename)
            dst = os.path.join(target_folder, filename)
            shutil.copy2(src, dst)
            print(f"⚠️  无数字可格式化，直接复制: {filename}")
            unchanged_count += 1
            continue
        
        number_str = match.group(1)          # 提取的数字字符串，如 "1"、"12"
        non_number_prefix = prefix[:-len(number_str)]  # 前缀非数字部分，如 "cq"、"0411cq"
        
        # 格式化为三位数
        formatted_number = number_str.zfill(3)
        
        # 构建新文件名
        new_prefix = non_number_prefix + formatted_number
        new_filename = new_prefix + '_' + suffix
        new_filepath = os.path.join(target_folder, new_filename)
        
        # 复制文件
        src_filepath = os.path.join(source_folder, filename)
        shutil.copy2(src_filepath, new_filepath)
        
        print(f"✅ 重命名并复制: {filename} → {new_filename}")
        renamed_count += 1
    
    print(f"\n🎉 处理完成！")
    print(f"  总文件数: {len(png_files)}")
    print(f"  重命名数: {renamed_count}")
    print(f"  未修改数: {unchanged_count}")

# ========================
# 使用示例
# ========================
if __name__ == "__main__":
    # ⚠️ 请替换为你的实际路径
    SOURCE_DIR = r"/data2/gaojiahao/Ultrasound_examination/2_substantia_nigra_Segmentation/0_sn_data/raw_data/images"
    TARGET_DIR = r"/data2/gaojiahao/Ultrasound_examination/2_substantia_nigra_Segmentation/0_sn_data/raw_data/images000"
    
    rename_and_copy_png_files(SOURCE_DIR, TARGET_DIR)