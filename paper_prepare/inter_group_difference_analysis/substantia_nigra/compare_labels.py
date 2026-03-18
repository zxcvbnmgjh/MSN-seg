import os
from PIL import Image
import numpy as np

def calculate_dice(mask1, mask2):
    """计算两个二值掩码的 Dice 系数"""
    mask1 = np.array(mask1).astype(bool)
    mask2 = np.array(mask2).astype(bool)
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = mask1.sum() + mask2.sum()
    
    if union == 0:
        return 1.0  # 两者都为空，视为完全一致
    
    dice = (2. * intersection) / union
    return dice

# 👇 设置路径（请替换为你自己的路径）
folder1_path = "/data2/gaojiahao/paper_prepare/inter_group_difference_analysis/substantia_nigra/stu_masks_old"      # 原始标签文件夹1
folder2_path = "/data2/gaojiahao/paper_prepare/inter_group_difference_analysis/substantia_nigra/stu_masks_new"      # 对比标签文件夹2
folder3_path = "/data2/gaojiahao/paper_prepare/inter_group_difference_analysis/substantia_nigra/diff_labels_v1" # 存放 folder1 中 Dice≠1 的文件
folder4_path = "/data2/gaojiahao/paper_prepare/inter_group_difference_analysis/substantia_nigra/diff_labels_v2" # 存放 folder2 中 Dice≠1 的文件

# 创建输出文件夹（如果不存在）
os.makedirs(folder3_path, exist_ok=True)
os.makedirs(folder4_path, exist_ok=True)

# 获取两个文件夹中的 PNG 文件列表
files1 = {f for f in os.listdir(folder1_path) if f.lower().endswith('.png')}
files2 = {f for f in os.listdir(folder2_path) if f.lower().endswith('.png')}

# 找出交集（同名文件）
common_files = files1 & files2

print(f"共找到 {len(common_files)} 个同名文件")

diff_count = 0

for filename in sorted(common_files):
    path1 = os.path.join(folder1_path, filename)
    path2 = os.path.join(folder2_path, filename)
    
    try:
        img1 = Image.open(path1).convert('L')  # 转为灰度图
        img2 = Image.open(path2).convert('L')
        
        dice = calculate_dice(img1, img2)
        
        if abs(dice - 1.0) > 1e-6:  # 避免浮点误差
            diff_count += 1
            print(f"⚠️ {filename}: Dice = {dice:.6f} ≠ 1.0")
            
            # 复制到目标文件夹
            dest1 = os.path.join(folder3_path, filename)
            dest2 = os.path.join(folder4_path, filename)
            
            img1.save(dest1)
            img2.save(dest2)
        else:
            print(f"✅ {filename}: Dice = {dice:.6f} (完全一致)")
            
    except Exception as e:
        print(f"❌ {filename} 处理失败: {e}")

print(f"\n✅ 完成！共发现 {diff_count} 个不一致的文件，已保存至 {folder3_path} 和 {folder4_path}")