import os
import re

# ✅ 请修改为你的实际文件夹路径
folder_path = "/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/5_nnunet/predict_results/test/fold0"

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.lower().endswith('.png'):
        # 使用正则匹配 midbrain_ 后跟数字（支持任意位数，如 001, 1, 1234）
        match = re.match(r'^midbrain_(\d+)\.png$', filename, re.IGNORECASE)
        if match:
            number_part = match.group(1)  # 提取数字部分，如 "001"
            new_name = f"midbrain{number_part}_image_mask.png"
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)
            
            # 避免覆盖已存在的文件
            if os.path.exists(new_path):
                print(f"⚠️ 跳过（目标文件已存在）: {new_name}")
                continue
            
            # 执行重命名
            os.rename(old_path, new_path)
            print(f"✅ 重命名: {filename} → {new_name}")

print("✅ 所有匹配文件重命名完成。")