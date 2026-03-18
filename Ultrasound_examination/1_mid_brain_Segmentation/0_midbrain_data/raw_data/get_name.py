import os
import re

def natural_sort_key(filename):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', filename)]

folder_path = "/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/0_midbrain_data/raw_data/images000"
output_file = "filename_list.txt"

try:
    all_entries = os.listdir(folder_path)
    files = [f for f in all_entries if os.path.isfile(os.path.join(folder_path, f))]
    
    # 使用自然排序
    files_sorted = sorted(files, key=natural_sort_key)

    with open(output_file, 'w', encoding='utf-8') as f:
        for filename in files_sorted:
            f.write(filename + '\n')

    print(f"文件名已按自然顺序保存到 {output_file}")

except FileNotFoundError:
    print(f"错误：找不到文件夹 '{folder_path}'，请检查路径是否正确。")
except Exception as e:
    print(f"发生错误：{e}")