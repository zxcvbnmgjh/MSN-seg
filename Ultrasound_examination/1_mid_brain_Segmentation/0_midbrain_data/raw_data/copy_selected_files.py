import os
import shutil

def copy_files_listed_in_txt(txt_file_path, source_folder, destination_folder):
    """
    根据txt文件中的文件名列表，从源文件夹复制文件到目标文件夹。

    :param txt_file_path: 包含文件名列表的txt文件路径
    :param source_folder: 源文件夹路径（从中查找并复制文件）
    :param destination_folder: 目标文件夹路径（复制到此处）
    """
    # 检查txt文件是否存在
    if not os.path.exists(txt_file_path):
        raise FileNotFoundError(f"TXT文件不存在: {txt_file_path}")

    # 检查源文件夹是否存在
    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"源文件夹不存在: {source_folder}")

    # 创建目标文件夹（如果不存在）
    os.makedirs(destination_folder, exist_ok=True)

    # 读取txt文件中的所有文件名（去空格和换行）
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        file_names = [line.strip() for line in f if line.strip()]  # 去除空白行

    # 统计找到和未找到的文件
    copied_count = 0
    not_found = []

    # 遍历源文件夹中的文件
    source_files = {f: os.path.join(source_folder, f) for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))}

    # 复制匹配的文件
    for filename in file_names:
        if filename in source_files:
            src_path = source_files[filename]
            dst_path = os.path.join(destination_folder, filename)
            shutil.copy2(src_path, dst_path)
            print(f"已复制: {filename}")
            copied_count += 1
        else:
            not_found.append(filename)
            print(f"未找到: {filename} (在 {source_folder} 中)")

    # 打印汇总
    print(f"\n✅ 完成！共复制 {copied_count} 个文件。")
    if not_found:
        print(f"⚠️ 未找到 {len(not_found)} 个文件:")
        for fname in not_found:
            print(f"   - {fname}")

# ================================
# ✅ 使用示例（请修改路径）
# ================================

if __name__ == "__main__":
    txt_path = "/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/0_midbrain_data/raw_data/low_dice_label.txt"           # txt文件路径，每行一个文件名
    src_folder = "/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/0_midbrain_data/raw_data/overlapped_images"         # 源文件夹（包含所有文件）
    dest_folder = "/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/0_midbrain_data/raw_data/need_doctor_test/labels"   # 目标文件夹（复制到此处）

    copy_files_listed_in_txt(txt_path, src_folder, dest_folder)