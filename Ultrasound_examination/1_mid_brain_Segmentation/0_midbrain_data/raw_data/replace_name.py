import os

def replace_mask_with_label(input_file_path, output_file_path):
    """
    将文本文件中的所有 '_mask' 替换为 '_label'，并保存到新文件。

    :param input_file_path: 输入文件路径（原文件）
    :param output_file_path: 输出文件路径（保存修改后的文件）
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"输入文件不存在: {input_file_path}")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 读取原文件内容
    with open(input_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 替换 '_mask' 为 '_label'
    modified_content = content.replace('_label', '_image')

    # 写入新文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)

    print(f"已成功将 '{input_file_path}' 中的 '_label' 替换为 '_image'")
    print(f"修改后的文件已保存至: '{output_file_path}'")

# ================================
# ✅ 使用示例（请修改路径）
# ================================

if __name__ == "__main__":
    input_path = "/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/0_midbrain_data/raw_data/low_dice_label.txt"      # 修改为你的输入文件路径
    output_path = "/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/0_midbrain_data/raw_data/low_dice_image.txt"    # 修改为你的输出文件路径

    replace_mask_with_label(input_path, output_path)