import os
import re

def extract_letters_from_prefixes(folder_path, output_file_path):
    """
    遍历指定文件夹中的文件，提取文件名中 '_' 之前的前缀，
    从中过滤出字母部分（去除数字、符号），去重并按首次出现顺序保存到文本文件。

    参数：
    - folder_path: 文件夹路径（字符串）
    - output_file_path: 输出文本文件路径（字符串）
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"指定的文件夹不存在: {folder_path}")

    result = []
    seen = set()

    # 遍历文件，保持顺序（按文件名排序）
    for filename in sorted(os.listdir(folder_path)):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            name = os.path.splitext(filename)[0]  # 去掉扩展名
            if '_' in name:
                prefix = name.split('_')[0]  # 取 '_' 之前的部分
            else:
                prefix = name  # 没有 '_'，整个作为前缀

            # 提取前缀中的字母部分（连续字母，或所有字母拼接）
            letters = ''.join(re.findall(r'[a-zA-Z]', prefix))

            # 如果提取后为空（如全是数字），可跳过或保留原样，这里选择跳过
            if not letters:
                continue

            # 按首次出现顺序去重
            if letters not in seen:
                seen.add(letters)
                result.append(letters)

    # 写入输出文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for word in result:
            f.write(word + '\n')

    print(f"已将 {len(result)} 个唯一字母前缀按顺序保存至: {output_file_path}")

# ========================
# 使用示例
# ========================
if __name__ == "__main__":
    folder_path = r"/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/0_midbrain_data/raw_data/images"          # 替换为你的实际文件夹路径
    output_path = r"/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/0_midbrain_data/raw_data/letter_prefixes.txt"     # 替换为输出文件路径

    extract_letters_from_prefixes(folder_path, output_path)