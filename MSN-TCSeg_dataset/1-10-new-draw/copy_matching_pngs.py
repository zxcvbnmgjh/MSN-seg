import os
import shutil
import argparse

def copy_matching_pngs(source_folder, target_folder, prefix_txt):
    # 读取前缀列表，去除空白字符和空行
    with open(prefix_txt, 'r') as f:
        valid_prefixes = set(line.strip() for line in f if line.strip())
    
    # 确保目标文件夹存在
    os.makedirs(target_folder, exist_ok=True)

    copied_count = 0
    for filename in os.listdir(source_folder):
        if filename.endswith('_image.png'):
            prefix = filename[:-len('_image.png')]
            if prefix in valid_prefixes:
                src_path = os.path.join(source_folder, filename)
                dst_path = os.path.join(target_folder, filename)
                shutil.copy2(src_path, dst_path)
                copied_count += 1

    print(f"共复制 {copied_count} 个文件到 {target_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="复制前缀在文本文档中出现的 _image.png 文件")
    parser.add_argument("--source", type=str, default="/data2/gaojiahao/MSN-TCSeg_dataset/split/2-images-before-tuomin", help="源文件夹路径（包含 .png 文件）")
    parser.add_argument("--target", type=str, default="/data2/gaojiahao/MSN-TCSeg_dataset/1-10-new-draw/new_image",help="目标文件夹路径（用于保存匹配的文件）")
    parser.add_argument("--prefix_txt", type=str, default="/data2/gaojiahao/MSN-TCSeg_dataset/1-10-new-draw/prefixes.txt", help="包含有效前缀的文本文档路径")
    args = parser.parse_args()

    copy_matching_pngs(args.source, args.target, args.prefix_txt)