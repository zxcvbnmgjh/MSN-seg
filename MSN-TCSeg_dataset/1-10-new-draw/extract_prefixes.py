import os
import argparse

def extract_prefixes_from_nrrd(folder_path, output_txt):
    prefixes = []
    for filename in os.listdir(folder_path):
        if filename.endswith('_mask.nrrd'):
            # 移除 '_image.nrrd' 后缀，获取前缀
            prefix = filename[:-len('_mask.nrrd')]
            prefixes.append(prefix)
    
    # 写入文本文档，每行一个前缀
    with open(output_txt, 'w') as f:
        for p in sorted(prefixes):  # 可选：排序以保持一致性
            f.write(p + '\n')
    
    print(f"共提取 {len(prefixes)} 个前缀，已保存至 {output_txt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取 _mask.nrrd 文件的前缀并保存到文本文件")
    parser.add_argument("--folder", type=str, default="/data2/gaojiahao/MSN-TCSeg_dataset/1-10-new-draw/nrrd_mask")
    parser.add_argument("--output", type=str, default="/data2/gaojiahao/MSN-TCSeg_dataset/1-10-new-draw/prefixes.txt", help="输出的文本文档路径（默认: prefixes.txt）")
    args = parser.parse_args()

    extract_prefixes_from_nrrd(args.folder, args.output)