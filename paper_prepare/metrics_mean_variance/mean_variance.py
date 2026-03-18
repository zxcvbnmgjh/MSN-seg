

import os
import re
import sys
import numpy as np


def read_and_parse_metrics(file_path):
    """
    读取单个文本文件并解析其分割指标。

    Args:
        file_path (str): 文本文件的路径。

    Returns:
        dict: 包含指标名称作为键、数值作为值的字典。
              如果文件不存在或解析失败，返回空字典。
    """
    metrics = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # 使用正则表达式匹配指标行，例如 "Dice: 0.8620"
                match = re.match(r'^(\w+):\s*(\S+)$', line.strip())
                if match:
                    metric_name = match.group(1)
                    metric_value = float(match.group(2))
                    metrics[metric_name] = metric_value
    except Exception as e:
        print(f"Error reading or parsing {file_path}: {e}")
        return {}
    return metrics


def main():
    """主函数，执行整个流程。"""
    # 检查命令行参数
    if len(sys.argv) != 2:
        print("用法: python calculate_metrics_stats.py <目录路径>")
        sys.exit(1)

    directory_path = sys.argv[1]

    # 检查目录是否存在
    if not os.path.exists(directory_path):
        print(f"错误：目录 '{directory_path}' 不存在。")
        sys.exit(1)

    if not os.path.isdir(directory_path):
        print(f"错误：'{directory_path}' 不是一个目录。")
        sys.exit(1)

    # 获取目录中所有的 .txt 文件
    txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    print(f"找到 {len(txt_files)} 个文本文件。")

    # 如果没有找到文件，退出
    if len(txt_files) == 0:
        print("错误：目录中未找到任何 .txt 文件。")
        sys.exit(1)

    # 如果文件数量不是5个，给出警告但继续（可选）
    if len(txt_files) != 5:
        print(f"警告：目录中有 {len(txt_files)} 个文件，而不是预期的5个。将处理所有找到的文件。")

    # 存储所有文件的指标数据
    all_metrics_data = []

    # 遍历每个文件，读取并解析指标
    for filename in txt_files:
        file_path = os.path.join(directory_path, filename)
        print(f"正在处理: {filename}")
        metrics = read_and_parse_metrics(file_path)
        if metrics:
            all_metrics_data.append(metrics)
        else:
            print(f"跳过文件 {filename}，因为未能成功解析。")

    # 检查是否成功读取了至少一个文件
    if len(all_metrics_data) == 0:
        print("错误：未能成功解析任何文件中的指标。")
        sys.exit(1)

    # 找出所有出现过的指标名称（即所有键的并集）
    all_metric_names = set()
    for metrics in all_metrics_data:
        all_metric_names.update(metrics.keys())

    # 为每个指标创建一个列表来存储其数值
    metric_values_dict = {name: [] for name in all_metric_names}

    # 将每个文件中对应指标的数值填入对应的列表
    for metrics in all_metrics_data:
        for name, value in metrics.items():
            metric_values_dict[name].append(value)

    # 计算每个指标的均值和方差
    print("\n" + "="*50)
    print("Metrics Statistics (Mean ± Std Dev)")
    print("-"*50)
    for name in sorted(metric_values_dict.keys()):
        values = np.array(metric_values_dict[name])
        mean = np.mean(values)
        variance = np.var(values, ddof=1)  # 样本方差 (ddof=1)
        std_dev = np.sqrt(variance)
        print(f"{name:<20} {mean:>10.4f} ± {std_dev:>7.4f}")


if __name__ == "__main__":
    main()