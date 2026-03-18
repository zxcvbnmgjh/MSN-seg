import os
import numpy as np
from PIL import Image
from skimage.metrics import hausdorff_distance

# -----------------------------
# 指标计算函数
# -----------------------------

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true & y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

def iou(y_true, y_pred):
    intersection = np.sum(y_true & y_pred)
    union = np.sum(y_true | y_pred)
    return intersection / union if union != 0 else 0

def precision(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    return true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0

def recall(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    return true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

def f1_score(prec, rec):
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

def hausdorff_distance_metric(y_true, y_pred):
    return hausdorff_distance(y_true, y_pred)

# -----------------------------
# 图像加载函数
# -----------------------------
def load_binary_image(path):
    """加载RGB图像并转换为二值数组（0或1），基于像素是否为(0,0,0)或(1,1,1)"""
    img = Image.open(path).convert('RGB')  # 强制转为RGB格式
    img_array = np.array(img)

    # 创建一个全零的二维数组作为二值图像
    binary_array = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=bool)

    # 遍历每个像素点，判断是否为 (1,1,1)
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            pixel = tuple(img_array[i, j])
            if pixel == (1, 1, 1):
                binary_array[i, j] = True  # 1 表示目标区域
            elif pixel == (0, 0, 0):
                binary_array[i, j] = False  # 0 表示背景
            else:
                raise ValueError(f"无效像素值 {pixel} 在图像 {path} 中出现。只允许 (0,0,0) 和 (1,1,1)")

    return binary_array

# -----------------------------
# 主评估函数
# -----------------------------
def evaluate_folder_metrics(folder1, folder2):
    files1 = sorted([f for f in os.listdir(folder1) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))])
    files2 = sorted([f for f in os.listdir(folder2) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))])

    if len(files1) != len(files2):
        raise ValueError("两个文件夹中的图像数量不一致，请检查是否一一对应。")

    metrics = {
        'dice': [],
        'iou': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'hausdorff_distance': []
    }

    print(f"开始评估 {len(files1)} 对图像...\n")
    for idx, (file1, file2) in enumerate(zip(files1, files2)):
        print(f"正在处理第 {idx+1}/{len(files1)} 对: {file1} - {file2}")
        path1 = os.path.join(folder1, file1)
        path2 = os.path.join(folder2, file2)

        img1 = load_binary_image(path1)
        img2 = load_binary_image(path2)

        # 计算各项指标
        d = dice_coefficient(img1, img2)
        i = iou(img1, img2)
        p = precision(img1, img2)
        r = recall(img1, img2)
        f = f1_score(p, r)
        h = hausdorff_distance_metric(img1, img2)

        metrics['dice'].append(d)
        metrics['iou'].append(i)
        metrics['precision'].append(p)
        metrics['recall'].append(r)
        metrics['f1_score'].append(f)
        metrics['hausdorff_distance'].append(h)

    # 平均指标
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    return avg_metrics

# -----------------------------
# 保存到文本文件
# -----------------------------
def save_metrics_to_file(metrics_dict, output_path):
    with open(output_path, 'w') as f:
        f.write("Evaluation Metrics:\n")  # 添加标题行
        for key, value in metrics_dict.items():
            line = f"{key.replace('_', ' ').capitalize()}: {value:.4f}\n"
            f.write(line)
    print(f"\n评估完成，结果已保存至: {output_path}")

# -----------------------------
# 示例调用部分
# -----------------------------
if __name__ == '__main__':
    # 设置路径
    folder1 = "/data2/gaojiahao/nnU-net/nnUNet/DATASET/nnUNet_raw/Dataset134_MidbrainSegmentation/labelsTs"  # 第一个文件夹（真实标签）
    folder2 = "/data2/gaojiahao/nnU-net/nnUNet/DATASET/Results/134midbrain_2D_results/midbrain_2d_predict_pp"   # 第二个文件夹（预测结果）
    output_file = "/data2/gaojiahao/nnU-net/nnUNet/DATASET/Results/134midbrain_2D_results/post_process/results.txt"  # 输出文件路径

    # 计算指标
    results = evaluate_folder_metrics(folder1, folder2)

    # 保存结果
    save_metrics_to_file(results, output_file)