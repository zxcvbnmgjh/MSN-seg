import torch
import numpy as np
from PIL import Image
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.spatial.distance import cdist  # 更推荐使用 cdist 计算 HD95

def calculate_metrics(pred_mask, true_mask):
    """
    计算医学图像分割的常见指标。
    :param pred_mask: 模型预测的分割掩码 (numpy array)
    :param true_mask: 真实标签的分割掩码 (numpy array)
    :return: 包含各项指标的字典
    """
    # 将掩码二值化（假设是二分类任务）
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    true_mask = (true_mask > 0.5).astype(np.uint8)

    # 计算 Dice 系数
    intersection = np.sum(pred_mask * true_mask)
    dice = (2. * intersection) / (np.sum(pred_mask) + np.sum(true_mask) + 1e-8)

    # 计算 IoU（交并比）
    union = np.sum(pred_mask) + np.sum(true_mask) - intersection
    iou = intersection / (union + 1e-8)

    # 计算 Precision 和 Recall
    precision = precision_score(true_mask.flatten(), pred_mask.flatten(), zero_division=0)
    recall = recall_score(true_mask.flatten(), pred_mask.flatten(), zero_division=0)

    # 计算 F1 分数
    f1 = f1_score(true_mask.flatten(), pred_mask.flatten(), zero_division=0)

    # ========== 计算 95% Hausdorff Distance ==========
    coords_pred = np.argwhere(pred_mask == 1)
    coords_true = np.argwhere(true_mask == 1)

    # 处理空预测或空真实标签的情况
    if len(coords_pred) == 0 or len(coords_true) == 0:
        hd95 = float('inf')  # 表示无效
    else:
        # 使用 cdist 计算所有点对距离（比 directed_hausdorff 更精确用于 HD95）
        dist_matrix = cdist(coords_pred, coords_true)

        # 取每个方向上最近距离的 95% 分位数
        percent_95_pred_to_true = np.percentile(np.min(dist_matrix, axis=1), 95)
        percent_95_true_to_pred = np.percentile(np.min(dist_matrix, axis=0), 95)

        # HD95 是两个方向的最大值
        hd95 = max(percent_95_pred_to_true, percent_95_true_to_pred)

    return {
        "Dice": dice,
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "Hausdorff_Distance 95%": hd95
    }


def calculate_test_metrics(model, data_loader, device, save_dir=None):
    model.eval()
    metrics_list = []
    image_idx = 0  # 用于命名保存的文件

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)

            # 模型预测
            outputs = model(images)
            outputs = outputs['out']
            preds = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # 前景通道概率
            targets = targets.cpu().numpy()

            # 遍历每个样本
            for pred, target in zip(preds, targets):
                # 计算指标
                metrics = calculate_metrics(pred.squeeze(), target.squeeze())
                metrics_list.append(metrics)

                """
                # 保存预测和标签图（可选）
                if save_dir is not None:
                    os.makedirs(save_dir, exist_ok=True)
                    pred_path = os.path.join(save_dir, f"pred_{image_idx}.png")
                    label_path = os.path.join(save_dir, f"label_{image_idx}.png")
                    save_prediction_mask(pred, pred_path)
                    save_prediction_mask(target, label_path)
                    image_idx += 1
                """
                

    # ========== 计算平均指标（关键：HD95 跳过 inf）==========
    avg_metrics = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]

        if key == "Hausdorff_Distance 95%":
            # 过滤掉 inf（空 mask 导致）
            valid_values = [v for v in values if not (isinstance(v, float) and np.isinf(v))]
            if len(valid_values) == 0:
                avg_metrics[key] = float('nan')
            else:
                avg_metrics[key] = np.mean(valid_values)
        else:
            # 其他指标正常平均（也可过滤 nan，但一般不会有）
            valid_values = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
            avg_metrics[key] = np.mean(valid_values) if valid_values else float('nan')

    return avg_metrics


def save_prediction_mask(mask, path):
    """
    保存单张 mask 图像为 PNG 格式
    :param mask: numpy array (H, W)
    :param path: 保存路径
    """
    mask = (mask > 0.5).astype(np.uint8) * 255
    img = Image.fromarray(mask)
    img.save(path)