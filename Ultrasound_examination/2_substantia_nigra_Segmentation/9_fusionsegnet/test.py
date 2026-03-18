
"""
import os
import torch
from src import UNet
from my_dataset import DriveDataset_test
import transforms as T
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.spatial.distance import directed_hausdorff


class SegmentationPresetTest:
    def __init__(self, mean=(0.12097807,0.12098149,0.12098066), std=(0.22976802,0.22977178,0.22977111)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)

def get_transform(mean=(0.12097807,0.12098149,0.12098066), std=(0.22976802,0.22977178,0.22977111)):
    return SegmentationPresetTest(mean=mean, std=std)


def calculate_metrics(pred_mask, true_mask):
    
    #计算医学图像分割的常见指标。
    #:param pred_mask: 模型预测的分割掩码 (numpy array)
    #:param true_mask: 真实标签的分割掩码 (numpy array)
    #:return: 包含各项指标的字典

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

    # 计算 Hausdorff 距离
    coords_pred = np.argwhere(pred_mask == 1)
    coords_true = np.argwhere(true_mask == 1)
    hausdorff_dist = max(directed_hausdorff(coords_pred, coords_true)[0],
                         directed_hausdorff(coords_true, coords_pred)[0])

    return {
        "Dice": dice,
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "Hausdorff_Distance": hausdorff_dist
    }


def evaluate_model(model, data_loader, device, num_classes):
    model.eval()
    metrics_list = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)

            # 模型预测
            outputs = model(images)
            outputs = outputs['out']
            # preds = torch.sigmoid(outputs).cpu().numpy()  # 假设是二分类任务
            preds = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # 选择第二个通道
            targets = targets.cpu().numpy()

            # 遍历每个样本
            for pred, target in zip(preds, targets):
                # 计算指标
                metrics = calculate_metrics(pred.squeeze(), target.squeeze())
                metrics_list.append(metrics)

    # 计算平均指标
    avg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0].keys()}
    return avg_metrics


def main(args):
    # 设备设置
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 加载测试数据集
    test_dataset = DriveDataset_test(args.data_path,
                                test=True,
                                transforms=get_transform())
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              pin_memory=True,
                                              collate_fn=test_dataset.collate_fn)

    # 创建模型
    model = UNet(in_channels=3,num_classes=args.num_classes + 1, base_c=32)
    model.to(device)

    # 加载预训练权重
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # 评估模型
    metrics = evaluate_model(model, test_loader, device, num_classes=args.num_classes + 1)

    # 打印结果
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")





def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")
    parser.add_argument("--data-path", default="/data2/gaojiahao/Ultrasound_examination/Segmentation/unet", help="DRIVE root")
    parser.add_argument('--resume', default='/data2/gaojiahao/Ultrasound_examination/Segmentation/unet/save_weights/best_model.pth', help='resume from checkpoint')
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
"""

import os
import torch
from src import UNet
from my_dataset import MidBrainDataset_Test
import transforms as T
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.spatial.distance import directed_hausdorff


class SegmentationPresetTest:
    def __init__(self, mean=(0.16666081,0.16667844,0.16667177), std=(0.26213387,0.26214503,0.26214192)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(mean=(0.16666081,0.16667844,0.16667177), std=(0.26213387,0.26214503,0.26214192)):
    return SegmentationPresetTest(mean=mean, std=std)


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

    # 计算 Hausdorff 距离
    coords_pred = np.argwhere(pred_mask == 1)
    coords_true = np.argwhere(true_mask == 1)
    hausdorff_dist = max(directed_hausdorff(coords_pred, coords_true)[0],
                         directed_hausdorff(coords_true, coords_pred)[0])

    return {
        "Dice": dice,
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "Hausdorff_Distance": hausdorff_dist
    }


def evaluate_model(model, data_loader, device, num_classes):
    model.eval()
    metrics_list = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)

            # 模型预测
            outputs = model(images)
            outputs = outputs['out']
            preds = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # 选择第二个通道
            targets = targets.cpu().numpy()

            # 遍历每个样本
            for pred, target in zip(preds, targets):
                # 计算指标
                metrics = calculate_metrics(pred.squeeze(), target.squeeze())
                metrics_list.append(metrics)

    # 计算平均指标
    avg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0].keys()}
    return avg_metrics


def main(args):
    # 设备设置
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 加载测试数据集
    test_dataset = MidBrainDataset_Test(args.data_path,
                                     test=True,
                                     transforms=get_transform())
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              pin_memory=True,
                                              collate_fn=test_dataset.collate_fn)

    # 创建模型
    model = UNet(in_channels=3, num_classes=args.num_classes + 1, base_c=32)
    model.to(device)

    # 加载预训练权重
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # 评估模型
    metrics = evaluate_model(model, test_loader, device, num_classes=args.num_classes + 1)

    # 打印结果
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # 将结果保存到文件
    result_file = os.path.join(os.path.dirname(args.resume), "test_result.txt")
    with open(result_file, "w") as f:
        f.write("Evaluation Metrics:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    print(f"Results saved to {result_file}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")
    parser.add_argument("--data-path", default="/data2/gaojiahao/Ultrasound_examination/Segmentation/unet-plpa", help="DRIVE root")
    parser.add_argument('--resume', default='/data2/gaojiahao/Ultrasound_examination/Segmentation/unet-plpa/save_weights/best_model.pth', help='resume from checkpoint')
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda:1", help="training device")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)