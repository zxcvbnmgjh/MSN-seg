
import os
import torch
from transunet_official_net.vit_seg_modeling import VisionTransformer as ViT_seg 
from transunet_official_net.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from my_dataset import MidBrain_fold2_test
import transforms as T
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.spatial.distance import directed_hausdorff

class SegmentationPresetTest:
    def __init__(self, crop_size,mean = (0.17225,0.17229316 ,0.17226526) ,std = (0.22833531, 0.22835146, 0.22830528)):
        self.transforms = T.Compose([
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform_test(crop_size , mean = (0.17225,0.17229316 ,0.17226526), std = (0.22833531, 0.22835146, 0.22830528)):
    crop_size = 512
    return SegmentationPresetTest(crop_size=crop_size, mean=mean, std=std)


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

    # ========== 修改：计算 95% Hausdorff Distance ==========
    coords_pred = np.argwhere(pred_mask == 1)
    coords_true = np.argwhere(true_mask == 1)

    # 处理空预测或空真实标签的情况
    if len(coords_pred) == 0 or len(coords_true) == 0:
        hd95 = float('inf')  # 或 np.nan，根据你的需求
    else:
        # 计算两个方向的点到点距离
        distances_pred_to_true = directed_hausdorff(coords_pred, coords_true)[0]
        distances_true_to_pred = directed_hausdorff(coords_true, coords_pred)[0]

        # 获取所有成对距离（近似计算 HD95）
        # 注意：scipy.spatial.distance.cdist 可以计算所有点对距离
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(coords_pred, coords_true)
        
        # 取每个方向的 95% 分位数
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
        "Hausdorff_Distance 95%": hd95  # 已改为 HD95
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
    test_dataset = MidBrain_fold2_test(args.data_path,
                                       transforms=get_transform_test(crop_size= 512,mean = (0.17225,0.17229316 ,0.17226526), std = (0.22833531, 0.22835146, 0.22830528)))
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              pin_memory=True,
                                              collate_fn=test_dataset.collate_fn)

    # 创建模型，可修改
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)
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
    parser.add_argument("--data-path", default="./", help="DRIVE root")
    # exclude background
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument('--img_size', type=int,default=512, help='input patch size of network input')
    parser.add_argument("--pretrained", default=False, type=bool)
    parser.add_argument("--device", default="cuda:3", help="training device")
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument('--n_skip', type=int,
                        default=3, help='using number of skip-connect, default is num')
    # 修改最优权重路径
    parser.add_argument('--resume', default='/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/3_transunet/save_weights/5fold/fold_2/best_model.pth', help='resume from checkpoint')
   
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)