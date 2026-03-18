# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from timm.models import create_model

# 你的工程内模块（与 main.py 一致）
import utils
import MedViT  # 关键：触发模型注册（MedViT_small 等）
from datasets import build_dataset  # 复用你的数据构建与 transform


def parse_args():
    parser = argparse.ArgumentParser("MedViT test inference + metrics")

    # 模型/权重
    parser.add_argument("--model", default="MedViT_small", type=str)
    parser.add_argument("--ckpt", default="/data2/gaojiahao/SN+project/MedViT-main/medvit_cls_exp/1228_143044/checkpoint_best.pth", type=str, help="checkpoint_best.pth 或 checkpoint.pth 路径")
    parser.add_argument("--device", default="cuda", type=str)

    # 数据
    parser.add_argument("--data-set", default="IMNET-test",
                        choices=["CIFAR", "IMNET", "INAT", "INAT19", "image_folder"],
                        type=str)
    parser.add_argument("--data-path", default="/data2/gaojiahao/SN+project/MedViT-main/TCSdatatset", type=str, help="数据根目录")
    parser.add_argument("--input-size", default=224, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--pin-mem", action="store_true", default=True)

    # 仅当 data-set=image_folder 时需要
    parser.add_argument("--nb-classes", default=None, type=int,
                        help="仅在 --data-set image_folder 时必填，如 4/5 等")

    # 输出
    parser.add_argument("--out-dir", default="/data2/gaojiahao/SN+project/MedViT-main/Results/1228_143044", type=str)
    parser.add_argument("--save-preds-csv", action="store_true",
                        help="是否保存逐样本预测到 CSV（含路径/真值/预测）")
    parser.add_argument("--normalize-cm", action="store_true",
                        help="混淆矩阵按行归一化（显示每类召回分布）")

    return parser.parse_args()


@torch.no_grad()
def infer_collect(model: torch.nn.Module, loader: DataLoader, device: torch.device
                  ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """返回：y_true, y_pred, paths"""
    model.eval()

    y_true = []
    y_pred = []
    paths = []

    # torchvision.datasets.ImageFolder 会把样本路径放在 dataset.samples 里
    # DataLoader 默认不会返回 path，因此这里用一个索引技巧：
    # 我们通过 dataset.samples 顺序与 DataLoader 顺序一致来取路径（前提：sampler=SequentialSampler）
    # 为确保一致，本脚本使用 build_dataset(is_train=False) + SequentialSampler（datasets.py 内部对非分布式就是顺序采样）
    sample_paths = [p for (p, _) in loader.dataset.samples]
    cursor = 0

    for images, targets in loader:
        bs = images.size(0)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        y_true.append(targets.cpu().numpy())
        y_pred.append(preds.cpu().numpy())

        # 对齐路径
        paths.extend(sample_paths[cursor:cursor + bs])
        cursor += bs

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    return y_true, y_pred, paths


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def precision_recall_f1_from_cm(cm: np.ndarray) -> Dict[str, np.ndarray]:
    """返回每类 precision/recall/f1，以及 macro/micro/weighted"""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    support = cm.sum(axis=1).astype(np.float64)  # 每类真值数量

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    # macro
    macro_p = precision.mean()
    macro_r = recall.mean()
    macro_f1 = f1.mean()

    # micro（等价于整体 TP/FP/FN 聚合）
    TP = tp.sum()
    FP = fp.sum()
    FN = fn.sum()
    micro_p = TP / (TP + FP + 1e-12)
    micro_r = TP / (TP + FN + 1e-12)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-12)

    # weighted（按 support 加权）
    w = support / (support.sum() + 1e-12)
    weighted_p = (precision * w).sum()
    weighted_r = (recall * w).sum()
    weighted_f1 = (f1 * w).sum()

    return {
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "support_per_class": support,
        "macro": np.array([macro_p, macro_r, macro_f1]),
        "micro": np.array([micro_p, micro_r, micro_f1]),
        "weighted": np.array([weighted_p, weighted_r, weighted_f1]),
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: str, normalize: bool = False):
    cm_to_show = cm.astype(np.float64)
    if normalize:
        row_sum = cm_to_show.sum(axis=1, keepdims=True) + 1e-12
        cm_to_show = cm_to_show / row_sum

    fig = plt.figure(figsize=(7, 6))
    plt.imshow(cm_to_show, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # 数值标注
    thresh = cm_to_show.max() * 0.6
    for i in range(cm_to_show.shape[0]):
        for j in range(cm_to_show.shape[1]):
            val = cm_to_show[i, j]
            if normalize:
                text = f"{val:.2f}"
            else:
                text = f"{int(val)}"
            plt.text(j, i, text, ha="center", va="center",
                     color="white" if val > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    # -------- 构建测试集（复用你的 datasets.py / transform）--------
    # build_dataset(is_train=False) 在 IMNET 分支会读取 data_path/val
    # image_folder 分支则直接读取 args.data_path
    if args.data_set == "image_folder":
        if args.nb_classes is None:
            raise ValueError("当 --data-set image_folder 时，必须显式提供 --nb-classes")
        # build_dataset 会用 args.nb_classes 对齐校验
        args.nb_classes = int(args.nb_classes)

    dataset_test, nb_classes = build_dataset(is_train=False, args=args)
    class_names = dataset_test.classes  # ImageFolder 会提供

    test_loader = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # -------- 创建模型并加载权重（与你 main.py 的逻辑一致）--------
    print(f"[INFO] Creating model: {args.model}  num_classes={nb_classes}")
    model = create_model(args.model, num_classes=nb_classes)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    utils.load_state_dict(model, state)

    model.to(device)
    model.eval()

    # -------- 推理 + 收集预测 --------
    y_true, y_pred, paths = infer_collect(model, test_loader, device)

    # -------- 指标：Acc / Precision / Recall / F1 --------
    acc = float((y_true == y_pred).mean())
    cm = confusion_matrix_np(y_true, y_pred, num_classes=nb_classes)
    metrics = precision_recall_f1_from_cm(cm)

    print("\n================= Test Metrics =================")
    print(f"Accuracy: {acc * 100:.2f}%")

    macro_p, macro_r, macro_f1 = metrics["macro"]
    micro_p, micro_r, micro_f1 = metrics["micro"]
    w_p, w_r, w_f1 = metrics["weighted"]

    print(f"Macro   - P: {macro_p:.4f}  R: {macro_r:.4f}  F1: {macro_f1:.4f}")
    print(f"Micro   - P: {micro_p:.4f}  R: {micro_r:.4f}  F1: {micro_f1:.4f}")
    print(f"Weighted- P: {w_p:.4f}  R: {w_r:.4f}  F1: {w_f1:.4f}")

    print("\nPer-class:")
    for i, name in enumerate(class_names):
        p = metrics["precision_per_class"][i]
        r = metrics["recall_per_class"][i]
        f1 = metrics["f1_per_class"][i]
        sup = int(metrics["support_per_class"][i])
        print(f"  [{i}] {name:>15s} | support={sup:4d} | P={p:.4f} R={r:.4f} F1={f1:.4f}")

    # -------- 混淆矩阵可视化 --------
    cm_path = os.path.join(args.out_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path, normalize=False)

    cmn_path = os.path.join(args.out_dir, "confusion_matrix_norm.png")
    plot_confusion_matrix(cm, class_names, cmn_path, normalize=args.normalize_cm)

    print(f"\n[INFO] Confusion matrix saved to:\n  {cm_path}\n  {cmn_path}")

    # -------- 可选：保存逐样本预测 --------
    if args.save_preds_csv:
        import csv
        csv_path = os.path.join(args.out_dir, "predictions.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "y_true", "y_pred", "true_name", "pred_name"])
            for pth, t, pr in zip(paths, y_true.tolist(), y_pred.tolist()):
                writer.writerow([pth, t, pr, class_names[t], class_names[pr]])
        print(f"[INFO] Predictions CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
