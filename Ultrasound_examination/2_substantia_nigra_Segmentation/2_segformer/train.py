import os
import time
import datetime

import torch
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from nets.segformer import SegFormer
from my_dataset import DriveDataset
import transforms as T
import numpy as np
import matplotlib.pyplot as plt


class SegmentationPresetTrain:
    def __init__(self,crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.16666081,0.16667844,0.16667177), std=(0.26213387,0.26214503,0.26214192)):

        # trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans = [T.RandomHorizontalFlip(hflip_prob)]
        if vflip_prob > 0:
            trans = [T.RandomVerticalFlip(vflip_prob)]
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0.16666081,0.16667844,0.16667177), std=(0.26213387,0.26214503,0.26214192)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.16666081,0.16667844,0.16667177), std=(0.26213387,0.26214503,0.26214192)):
    crop_size = 512

    if train:
        return SegmentationPresetTrain(crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


def create_model(num_classes,phi,pretrained):
    model   = SegFormer(num_classes=num_classes, phi=phi, pretrained=pretrained)
    return model


def main(args):
    #device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    phi = args.phi
    pretrained = args.pretrained

    # using compute_mean_std.py
    mean=(0.16666081,0.16667844,0.16667177)
    std=(0.26213387,0.26214503,0.26214192)

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_dataset = DriveDataset(args.data_path,
                                 train=True,
                                 transforms=get_transform(train=True, mean=mean, std=std))

    val_dataset = DriveDataset(args.data_path,
                               train=False,
                               transforms=get_transform(train=False, mean=mean, std=std))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=num_classes,phi=phi,pretrained=pretrained)
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    # scaler = torch.cuda.amp.GradScaler() if args.amp else None
    scaler = torch.amp.GradScaler("cuda") if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    best_dice = 0.
    best_epoch = 0  # 新增：记录最佳 Dice 对应的 epoch
    start_time = time.time()

    train_losses = []
    lr_es = []
    dies_es = []
    # val_losses = []



    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        train_losses.append(mean_loss)
        
        lr_es.append(lr)

        confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
        # val_loss, confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
        # val_losses.append(val_loss)
        dies_es.append(dice)
        val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {dice:.3f}\n"
            f.write(train_info + val_info + "\n\n")

        if args.save_best is True:
            if best_dice < dice:
                best_dice = dice
                best_epoch = epoch  # 更新最佳 epoch
            else:
                continue

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if args.save_best is True:
            torch.save(save_file, "save_weights/best_model.pth")
        else:
            torch.save(save_file, "save_weights/model_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))
    # 在日志文件末尾记录最佳 epoch 和对应的 Dice 系数
    with open(results_file, "a") as f:
        f.write(f"\nBest Dice Coefficient: {best_dice:.3f}, Achieved at Epoch: {best_epoch}\n")
    # return train_losses,lr_es,val_losses,dies_es
    return train_losses,lr_es,dies_es


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default="../", help="DRIVE root")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--phi", default="b3", help="transformer encoder")
    parser.add_argument("--pretrained", default=False, type=bool)
    parser.add_argument("--device", default="cuda:0", help="training device")
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=500, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    # TL,LR,VL,DICE = main(args)
    TL,LR,DICE = main(args)

    TLes = np.array(TL)
    RLes = np.array(LR)
    # VLes = np.array(VL)
    DICEes = np.array(DICE)

    #绘制损失图
    plt.figure(figsize=(30, 10))
    plt.subplot(1, 3, 1)
    plt.plot(TLes, label='Train_Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Train_Losses')
    plt.title('Training Loss')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(RLes, label='lr')
    plt.xlabel('Epoch')
    plt.ylabel('lr')
    plt.title('lr')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(DICEes, label='dice')
    plt.xlabel('Epoch')
    plt.ylabel('dice')
    plt.title('dice')
    plt.legend()

    plt.tight_layout()
    #---(改路径)---
    save_path = "/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/2_segformer-plpa"
    file_name='para_changing.png'
    full_path = os.path.join(save_path, file_name)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(full_path) 
    plt.savefig(full_path)  
    plt.show()