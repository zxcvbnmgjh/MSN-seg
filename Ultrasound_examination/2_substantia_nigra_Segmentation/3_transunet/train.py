import os
import time
import datetime
import random
import torch
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from transunet_official_net.vit_seg_modeling import VisionTransformer as ViT_seg 
from transunet_official_net.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from my_dataset import MidBrainDataset_Train
import transforms as T
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# from transunet_unofficial_net.transunet import TransUNet
# from transunet_unofficial_net.config import cfg

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
    def __init__(self, crop_size,mean=(0.16666081,0.16667844,0.16667177), std=(0.26213387,0.26214503,0.26214192)):
        trans = [T.RandomCrop(crop_size)]
        trans.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


def set_seed(seed=42):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                 # 为CPU设置种子
    torch.cuda.manual_seed_all(seed)        # 为所有GPU设置种子
    torch.backends.cudnn.deterministic = True  # 确保卷积操作可复现
    torch.backends.cudnn.benchmark = False     # 关闭自动优化，提升可复现性
    os.environ['PYTHONHASHSEED'] = str(seed)   # 固定Python哈希种子


def get_transform(train, mean=(0.16666081,0.16667844,0.16667177), std=(0.26213387,0.26214503,0.26214192)):
    crop_size = 512

    if train:
        return SegmentationPresetTrain(crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(crop_size,mean=mean, std=std)

def main(args):
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_classes = args.num_classes 
    # using compute_mean_std.py
    mean=(0.16666081,0.16667844,0.16667177)
    std=(0.26213387,0.26214503,0.26214192)
    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    train_dataset = MidBrainDataset_Train(args.data_path,
                                 train=True,
                                 transforms=get_transform(train=True, mean=mean, std=std))
    val_dataset =   MidBrainDataset_Train(args.data_path,
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
    model.to(device)
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

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
    
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        train_losses.append(mean_loss)
        
        lr_es.append(lr)

        confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes )
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
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument('--img_size', type=int,default=512, help='input patch size of network input')
    parser.add_argument("--pretrained", default=False, type=bool)
    parser.add_argument("--device", default="cuda:0", help="training device")
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument("--epochs", default=300, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=4, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument('--n_skip', type=int,
                        default=3, help='using number of skip-connect, default is num')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip

    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    
    
    model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)
    # model.load_from(weights=np.load(config_vit.pretrained_path))

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
    save_path = "/data2/gaojiahao/1_mid_brain_Segmentation/3_TransUNet-plpa"
    file_name='para_changing.png'
    full_path = os.path.join(save_path, file_name)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(full_path) 
    plt.savefig(full_path)  
    plt.show()