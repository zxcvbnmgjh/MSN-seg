# 对测试集进行预测，并保存结果
import argparse
import os
from ssl import OP_NO_TLSv1
import nibabel as nib
# from visdom import Visdom
# viz = Visdom(port=8850)
import sys
import random
sys.path.append(".")
import numpy as np
import time
import torch as th
from PIL import Image
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.midbrainloader import midbrainDataset
import torchvision.utils as vutils
from guided_diffusion.utils import staple
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torchvision.transforms as transforms
from torchsummary import summary
import re  # 添加正则模块

seed = 10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def visualize(img):  # 图像归一化函数
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img


def main():
    # 加载一个预训练的扩散模型，对测试集中的图像进行分割预测，并通过集成学习（Ensemble）提升结果的鲁棒性和准确性
    args = create_argparser().parse_args()  # 定义推理时的参数，解析命令行参数
    dist_util.setup_dist(args)  # 初始化分布式环境，即使在单卡上运行也需调用
    logger.configure(dir=args.out_dir)  # 设置日志和结果的输出目录

    if args.data_name == 'midbrain':
        tran_list = [transforms.CenterCrop(args.image_size), transforms.ToTensor()]
        transform_test = transforms.Compose(tran_list)

        ds = midbrainDataset(args, args.data_dir, transform_test, mode='val')
        args.in_ch = 4

    # 创建 DataLoader 并将其转换为迭代器 data，以便在循环中使用 next(data) 获取数据
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True
    )
    data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(  # 创建与训练时一致的 U-Net + Diffusion 模型
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    all_images = []

    # 加载预训练权重
    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    for _ in range(len(data)):
        b, m, path = next(data)  # b: 原始图像, m: 真实掩码 (用于 debug), path: 文件路径
        c = th.randn_like(b[:, :1, ...])  # 生成一个与输入通道数相同的随机噪声
        img = th.cat((b, c), dim=1)  # 拼接，输入变为 [B, C+1, H, W]

        # ✅ 修复：正确提取 midbrainX_image.png 中的 X（如 160）
        filename = os.path.basename(path[0])
        match = re.search(r'midbrain(\d+)', filename)
        if not match:
            raise ValueError(f"无法从文件名 {filename} 中提取编号，请检查文件命名格式。")
        slice_ID = match.group(1)  # 如 '160'

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)

        # 采样与集成
        enslist = []
        for i in range(args.num_ensemble):  # this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org, cal, cal_out = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                step=args.diffusion_steps,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))

            co = cal_out.clone().detach()

            enslist.append(sample[:, -1:, :, :])
            """
            if args.version == 'new':
                enslist.append(sample[:, -1, :, :])
            else:
                enslist.append(co)
            """
            

            if args.debug:
                if args.data_name == 'midbrain':
                    s = sample[:, -1, :, :]
                    b, h, w = s.size()
                    ss = s.clone()
                    ss = ss.view(s.size(0), -1)
                    ss -= ss.min(1, keepdim=True)[0]
                    ss /= ss.max(1, keepdim=True)[0]
                    ss = ss.view(b, h, w)
                    ss = ss.unsqueeze(1).repeat(1, 3, 1, 1)
                    o = th.tensor(org)[:, :-1, :, :]
                    c = th.tensor(cal).repeat(1, 3, 1, 1)
                    tup = (ss, o, c)
                elif args.data_name == 'BRATS':
                    s = th.tensor(sample)[:, -1, :, :].unsqueeze(1)
                    m = th.tensor(m.to(device='cuda:0'))[:, 0, :, :].unsqueeze(1)
                    o1 = th.tensor(org)[:, 0, :, :].unsqueeze(1)
                    o2 = th.tensor(org)[:, 1, :, :].unsqueeze(1)
                    o3 = th.tensor(org)[:, 2, :, :].unsqueeze(1)
                    o4 = th.tensor(org)[:, 3, :, :].unsqueeze(1)
                    c = th.tensor(cal)
                    tup = (o1/o1.max(), o2/o2.max(), o3/o3.max(), o4/o4.max(), m, s, c, co)

                compose = th.cat(tup, 0)
                vutils.save_image(compose, fp=os.path.join(args.out_dir, f"{slice_ID}_output_{i}.png"), nrow=1, padding=10)

        # 将多次采样（Ensemble）得到的多个分割掩码结果进行融合，生成一个更鲁棒、更可靠的最终分割图，并将其保存为图像文件。
        ensres = staple(th.stack(enslist, dim=0)).squeeze(0)
        vutils.save_image(ensres, fp=os.path.join(args.out_dir, f"{slice_ID}_output_ens.png"), nrow=1, padding=10)


def create_argparser():
    defaults = dict(
        data_name='midbrain',
        data_dir="./data/midbrain/fold0/val",  # 测试集数据目录路径
        clip_denoised=True,  # 是否将去噪结果裁剪到有效范围（如 [0,1]）
        num_samples=1,  # 每张图像生成的样本数量（通常为 1）
        batch_size=1,
        use_ddim=False,  # 是否使用DDIM采样（加速推理），False表示使用标准 DDPM
        model_path="/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/5_medsegdiff/weights_results/fold0/emasavedmodel_0.9999_021000.pt",
        num_ensemble=5,  # number of samples in the ensemble ，集成预测次数：对同一张图像运行多次去噪，提升分割鲁棒性
        gpu_dev="0",
        out_dir='./sample_results/fold0/',
        multi_gpu="0,1",  # "0,1,2"
        debug=False  # 是否开启调试模式，保存中间可视化结果（原始图、GT、预测图等）
        
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()