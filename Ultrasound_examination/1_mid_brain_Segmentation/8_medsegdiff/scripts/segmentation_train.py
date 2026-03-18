# 训练过程的数据准备、参数配置
import sys
import argparse
sys.path.append("../")
sys.path.append("./")
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.midbrainloader import midbrainDataset
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from pathlib import Path
from guided_diffusion.train_util import TrainLoop
from visdom import Visdom
viz = Visdom(port=8850)
import torchvision.transforms as transforms

def main():
    args = create_argparser().parse_args() # 调用定义的 create_argparser()，解析命令行参数

    

    dist_util.setup_dist(args) # 设置GPU分布式训练环境
    logger.configure(dir = args.out_dir) # 配置日志输出路径

    logger.log("creating data loader...")

    
    if args.data_name == 'midbrain': 
        tran_list = [transforms.RandomCrop(args.image_size), transforms.ToTensor(),]
        transform_train = transforms.Compose(tran_list)
        ds = midbrainDataset(args, args.data_dir, transform_train)
        args.in_ch = 4

    datal= th.utils.data.DataLoader( # 创建 PyTorch 数据加载器
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    data = iter(datal)  # 将其转换为可迭代对象，供 TrainLoop 使用

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(  # 构建unet和diffusion模型
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    
    if args.multi_gpu:
        model = th.nn.DataParallel(model,device_ids=[int(id) for id in args.multi_gpu.split(',')])
        model.to(device = th.device('cuda', int(args.gpu_dev)))
    else:
        model.to(dist_util.dev())
    
    



    # timestep采样器，在每一轮训练迭代中，决定从 1 到 T（总扩散步数）中选择哪个时间步（timestep）的噪声图像来计算损失
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=args.diffusion_steps)


    logger.log("training...")
    # 开始训练
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data, # 数据迭代器
        dataloader=datal, # 原始 DataLoader（用于保存 batch）
        batch_size=args.batch_size,
        microbatch=args.microbatch, # 微批次大小，用于梯度累积（-1表示禁用）
        lr=args.lr,
        ema_rate=args.ema_rate, # 指数移动平均率
        log_interval=args.log_interval, # 每 N 步打印一次日志
        save_interval=args.save_interval, # 每 N 步保存一次 checkpoint
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler, # timestep 采样器
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser(): # 配置命令行参数
    defaults = dict( # 定义默认参数
        data_name = 'midbrain',
        data_dir="./data/midbrain/fold0/train",
        schedule_sampler="uniform", # 采样策略（uniform：均匀采样|loss-aware:基于损失的采样）
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=120000,  # 学习率衰减步数，设置为0时启用线性衰减
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches 微批次大小，-1表示禁用微批次
        ema_rate="0.9999",  # comma-separated list of EMA values 指数移动平均率
        log_interval=500, # 每训练多少step输出一次日志
        save_interval=1500, # 每训练多少 step 保存一次模型 checkpoint
        resume_checkpoint=None, #"/results/pretrainedmodel.pt" 断电续训路径
        use_fp16=True, # 是否使用混合精度
        fp16_scale_growth=1e-3, # AMP（自动混合精度）中 loss scaling 的增长速率
        gpu_dev = "0", # 主GPU设备
        multi_gpu = "0,1,2", #"0,1,2" 多GPU设置
        out_dir='./weights_results/fold0_1' # 输出结果保存路径
    )
    defaults.update(model_and_diffusion_defaults()) # 合并模型和扩散过程的默认参数
    parser = argparse.ArgumentParser() # 返回对象
    add_dict_to_argparser(parser, defaults) # 将字典添加到解释器
    return parser


if __name__ == "__main__":
    main()
