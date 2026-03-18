import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
# from visdom import Visdom
# viz = Visdom(port=8850)
# loss_window = viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='Loss', title='loss'))
# grad_window = viz.line(Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(),
#                            opts=dict(xlabel='step', ylabel='amplitude', title='gradient'))


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

class TrainLoop:
    def __init__(
        self,
        *,
        model, # 主分割模型
        classifier, # 校准网络，提供分割结果的置信度或进行后处理
        diffusion, # GaussianDiffusion 或 SpacedDiffusion 的实例，定义了扩散过程的数学框架
        data, 
        dataloader,
        batch_size,
        microbatch, # 用于梯度累积的微批次大小
        lr,
        ema_rate, # 指数移动平均（EMA）的衰减率
        log_interval, # 记录日志的间隔步数
        save_interval, # 保存检查点的间隔步数
        resume_checkpoint,
        use_fp16=False, # 是否使用混合精度训练（FP16）
        fp16_scale_growth=1e-3, # 混合精度训练中，损失缩放因子的增长率
        schedule_sampler=None, # 时间步采样器，默认为 UniformSampler
        weight_decay=0.0, # 优化器的 L2 正则化系数，用于防止过拟合
        lr_anneal_steps=0, # 总训练步数。训练循环会一直运行，直到 self.step + self.resume_step 达到这个值。如果为0，则训练无限进行。
    ):
        self.model = model
        self.dataloader=dataloader
        self.classifier = classifier
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters() # 从 checkpoint 加载模型参数（如果 resume_checkpoint 存在）
        self.mp_trainer = MixedPrecisionTrainer( # 封装模型，支持 FP16 训练
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        # 从检查点（checkpoint）加载模型参数，并在分布式训练环境下同步所有 GPU 上的模型参数
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            print('resume model')
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_part_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        # 为指定的 EMA（指数移动平均）衰减率 rate，加载或初始化对应的 EMA 模型参数，并在分布式训练中进行同步。 
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        # 从检查点文件中加载优化器（optimizer）的状态，以便在恢复训练时保持学习率、动量、历史梯度等信息的连续性
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        # run_loop 方法是整个扩散模型训练的主循环（training loop），它控制了训练的流程：从数据加载、前向传播与反向传播（通过 run_step）、日志记录到模型保存
        # 持续运行训练步骤，直到达到预设的总训练步数（lr_anneal_steps），期间循环加载数据、执行训练、记录日志并定期保存模型
        
        i = 0 # 本地计数器（当前 epoch 内的 batch 数），仅用于调试或监控
        data_iter = iter(self.dataloader) # 将 DataLoader 转为 Python 迭代器，支持手动调用 next()
        
        while (
            # 如果 lr_anneal_steps == 0 → 训练无限进行（直到手动中断）
            # 当 self.step + self.resume_step >= lr_anneal_steps 时停止训练
            not self.lr_anneal_steps 
            or self.step + self.resume_step < self.lr_anneal_steps
        ):


            try:
                    batch, cond, name = next(data_iter) # 尝试获取下一个 batch
            except StopIteration: # 数据跑完了
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader
                    data_iter = iter(self.dataloader) # 重新创建 data_iter = iter(dataloader)，继续训练（相当于进入下一个 epoch）
                    batch, cond, name = next(data_iter) 
                    # batch：输入图像张量，shape (B, C, H, W)，已归一化
                    # cond：条件信息（可选），如类别标签、分割图、文本嵌入等
                    # name：样本名称或路径（用于调试或可视化）

            self.run_step(batch, cond)

           
            i += 1
          
            if self.step % self.log_interval == 0:
                logger.dumpkvs() # 每隔 log_interval 步，调用 logger.dumpkvs()，将当前缓存的所有标量（loss、lr、time 等）写入日志系统
            
            if self.step % self.save_interval == 0: # 每隔 save_interval 步，调用 self.save() 保存 checkpoint
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        # 保存最终模型
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        batch=th.cat((batch, cond), dim=1)
        # 将输入的图像 (batch) 和条件分割图 (cond) 沿着通道维度拼接，形成模型的输入 
        # batch: [B, 3, H, W] —— RGB 图像；cond: [B, 1, H, W] —— 单通道分割图或语义标签图
        cond={} # 条件只通过拼接输入提供
        
        sample = self.forward_backward(batch, cond) # 执行前向传播和反向传播，计算梯度
        
        took_step = self.mp_trainer.optimize(self.opt) # 调用优化器的 step() 方法，根据累积的梯度更新模型参数。
        # mp_trainer.optimize(opt) 是 MixedPrecisionTrainer 的方法

        if took_step:
            self._update_ema() # 只有当参数真正更新时，才更新 EMA 权重
        self._anneal_lr() # 线性衰减学习率
        self.log_step() # 记录当前的训练步数和已处理的样本总数
        return sample

    def forward_backward(self, batch, cond):

        self.mp_trainer.zero_grad() # 清空上一个 batch 的梯度

        for i in range(0, batch.shape[0], self.microbatch): # 循环处理 microbatch
            
            micro = batch[i : i + self.microbatch].to(dist_util.dev())# 将大 batch 拆成多个 microbatch，逐个计算梯度并累加
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            } # 将条件字典中的每个张量也按 microbatch 切片
            last_batch = (i + self.microbatch) >= batch.shape[0] 
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev()) # 从采样器中获取一个时间步 t 和用于重要性采样的权重 weights。

            compute_losses = functools.partial(
                self.diffusion.training_losses_segmentation,
                self.ddp_model,
                self.classifier,
                micro,
                t,
                model_kwargs=micro_cond,
            ) # 使用 functools.partial 创建一个预填充了大部分参数的函数，简化后续调用。

            if last_batch or not self.use_ddp:
                losses1 = compute_losses()

            else:
                with self.ddp_model.no_sync():
                    losses1 = compute_losses() # 最后一个 microbatch 才进行同步，触发全局梯度更新

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses1[0]["loss"].detach()
                )
            losses = losses1[0]
            sample = losses1[1]

            loss = (losses["loss"] * weights + losses['loss_cal'] * 10).mean()

            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)
            for name, param in self.ddp_model.named_parameters():
                if param.grad is None:
                    print(name)
            return  sample

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"savedmodel{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"emasavedmodel_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"optsavedmodel{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
