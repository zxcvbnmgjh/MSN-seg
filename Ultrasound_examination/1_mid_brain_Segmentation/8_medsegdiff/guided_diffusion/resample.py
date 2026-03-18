# 实现一个用于扩散模型的时间步采样器（Schedule Sampler）框架，智能地选择训练过程中使用的扩散时间步 t

from abc import ABC, abstractmethod

import numpy as np
import torch as th
import torch.distributed as dist


def create_named_schedule_sampler(name, diffusion, maxt):
    # 创建调度采样器，用于选择时间步长timesteps
    # 根据名称 name 创建并返回相应的采样器实例
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.采样器的名字
    :param diffusion: the diffusion object to sample for.扩散模型对象
    :param maxt: the maximum timestep to sample for.最大时间步长
    """
    if name == "uniform":
        return UniformSampler(diffusion, maxt)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    # 抽象基类，定义了所有采样器必须实现的接口
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.
        返回一个 NumPy 数组，其中每个元素对应一个时间步 t 的采样权重
        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        # 调用 weights() 获取权重数组 w
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p) # 根据概率 p 随机选择 batch_size 个时间步索引
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    # 实现最简单的均匀采样。
    def __init__(self, diffusion, maxt):
        self.diffusion = diffusion
        self._weights = np.ones([maxt])

    def weights(self):
        return self._weights


class LossAwareSampler(ScheduleSampler):
    # 引入了基于损失的学习能力，通过损失的历史记录来调整采样权重。
    def update_with_local_losses(self, local_ts, local_losses):
        """
        Update the reweighting using losses from a model.

        Call this method from each rank with a batch of timesteps and the
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting.

        :param local_ts: an integer Tensor of timesteps.
        :param local_losses: a 1D Tensor of losses.
        """
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # Pad all_gather batches to be the maximum batch size.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.

        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """


class LossSecondMomentResampler(LossAwareSampler):
    # LossAwareSampler 的一个具体实现
    # LossSecondMomentResampler 会更多地采样那些历史损失较大的时间步。因为损失大意味着模型在这些时间步上表现较差，需要更多的训练
    # 监控模型在每个时间步的损失，并动态调整采样概率，让模型更多地训练那些表现不佳的时间步，从而可能加快收敛速度
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()
