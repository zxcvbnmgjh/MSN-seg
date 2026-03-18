import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
from torch.utils.data import Dataset


def random_rot_flip(image, label): # 用于数据增强，在训练阶段随机旋转和翻转图像及其对应标签  
    k = np.random.randint(0, 4) # 取值范围为 [0, 3]
    image = np.rot90(image, k) # 会把图像旋转 90°×k 次。
    label = np.rot90(label, k)
    # 随机选择一个翻转方向：0=上下翻转，1=左右翻转
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label): # 数据增强函数，用于在训练中对图像和标签进行随机角度旋转。
    angle = np.random.randint(-20, 20) # 从区间 [-20, 20) 中随机取一个整数角度
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    # （要旋转的数组（图像或标签），旋转角度（正值表示逆时针旋转），插值方式，0 表示最近邻插值，保持输出尺寸与输入一致，不扩大画布）
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object): #数据增强与预处理模块
    """
    每次调用都会：
        随机进行旋转或翻转；
        调整到统一尺寸；
        转换为 torch.Tensor 格式返回。
    """
    def __init__(self, output_size):
        self.output_size = output_size # 初始化时传入一个输出尺寸 output_size,所有增强后的图像都会被缩放到该尺寸。
 
    def __call__(self, sample):
        image, label = sample['image'], sample['label'] # 从传入的字典中取出 image 和 label
 
        if random.random() > 0.5: # 有 50% 概率执行 随机旋转 + 翻转
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5: # 否则有 50% 概率执行 随机小角度旋转
            image, label = random_rotate(image, label)
        x, y = image.shape # 读取原始图像的大小 (x, y)
        if x != self.output_size[0] or y != self.output_size[1]: # 如果和目标尺寸不一致，就通过 scipy.ndimage.zoom 进行缩放。
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0) # 转换为 PyTorch 张量; unsqueeze(0) 增加一个通道维度，使形状从 [H, W] → [1, H, W]；
        label = torch.from_numpy(label.astype(np.float32)).unsqueeze(0)
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapsedataset(Dataset):
    """
    1.从文件系统中读取 .npy 或 .h5 格式的数据；
    2.根据训练/验证模式加载不同结构的样本；
    3.对训练集样本执行数据增强；
    4.返回 PyTorch 模型可直接使用的 (image, label) 张量。
    """
    def __init__(self, base_dir, list_dir, split="train", transform=transforms.Compose([RandomGenerator(output_size=[256, 256])])):
        # base_dir: 数据集根目录   /data2/gaojiahao/HiDiff-main/datasets/data/Synapse
        # list_dir: 存放样本列表（txt文件）的目录   /data2/gaojiahao/HiDiff-main/datasets/data/Synapse/list
        # split: 当前模式（train、val、test）
        # transform: 数据增强函数
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list) # 返回样本数量，方便 DataLoader 知道迭代次数

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')  
            data_path = os.path.join(self.data_dir, self.split, slice_name)
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = os.path.join(self.data_dir, self.split, vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform and self.split == "train":
            sample = self.transform(sample) #若为训练模式，则执行前面定义的 RandomGenerator，完成随机旋转/翻转/缩放等增强
            sample["label"] = torch.nn.functional.one_hot(sample["label"].long(), num_classes=9)[0].permute(2,0,1)
            # one_hot(..., num_classes=9)→ 把类别标签从整数（0–8）变成 one-hot 向量（[H,W,9]）；
            # [0]→ 因为 sample["label"] 原本形状是 [1,H,W]，去掉第一个维度；
            # .permute(2,0,1)→ 调整维度顺序为 [C,H,W]，以匹配 PyTorch 的卷积输入格式。
        return (sample["image"].float(), sample["label"].float())
