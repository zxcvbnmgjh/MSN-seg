import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class SubstantiaNigra_Train(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(SubstantiaNigra_Train, self).__init__()
        self.flag = "train" if train else "val" #若train为true则载入training下的数据集，若为false则载入test下数据集
        data_root = os.path.join(root, "0_sn_data", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms #数据的预处理方式
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".png")] #得到每张图片的名称
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names] #每张图片的路径 
        

        self.mask = [os.path.join(data_root, "masks", i.split("_")[0] + f"_mask.png")
                         for i in img_names] #得到每个mask文件的路径
        # check files，检测每个mask是否都存在
        for i in self.mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB') #打开索引为 idx 的图像文件，并将其转换为 RGB 模式
        mask = Image.open(self.mask[idx]).convert('L') #打开索引为 idx 的手动标注掩码文件，并将其转换为单通道的灰度图像（模式为 'L'）
        mask = np.array(mask) #将手动标注掩码转换为 NumPy 数组，并将像素值归一化到 [0, 1] 范围。(前景像素为1，背景像素为0)
        mask[mask == 255] = 1   
     

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


class SubstantiaNigra_Test(Dataset):
    def __init__(self, root: str, test: bool, transforms=None):
        super(SubstantiaNigra_Test, self).__init__()
        self.flag = "test" if test else "val" #若test为true则载入training下的数据集，若为false则载入test下数据集
        data_root = os.path.join(root, "0_sn_data", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms #数据的预处理方式
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".png")] #得到每张图片的名称
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names] #每张图片的路径 
        

        self.mask = [os.path.join(data_root, "masks", i.split("_")[0] + f"_mask.png")
                         for i in img_names] #得到每个mask文件的路径
        # check files，检测每个mask是否都存在
        for i in self.mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB') #打开索引为 idx 的图像文件，并将其转换为 RGB 模式
        mask = Image.open(self.mask[idx]).convert('L') #打开索引为 idx 的手动标注掩码文件，并将其转换为单通道的灰度图像（模式为 'L'）
        mask = np.array(mask) / 255 #将手动标注掩码转换为 NumPy 数组，并将像素值归一化到 [0, 1] 范围。(前景像素为1，背景像素为0)
     

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets
    
"""
class SubstantiaNigra_Full(Dataset):
    
    def __init__(self, root: str, transforms=None):
        super(SubstantiaNigra_Full, self).__init__()
        self.flag = "full" # 载入所有数据集
        data_root = os.path.join(root, "0_sn_data", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms #数据的预处理方式
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".png")] #得到每张图片的名称
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names] #每张图片的路径 
        self.mask = [os.path.join(data_root, "masks", i.split("_")[0] + f"_mask.png")
                         for i in img_names] #得到每个mask文件的路径
        # check files，检测每个mask是否都存在
        for i in self.mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")
    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB') #打开索引为 idx 的图像文件，并将其转换为 RGB 模式
        mask = Image.open(self.mask[idx]).convert('L') #打开索引为 idx 的手动标注掩码文件，并将其转换为单通道的灰度图像（模式为 'L'）
        mask = np.array(mask) #将手动标注掩码转换为 NumPy 数组，并将像素值归一化到 [0, 1] 范围。(前景像素为1，背景像素为0)
        mask[mask == 255] = 1   
    
        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        return img, mask

    def __len__(self):
        return len(self.img_list)
    
    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets
"""
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

# 假设 cat_list 来自 transforms 或其他地方；若未定义，请确保导入
# 例如：from transforms import cat_list

class SubstantiaNigra_Full(Dataset):
    
    def __init__(self, root: str, transforms=None):
        super(SubstantiaNigra_Full, self).__init__()
        self.flag = "full"  # 载入所有数据集
        data_root = os.path.join(root, "0_sn_data", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exist."
        
        self.transforms = transforms
        
        # 获取图像文件名（如 "001.png", "002.png" ...）
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".png")]
        img_names.sort()  # ✅ 强烈建议排序，确保顺序可复现
        
        # 保存文件名列表（关键！供外部打印使用）
        self.img_names = img_names  # <<< 新增：支持 dataset.img_names
        
        # 构建完整路径
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.mask = [os.path.join(data_root, "masks", i.split("_")[0] + f"_mask.png") for i in img_names]
        
        # 检查 mask 是否存在
        for i in self.mask:
            if not os.path.exists(i):
                raise FileNotFoundError(f"mask file {i} does not exist.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        mask = Image.open(self.mask[idx]).convert('L')
        mask = np.array(mask)
        mask[mask == 255] = 1
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)
    
    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets