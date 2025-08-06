import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import torch.nn.functional as F


class ImageToDEMDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, splitNum=0.9, split_text=None, **kwargs):
        """
        Args:
            root_dir (str): 主文件夹路径，如 '.../ImageToDEM-XXX'
            mode (str): 'train' 或 'test'，决定读哪个子文件夹
            transform (callable, optional): 对 PIL.Image 做额外变换
        """
        assert mode in ['train', 'val', 'test'], "mode 必须是 'train' 或 'val' 或 'test'"
        self.mode = mode
        # 根据 mode 选择文件夹
        sub = 'singleRGBNormalization' if mode == 'train' or mode == 'val' else 'singleRGBNormalizationTest'
        # sub = 'singleRGBNormalization'
        data_dir = os.path.join(root_dir, sub)
        self.rgb_dir = os.path.join(data_dir, 'png-stretched-unique')
        self.dem_dir = os.path.join(data_dir, 'DEM_255-unique')

        # 枚举所有 tile 索引（假设两个文件夹内的文件一一对应）
        self.indices = []
        for fn in os.listdir(self.rgb_dir):
            if fn.startswith('tile_') and fn.endswith('.png'):
                idx = fn[len('tile_'):-len('.png')]
                self.indices.append(idx)

        if split_text:
            if isinstance(split_text, str):
                tempArray = open(split_text, 'r').read().splitlines()
                for i in range(len(tempArray)):
                    if tempArray[i] not in self.indices:
                        raise ValueError(f"Index {tempArray[i]} in split text not found in dataset.")
                    tempArray[i] = int(tempArray[i])
                self.indices = tempArray
            else:
                tempIndices = []
                for i in range(len(split_text)):
                    tempArray = open(split_text[i], 'r').read().splitlines()
                    for j in range(len(tempArray)):
                        if tempArray[j] not in self.indices:
                            raise ValueError(f"Index {tempArray[j]} in split text not found in dataset.")
                        tempArray[j] = int(tempArray[j])
                    tempIndices.extend(tempArray)
                self.indices = tempIndices

        random.shuffle(self.indices)
        if mode == 'train':
            if kwargs.get('not_split', False):
                splitNum = 1.0
            split_idx = int(len(self.indices) * splitNum)
            self.indices = self.indices[:split_idx]
        elif mode == 'val':
            if kwargs.get('not_split', False):
                splitNum = 0
            split_idx = int(len(self.indices) * splitNum)
            self.indices = self.indices[split_idx:]
        self.transform = transform
        print(
            f"成功加载 {mode} 数据集，包含 {len(self.indices)} 个样本。RGB 路径: {self.rgb_dir}, DEM 路径: {self.dem_dir}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        # 拼路径
        rgb_path = os.path.join(self.rgb_dir, f'tile_{index}.png')
        dem_path = os.path.join(self.dem_dir, f'tile_{index}.tif')

        # 读取
        rgb = Image.open(rgb_path).convert('RGB')
        dem = Image.open(dem_path)  # 单通道

        # 仅在训练模式下做随机翻转增强
        if self.mode == 'train':
            # 随机左右翻转
            if torch.rand(1) < 0.5:
                rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
                dem = dem.transpose(Image.FLIP_LEFT_RIGHT)
            # 随机上下翻转
            if torch.rand(1) < 0.5:
                rgb = rgb.transpose(Image.FLIP_TOP_BOTTOM)
                dem = dem.transpose(Image.FLIP_TOP_BOTTOM)

        # 如果传入了 transform，就用它；否则手动转为 tensor
        if self.transform:
            rgb = self.transform(rgb)
            dem = self.transform(dem)
        else:
            rgb = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float()
            dem = torch.from_numpy(np.array(dem)).unsqueeze(0).float() / 255.0
        return rgb, dem, rgb_path, dem_path


class NYUDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, splitNum=0.9, split_text=None, **kwargs):
        """
        Args:
            root_dir (str): 主文件夹路径，如 '.../ImageToDEM-XXX'
            mode (str): 'train' 或 'test'，决定读哪个子文件夹
            transform (callable, optional): 对 PIL.Image 做额外变换
        """
        assert mode in ['train', 'val', 'test'], "mode 必须是 'train' 或 'val' 或 'test'"
        self.mode = mode
        # 根据 mode 选择文件夹
        sub = 'train' if mode == 'train' or mode == 'val' else 'test'
        # sub = 'singleRGBNormalization'
        data_dir = os.path.join(root_dir, sub)
        self.rgb_dir = os.path.join(data_dir, 'rgb')
        self.dem_dir = os.path.join(data_dir, 'depth')

        # 枚举所有 tile 索引（假设两个文件夹内的文件一一对应）
        self.indices = []
        for fn in os.listdir(self.rgb_dir):
            if fn.endswith('.png'):
                self.indices.append(fn[:-4])

        if split_text:
            if isinstance(split_text, str):
                tempArray = open(split_text, 'r').read().splitlines()
                for i in range(len(tempArray)):
                    if tempArray[i] not in self.indices:
                        raise ValueError(f"Index {tempArray[i]} in split text not found in dataset.")
                    tempArray[i] = int(tempArray[i])
                self.indices = tempArray
            else:
                tempIndices = []
                for i in range(len(split_text)):
                    tempArray = open(split_text[i], 'r').read().splitlines()
                    for j in range(len(tempArray)):
                        if tempArray[j] not in self.indices:
                            raise ValueError(f"Index {tempArray[j]} in split text not found in dataset.")
                        tempArray[j] = int(tempArray[j])
                    tempIndices.extend(tempArray)
                self.indices = tempIndices

        random.shuffle(self.indices)
        if mode == 'train':
            if kwargs.get('not_split', False):
                splitNum = 1.0
            split_idx = int(len(self.indices) * splitNum)
            self.indices = self.indices[:split_idx]
        elif mode == 'val':
            if kwargs.get('not_split', False):
                splitNum = 0
            split_idx = int(len(self.indices) * splitNum)
            self.indices = self.indices[split_idx:]
        self.transform = transform
        self.height = 512
        self.width = 512
        print(
            f"成功加载 {mode} 数据集，包含 {len(self.indices)} 个样本。RGB 路径: {self.rgb_dir}, DEM 路径: {self.dem_dir}，要将其resize为 {self.height}x{self.width}。")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        # 拼路径
        rgb_path = os.path.join(self.rgb_dir, f'{index}.png')
        dem_path = os.path.join(self.dem_dir, f'{index}.png')

        # 读取
        rgb = Image.open(rgb_path).convert('RGB')
        dem = Image.open(dem_path)  # 单通道

        # 仅在训练模式下做随机翻转增强
        if self.mode == 'train':
            # 随机左右翻转
            if torch.rand(1) < 0.5:
                rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
                dem = dem.transpose(Image.FLIP_LEFT_RIGHT)

        # 如果传入了 transform，就用它；否则手动转为 tensor
        if self.transform:
            rgb = self.transform(rgb)
            dem = self.transform(dem)
        else:
            rgb = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float()
            dem = torch.from_numpy(np.array(dem)).unsqueeze(0).float()

        if rgb.shape[1] != self.height or rgb.shape[2] != self.width:
            rgb = F.interpolate(rgb.unsqueeze(0), size=(self.height, self.width), mode='bilinear',
                                align_corners=False).squeeze(0)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255
        if dem.shape[1] != self.height or dem.shape[2] != self.width:
            dem = F.interpolate(dem.unsqueeze(0), size=(self.height, self.width), mode='bilinear',
                                align_corners=False).squeeze(0)
            dem = (dem - dem.min()) / (dem.max() - dem.min())

        return rgb, dem, rgb_path, dem_path
