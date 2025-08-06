import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import torch.nn.functional as F


class ImageToDEMDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, splitNum=0.9, split_text=None, **kwargs):

        assert mode in ['train', 'val', 'test'], "mode must be 'train', 'val' or 'test'"
        self.mode = mode

        sub = 'singleRGBNormalization' if mode == 'train' or mode == 'val' else 'singleRGBNormalizationTest'
        # sub = 'singleRGBNormalization'
        data_dir = os.path.join(root_dir, sub)
        self.rgb_dir = os.path.join(data_dir, 'png-stretched-unique')
        self.dem_dir = os.path.join(data_dir, 'DEM_255-unique')

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
            f"Successfully loaded {mode} dataset with {len(self.indices)} samples. "
            f"RGB path: {self.rgb_dir}, DEM path: {self.dem_dir}, ")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]

        rgb_path = os.path.join(self.rgb_dir, f'tile_{index}.png')
        dem_path = os.path.join(self.dem_dir, f'tile_{index}.tif')

        rgb = Image.open(rgb_path).convert('RGB')
        dem = Image.open(dem_path)

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
