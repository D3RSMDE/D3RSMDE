import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import random


class ImageToDEMDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, splitNum=0.9, split_text=None, stage1Flag=True,
                 **kwargs):
        assert mode in ['train', 'val', 'test'], "mode best be 'train', 'val' or 'test'"
        self.mode = mode
        sub = 'singleRGBNormalization' if mode == 'train' or mode == 'val' else 'singleRGBNormalizationTest'
        # sub = 'singleRGBNormalizationTest'  # todo
        data_dir = os.path.join(root_dir, sub)
        self.rgb_dir = os.path.join(data_dir, 'png-stretched-unique')
        self.dem_dir = os.path.join(data_dir, 'DEM_255-unique')
        self.stage1_depth_dir = os.path.join(data_dir, 'stage1_depth')

        self.indices = []
        for fn in os.listdir(self.rgb_dir):
            if fn.startswith('tile_') and fn.endswith('.png'):
                idx = fn[len('tile_'):-len('.png')]
                self.indices.append(idx)

        if split_text:
            tempArray = open(split_text, 'r').read().splitlines()
            for i in range(len(tempArray)):
                if tempArray[i] not in self.indices:
                    raise ValueError(f"Index {tempArray[i]} in split text not found in dataset.")
                tempArray[i] = int(tempArray[i])
            self.indices = tempArray
        random.seed(kwargs['seed'])
        random.shuffle(self.indices)
        if mode == 'train':
            if kwargs.get('not_split', False):
                splitNum = 1.0
            split_idx = int(len(self.indices) * splitNum)
            self.indices = self.indices[:split_idx]
        elif mode == 'val':
            split_idx = int(len(self.indices) * splitNum)
            self.indices = self.indices[split_idx:]
        self.transform = transform
        self.stage1Flag = stage1Flag

        print(f"successfully loaded {mode} dataset with {len(self.indices)} samples, "
              f"RGB path: {self.rgb_dir}, DEM path: {self.dem_dir}, Stage1 depth path: {self.stage1_depth_dir}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]

        rgb_path = os.path.join(self.rgb_dir, f'tile_{index}.png')
        dem_path = os.path.join(self.dem_dir, f'tile_{index}.tif')
        stage1_depth_path = os.path.join(self.stage1_depth_dir, f'tile_{index}_pred_depth.png')


        rgb = Image.open(rgb_path).convert('RGB')
        dem = Image.open(dem_path)
        if self.stage1Flag:
            stage1_depth = Image.open(stage1_depth_path)

            if self.mode == 'train':

                if torch.rand(1) < 0.5:
                    rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
                    dem = dem.transpose(Image.FLIP_LEFT_RIGHT)
                    stage1_depth = stage1_depth.transpose(Image.FLIP_LEFT_RIGHT)

                if torch.rand(1) < 0.5:
                    rgb = rgb.transpose(Image.FLIP_TOP_BOTTOM)
                    dem = dem.transpose(Image.FLIP_TOP_BOTTOM)
                    stage1_depth = stage1_depth.transpose(Image.FLIP_TOP_BOTTOM)
            if self.transform:
                rgb = self.transform(torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float())
                dem = self.transform(torch.from_numpy(np.array(dem)).unsqueeze(0).float())
                stage1_depth = self.transform(torch.from_numpy(np.array(stage1_depth)).unsqueeze(0).float())
            else:
                rgb = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float()
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min()) * 2.0 - 1.0
                dem = torch.from_numpy(np.array(dem)).unsqueeze(0).float() / 255.0 * 2.0 - 1.0
                stage1_depth = torch.from_numpy(np.array(stage1_depth)).unsqueeze(0).float() / 255.0 * 2.0 - 1.0

                return rgb, dem, stage1_depth, rgb_path, dem_path, stage1_depth_path
        else:
            if self.mode == 'train':

                if torch.rand(1) < 0.5:
                    rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
                    dem = dem.transpose(Image.FLIP_LEFT_RIGHT)

                if torch.rand(1) < 0.5:
                    rgb = rgb.transpose(Image.FLIP_TOP_BOTTOM)
                    dem = dem.transpose(Image.FLIP_TOP_BOTTOM)
            if self.transform:
                rgb = self.transform(rgb)
                # dem = self.transform(torch.from_numpy(np.array(dem)).unsqueeze(0).float())
                dem = torch.from_numpy(np.array(dem)).unsqueeze(0).float() / 255.0 * 2.0 - 1.0  # todo
            else:
                rgb = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float()
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min()) * 2.0 - 1.0
                dem = torch.from_numpy(np.array(dem)).unsqueeze(0).float() / 255.0 * 2.0 - 1.0

            return rgb, dem, rgb_path, dem_path
