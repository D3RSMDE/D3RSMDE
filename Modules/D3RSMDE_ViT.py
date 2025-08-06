import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from Modules.HDN_demo import HDN_interface
from .base_model import BaseModel
from .blocks import _make_encoder
from .dpt_depth import _make_fusion_block
from .vit import forward_vit
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler, LCMScheduler
from Modules.blocks import Interpolate


class DPTEncoder(BaseModel):
    """DPT-based 特征提取器，含 refinenet 融合块"""

    def __init__(
            self,
            backbone: str = "vitb_rn50_384",
            features: int = 256,
            readout: str = "project",
            use_bn: bool = False,
    ):
        super(DPTEncoder, self).__init__()

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # self.input_conv = nn.Conv2d(1, 3, kernel_size=1)

        # ViT 编码器及 reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            True,
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
        )
        # Reassemble 融合块
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
            # nn.Conv2d(features, features // 2, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(True),
            # nn.Conv2d(features // 2, features // 4, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(True),
            # nn.Conv2d(features // 4, 4, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(True),
            # nn.Identity(),
        )

        # self.decoder = nn.Sequential(
        #     nn.Flatten(),  # 【B,4,64,64】->【B,16384】 :contentReference[oaicite:7]{index=7}
        #     nn.Linear(4 * 64 * 64, 1 * 512 * 512),  # 【B,16384】->【B,1024】 :contentReference[oaicite:8]{index=8}
        #     nn.GELU(),  # 激活函数
        #     nn.Unflatten(1, (1, 64 * 8, 64 * 8))  # 重塑为【B,1,512,512】
        # )
        # self.decoder = EfficientKAN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 判断x的通道数
        # if x.shape[1] == 1:
        #     x = self.input_conv(x)  # [B,3,H,W]，将单通道图像转换为三通道
        # ViT 特征提取
        l1, l2, l3, l4 = forward_vit(self.pretrained, x)
        # Reassemble
        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)
        p4 = self.scratch.refinenet4(l4_rn)
        p3 = self.scratch.refinenet3(p4, l3_rn)
        p2 = self.scratch.refinenet2(p3, l2_rn)
        p1 = self.scratch.refinenet1(p2, l1_rn)
        # 投影
        # p1_ds = self.downsample(p1)  # [B,features, 64,  64]
        # p_feature = self.output_proj(p1_ds)  # [B,4,64,64]
        p_feature = self.scratch.output_conv(p1)
        # p_feature = self.decoder(p_feature)  # [B,1,512,512]
        return (p_feature - p_feature.min()) / (p_feature.max() - p_feature.min() + 1e-8)



class EfficientKAN(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv1x1(x)  # [B,1,64,64]
        return F.interpolate(out, scale_factor=8, mode='bilinear', align_corners=True)



class MyDit(BaseModel):
    """结合 DPT-ViT 和 Marigold diffusion 的深度估计模型"""

    def __init__(
            self,
            backbone: str = "vitb_rn50_384", features: int = 256,
            readout: str = "project", use_bn: bool = False,
            alpha=3, beta=1, loss_types=None, generator=None, **kwargs
    ):
        super(MyDit, self).__init__()
        if loss_types is None:
            loss_types = ["L1Loss"]
        self.alpha, self.beta = alpha, beta
        self.generator = generator
        # RGB 特征编码器
        self.rgb_encoder = DPTEncoder(
            backbone=backbone,
            features=features,
            readout=readout,
            use_bn=use_bn,
        )

        self.loss_types = loss_types
    def from_pretrained(self, path: str):
        """从预训练模型加载参数"""
        tempDict = torch.load(path, map_location='cpu')
        warnings = self.load_state_dict(tempDict['model_state_dict'], strict=False)
        print(f"从 {path} 加载模型成功，包含以下警告：{warnings}, 是第{tempDict['epoch']}个epoch的模型")
        # 清除tempDict['model_state_dict']
        del tempDict['model_state_dict']
        return self, tempDict

    def encode_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.rgb_encoder(rgb)

    def forward(self, rgb: torch.Tensor):
        return self.encode_rgb(rgb)

    def inference(self, rgb: torch.Tensor):
        return self.encode_rgb(rgb)

    def compute_loss(self, depth_pred, depth):
        return HDN_interface(depth_pred, depth)


# 包含以下警告：_IncompatibleKeys(missing_keys=['rgb_encoder.decoder.conv1x1.weight', 'rgb_encoder.decoder.conv1x1.bias'], unexpected_keys=['rgb_encoder.downsample.0.weight', 'rgb_encoder.downsample.0.bias', 'rgb_encoder.downsample.2.weight', 'rgb_encoder.downsample.2.bias', 'rgb_encoder.output_proj.weight', 'rgb_encoder.output_proj.bias']), 是第27个epoch的模型
