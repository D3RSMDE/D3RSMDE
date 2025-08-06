"""
Vision Foundation Model Aligned VAE.
It has exactly the same architecture as the LDM VAE (or VQGAN-KL).
Here we first provide its inference implementation with diffusers. 
The training code will be provided later. 

"LightningDiT + VA_VAE" achieves state-of-the-art Latent Diffusion System
with 0.27 rFID and 1.35 FID on ImageNet 256x256.

by Maple (Jingfeng Yao) from HUST-VL
"""

import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms
from Modules.tokenizer.autoencoder import AutoencoderKL


class VA_VAE:
    """Vision Foundation Model Aligned VAE Implementation"""

    def __init__(self, config='Modules/tokenizer/configs/vavae_f16d32.yaml', img_size=512, horizon_flip=0.5):
        """Initialize VA_VAE
        Args:
            config: Configuration dict containing img_size, horizon_flip and fp16 parameters
        """
        self.config = OmegaConf.load(config)
        self.embed_dim = self.config.model.params.embed_dim
        # self.embed_dim = self.config.model.in_chans
        self.ckpt_path = self.config.ckpt_path
        self.img_size = img_size
        self.horizon_flip = horizon_flip
        self.load()

    def load(self):
        """Load and initialize VAE model"""
        self.model = AutoencoderKL(
            embed_dim=self.embed_dim,
            ch_mult=(1, 1, 2, 2, 4),
            ckpt_path=self.ckpt_path
        ).cuda().eval()
        return self

    def img_transform(self):
        """Image preprocessing transforms
        Args:
            p_hflip: Probability of horizontal flip
            img_size: Target image size, use default if None
        Returns:
            transforms.Compose: Image transform pipeline
        """
        img_transforms = [
            # transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, img_size)),
            # transforms.RandomHorizontalFlip(p=p_hflip),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ]
        return transforms.Compose(img_transforms)

    def encode_images(self, images, device="cpu"):
        """Encode images to latent representations
        Args:
            images: Input image tensor
        Returns:
            torch.Tensor: Encoded latent representation
        """
        with torch.no_grad():
            posterior = self.model.encode(images.to(device))
        return posterior.sample()

    def decode_to_images(self, z, device="cpu"):
        """Decode latent representations to images
        Args:
            z: Latent representation tensor
        Returns:
            np.ndarray: Decoded image array
        """
        with torch.no_grad():
            images = self.model.decode(z.to(device))
            images = torch.clamp(127.5 * images + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu",
                                                                                        dtype=torch.uint8).numpy()
        return images


if __name__ == "__main__":
    vae = VA_VAE('tokenizer/configs/vavae_f16d32_vfdinov2.yaml')
    vae.load()
