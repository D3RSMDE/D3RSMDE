import torch
import torch.nn as nn


def encode_depth(model, depth_in: torch.Tensor) -> torch.Tensor:
    mean, logvar = torch.chunk(model.quant_conv(model.encoder(depth_in)), 2, dim=1)
    # scale latent
    return mean * 0.18215


def decode_depth(model, depth_latent: torch.Tensor) -> torch.Tensor:
    return model.decoder(model.post_quant_conv(depth_latent / 0.18215))


def encode_rgb(model, rgb_in: torch.Tensor) -> torch.Tensor:
    mean, logvar = torch.chunk(model.quant_conv(model.encoder(rgb_in)), 2, dim=1)
    # scale latent
    return mean * 0.18215


def decode_rgb(model, depth_latent: torch.Tensor) -> torch.Tensor:
    return model.decoder(model.post_quant_conv(depth_latent / 0.18215))


def compute_loss(depth_pred: torch.Tensor, target_latent: torch.Tensor) -> torch.Tensor:
    return nn.functional.l1_loss(depth_pred, target_latent, reduction='mean')
