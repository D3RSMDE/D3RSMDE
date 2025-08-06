from collections import OrderedDict
import torch
import torch.distributed as dist
import numpy as np
from mmengine.model import BaseModule
from .builder import DETECTORS, build_head, build_loss
from Modules.HDN_demo import HDN_interface
from diffusers import AutoencoderKL
from Modules.tokenizer.vavae import VA_VAE
from efficientvit.ae_model_zoo import DCAE_HF


def uniform_sampler(num_steps, batch_size, device):
    all_indices = np.arange(num_steps)
    indices_np = np.random.choice(all_indices, size=(batch_size,))
    indices = torch.from_numpy(indices_np).long().to(device)
    return indices


@DETECTORS.register_module()
class D3_Diffusion(BaseModule):
    """Base class for detectors."""

    def __init__(self, task, diffusion_cfg, device, vae_type=None, vae_path=None, **kwargs):
        super(D3_Diffusion, self).__init__()
        self.task = task

        self.device = device
        if vae_type == "AutoencoderKL":
            # denoise_model = dict(type='DenoiseUNet', in_channels=8, out_channels=4, model_channels=128,
            #                      num_res_blocks=2,
            #                      num_heads=4, num_heads_upsample=-1, attention_strides=(4, 8), learn_time_embd=True,
            #                      channel_mult=(1, 2, 4, 4), dropout=0.0,
            #                      num_timesteps=diffusion_cfg['betas']['num_timesteps'])  # SA
            denoise_model = dict(type='DenoiseUNet', in_channels=8, out_channels=4, model_channels=128,
                                 num_res_blocks=2,
                                 num_heads=4, num_heads_upsample=-1, attention_strides=(1, 2, 4, 8),
                                 learn_time_embd=True,
                                 channel_mult=(1, 2, 4, 4),
                                 dropout=kwargs['dropout'] if "dropout" in kwargs and kwargs['dropout'] else 0.0,
                                 num_timesteps=diffusion_cfg['betas']['num_timesteps'])  # J_K / Med / Swi /Ast

            self.denoise_model = build_head(denoise_model)
            self._diffusion_init(diffusion_cfg)
            self.vae = AutoencoderKL.from_pretrained(vae_path)
        elif vae_type == "DC_AE":
            denoise_model = dict(type='DenoiseUNet', in_channels=64, out_channels=32, model_channels=256,
                                 num_res_blocks=2,
                                 num_heads=4, num_heads_upsample=-1, attention_strides=(1, 2, 4), learn_time_embd=True,
                                 channel_mult=(1, 2, 4), dropout=0.1,
                                 num_timesteps=diffusion_cfg['betas']['num_timesteps'])
            self.denoise_model = build_head(denoise_model)
            self._diffusion_init(diffusion_cfg)
            self.vae = DCAE_HF.from_pretrained(
                "/nfs5/wrz/.cache/huggingface/hub/models--mit-han-lab--dc-ae-f32c32-sana-1.0/",
                model_name="dc-ae-f32c32-sana-1.0")
        elif vae_type == "VA_VAE":

            denoise_model = dict(type='DenoiseUNet', in_channels=64, out_channels=32, model_channels=192,
                                 num_res_blocks=3,
                                 num_heads=8, num_heads_upsample=-1, attention_strides=(2, 4, 8), learn_time_embd=True,
                                 channel_mult=(1, 2, 3, 4), dropout=0.1, conv_resample=True, use_scale_shift_norm=True,
                                 num_timesteps=diffusion_cfg['betas']['num_timesteps'])
            self.denoise_model = build_head(denoise_model)
            self._diffusion_init(diffusion_cfg)
            self.vae = VA_VAE()
            self.vae.model = self.vae.model.to(device)
        else:
            raise NotImplementedError(f"VAE type {vae_type} is not implemented.")


    def _diffusion_init(self, diffusion_cfg):
        betas = diffusion_cfg['betas']
        self.betas_cumprod = np.linspace(
            betas['start'], betas['stop'],
            betas['num_timesteps'])
        betas_cumprod_prev = self.betas_cumprod[:-1]
        self.betas_cumprod_prev = np.insert(betas_cumprod_prev, 0, 1)
        self.betas = self.betas_cumprod / self.betas_cumprod_prev
        self.num_timesteps = self.betas_cumprod.shape[0]

    def sdVAE_encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:

        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)

        rgb_latent = mean * 0.18215
        return rgb_latent

    def sdVAE_encode_depth(self, depth_in: torch.Tensor) -> torch.Tensor:

        depth_in = depth_in.repeat(1, 3, 1, 1)

        return self.sdVAE_encode_rgb(depth_in)

    def forward(self, img, x_last, target):
        with torch.no_grad():
            if isinstance(self.vae, AutoencoderKL):
                img_latent = self.sdVAE_encode_rgb(img)
                x_last_latent = self.sdVAE_encode_depth(x_last)
                target_latent = self.sdVAE_encode_depth(target)
            elif isinstance(self.vae, DCAE_HF):
                img_latent = self.vae.encode(img)
                x_last = x_last.repeat(1, 3, 1, 1)
                target = target.repeat(1, 3, 1, 1)
                x_last_latent = self.vae.encode(x_last)
                target_latent = self.vae.encode(target)
            elif isinstance(self.vae, VA_VAE):
                img_latent = self.vae.encode_images(img, device=self.device)
                x_last = x_last.repeat(1, 3, 1, 1)
                target = target.repeat(1, 3, 1, 1)
                x_last_latent = self.vae.encode_images(x_last, device=self.device)
                target_latent = self.vae.encode_images(target, device=self.device)
            else:
                raise NotImplementedError(f"VAE type {type(self.vae)} is not implemented.")
        t = uniform_sampler(self.num_timesteps, img.shape[0], self.device)
        x_latent_t = self.q_sample(target_latent, x_last_latent, t, self.device)
        z_t = torch.cat((img_latent, x_latent_t), dim=1)
        pred_depth = self.denoise_model(z_t, t)
        return pred_depth, target_latent

    def q_sample(self, x_start, x_last, t, current_device):
        q_ori_probs = torch.tensor(self.betas_cumprod, device=current_device)  # [0, 0.8]
        q_ori_probs = q_ori_probs[t]
        q_ori_probs = q_ori_probs.reshape(-1, 1, 1, 1)

        transition_map = torch.ones_like(q_ori_probs) * q_ori_probs
        sample = transition_map * x_start + (1 - transition_map) * x_last
        return sample.to(x_last)

    @torch.no_grad()
    def inference(self, img, stage1_depth, process_depth_flag=False):

        indices = list(range(self.num_timesteps))[::-1]
        if not indices:
            indices = [0]

        process_depth = []
        if process_depth_flag:
            if isinstance(self.vae, AutoencoderKL):
                xs = [(self.sdVAE_encode_depth(stage1_depth), self.sdVAE_encode_rgb(img))]
                pred_depth_latent, process_depth_latent = self.p_sample_loop(xs, indices, self.device,
                                                                             process_depth_flag)
                pred_depth = self.sdVAE_decode_depth(pred_depth_latent)
                for i in range(process_depth_latent.shape[0]):
                    # Normalize each tensor and append to the list
                    process_depth.append(
                        self.sdVAE_decode_depth(process_depth_latent[i]).squeeze().squeeze())
                process_depth = torch.stack(process_depth, dim=0)
            elif isinstance(self.vae, DCAE_HF):
                xs = [(self.vae.encode(stage1_depth.repeat(1, 3, 1, 1)), self.vae.encode(img))]
                pred_depth_latent, process_depth_latent = self.p_sample_loop(xs, indices, self.device,
                                                                             process_depth_flag)
                pred_depth = self.vae.decode(pred_depth_latent).mean(dim=1, keepdim=True)
                for i in range(process_depth_latent.shape[0]):
                    # Normalize each tensor and append to the list
                    process_depth.append(self.vae.decode(process_depth_latent[i].unsqueeze(0)).squeeze().squeeze())
                process_depth = torch.stack(process_depth, dim=0)
            elif isinstance(self.vae, VA_VAE):
                xs = [(self.vae.encode_images(stage1_depth.repeat(1, 3, 1, 1), device=self.device),
                       self.vae.encode_images(img, device=self.device))]
                pred_depth_latent, process_depth_latent = self.p_sample_loop(xs, indices, self.device,
                                                                             process_depth_flag)
                pred_depth = self.vae.decode_to_images(pred_depth_latent, device=self.device)
                for i in range(process_depth_latent.shape[0]):
                    process_depth.append(
                        self.vae.decode_to_images(process_depth_latent[i].unsqueeze(0),
                                                  device=self.device).squeeze().squeeze())
                process_depth = torch.stack(process_depth, dim=0)
                pred_depth = torch.tensor(pred_depth).permute(0, 3, 1, 2).float()
                pred_depth = pred_depth.mean(dim=1, keepdim=True)
            else:
                raise NotImplementedError(f"VAE type {type(self.vae)} is not implemented.")
            for i in range(process_depth.shape[0]):
                # Normalize each tensor and append to the list
                process_depth[i] = (process_depth[i] - process_depth[i].min()) / (
                        process_depth[i].max() - process_depth[i].min())
        else:
            if isinstance(self.vae, AutoencoderKL):
                xs = [(self.sdVAE_encode_depth(stage1_depth), self.sdVAE_encode_rgb(img))]
                pred_depth = self.sdVAE_decode_depth(self.p_sample_loop(xs, indices, self.device))
            elif isinstance(self.vae, DCAE_HF):
                xs = [(self.vae.encode(stage1_depth.repeat(1, 3, 1, 1)), self.vae.encode(img))]
                pred_depth = self.vae.decode(self.p_sample_loop(xs, indices, self.device)).mean(dim=1, keepdim=True)
            elif isinstance(self.vae, VA_VAE):
                xs = [(self.vae.encode_images(stage1_depth.repeat(1, 3, 1, 1), device=self.device),
                       self.vae.encode_images(img, device=self.device))]
                pred_depth = self.vae.decode_to_images(self.p_sample_loop(xs, indices, self.device), device=self.device)
                pred_depth = torch.tensor(pred_depth).permute(0, 3, 1, 2).float()
                pred_depth = pred_depth.mean(dim=1, keepdim=True)
            else:
                raise NotImplementedError(f"VAE type {type(self.vae)} is not implemented.")

        for i in range(pred_depth.shape[0]):
            #     # Normalize each tensor and append to the list
            pred_depth[i] = (pred_depth[i] - pred_depth[i].min()) / (pred_depth[i].max() - pred_depth[i].min())
        return pred_depth, process_depth

    def sdVAE_decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:

        depth_latent = depth_latent / 0.18215

        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)

        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean

    def sdVAE_decode_rgb(self, rgb_latent: torch.Tensor) -> torch.Tensor:

        rgb_latent = rgb_latent / 0.18215

        z = self.vae.post_quant_conv(rgb_latent)
        return self.vae.decoder(z)

    def p_sample_loop(self, xs, indices, current_device, process_depth_flag=False):
        res = []
        if process_depth_flag:
            process_depth = []
        for x, img in xs:
            x_original = x.clone()
            for i in indices:
                t = torch.tensor([i] * x.shape[0], device=current_device)
                model_input = torch.cat((img, x), dim=1)

                x = self.p_sample(model_input, x_original, t)
                # x = (x - x.min()) / (x.max() - x.min())
                if process_depth_flag:
                    process_depth.append(x)
            res.append(x)
        res = torch.cat(res, dim=0)
        if process_depth_flag:
            return res, torch.stack(process_depth)
        return res

    def p_sample(self, model_input, x_original, t):
        pred_logits = self.denoise_model(model_input, t)

        t_val = t[0]
        # alpha_bar = torch.tensor(self.betas_cumprod[t_val], device=x.device, dtype=x.dtype)
        alpha_bar_prev = torch.tensor(self.betas_cumprod_prev[t_val], device=x_original.device, dtype=x_original.dtype)

        return pred_logits * alpha_bar_prev + x_original * (1 - alpha_bar_prev)

    def compute_loss(self, pred_depth, target_depth):
        return torch.nn.functional.l1_loss(pred_depth, target_depth, reduction='mean')
        # return HDN_interface(pred_depth, target_depth)
