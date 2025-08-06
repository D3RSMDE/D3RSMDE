import argparse
import os
import json
import shutil

import lpips
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from PIL import Image
import numpy as np
from tqdm import tqdm
import time
from Utils.calculateDiff import calculate_single
from Utils.loadDiffusionDatasets import ImageToDEMDataset
from matplotlib.colors import Normalize
from matplotlib import pyplot as plt
from Modules.D3RSMDE_diffusion import D3_Diffusion
from Utils.autoSelectGPU import select_best_gpu


def generate_heatmap(diff):
    diff = Normalize()(diff) * 255
    diff = diff.astype(np.uint8)


    cmap = plt.get_cmap('jet')
    heatmap = cmap(diff / 255.0)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    return heatmap


# Evaluate function
def evaluate(config_path: str):
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    print(f"Loading configuration from {config_path}, content: {cfg}")

    # Extract configuration parameters
    dataset_path = cfg['dataset_path']
    output_dir = cfg['output_dir']
    checkpoint_path = cfg['checkpoint_path']
    seed = cfg['seed']

    # Set random seed
    torch.manual_seed(seed)

    # Prepare output directory
    timestamp = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(os.path.join(output_dir, "compare"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth_raw"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth_npy"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth_process"), exist_ok=True)
    shutil.copy(config_path, os.path.join(output_dir, os.path.basename(config_path)))
    shutil.copy(__file__, output_dir)
    shutil.copy("Utils/loadDiffusionDatasets.py", output_dir)
    shutil.copy("Modules/segrefiner_base.py", output_dir)
    # print(f"Configuration: {cfg}")
    print(f"输出文件夹：{output_dir}")

    # Load test dataset
    test_dataset = ImageToDEMDataset(root_dir=dataset_path, mode='test', seed=seed)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'])

    # Initialize model
    device = select_best_gpu()
    # device = torch.device(f'cuda:{0}')
    model = D3_Diffusion(device=device, **cfg).to(device)


    if cfg.get("pretrained_vae_path") and os.path.exists(cfg['pretrained_vae_path']):
        if cfg['vae_type'] == 'VA_VAE':
            m1, m2 = model.vae.model.load_state_dict(
                torch.load(cfg['pretrained_vae_path'], map_location="cpu")['vae_state_dict'], strict=False)
        else:
            m1, m2 = model.vae.load_state_dict(
                torch.load(cfg['pretrained_vae_path'], map_location="cpu")['vae_state_dict'], strict=False)
        print(f"loading pretrained VAE from {cfg['pretrained_vae_path']}, "
                f"model warnings: {m1}, {m2}")
    tempDict = torch.load(checkpoint_path, map_location='cpu')
    warnings1 = model.denoise_model.load_state_dict(tempDict['denoise_model_state_dict'], strict=False)
    # warnings2 = model.vae.load_state_dict(tempDict['vae_state_dict'], strict=False)
    print(f"Loading pretrained model from {checkpoint_path}, "
          f"model warnings: {warnings1}, epoch: {tempDict['epoch']}, best_val_loss: {tempDict['best_val_loss']:.6f}")
    del tempDict['denoise_model_state_dict']
    model.eval()

    allMetrics = []
    original_allMetrics = []
    total_time = 0
    original_average_mae, pred_average_mae = 0.0, 0.0
    lpips_model = lpips.LPIPS(net='alex')
    process_depth_flag = False
    # Iterate through the test set
    for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating"):
        # Get input image
        rgb_image = batch[0].to(device)
        stage1_depth = batch[2].to(device)  # [B, 1, H, W]
        # if "11081" not in batch[3][0]:
        #     continue
        # Run inference
        with torch.no_grad():
            time1 = time.time()
            pred_depth, process_depth = model.inference(rgb_image, stage1_depth, process_depth_flag=process_depth_flag)
            total_time += time.time() - time1

            stage1_depth = ((stage1_depth - stage1_depth.min()) / (stage1_depth.max() - stage1_depth.min())).cpu()
            pred_depth = pred_depth.cpu()
            rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min()) * 255
            rgb_pil = Image.fromarray((rgb_image.squeeze(0).permute(1, 2, 0).cpu().numpy()).astype(np.uint8))
            stage1_depth_pil = Image.fromarray((stage1_depth.squeeze(0).squeeze(0).numpy() * 255).astype(np.uint8))

            # Calculate evaluation metrics
            target_depth = batch[1].squeeze(0).squeeze(0).cpu()
            target_depth_numpy = ((target_depth - target_depth.min()) / (
                    target_depth.max() - target_depth.min())).numpy()
            pred_depth_numpy = pred_depth.squeeze(0).squeeze(0).numpy()
            allMetrics.append(calculate_single(target_depth_numpy, pred_depth_numpy, lpips_model))
            original_allMetrics.append(
                calculate_single(target_depth_numpy, stage1_depth.squeeze(0).squeeze(0), lpips_model))

            image_name = os.path.basename(batch[3][0].replace('.png', ''))
            np.save(os.path.join(output_dir, "depth_npy", f'{image_name}_pred_depth.npy'), pred_depth_numpy)

            # Save results
            # Save depth map
            depth_img = (pred_depth_numpy * 255).astype(np.uint8)
            Image.fromarray(depth_img).save(os.path.join(output_dir, "depth_raw", f'{image_name}_pred_depth.png'))

            if process_depth_flag:
                for i in range(process_depth.shape[0]):
                    process_img = Image.fromarray((process_depth[i].squeeze().cpu().numpy() * 255).astype(np.uint8))
                    process_img.save(os.path.join(output_dir, "depth_process", f'{image_name}_process_depth_{i}.png'))

            # Save comparison image (RGB - Predicted depth - Diff heatmap)
            heatmap = generate_heatmap(np.abs(pred_depth_numpy - target_depth_numpy))
            comparison_img = Image.new('RGB', (rgb_pil.width * 5, rgb_pil.height))
            comparison_img.paste(rgb_pil, (0, 0))
            comparison_img.paste(Image.fromarray((target_depth_numpy * 255).astype(np.uint8)), (rgb_pil.width, 0))
            comparison_img.paste(stage1_depth_pil, (rgb_pil.width * 2, 0))
            comparison_img.paste(Image.fromarray(depth_img), (rgb_pil.width * 3, 0))
            comparison_img.paste(Image.fromarray(heatmap), (rgb_pil.width * 4, 0))
            comparison_img.save(os.path.join(output_dir, "compare", f'{image_name}_compare_{allMetrics[-1]["mae"] - original_allMetrics[-1]["mae"]}.png'))

        # Log metrics to file
        with open(os.path.join(output_dir, 'loss.txt'), 'a') as f:
            f.write(f"{image_name}: {allMetrics[-1]}\n")
    print(f"Total evaluation time: {total_time:.6f} seconds")
    with open(os.path.join(output_dir, 'loss.txt'), 'a') as f:
        f.write(f"\nTotal evaluation time: {total_time:.6f} seconds\n")

    print(f"Evaluation completed. Results saved to {output_dir}")
    avgMetrics = {k: np.mean([m[k] for m in allMetrics]) for k in allMetrics[0].keys()}
    original_avgMetrics = {k: np.mean([m[k] for m in original_allMetrics]) for k in original_allMetrics[0].keys()}
    print("Average Metrics:", avgMetrics)
    with open(os.path.join(output_dir, 'loss.txt'), 'a') as f:
        f.write(f"original_average_mae: {original_average_mae / len(test_loader):.4f}\n")
        f.write(f"pred_average_mae: {pred_average_mae / len(test_loader):.4f}\n")
        for k, v in avgMetrics.items():
            f.write(f"{k}: {v}\n")
        f.write(f"original_avgMetrics: {original_avgMetrics}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Config/testStage2ImageToDEM-Swi2_512.json')
    args = parser.parse_args()
    evaluate(args.config)  # Adjust config path if necessary
