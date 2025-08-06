import argparse
import os
import json
import shutil
import time

from Utils.randomChoose import init_seed, seed_worker
import lpips
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from PIL import Image
import numpy as np
from tqdm import tqdm

from Utils.calculateDiff import calculate_single
from Utils.loadDatasets import ImageToDEMDataset
from matplotlib.colors import Normalize
from matplotlib import pyplot as plt
from Modules.MyVit import MyDit
from Utils.autoSelectGPU import select_best_gpu


def generate_heatmap(diff):
    # 将差值标准化到 0-255
    diff = Normalize()(diff) * 255
    diff = diff.astype(np.uint8)

    # 使用 matplotlib 的热力图 colormap
    cmap = plt.get_cmap('jet')
    heatmap = cmap(diff / 255.0)[:, :, :3]  # 转为 RGB 格式 (H, W, 3)
    heatmap = (heatmap * 255).astype(np.uint8)  # 确保类型为 uint8
    return heatmap


# Evaluate function
def evaluate(config_path: str):
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    print(f"加载配置文件: {config_path}，配置内容: {cfg}")
    init_seed(cfg['seed'])  # 设置随机种子
    generator = torch.Generator().manual_seed(cfg['seed'])  # 设置随机种子, todo: 之后加加看
    # Extract configuration parameters
    dataset_path = cfg['dataset_path']
    output_dir = cfg['output_dir']
    checkpoint_path = cfg['checkpoint_path']
    device = select_best_gpu()

    # Prepare output directory
    timestamp = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(os.path.join(output_dir, "compare"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth_raw"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth_npy"), exist_ok=True)
    shutil.copy(config_path, os.path.join(output_dir, os.path.basename(config_path)))
    shutil.copy(__file__, output_dir)  # 复制当前脚本文件
    shutil.copy("Utils/loadDatasets.py", output_dir)  # 复制数据集加载脚本

    print(f"Configuration: {cfg}")
    print(f"输出文件夹：{output_dir}")

    # Load test dataset
    # test_dataset = ImageToDEMDataset(root_dir=dataset_path, mode='test', worker_init_fn=seed_worker,
    #                                  generator=generator)
    test_dataset = ImageToDEMDataset(root_dir=dataset_path, mode='val',
                                     split_text=cfg['test_split_text'] if 'test_split_text' in cfg else None,
                                     not_split=cfg.get('not_split', True))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'])

    # Initialize model
    model, tempDict = MyDit.from_pretrained(MyDit(device=device, **cfg), checkpoint_path, )
    model.eval()

    # Set device
    model.to(device)

    allMetrics = []
    lpips_model = lpips.LPIPS(net='alex')
    total_time = 0
    # Iterate through the test set
    for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating"):
        # Get input image
        rgb_image = batch[0].to(device)
        rgb_pil = Image.fromarray((rgb_image.squeeze(0).permute(1, 2, 0).cpu().numpy()).astype(np.uint8))

        # Run inference
        with torch.no_grad():
            time1 = time.time()
            pred_depth = model.inference(rgb_image)
            total_time += time.time() - time1

            # Calculate evaluation metrics
            target_depth_numpy = batch[1].squeeze(0).squeeze(0).cpu().numpy()  # batch[1] 范围是 [0, 1]
            pred_depth_numpy = pred_depth.squeeze(0).squeeze(0).cpu().numpy()  # 预测深度范围也是 [0, 1]
            allMetrics.append(calculate_single(target_depth_numpy, pred_depth_numpy, lpips_model))  # todo: 检查loss计算

            # Save results
            image_name = os.path.basename(batch[2][0].replace('.png', ''))
            np.save(os.path.join(output_dir, "depth_npy", f'{image_name}_pred_depth.npy'), pred_depth_numpy)
            # Save depth map
            depth_img = (pred_depth_numpy * 255).astype(np.uint8)
            Image.fromarray(depth_img).save(os.path.join(output_dir, "depth_raw", f'{image_name}_pred_depth.png'))

            # Save comparison image (RGB - Predicted depth - Diff heatmap)
            heatmap = generate_heatmap(np.abs(pred_depth_numpy - target_depth_numpy))
            comparison_img = Image.new('RGB', (rgb_pil.width * 4, rgb_pil.height))
            comparison_img.paste(rgb_pil, (0, 0))
            comparison_img.paste(Image.fromarray((target_depth_numpy * 255).astype(np.uint8)), (rgb_pil.width, 0))
            comparison_img.paste(Image.fromarray(depth_img), (rgb_pil.width * 2, 0))
            comparison_img.paste(Image.fromarray(heatmap), (rgb_pil.width * 3, 0))
            comparison_img.save(os.path.join(output_dir, "compare", f'{image_name}_compare.png'))

        # Log metrics to file
        with open(os.path.join(output_dir, 'loss.txt'), 'a') as f:
            f.write(f"{image_name}: {allMetrics[-1]}\n")

    print(f"Total evaluation time: {total_time:.6f} seconds")
    print(f"Evaluation completed. Results saved to {output_dir}")
    avgMetrics = {k: np.mean([m[k] for m in allMetrics]) for k in allMetrics[0].keys()}
    print("Average Metrics:", avgMetrics)
    with open(os.path.join(output_dir, 'loss.txt'), 'a') as f:
        f.write("\nTotal evaluation time: {:.6f} seconds\n".format(total_time))
        for k, v in avgMetrics.items():
            f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Config/testImageToDEM-瑞士2_512.json')
    args = parser.parse_args()
    evaluate(args.config)  # Adjust config path if necessary
