import os
import json
import shutil
from datetime import datetime
from tqdm import tqdm
import torch

from Modules.tokenizer.vavae import VA_VAE
from torch.utils.data import DataLoader

import argparse
from Utils.loadDatasets import ImageToDEMDataset
from Utils.autoSelectGPU import select_best_gpu
# from Modules.segrefiner_base_SD_Unet import SegRefiner
from Modules.D3RSMDE_diffusion import SegRefiner  # 确保这个路径正确
from Utils.chooseRandom import init_seed, seed_worker


# from Modules.segrefiner_base_multiTrainMethod import SegRefiner

def train(config_path: str):
    # 1. 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    print(f"加载配置文件: {config_path}，配置内容: {cfg}")
    init_seed(cfg['seed'])
    generator = torch.Generator().manual_seed(cfg['seed'])  # 设置随机种子, todo: 之后加加看
    dataset_path = cfg['dataset_path']
    batch_size = cfg['batch_size']
    weight_decay = cfg['weight_decay']
    epochs = cfg['epochs']

    # 2. 准备输出目录
    timestamp = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    output_dir = os.path.join('output', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    # 复制配置文件和相关代码
    shutil.copy(config_path, output_dir)
    shutil.copy(__file__, output_dir)  # 复制当前脚本文件
    shutil.copy("Modules/D3RSMDE_diffusion.py", output_dir)  # 复制模型定义文件
    shutil.copy("Utils/loadDatasets.py", output_dir)  # 复制数据集加载脚本
    # 创建日志文件
    train_log = open(os.path.join(output_dir, 'trainLoss.txt'), 'w', encoding='utf-8')
    val_log = open(os.path.join(output_dir, 'valLoss.txt'), 'w', encoding='utf-8')
    # 打印出当前超参数配置和输出目录
    # print(f"Configuration: {cfg}")
    print(f"输出文件夹：{output_dir}")

    # 3. 数据集与 DataLoader
    # train_dataset = ImageToDEMDataset(root_dir=dataset_path, mode='train', seed=cfg['seed'])
    # val_dataset = ImageToDEMDataset(root_dir=dataset_path, mode='val', seed=cfg['seed'])
    train_dataset = ImageToDEMDataset(root_dir=dataset_path, mode='train', seed=cfg['seed'], not_split=True)
    val_dataset = ImageToDEMDataset(root_dir=dataset_path, mode='test', seed=cfg['seed'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=cfg['num_workers'],
                              worker_init_fn=seed_worker, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=cfg['num_workers'],
                            worker_init_fn=seed_worker, generator=generator)

    # 4. 模型、优化器与损失
    device = select_best_gpu()
    # device = torch.device(f'cuda:{0}')
    model = SegRefiner(device=device, **cfg).to(device)  # todo
    optimizer = torch.optim.AdamW(model.denoise_model.parameters(), lr=cfg['lr'], weight_decay=weight_decay)
    # 添加vae的参数
    # if isinstance(model.vae, VA_VAE):
    #     optimizer.add_param_group(
    #         {'params': model.vae.model.parameters(), 'lr': cfg['vae_lr'], 'weight_decay': weight_decay})
    #     print(f"Using VA_VAE with lr {cfg['vae_lr']} and no weight decay for VAE parameters.")
    # else:
    #     optimizer.add_param_group({'params': model.vae.parameters(), 'lr': cfg['vae_lr'], 'weight_decay': weight_decay})
    if cfg.get("pretrained_vae_path") and os.path.exists(cfg['pretrained_vae_path']):
        if cfg['vae_type'] == 'VA_VAE':
            m1, m2 = model.vae.model.load_state_dict(
                torch.load(cfg['pretrained_vae_path'], map_location="cpu")['vae_state_dict'], strict=False)
        else:
            m1, m2 = model.vae.load_state_dict(
                torch.load(cfg['pretrained_vae_path'], map_location="cpu")['vae_state_dict'], strict=False)
        print(f"成功加载预训练的vae模型，warnings: {m1}, {m2}")

    if cfg.get('pretrained_model_path') and os.path.exists(cfg['pretrained_model_path']):
        tempWeight = torch.load(cfg['pretrained_model_path'], map_location="cpu")
        m1, m2 = model.denoise_model.load_state_dict(tempWeight['denoise_model_state_dict'], strict=False)
        # if isinstance(model.vae, VA_VAE):
        #     m3, m4 = model.vae.model.load_state_dict(tempWeight['vae_state_dict'], strict=False)
        # else:
        #     m3, m4 = model.vae.load_state_dict(tempWeight['vae_state_dict'], strict=False)
        # optimizer.load_state_dict(tempWeight['optimizer_state_dict'])  # todo: 之后加加看
        best_val_loss = tempWeight['best_val_loss']
        # best_val_loss = float('inf')
        start_epoch = tempWeight['epoch'] + 1  # 从下一个epoch开始
        print(f"Loading pretrained model from {cfg['pretrained_model_path']}, "
              f"best validation loss: {best_val_loss:.8f}, start at epoch {start_epoch}."
              f"model warnings: {m1}, {m2}, 当前学习率: {[p['lr'] for p in optimizer.param_groups]}")
    else:
        print("No pretrained model path provided, starting from scratch.")
        best_val_loss = float('inf')
        start_epoch = 1

    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    if cfg['threshold_epoch'] > 0:
        threshold_epoch = cfg['threshold_epoch']
        downgrade_lr_ratio = cfg['downgrade_lr_ratio']
    else:
        threshold_epoch = float('inf')  # 如果没有设置阈值，则不限制
        downgrade_lr_ratio = 1.0  # 不降学习率
    i_epoch = 0  # 用于记录当前epoch
    # 5. 训练与验证循环
    for epoch in range(start_epoch, epochs + 1):
        model.denoise_model.train()  # 设置模型为训练模式
        # model.vae.model.train() if isinstance(model.vae, VA_VAE) else model.vae.train()
        train_loss_sum, feature_loss_sum, image_loss_sum = 0.0, 0.0, 0.0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Train")):
            # if i < 707:
            #     continue  # debug
            rgb = batch[0].to(device)
            depth = batch[1].to(device)
            stage1_depth = batch[2].to(device)

            optimizer.zero_grad()
            # 调用模型进行训练
            depth_pred, target_latent = model(rgb, stage1_depth, depth)

            # 计算损失
            total_loss = model.compute_loss(depth_pred, target_latent)
            total_loss.backward()
            optimizer.step()
            # torch.cuda.empty_cache()  # 清理缓存

            train_loss_sum += total_loss.item()
            if torch.isnan(total_loss).any():
                raise ValueError("Training loss is NaN, check your model or data.")

            train_log.write(
                f"{datetime.now().strftime('%y-%m-%d_%H:%M:%S')} [{epoch}/{i}] avg_loss: {total_loss:.8f}\n")
            train_log.flush()  # 确保日志文件及时写入
            # break  # todo
        avg_train_loss = train_loss_sum / len(train_loader)
        print(f"Epoch {epoch} Train Loss: {avg_train_loss:.8f}")
        i_epoch += 1
        if i_epoch >= threshold_epoch:
            # 降低学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] *= downgrade_lr_ratio
            print(f"Epoch {epoch}: Learning rate reduced to {optimizer.param_groups[0]['lr']:.8f}")
            train_log.write(
                f"{datetime.now().strftime('%y-%m-%d_%H:%M:%S')} [{epoch}] Learning rate reduced to {optimizer.param_groups[0]['lr']:.8f}\n")
            train_log.flush()
            i_epoch = 0

        # 验证
        model.eval()
        model.denoise_model.eval()  # 设置模型为评估模式
        # model.vae.model.train() if isinstance(model.vae, VA_VAE) else model.vae.train()
        with torch.no_grad():
            val_loss_sum, val_feature_loss_sum, val_image_loss_sum = 0.0, 0.0, 0.0
            for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch} Val")):
                rgb = batch[0].to(device)
                depth = batch[1].to(device)
                stage1_depth = batch[2].to(device)
                # depth_pred, target_latent = model(rgb, stage1_depth, depth)
                pred_depth, _ = model.inference(rgb, stage1_depth)
                pred_depth = pred_depth.to(device)
                for j in range(pred_depth.shape[0]):
                    pred_depth[j] = (pred_depth[j] - 0.5) / 0.5

                # 计算损失
                total_loss = model.compute_loss(depth, pred_depth)
                # 判断nan
                if torch.isnan(total_loss).any():
                    raise ValueError("Validation loss is NaN, check your model or data.")

                val_loss_sum += total_loss.item()

                # break  # todo

            avg_val_loss = val_loss_sum / len(val_loader)
            print(f"Epoch {epoch} Val Loss: {avg_val_loss:.8f}")

            val_log.write(
                f"{datetime.now().strftime('%y-%m-%d_%H:%M:%S')} [{epoch}] avg_val_loss: {avg_val_loss:.8f}\n")
            val_log.flush()

        # 保存最优模型
        if avg_val_loss < best_val_loss:
            i_epoch = 0  # 重置epoch计数器
            print(f"Validation loss improved from {best_val_loss:.8f} to {avg_val_loss:.8f}, saving model.")
            val_log.write(
                f"Validation loss improved from {best_val_loss:.8f} to {avg_val_loss:.6f}, saving model.\n")
            val_log.flush()
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'denoise_model_state_dict': model.denoise_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'val_loss': avg_val_loss,
                'seed': cfg['seed'],
            }, os.path.join(ckpt_dir, f'best_epoch.pth'))

        # 每隔10个epoch保存一次
        if epoch % cfg['save_n_epoch'] == 0:
            torch.save({
                'epoch': epoch,
                'denoise_model_state_dict': model.denoise_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'val_loss': avg_val_loss,
                'seed': cfg['seed'],
            }, os.path.join(ckpt_dir, f'epoch_{epoch}.pth'))

        if "save_i_epoch" in cfg and epoch in cfg['save_i_epoch']:
            torch.save({
                'epoch': epoch,
                'denoise_model_state_dict': model.denoise_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'val_loss': avg_val_loss,
                'seed': cfg['seed'],
            }, os.path.join(ckpt_dir, f'epoch_{epoch}.pth'))

    train_log.close()
    val_log.close()


if __name__ == '__main__':
    # 设置环境变量
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Config/trainStage2ImageToDEM-澳大利亚.json')
    args = parser.parse_args()
    train(args.config)
