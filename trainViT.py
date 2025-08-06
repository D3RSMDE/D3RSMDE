import os
import json
import random
import shutil
from datetime import datetime
from Utils.randomChoose import init_seed, seed_worker
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import argparse
from Utils.loadDatasets import ImageToDEMDataset
from Utils.autoSelectGPU import select_best_gpu
from Modules.MyVit import MyDit


def train(config_path: str):
    # 1. 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    print(f"加载配置文件: {config_path}，配置内容: {cfg}")
    device = select_best_gpu()
    init_seed(cfg['seed'])  # 设置随机种子
    generator = torch.Generator().manual_seed(cfg['seed'])  # 设置随机种子, todo: 之后加加看
    dataset_path = cfg['dataset_path']
    batch_size = cfg['batch_size']
    weight_decay = cfg['weight_decay']
    epochs = cfg['epochs']

    # 2. 准备输出目录
    timestamp = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    output_dir = os.path.join('output', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    # 复制配置文件
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    shutil.copy(config_path, output_dir)
    shutil.copy("Modules/MyVit.py", output_dir)  # 复制模型定义文件
    shutil.copy(__file__, output_dir)  # 复制训练脚本
    shutil.copy("Utils/loadDatasets.py", output_dir)  # 复制数据集加载脚本
    # 创建日志文件
    train_log = open(os.path.join(output_dir, 'trainLoss.txt'), 'w', encoding='utf-8')
    val_log = open(os.path.join(output_dir, 'valLoss.txt'), 'w', encoding='utf-8')
    # 打印出当前超参数配置和输出目录
    print(f"Configuration: {cfg}")
    print(f"Output directory: {output_dir}")

    # 3. 数据集与 DataLoader
    train_dataset = ImageToDEMDataset(root_dir=dataset_path, mode='train')
    # train_dataset = ImageToDEMDataset(root_dir=dataset_path, mode='train', not_split=True)
    # val_dataset = ImageToDEMDataset(root_dir=dataset_path, mode='val')
    val_dataset = ImageToDEMDataset(root_dir=dataset_path, mode='test')
    # train_dataset = ImageToDEMDataset(root_dir=dataset_path, mode='train',
    #                                   split_text=cfg['split_text'] if 'split_text' in cfg else None,
    #                                   not_split=cfg.get('not_split', False))
    # val_dataset = ImageToDEMDataset(root_dir=dataset_path, mode='val',
    #                                 split_text=cfg['val_split_text'] if 'split_text' in cfg else None,
    #                                 not_split=cfg.get('not_split', False))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=cfg['num_workers'],
                              worker_init_fn=seed_worker, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=cfg['num_workers'],
                            worker_init_fn=seed_worker, generator=generator)

    # 4. 模型、优化器与损失
    model = MyDit(device=device, **cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=weight_decay)
    if cfg.get('pretrained_model_path'):
        tempWeight = torch.load(cfg['pretrained_model_path'], map_location="cpu")
        m1, m2 = model.load_state_dict(tempWeight['model_state_dict'], strict=False)
        # m3, m4 = optimizer.load_state_dict(tempWeight['optimizer_state_dict'], strict=False)  # todo: 之后加加看
        best_val_loss = tempWeight['best_val_loss']
        start_epoch = tempWeight['epoch'] + 1  # 从下一个epoch开始
        print(f"Loading pretrained model from {cfg['pretrained_model_path']}, "
              f"best validation loss: {best_val_loss:.6f}, start at epoch {start_epoch}."
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
        model.train()
        train_loss_sum, feature_loss_sum, image_loss_sum = 0.0, 0.0, 0.0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Train")):
            rgb = batch[0].to(device)
            depth = batch[1].to(device)

            optimizer.zero_grad()
            # 调用模型进行训练
            depth_pred = model(rgb)

            # 计算损失
            total_loss = model.compute_loss(depth_pred, depth)
            total_loss.backward()
            optimizer.step()
            # torch.cuda.empty_cache()  # 清理缓存

            train_loss_sum += total_loss.item()
            if torch.isnan(total_loss).any():
                raise ValueError("Training loss is NaN, check your model or data.")

            train_log.write(
                f"{datetime.now().strftime('%y-%m-%d_%H:%M:%S')} [{epoch}/{i}] avg_loss: {total_loss:.6f}\n")
            train_log.flush()  # 确保日志文件及时写入
            # break  # todo
        avg_train_loss = train_loss_sum / len(train_loader)
        print(f"Epoch {epoch} Train Loss: {avg_train_loss:.6f}")
        i_epoch += 1
        if i_epoch >= threshold_epoch:
            # 降低学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] *= downgrade_lr_ratio
            print(f"Epoch {epoch}: Learning rate reduced to {optimizer.param_groups[0]['lr']:.6f}")
            train_log.write(
                f"{datetime.now().strftime('%y-%m-%d_%H:%M:%S')} [{epoch}] Learning rate reduced to {optimizer.param_groups[0]['lr']:.6f}\n")
            train_log.flush()
            i_epoch = 0

        # 验证
        model.eval()
        with torch.no_grad():
            val_loss_sum, val_feature_loss_sum, val_image_loss_sum = 0.0, 0.0, 0.0
            for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch} Val")):
                rgb = batch[0].to(device)
                depth = batch[1].to(device)
                depth_pred = model(rgb)

                # 计算损失
                total_loss = model.compute_loss(depth_pred, depth)
                # 判断nan
                if torch.isnan(total_loss).any():
                    raise ValueError("Validation loss is NaN, check your model or data.")

                val_loss_sum += total_loss.item()

                # break  # todo

            avg_val_loss = val_loss_sum / len(val_loader)
            print(f"Epoch {epoch} Val Loss: {avg_val_loss:.6f}")

            val_log.write(
                f"{datetime.now().strftime('%y-%m-%d_%H:%M:%S')} [{epoch}] avg_val_loss: {avg_val_loss:.6f}\n")
            val_log.flush()

        # 保存最优模型
        if avg_val_loss < best_val_loss:
            i_epoch = 0  # 重置epoch计数器
            print(f"Validation loss improved from {best_val_loss:.6f} to {avg_val_loss:.6f}, saving model.")
            val_log.write(
                f"Validation loss improved from {best_val_loss:.6f} to {avg_val_loss:.6f}, saving model.\n")
            val_log.flush()
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'seed': cfg['seed'],
            }, os.path.join(ckpt_dir, 'best.pth'))

        # 每隔10个epoch保存一次
        if epoch % cfg['save_n_epoch'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'val_loss': avg_val_loss,
                'seed': cfg['seed'],
            }, os.path.join(ckpt_dir, f'epoch_{epoch}.pth'))

        if "save_i_epoch" in cfg and epoch in cfg['save_i_epoch']:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'val_loss': avg_val_loss,
                'seed': cfg['seed'],
            }, os.path.join(ckpt_dir, f'epoch_{epoch}.pth'))

    train_log.close()
    val_log.close()


# loader_generator
# generator
# rand_num_generator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Config/trainImageToDEM-瑞士2_512.json')
    args = parser.parse_args()
    train(args.config)
