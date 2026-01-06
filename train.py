import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import shutil
import gc

# 导入现有模块
from process_raw_match_data import process_match_data
from dual_network import DualNetwork
from data_loader import BilliardsDataset, StatePreprocessor


def train(args):
    """
    训练双网络模型（新增复用已生成训练数据的逻辑）
    """
    # 1. 处理对局数据（新增：根据参数决定是否复用已有trainable_data.json）
    if args.use_existing_train_data:
        # 检查已有文件是否存在
        if not os.path.exists(args.train_data_file):
            raise FileNotFoundError(
                f"指定的训练数据文件不存在: {args.train_data_file}\n请先运行无--use_existing_train_data参数的命令生成数据")
        print(f"✅ 复用已生成的训练数据: {args.train_data_file}")
    else:
        # 原有逻辑：重新生成训练数据
        print(f"Processing match data from {args.match_dir}...")
        process_match_data(args.match_dir, args.train_data_file)
        print(f"Training data generated: {args.train_data_file}")

    # 2. 准备数据加载
    print("Loading training data...")

    # 创建状态预处理器
    preprocessor = StatePreprocessor()

    # 创建数据集（修改为支持单个文件）
    # 首先将生成的训练数据文件移动到临时目录，然后使用该目录
    temp_data_dir = os.path.join(
        os.path.dirname(args.train_data_file), 'temp_data')
    os.makedirs(temp_data_dir, exist_ok=True)
    temp_data_file = os.path.join(
        temp_data_dir, os.path.basename(args.train_data_file))
    shutil.copy(args.train_data_file, temp_data_file)

    # 创建数据集并传递预处理器
    dataset = BilliardsDataset(temp_data_dir, transform=preprocessor)

    # 创建数据加载器（优化：根据是否使用GPU调整num_workers，减少内存占用）
    num_workers = 0 if torch.cuda.is_available(
    ) and os.name == 'nt' else 4  # Windows+GPU下关闭多进程避免内存爆炸
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {len(dataloader)}")
    print(f"DataLoader num_workers: {num_workers}")

    # 3. 初始化模型
    print("Initializing dual network model...")
    model = DualNetwork()

    # 检查CUDA可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # 4. 定义损失函数和优化器
    # 策略损失：均方误差
    policy_criterion = nn.MSELoss()
    # 价值损失：均方误差
    value_criterion = nn.MSELoss()

    # 优化器：Adam
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma
    )

    # 5. 训练循环
    print("Starting training...")

    for epoch in range(args.epochs):
        model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0

        # 使用tqdm显示进度
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch_idx, (states, policy_targets, value_targets) in enumerate(pbar):
                # 数据移到设备上
                states = states.to(device, non_blocking=True)
                policy_targets = policy_targets.to(device, non_blocking=True)
                value_targets = value_targets.to(device, non_blocking=True)

                # 前向传播
                outputs = model(states)
                policy_output = outputs['policy_output']
                value_output = outputs['value_output']

                # 计算损失
                policy_loss = policy_criterion(policy_output, policy_targets)
                value_loss = value_criterion(value_output, value_targets)

                # 总损失：策略损失 + 价值损失
                loss = policy_loss + value_loss

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 累积损失
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_loss += loss.item()

                # 更新进度条
                pbar.set_postfix({
                    'Policy Loss': f'{policy_loss.item():.6f}',
                    'Value Loss': f'{value_loss.item():.6f}',
                    'Total Loss': f'{loss.item():.6f}'
                })

        # 学习率调度
        scheduler.step()

        # 打印 epoch 结果
        avg_policy_loss = total_policy_loss / len(dataloader)
        avg_value_loss = total_value_loss / len(dataloader)
        avg_total_loss = total_loss / len(dataloader)

        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Average Policy Loss: {avg_policy_loss:.6f}")
        print(f"  Average Value Loss: {avg_value_loss:.6f}")
        print(f"  Average Total Loss: {avg_total_loss:.6f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']}")

        # 保存模型检查点
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = os.path.join(
                args.model_dir, f"dual_network_epoch_{epoch+1}.pt")
            model.save(checkpoint_path)
            print(f"Model checkpoint saved: {checkpoint_path}")

        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 保存最终模型
    final_model_path = os.path.join(args.model_dir, "dual_network_final.pt")
    model.save(final_model_path)
    print(f"Final model saved: {final_model_path}")

    # 清理临时文件
    shutil.rmtree(temp_data_dir, ignore_errors=True)
    print("Training completed!")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="Train dual network model using processed match data")

    # 数据相关参数
    parser.add_argument('--match_dir', type=str, default='match_data',
                        help='Directory containing match data files')
    parser.add_argument('--train_data_file', type=str, default='trainable_data.json',
                        help='Output file path for trainable data')
    # 新增：复用已生成训练数据的参数
    parser.add_argument('--use_existing_train_data', action='store_true',
                        help='Use existing trainable_data.json instead of reprocessing match data (skips process_match_data)')

    # 模型相关参数
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save trained models')

    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for regularization')
    parser.add_argument('--lr_step_size', type=int, default=50,
                        help='Step size for learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.5,
                        help='Gamma value for learning rate decay')
    parser.add_argument('--save_interval', type=int, default=20,
                        help='Interval for saving model checkpoints')

    args = parser.parse_args()

    # 创建必要的目录
    os.makedirs(args.model_dir, exist_ok=True)

    # 开始训练
    train(args)


if __name__ == '__main__':
    main()
