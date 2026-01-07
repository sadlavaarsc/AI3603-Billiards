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
    训练双网络模型（支持续训预训练模型，兼容嵌套模块权重格式）
    核心修改：1.策略分支改为MSE损失 2.删除错误标签转换 3.策略标签即时归一化（不改动原始数据）
    """
    # 1. 处理对局数据
    if args.use_existing_train_data:
        if not os.path.exists(args.train_data_file):
            raise FileNotFoundError(
                f"指定的训练数据文件不存在: {args.train_data_file}")
        print(f"✅ 复用已生成的训练数据: {args.train_data_file}")
    else:
        print(f"Processing match data from {args.match_dir}...")
        process_match_data(args.match_dir, args.train_data_file)
        print(f"Training data generated: {args.train_data_file}")

    # 2. 准备数据加载
    print("Loading training data...")
    preprocessor = StatePreprocessor()
    temp_data_dir = os.path.join(
        os.path.dirname(args.train_data_file), 'temp_data')
    os.makedirs(temp_data_dir, exist_ok=True)
    temp_data_file = os.path.join(
        temp_data_dir, os.path.basename(args.train_data_file))
    shutil.copy(args.train_data_file, temp_data_file)
    dataset = BilliardsDataset(temp_data_dir, transform=preprocessor)

    num_workers = 0 if torch.cuda.is_available() and os.name == 'nt' else 4
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

    # 3. 初始化/加载模型 -------------------------- 核心修复：适配嵌套模块权重 --------------------------
    print("Initializing/loading dual network model...")
    model = DualNetwork()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 记录初始Epoch（续训时从指定Epoch开始）
    start_epoch = 1
    checkpoint = None
    # 加载预训练模型
    if args.resume_from:
        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(f"预训练模型文件不存在: {args.resume_from}")

        # 加载模型文件
        checkpoint = torch.load(args.resume_from, map_location=device)
        try:
            # 尝试1：加载新格式（带model_state_dict）
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(
                    checkpoint['model_state_dict'], strict=False)
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                print(
                    f"✅ 加载新格式预训练模型: {args.resume_from}，从Epoch {start_epoch} 开始续训")
            # 尝试2：加载旧格式（直接权重）
            else:
                # 适配嵌套模块的权重格式
                model.load_state_dict(checkpoint, strict=False)
                print(f"✅ 加载旧格式预训练模型: {args.resume_from}（兼容嵌套模块）")

            # 打印权重加载情况，方便调试
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'] if (
                isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint) else checkpoint, strict=False)
            if missing_keys:
                print(f"⚠️ 权重中缺失的键（可忽略）: {missing_keys[:5]}...")  # 只打印前5个避免刷屏
            if unexpected_keys:
                print(f"⚠️ 权重中多余的键（可忽略）: {unexpected_keys[:5]}...")

        except Exception as e:
            # 终极方案：手动遍历权重，匹配层名
            print(f"⚠️ 直接加载权重失败，尝试手动匹配: {str(e)}")
            model_dict = model.state_dict()
            # 过滤出匹配的权重
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                pretrained_dict = checkpoint['model_state_dict']
            else:
                pretrained_dict = checkpoint

            # 适配嵌套模块的权重名
            new_pretrained_dict = {}
            for k, v in pretrained_dict.items():
                # 如果权重名直接匹配，保留
                if k in model_dict:
                    new_pretrained_dict[k] = v
                # 如果是嵌套模块（如feature_extractor.spatial_fc1.weight）
                else:
                    # 尝试去掉顶层模块名（如feature_extractor.）
                    parts = k.split('.', 1)
                    if len(parts) > 1 and parts[1] in model_dict:
                        new_pretrained_dict[parts[1]] = v
                    # 尝试添加顶层模块名
                    elif f"feature_extractor.{k}" in model_dict:
                        new_pretrained_dict[f"feature_extractor.{k}"] = v
                    elif f"policy_head.{k}" in model_dict:
                        new_pretrained_dict[f"policy_head.{k}"] = v
                    elif f"value_head.{k}" in model_dict:
                        new_pretrained_dict[f"value_head.{k}"] = v

            # 更新模型权重
            model_dict.update(new_pretrained_dict)
            model.load_state_dict(model_dict)
            print("✅ 手动匹配权重成功，忽略层名不匹配的部分")
    else:
        print("🔄 初始化全新模型，从头开始训练")

    print(f"Using device: {device}")

    # 4. 定义损失函数和优化器
    # 核心修改1：策略分支改为MSE损失（适配5维连续动作回归）
    policy_criterion = nn.MSELoss()
    value_criterion = nn.MSELoss()

    # 优化器
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

    # 续训时恢复优化器和调度器状态（仅新格式支持）
    if args.resume_from and checkpoint is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✅ 恢复优化器和学习率调度器状态")
        except:
            print("⚠️ 无法恢复优化器/调度器状态，使用全新的优化器配置")
    elif args.resume_from:
        print("⚠️ 旧格式模型文件无优化器状态，使用全新的优化器配置")

    # 5. 训练循环
    print(f"Starting training from Epoch {start_epoch}...")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0

        with tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}") as pbar:
            for batch_idx, (states, policy_targets, value_targets) in enumerate(pbar):
                # 数据移到设备上
                states = states.to(device, non_blocking=True)
                policy_targets = policy_targets.to(device, non_blocking=True)
                value_targets = value_targets.to(device, non_blocking=True)

                # 核心修改4：策略标签即时归一化（关键！不改动原始数据，仅在计算损失前处理）
                # 定义每个动作维度的原始物理范围（根据你的业务逻辑调整）
                # 速度/水平角/垂直角/x偏移/y偏移最小值
                action_min = torch.tensor(
                    [0.5, 0.0, 0.0, -0.5, -0.5], device=device)
                action_max = torch.tensor(
                    [8.0, 360.0, 90.0, 0.5, 0.5], device=device)  # 对应最大值
                # 归一化到0~1区间（匹配模型sigmoid输出）
                policy_targets = (policy_targets - action_min) / \
                    (action_max - action_min)
                # 裁剪异常值，避免数据错误导致损失异常
                policy_targets = torch.clamp(policy_targets, 0.0, 1.0)

                # 核心修改2：删除错误的策略标签格式转换（已移除）

                # 前向传播
                outputs = model(states)
                policy_logits = outputs['policy_output']
                value_output = outputs['value_output']

                # 计算损失（此时policy_logits和policy_targets都是0~1，尺度匹配）
                policy_loss = policy_criterion(policy_logits, policy_targets)
                value_loss = value_criterion(value_output, value_targets)
                loss = args.policy_weight * policy_loss + args.value_weight * value_loss

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=args.clip_grad_norm)
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

        print(f"Epoch {epoch}/{args.epochs}:")
        print(f"  Average Policy Loss: {avg_policy_loss:.6f}")
        print(f"  Average Value Loss: {avg_value_loss:.6f}")
        print(f"  Average Total Loss: {avg_total_loss:.6f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']}")

        # 保存模型检查点（新格式，包含完整状态）
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(
                args.model_dir, f"dual_network_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'policy_loss': avg_policy_loss,
                'value_loss': avg_value_loss
            }, checkpoint_path)
            print(f"Model checkpoint saved: {checkpoint_path}")

        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 保存最终模型（新格式）
    final_model_path = os.path.join(args.model_dir, "dual_network_final.pt")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, final_model_path)
    print(f"Final model saved: {final_model_path}")

    # 清理临时文件
    shutil.rmtree(temp_data_dir, ignore_errors=True)
    print("Training completed!")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="Train dual network model (support resume training)")

    # 数据相关参数
    parser.add_argument('--match_dir', type=str, default='match_data',
                        help='Directory containing match data files')
    parser.add_argument('--train_data_file', type=str, default='trainable_data.json',
                        help='Output file path for trainable data')
    parser.add_argument('--use_existing_train_data', action='store_true',
                        help='Use existing trainable_data.json (skip reprocessing)')

    # 模型相关参数
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save trained models')
    parser.add_argument('--resume_from', type=str, default='',
                        help='Path to pre-trained model (e.g., models/dual_network_epoch_100.pt) for resume training')

    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='Total number of training epochs (include resumed epochs)')
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
    # 核心修改3：调整损失权重默认值（适配MSE损失）
    parser.add_argument('--policy_weight', type=float, default=5.0,  # 归一化后调为5.0更合理
                        help='Weight for policy loss (balance with value loss)')
    parser.add_argument('--value_weight', type=float, default=1.0,
                        help='Weight for value loss (balance with policy loss)')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0,
                        help='Max norm for gradient clipping (0 to disable)')

    args = parser.parse_args()

    # 创建必要的目录
    os.makedirs(args.model_dir, exist_ok=True)

    # 开始训练
    train(args)


if __name__ == '__main__':
    main()
