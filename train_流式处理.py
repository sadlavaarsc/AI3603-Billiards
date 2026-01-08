import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gc
import shutil
import time
import sys

# 修复Windows路径问题的核心工具函数


def get_absolute_path(path):
    """获取绝对路径，处理Windows路径分隔符"""
    # 转换为绝对路径
    abs_path = os.path.abspath(path)
    # 处理Windows路径分隔符
    abs_path = abs_path.replace('/', '\\')
    return abs_path


def ensure_dir_exists(path):
    """确保目录存在（处理文件路径和目录路径两种情况）"""
    # 如果是文件路径，取其目录
    if os.path.splitext(path)[1] != '':
        dir_path = os.path.dirname(path)
    else:
        dir_path = path

    # 如果目录为空（当前目录），直接返回
    if not dir_path:
        return True

    # 创建目录（递归创建多级目录）
    try:
        os.makedirs(dir_path, exist_ok=True)
        return True
    except Exception as e:
        print(f"❌ 创建目录失败: {dir_path} | 错误: {str(e)[:100]}")
        return False

# 模拟依赖模块


def process_match_data(match_dir, output_file):
    """模拟生成训练数据"""
    # 确保输出目录存在
    ensure_dir_exists(output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        samples = []
        for i in range(10):
            samples.append({
                "states": [[float(i)]*81, [float(i+1)]*81, [float(i+2)]*81],
                "action": [0.1*i, 0.2*i, 0.3*i, 0.4*i, 0.5*i],
                "value": 0.8*i
            })
        json.dump(samples, f, ensure_ascii=False)


class DualNetwork(nn.Module):
    """模拟双网络模型"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3*81, 128)
        self.policy_head = nn.Linear(128, 5)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return {
            'policy_output': self.policy_head(x),
            'value_output': self.value_head(x)
        }

# ========== 核心修复：Windows路径兼容的流式数据集 ==========


class StreamingBilliardsDataset(Dataset):
    def __init__(self, json_file, preprocessor=None, use_existing_index=True):
        # 核心修复1：统一转换为绝对路径
        self.json_file = get_absolute_path(json_file)
        self.preprocessor = preprocessor
        self.sample_indices = []

        # 检查数据文件是否存在
        if not os.path.exists(self.json_file):
            print(f"❌ 数据文件不存在: {self.json_file}")
            self.file_size = 0
        else:
            self.file_size = os.path.getsize(self.json_file)

        # 核心修复2：索引文件使用绝对路径
        self.index_file = get_absolute_path(f"{self.json_file}.index.json")
        self.use_existing_index = use_existing_index
        self.chunk_size = min(512 * 1024 * 1024, self.file_size //
                              4 if self.file_size > 0 else 512 * 1024 * 1024)

        # 预扫描文件
        self._scan_sample_positions()

    def _load_index_file(self):
        """加载索引文件（Windows路径兼容）"""
        try:
            if not os.path.exists(self.index_file):
                return False

            # 验证文件可读取
            if not os.access(self.index_file, os.R_OK):
                print(f"❌ 无读取权限: {self.index_file}")
                return False

            # 流式读取JSON
            with open(self.index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)

            # 校验
            current_file_size = os.path.getsize(
                self.json_file) if os.path.exists(self.json_file) else 0
            if index_data.get('file_size', 0) != current_file_size:
                print(f"⚠️  索引与数据文件大小不匹配，重新生成...")
                return False

            self.sample_indices = index_data.get('sample_indices', [])
            if len(self.sample_indices) == 0:
                print(f"⚠️  索引文件无有效数据，重新生成...")
                return False

            print(f"✅ 成功加载索引: {self.index_file}")
            print(f"   样本数: {len(self.sample_indices)}")
            return True

        except Exception as e:
            print(f"⚠️  加载索引失败: {str(e)[:100]}，重新生成...")
            return False

    def _save_index_file(self):
        """保存索引文件（Windows路径核心修复）"""
        try:
            # 核心修复3：强制确保索引文件目录存在
            if not ensure_dir_exists(self.index_file):
                return False

            # 验证目录可写入
            index_dir = os.path.dirname(self.index_file)
            if not os.access(index_dir, os.W_OK):
                print(f"❌ 无写入权限: {index_dir}")
                return False

            # 准备索引数据
            index_data = {
                'version': '1.0',
                'file_size': self.file_size,
                'create_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_samples': len(self.sample_indices),
                'sample_indices': self.sample_indices,
                'json_file_path': self.json_file,
                'index_file_path': self.index_file
            }

            # 核心修复4：Windows下使用正确的写入方式
            with open(self.index_file, 'w', encoding='utf-8', newline='') as f:
                # 格式化JSON，便于查看
                json_str = json.dumps(index_data, ensure_ascii=False, indent=2)
                # 分块写入，避免大文件问题
                chunk_size = 4096
                for i in range(0, len(json_str), chunk_size):
                    f.write(json_str[i:i+chunk_size])
                f.flush()  # 强制刷盘
                os.fsync(f.fileno())  # Windows下强制写入磁盘

            # 最终验证
            if not os.path.exists(self.index_file):
                print(f"❌ 索引文件保存后不存在: {self.index_file}")
                return False

            file_size = os.path.getsize(self.index_file)
            print(f"✅ 索引保存成功: {self.index_file}")
            print(f"   大小: {file_size / 1024:.2f} KB")
            return True

        except PermissionError:
            print(f"❌ 保存失败：无写入权限，请以管理员身份运行")
            return False
        except Exception as e:
            print(f"❌ 保存索引失败: {str(e)[:100]}")
            # 清理残缺文件
            if os.path.exists(self.index_file):
                try:
                    os.remove(self.index_file)
                except:
                    pass
            return False

    def _scan_sample_positions(self):
        """扫描样本位置（Windows兼容）"""
        # 优先加载索引
        if self.use_existing_index and self._load_index_file():
            return

        # 检查数据文件
        if not os.path.exists(self.json_file):
            print(f"❌ 数据文件不存在，跳过扫描")
            return
        if self.file_size == 0:
            print(f"❌ 数据文件为空，跳过扫描")
            return

        print(f"\n📊 扫描数据文件: {self.json_file}")
        print(f"   大小: {self.file_size / (1024*1024):.2f} MB")
        print(f"   块大小: {self.chunk_size / (1024*1024):.2f} MB")

        sample_indices = []
        depth = 0
        current_pos = 0

        # 二进制模式读取，避免编码问题
        with open(self.json_file, 'rb') as f:
            # 跳过开头的[
            while True:
                byte = f.read(1)
                current_pos += 1
                if not byte:
                    break
                char = byte.decode('utf-8', errors='ignore')
                if char == '[':
                    break
                if char not in [' ', '\n', '\r', '\t']:
                    print(f"⚠️  数据文件不是JSON数组")
                    return

            # 进度条
            pbar = tqdm(total=self.file_size, desc="解析样本位置",
                        unit="B", unit_scale=True)
            pbar.update(current_pos)

            # 分块解析
            buffer = b''
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                buffer += chunk
                buffer_pos = 0
                buffer_len = len(buffer)

                while buffer_pos < buffer_len:
                    byte = buffer[buffer_pos:buffer_pos+1]
                    try:
                        char = byte.decode('utf-8')
                    except:
                        char = ''
                    buffer_pos += 1
                    current_pos += 1

                    if current_pos % 1000 == 0:
                        pbar.update(1000)

                    if char in [' ', '\n', '\r', '\t', ',']:
                        continue

                    if char == '{':
                        depth += 1
                        if depth == 1:
                            sample_start = current_pos - \
                                1 - len(buffer) + buffer_pos
                            sample_indices.append(sample_start)
                    elif char == '}':
                        depth -= 1

                buffer = buffer[-1000:] if buffer_len > 1000 else b''

            pbar.update(self.file_size - pbar.n)
            pbar.close()

        self.sample_indices = sample_indices
        print(f"\n✅ 扫描完成！样本数: {len(sample_indices)}")

        # 保存索引
        self._save_index_file()

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        """读取样本（Windows兼容）"""
        try:
            if idx < 0 or idx >= len(self.sample_indices):
                raise IndexError(f"索引 {idx} 超出范围")

            start_pos = self.sample_indices[idx]

            with open(self.json_file, 'r', encoding='utf-8') as f:
                f.seek(start_pos)
                sample_str = ''
                depth = 0
                while True:
                    char = f.read(1)
                    if not char:
                        break
                    sample_str += char

                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            break

                sample = json.loads(sample_str)

            states = np.array(sample['states'], dtype=np.float32)
            action = np.array(sample['action'], dtype=np.float32)
            value = np.array([sample['value']], dtype=np.float32)

            if self.preprocessor is not None:
                states = self.preprocessor(states)

            return (
                torch.from_numpy(states),
                torch.from_numpy(action),
                torch.from_numpy(value)
            )

        except Exception as e:
            print(f"⚠️  加载样本 {idx} 失败: {str(e)[:100]}")
            return (
                torch.zeros(3, 81, dtype=torch.float32),
                torch.zeros(5, dtype=torch.float32),
                torch.zeros(1, dtype=torch.float32)
            )

# ========== 状态预处理器 ==========


class StatePreprocessor:
    def __call__(self, states):
        states[:, 64] = states[:, 64] / 2.540
        states[:, 65] = states[:, 65] / 2.540
        states[:, :64:4] = states[:, :64:4] / 2.540
        states[:, 1:64:4] = states[:, 1:64:4] / 1.270
        return states

# ========== 训练函数 ==========


def train(args):
    """训练函数（Windows路径兼容）"""
    # 转换为绝对路径
    args.train_data_file = get_absolute_path(args.train_data_file)
    args.model_dir = get_absolute_path(args.model_dir)
    args.match_dir = get_absolute_path(args.match_dir)

    # 确保模型目录存在
    ensure_dir_exists(args.model_dir)

    # 数据处理
    if args.use_existing_data:
        if os.path.exists(args.train_data_file):
            print(f"✅ 使用已有数据: {args.train_data_file}")
        else:
            print(f"❌ 数据文件不存在: {args.train_data_file}")
            if args.auto_generate_if_missing:
                print(f"🔄 自动生成数据...")
                process_match_data(args.match_dir, args.train_data_file)
            else:
                return
    else:
        print(f"🔄 重新生成数据...")
        process_match_data(args.match_dir, args.train_data_file)
        # 删除旧索引
        index_file = get_absolute_path(f"{args.train_data_file}.index.json")
        if os.path.exists(index_file):
            try:
                os.remove(index_file)
                print(f"🗑️ 删除旧索引: {index_file}")
            except:
                print(f"⚠️ 删除旧索引失败")

    gc.collect()

    # 加载数据集
    print("\n📥 加载数据集...")
    preprocessor = StatePreprocessor()
    dataset = StreamingBilliardsDataset(
        json_file=args.train_data_file,
        preprocessor=preprocessor,
        use_existing_index=args.use_existing_index
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows下必须设为0
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )

    print(f"📊 数据集信息:")
    print(f"   总样本: {len(dataset)}")
    print(f"   批次大小: {args.batch_size}")
    print(f"   总批次: {len(dataloader)}")

    # 初始化模型
    print("\n🔧 初始化模型...")
    model = DualNetwork()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"💻 设备: {device}")

    # 损失函数和优化器
    policy_criterion = nn.MSELoss()
    value_criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma
    )

    # 训练循环
    print("\n🚀 开始训练...")
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    for epoch in range(args.epochs):
        model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0

        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch_idx, (states, policy_targets, value_targets) in enumerate(pbar):
                states = states.to(device, non_blocking=True)
                policy_targets = policy_targets.to(device, non_blocking=True)
                value_targets = value_targets.to(device, non_blocking=True)

                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(states)
                        policy_loss = policy_criterion(
                            outputs['policy_output'], policy_targets)
                        value_loss = value_criterion(
                            outputs['value_output'], value_targets)
                        loss = policy_loss + value_loss
                else:
                    outputs = model(states)
                    policy_loss = policy_criterion(
                        outputs['policy_output'], policy_targets)
                    value_loss = value_criterion(
                        outputs['value_output'], value_targets)
                    loss = policy_loss + value_loss

                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_loss += loss.item()

                pbar.set_postfix({
                    'Policy Loss': f'{policy_loss.item():.6f}',
                    'Value Loss': f'{value_loss.item():.6f}',
                    'Total Loss': f'{loss.item():.6f}'
                })

                if batch_idx % 100 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        scheduler.step()

        avg_policy_loss = total_policy_loss / len(dataloader)
        avg_value_loss = total_value_loss / len(dataloader)
        avg_total_loss = total_loss / len(dataloader)

        print(f"\n📈 Epoch {epoch+1} 结果:")
        print(f"   平均策略损失: {avg_policy_loss:.6f}")
        print(f"   平均价值损失: {avg_value_loss:.6f}")
        print(f"   平均总损失: {avg_total_loss:.6f}")
        print(f"   学习率: {optimizer.param_groups[0]['lr']:.6e}")

        # 保存模型
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = os.path.join(
                args.model_dir, f"dual_network_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_total_loss,
            }, checkpoint_path)
            print(f"💾 模型保存: {checkpoint_path}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 保存最终模型
    final_model_path = os.path.join(args.model_dir, "dual_network_final.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"\n🏁 训练完成！最终模型: {final_model_path}")

# ========== 主函数 ==========


def main():
    parser = argparse.ArgumentParser(description="台球双网络训练（Windows兼容版）")

    # 核心参数
    parser.add_argument('--use_existing_data', action='store_true',
                        help='使用已有训练数据')
    parser.add_argument('--auto_generate_if_missing', action='store_true',
                        help='数据缺失时自动生成')
    parser.add_argument('--use_existing_index', action='store_true', default=True,
                        help='使用已有索引文件')

    # 路径参数
    parser.add_argument('--match_dir', type=str, default='match_data',
                        help='原始对局数据目录')
    parser.add_argument('--train_data_file', type=str, default='trainable_data.json',
                        help='训练数据文件路径')

    # 模型参数
    parser.add_argument('--model_dir', type=str, default='models',
                        help='模型保存目录')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=3,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减')
    parser.add_argument('--lr_step_size', type=int, default=10,
                        help='学习率衰减步长')
    parser.add_argument('--lr_gamma', type=float, default=0.5,
                        help='学习率衰减系数')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='模型保存间隔')

    args = parser.parse_args()

    # Windows下的额外检查
    if sys.platform == 'win32':
        print(f"🔍 Windows系统检测，自动处理路径问题")
        # 确保当前目录可写
        if not os.access('.', os.W_OK):
            print(f"⚠️ 当前目录无写入权限，请以管理员身份运行")

    train(args)


if __name__ == '__main__':
    main()
