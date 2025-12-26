import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
from dual_network import DualNetwork


class DualNetworkTrainer:
    """
    双网络训练器：负责训练行为网络和价值网络
    """
    def __init__(self, model=None, lr=1e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # 如果没有提供模型，创建一个新模型
        if model is None:
            self.model = DualNetwork().to(self.device)
        else:
            self.model = model.to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # 损失函数
        self.policy_loss_fn = nn.MSELoss()  # 行为网络使用MSE
        self.value_loss_fn = nn.BCELoss()   # 价值网络使用BCE
    
    def train_step(self, states, policy_targets, value_targets):
        """
        单步训练
        
        参数：
            states: 输入状态，形状[batch_size, 3, 81]
            policy_targets: 行为目标，形状[batch_size, 5]
            value_targets: 价值目标，形状[batch_size, 1]
        
        返回：
            总损失、策略损失、价值损失
        """
        # 转换为张量
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        policy_targets = torch.tensor(policy_targets, dtype=torch.float32).to(self.device)
        value_targets = torch.tensor(value_targets, dtype=torch.float32).to(self.device)
        
        # 前向传播
        outputs = self.model(states)
        policy_outputs = outputs['policy_output']
        value_outputs = outputs['value_output']
        
        # 计算损失
        policy_loss = self.policy_loss_fn(policy_outputs, policy_targets)
        value_loss = self.value_loss_fn(value_outputs, value_targets)
        
        # 总损失（权重可以调整）
        total_loss = policy_loss + value_loss
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
    
    def train(self, train_loader, epochs=100, save_path=None, save_interval=10):
        """
        完整训练过程
        
        参数：
            train_loader: 数据加载器，返回(states, policy_targets, value_targets)
            epochs: 训练轮数
            save_path: 模型保存路径
            save_interval: 每多少轮保存一次模型
        """
        # 创建保存目录
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 训练历史记录
        history = {
            'total_loss': [],
            'policy_loss': [],
            'value_loss': []
        }
        
        # 训练循环
        for epoch in range(epochs):
            epoch_losses = {
                'total_loss': 0,
                'policy_loss': 0,
                'value_loss': 0
            }
            
            # 进度条
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            
            for i, (states, policy_targets, value_targets) in pbar:
                # 单步训练
                losses = self.train_step(states, policy_targets, value_targets)
                
                # 累计损失
                epoch_losses['total_loss'] += losses['total_loss']
                epoch_losses['policy_loss'] += losses['policy_loss']
                epoch_losses['value_loss'] += losses['value_loss']
                
                # 更新进度条
                pbar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {losses['total_loss']:.6f}")
            
            # 计算平均损失
            avg_total_loss = epoch_losses['total_loss'] / len(train_loader)
            avg_policy_loss = epoch_losses['policy_loss'] / len(train_loader)
            avg_value_loss = epoch_losses['value_loss'] / len(train_loader)
            
            # 记录历史
            history['total_loss'].append(avg_total_loss)
            history['policy_loss'].append(avg_policy_loss)
            history['value_loss'].append(avg_value_loss)
            
            # 打印 epoch 信息
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Total Loss: {avg_total_loss:.6f}")
            print(f"  Policy Loss: {avg_policy_loss:.6f}")
            print(f"  Value Loss: {avg_value_loss:.6f}")
            
            # 保存模型
            if save_path and (epoch + 1) % save_interval == 0:
                model_path = save_path.replace('.pt', f'_{epoch+1}.pt')
                self.model.save(model_path)
                print(f"Model saved to {model_path}")
        
        # 保存最终模型
        if save_path:
            self.model.save(save_path)
            print(f"Final model saved to {save_path}")
        
        return history
    
    def evaluate(self, test_loader):
        """
        评估模型性能
        """
        self.model.eval()
        total_loss = 0
        policy_loss = 0
        value_loss = 0
        
        with torch.no_grad():
            for states, policy_targets, value_targets in test_loader:
                # 转换为张量
                states = torch.tensor(states, dtype=torch.float32).to(self.device)
                policy_targets = torch.tensor(policy_targets, dtype=torch.float32).to(self.device)
                value_targets = torch.tensor(value_targets, dtype=torch.float32).to(self.device)
                
                # 前向传播
                outputs = self.model(states)
                policy_outputs = outputs['policy_output']
                value_outputs = outputs['value_output']
                
                # 计算损失
                policy_loss += self.policy_loss_fn(policy_outputs, policy_targets).item()
                value_loss += self.value_loss_fn(value_outputs, value_targets).item()
                total_loss += (policy_loss + value_loss)
        
        # 恢复训练模式
        self.model.train()
        
        return {
            'total_loss': total_loss / len(test_loader),
            'policy_loss': policy_loss / len(test_loader),
            'value_loss': value_loss / len(test_loader)
        }


def create_dataloader(data, batch_size=32, shuffle=True):
    """
    创建数据加载器
    
    参数：
        data: 包含states, policy_targets, value_targets的字典
        batch_size: 批次大小
        shuffle: 是否打乱数据
    """
    # 转换为numpy数组
    states = np.array(data['states'])
    policy_targets = np.array(data['policy_targets'])
    value_targets = np.array(data['value_targets'])
    
    # 创建索引
    indices = np.arange(len(states))
    if shuffle:
        np.random.shuffle(indices)
    
    # 批次生成器
    def batch_generator():
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            yield (
                states[batch_indices],
                policy_targets[batch_indices],
                value_targets[batch_indices]
            )
    
    # 返回生成器和长度
    return batch_generator(), len(range(0, len(indices), batch_size))


# 示例使用代码
if __name__ == "__main__":
    # 创建示例数据
    batch_size = 32
    example_states = np.random.rand(batch_size, 3, 81)  # 3局×81维
    example_policy_targets = np.random.rand(batch_size, 5)  # 5维动作
    example_value_targets = np.random.randint(0, 2, (batch_size, 1))  # 0或1
    
    # 创建数据加载器
    data = {
        'states': [example_states[0]],  # 示例数据
        'policy_targets': [example_policy_targets[0]],
        'value_targets': [example_value_targets[0]]
    }
    train_loader, _ = create_dataloader(data, batch_size=1)
    
    # 创建训练器
    trainer = DualNetworkTrainer()
    
    # 训练模型（仅作演示）
    print("Starting training (demo mode)...")
    trainer.train(train_loader, epochs=1, save_path='models/dual_network_demo.pt')
    print("Training completed (demo mode)")
