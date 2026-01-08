import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
from poolenv import PoolEnv
from agents import NewAgent
import os
import time

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.position = 0
    
    def add(self, state, action, reward, next_state, done):
        """添加经验"""
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.buffer_size
    
    def sample(self, batch_size):
        """采样经验"""
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in batch:
            states.append(self.buffer[idx][0])
            actions.append(self.buffer[idx][1])
            rewards.append(self.buffer[idx][2])
            next_states.append(self.buffer[idx][3])
            dones.append(self.buffer[idx][4])
        
        # 转换为PyTorch张量
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def size(self):
        """返回缓冲区大小"""
        return len(self.buffer)

class PPOAgent(NewAgent):
    """基于PPO算法的强化学习智能体"""
    def __init__(self, lr=1e-4, gamma=0.99, clip_epsilon=0.2, K_epochs=4, batch_size=64):
        super().__init__()
        # PPO参数
        self.lr = lr
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        
        # 优化器
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        
        # 动作分布的标准差
        self.log_std = torch.tensor([-0.5] * 5, requires_grad=True)
        self.optimizer.add_param_group({'params': self.log_std})
        
        print("PPOAgent 已初始化。")
    
    def get_action(self, state):
        """获取动作和对数概率"""
        state_tensor = torch.tensor(state).unsqueeze(0)
        with torch.no_grad():
            mean = self.net(state_tensor)[0]
            std = torch.exp(self.log_std)
            normal_dist = Normal(mean, std)
            action = normal_dist.sample()
            log_prob = normal_dist.log_prob(action).sum(dim=-1)
            
            # 确保动作在合法范围内
            action = self._normalize_action(action.numpy())
        
        return action, log_prob.item()
    
    def _normalize_action(self, action):
        """将网络输出映射到合法的动作范围"""
        normalized_action = np.zeros_like(action)
        normalized_action[0] = F.sigmoid(torch.tensor(action[0])).item() * 7.5 + 0.5  # V0: 0.5-8.0
        normalized_action[1] = F.sigmoid(torch.tensor(action[1])).item() * 360  # phi: 0-360
        normalized_action[2] = F.sigmoid(torch.tensor(action[2])).item() * 90  # theta: 0-90
        normalized_action[3] = torch.tanh(torch.tensor(action[3])).item() * 0.5  # a: -0.5-0.5
        normalized_action[4] = torch.tanh(torch.tensor(action[4])).item() * 0.5  # b: -0.5-0.5
        return normalized_action
    
    def evaluate(self, states, actions):
        """评估动作的对数概率和值函数"""
        mean = self.net(states)
        std = torch.exp(self.log_std)
        normal_dist = Normal(mean, std)
        
        # 计算对数概率
        log_probs = normal_dist.log_prob(actions).sum(dim=-1, keepdim=True)
        
        # 计算熵（用于探索）
        entropy = normal_dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_probs, entropy
    
    def update(self, replay_buffer):
        """更新模型参数"""
        if replay_buffer.size() < self.batch_size:
            return
        
        for _ in range(self.K_epochs):
            # 采样批次经验
            states, actions, rewards, next_states, dones = replay_buffer.sample(self.batch_size)
            
            # 计算旧动作的对数概率
            old_log_probs, _ = self.evaluate(states, actions)
            
            # 计算优势函数
            with torch.no_grad():
                # 计算TD目标
                td_targets = rewards + self.gamma * (1 - dones) * self.net(next_states).mean(dim=-1, keepdim=True)
                
                # 计算值函数估计
                value_estimates = self.net(states).mean(dim=-1, keepdim=True)
                
                # 计算优势
                advantages = td_targets - value_estimates
                
                # 标准化优势
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 计算新动作的对数概率和熵
            new_log_probs, entropy = self.evaluate(states, actions)
            
            # 计算概率比
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            
            # 计算PPO损失
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            
            # 计算值函数损失
            critic_loss = F.mse_loss(self.net(states).mean(dim=-1, keepdim=True), td_targets.detach())
            
            # 计算总损失
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()
            
            # 更新模型
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()
    
    def save_model(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'log_std': self.log_std,
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        print(f"模型已保存到 {path}")
    
    def load_model(self, path):
        """加载模型"""
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.log_std = checkpoint['log_std']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"模型已从 {path} 加载")
        else:
            print(f"模型文件 {path} 不存在")

def train():
    """训练主函数"""
    # 环境和智能体初始化
    env = PoolEnv(verbose=False)
    agent = PPOAgent(lr=1e-4, gamma=0.99, clip_epsilon=0.2, K_epochs=4, batch_size=64)
    
    # 经验回放缓冲区
    buffer_size = 10000
    replay_buffer = ReplayBuffer(buffer_size)
    
    # 训练参数
    num_episodes = 1000
    max_steps_per_episode = 100
    
    # 开始训练
    for episode in range(num_episodes):
        # 重置环境，设置玩家A的目标球型为实心球(1-7)
        env.reset(target_ball='solid')
        state = env.get_observation('A')
        
        # 预处理状态
        state_features = agent._preprocess_observation(state[0], state[1], state[2])
        
        episode_reward = 0
        done = False
        
        for step in range(max_steps_per_episode):
            # 获取动作
            action, log_prob = agent.get_action(state_features)
            
            # 转换为环境可接受的动作格式
            action_dict = {
                'V0': action[0],
                'phi': action[1],
                'theta': action[2],
                'a': action[3],
                'b': action[4]
            }
            
            # 执行动作（确保所有参数都是float64类型）
            action_dict_float64 = {k: float(v) for k, v in action_dict.items()}
            step_info = env.take_shot(action_dict_float64)
            
            # 检查是否结束
            done, info = env.get_done()
            
            # 获取下一个状态
            next_state = env.get_observation('A')
            next_state_features = agent._preprocess_observation(next_state[0], next_state[1], next_state[2])
            
            # 计算奖励
            reward = calculate_reward(step_info, done, info)
            episode_reward += reward
            
            # 添加经验到回放缓冲区
            replay_buffer.add(state_features, action, reward, next_state_features, done)
            
            # 更新状态
            state_features = next_state_features
            
            if done:
                break
        
        # 每局更新一次模型
        agent.update(replay_buffer)
        
        # 打印训练信息
        print(f"Episode: {episode+1}, Reward: {episode_reward:.2f}, Steps: {step+1}")
        
        # 每100局保存一次模型
        if (episode + 1) % 100 == 0:
            agent.save_model(f"./models/ppo_agent_{episode+1}.pt")
    
    # 训练结束后保存最终模型
    agent.save_model("./models/ppo_agent_final.pt")

def calculate_reward(step_info, done, info):
    """计算奖励，完全对齐台球规则"""
    reward = 0
    
    # 1. 进球奖励
    if 'ME_INTO_POCKET' in step_info and step_info['ME_INTO_POCKET']:
        reward += len(step_info['ME_INTO_POCKET']) * 50
    
    # 2. 黑8进球奖励
    if done and info['winner'] == 'A':
        reward += 100  # 合法黑8进球奖励
    
    # 3. 对方进球惩罚
    if 'ENEMY_INTO_POCKET' in step_info and step_info['ENEMY_INTO_POCKET']:
        reward -= len(step_info['ENEMY_INTO_POCKET']) * 20
    
    # 4. 白球进袋惩罚
    if 'FOUL_CUE_IN_POCKET' in step_info and step_info['FOUL_CUE_IN_POCKET']:
        if 'ME_INTO_POCKET' in step_info and '8' in step_info['ME_INTO_POCKET']:
            reward -= 150  # 白球+黑8同时进袋，严重犯规
        else:
            reward -= 100  # 白球进袋
    
    # 5. 非法黑8惩罚
    if not done and 'ME_INTO_POCKET' in step_info and '8' in step_info['ME_INTO_POCKET']:
        reward -= 150  # 清台前误打黑8，判负
    
    # 6. 其他犯规惩罚
    if 'FOUL_FIRST_HIT' in step_info and step_info['FOUL_FIRST_HIT']:
        reward -= 30  # 首球犯规
    if 'FOUL_NO_RAIL' in step_info and step_info['FOUL_NO_RAIL']:
        reward -= 30  # 碰库犯规
    if 'FOUL_NO_HIT' in step_info and step_info['FOUL_NO_HIT']:
        reward -= 30  # 未击中任何球
    
    # 7. 合法无进球小奖励
    if reward == 0:
        reward += 10  # 合法无进球
    
    # 8. 时间惩罚
    reward -= 0.1
    
    return reward

if __name__ == "__main__":
    # 创建模型保存目录
    os.makedirs("./models", exist_ok=True)
    
    # 开始训练
    train()
