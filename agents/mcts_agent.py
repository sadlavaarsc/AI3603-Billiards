import numpy as np
import torch
import collections
from .agent import Agent
from MCTS import MCTS
from dual_network import DualNetwork
from data_loader import StatePreprocessor

class MCTSAgent(Agent):
    """基于 MCTS 的智能 Agent"""
    
    def __init__(self, model=None, env=None, n_simulations=30, n_action_samples=8, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.env = env
        self.device = device
        self.model = model
        
        if model is not None and env is not None:
            self.mcts = MCTS(
                model=model,
                env=env,
                n_simulations=n_simulations,
                n_action_samples=n_action_samples,
                device=device
            )
        else:
            self.mcts = None
        
        self.state_buffer = collections.deque(maxlen=3)
        self.n_simulations = n_simulations
        self.n_action_samples = n_action_samples
    
    def set_env(self, env):
        """设置环境"""
        self.env = env
        if self.model is not None:
            self.mcts = MCTS(
                model=self.model,
                env=env,
                n_simulations=self.n_simulations,
                n_action_samples=self.n_action_samples,
                device=self.device
            )
    
    def set_model(self, model):
        """设置模型"""
        self.model = model
        if self.env is not None:
            self.mcts = MCTS(
                model=model,
                env=self.env,
                n_simulations=self.n_simulations,
                n_action_samples=self.n_action_samples,
                device=self.device
            )
    
    def load_model(self, model_path):
        """加载模型"""
        model = DualNetwork()
        model.load(model_path)
        model.to(self.device)
        model.eval()
        self.set_model(model)
    
    def clear_buffer(self):
        """清空状态缓冲区"""
        self.state_buffer.clear()
    
    def set_simulations(self, n_simulations):
        """设置模拟次数"""
        self.n_simulations = n_simulations
        if self.mcts is not None:
            self.mcts.n_simulations = n_simulations
    
    def set_action_samples(self, n_action_samples):
        """设置动作采样数量"""
        self.n_action_samples = n_action_samples
        if self.mcts is not None:
            self.mcts.n_action_samples = n_action_samples
    
    def decision(self, balls_state, my_targets=None, table=None):
        """使用 MCTS 进行决策
        
        参数：
            balls_state: 球状态字典
            my_targets: 目标球ID列表
            table: 球桌对象
        
        返回：
            dict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
        """
        # 维护最近三杆
        state_vec = self.mcts._balls_state_to_81(balls_state, my_targets, table)
        if len(self.state_buffer) < 3:
            self.state_buffer.append(state_vec)
            while len(self.state_buffer) < 3:
                self.state_buffer.append(state_vec)
        else:
            self.state_buffer.append(state_vec)
        state_seq = list(self.state_buffer)
        action = self.mcts.search(state_seq)
        return {
            "V0": float(action[0]),
            "phi": float(action[1]),
            "theta": float(action[2]),
            "a": float(action[3]),
            "b": float(action[4]),
        }

class DirectModelAgent(Agent):
    """直接输出模型预测结果的 Agent，不进行 MCTS 模拟和采样"""
    def __init__(self, model=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.model = model
        self.device = device
        self.state_buffer = collections.deque(maxlen=3)
        self.state_preprocessor = StatePreprocessor()
        
        # 与 MCTS 一致的动作范围
        self.action_min = np.array([0.5, 0.0, 0.0, -0.5, -0.5], dtype=np.float32)
        self.action_max = np.array([8.0, 360.0, 90.0, 0.5, 0.5], dtype=np.float32)
    
    def set_model(self, model):
        """设置模型"""
        self.model = model
    
    def load_model(self, model_path):
        """加载模型"""
        model = DualNetwork()
        model.load(model_path)
        model.to(self.device)
        model.eval()
        self.model = model
    
    def clear_buffer(self):
        """清空状态缓冲区"""
        self.state_buffer.clear()
    
    def _denormalize_action(self, action_norm):
        """将 [0,1] 策略输出映射回真实物理动作"""
        action_norm = np.clip(action_norm, 0.0, 1.0)
        return action_norm * (self.action_max - self.action_min) + self.action_min
    
    def _balls_state_to_81(self, balls_state, my_targets, table):
        """将球状态转换为81维向量"""
        state = np.zeros(81, dtype=np.float32)
        
        ball_order = [
            'cue', '1', '2', '3', '4', '5', '6', '7', '8',
            '9', '10', '11', '12', '13', '14', '15'
        ]
        
        STANDARD_TABLE_WIDTH = 2.845
        STANDARD_TABLE_LENGTH = 1.4225
        BALL_RADIUS = 0.0285
        
        for i, ball_id in enumerate(ball_order):
            base = i * 4
            
            if ball_id in balls_state:
                ball = balls_state[ball_id]
                rvw = ball.state.rvw
                pos = rvw[0]
                
                if ball.state.s == 4:  # 进袋
                    state[base + 0] = -1.0
                    state[base + 1] = -1.0
                    state[base + 2] = -1.0
                    state[base + 3] = 1.0
                else:
                    # 球坐标归一化
                    state[base + 0] = pos[0] / STANDARD_TABLE_WIDTH
                    state[base + 1] = pos[1] / STANDARD_TABLE_LENGTH
                    state[base + 2] = pos[2] / (2 * BALL_RADIUS)
                    state[base + 3] = 0.0
            else:
                state[base + 0] = -1.0
                state[base + 1] = -1.0
                state[base + 2] = -1.0
                state[base + 3] = 1.0
        
        # 球桌尺寸
        if table is not None:
            state[64] = table.w
            state[65] = table.l
        else:
            state[64] = 2.540
            state[65] = 1.270
        
        # 目标球 one-hot
        if my_targets:
            for t in my_targets:
                idx = int(t) - 1
                state[66 + idx] = 1.0
        
        return state
    
    def decision(self, balls_state, my_targets=None, table=None):
        """直接使用模型预测进行决策，不进行 MCTS 模拟
        
        参数：
            balls_state: 球状态字典
            my_targets: 目标球ID列表
            table: 球桌对象
        
        返回：
            dict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
        """
        # 维护最近三杆
        state_vec = self._balls_state_to_81(balls_state, my_targets, table)
        if len(self.state_buffer) < 3:
            self.state_buffer.append(state_vec)
            while len(self.state_buffer) < 3:
                self.state_buffer.append(state_vec)
        else:
            self.state_buffer.append(state_vec)
        state_seq = list(self.state_buffer)
        
        # 转换为模型输入格式
        states = np.stack(state_seq, axis=0)
        states = self.state_preprocessor(states)
        state_tensor = torch.from_numpy(states).float().unsqueeze(0).to(self.device)
        
        # 直接使用模型预测
        with torch.no_grad():
            out = self.model(state_tensor)
            action_norm = out["policy_output"][0].cpu().numpy()
        
        # 反归一化得到真实物理动作
        action = self._denormalize_action(action_norm)
        
        return {
            "V0": float(action[0]),
            "phi": float(action[1]),
            "theta": float(action[2]),
            "a": float(action[3]),
            "b": float(action[4]),
        }
