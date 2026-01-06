import torch
import numpy as np
import math
import copy
from collections import defaultdict
from dual_network import DualNetwork
from poolenv import PoolEnv, save_balls_state, restore_balls_state

# MCTS节点类
class MCTSNode:
    def __init__(self, state, parent=None, action=None, player='A'):
        """
        初始化MCTS节点
        :param state: 环境状态 (balls, my_targets, table)
        :param parent: 父节点
        :param action: 到达该节点的动作
        :param player: 当前节点的玩家 ('A'/'B')
        """
        self.state = state  # (balls, my_targets, table)
        self.parent = parent
        self.action = action
        self.player = player
        
        # 统计信息
        self.visits = 0  # 访问次数
        self.value = 0.0  # 累计价值
        self.children = {}  # 子节点: action -> MCTSNode
        self.untried_actions = None  # 未尝试的动作
        
        # 先验概率（由行为网络给出）
        self.prior = 0.0
        
        # 环境相关
        self.env_snapshot = None  # 环境快照，用于快速恢复
        self.done = False
        self.winner = None

    def is_terminal(self):
        """判断是否为终止节点"""
        return self.done
    
    def is_fully_expanded(self):
        """判断是否完全扩展"""
        return len(self.untried_actions) == 0
    
    def update(self, value):
        """更新节点价值和访问次数"""
        self.visits += 1
        self.value += value
    
    def uct_value(self, exploration_weight=1.414):
        """
        计算UCT值（Upper Confidence Bound for Trees）
        :param exploration_weight: 探索权重
        :return: UCT值
        """
        if self.visits == 0:
            return float('inf')
        
        # 计算利用项 + 探索项
        exploitation = self.value / self.visits
        exploration = exploration_weight * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return exploitation + exploration


# MCTS核心类
class MCTS:
    def __init__(self, model: DualNetwork, env: PoolEnv, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化MCTS
        :param model: 双网络模型
        :param env: 台球环境
        :param device: 计算设备
        """
        self.model = model
        self.env = env
        self.device = device
        self.model.eval()  # 评估模式
        
        # 动作探索参数
        self.action_noise = 0.1  # 动作噪声标准差
        self.num_simulations = 100  # 每次决策的模拟次数
        self.action_expansion_num = 10  # 每个节点扩展的动作数
        
        # 状态历史缓存（存储最近3局状态）
        self.state_history = []

    def encode_state(self, observation):
        """
        将环境观测编码为网络输入格式 (Batch × 3 × 81)
        :param observation: (balls, my_targets, table)
        :return: 81维状态向量
        """
        balls, my_targets, table = observation
        
        # 初始化81维特征向量
        state_vec = np.zeros(81)
        
        # 球位置特征 (16球 × 3维坐标 = 48维)
        ball_ids = ['cue', '1', '2', '3', '4', '5', '6', '7', '8', 
                    '9', '10', '11', '12', '13', '14', '15']
        for i, bid in enumerate(ball_ids):
            if bid in balls:
                pos = balls[bid].state.rvw[0]  # (x, y, z)
                state_vec[i*3:i*3+3] = pos / table.l  # 归一化到球桌长度
        
        # 球速度特征 (16球 × 3维速度 = 48维，取前33维避免越界)
        for i, bid in enumerate(ball_ids):
            if bid in balls and i*3+3 < 81:
                vel = balls[bid].state.rvw[1]
                state_vec[48 + i*3:48 + i*3+3] = vel / 10.0  # 速度归一化
        
        # 目标球掩码 (剩余维度填充)
        target_mask = np.zeros(81 - 84) if 81 - 84 > 0 else np.array([])
        for i, bid in enumerate(my_targets):
            if i < len(target_mask):
                target_mask[i] = 1.0
        
        if len(target_mask) > 0:
            state_vec[84:] = target_mask
        
        return state_vec.astype(np.float32)

    def get_state_for_network(self):
        """
        获取最近3局状态，填充到网络输入格式
        :return: (1 × 3 × 81) 张量
        """
        # 确保有3个状态，不足则补0
        while len(self.state_history) < 3:
            self.state_history.insert(0, np.zeros(81))
        
        # 取最近3个状态
        states = np.array(self.state_history[-3:])
        return torch.tensor(states).unsqueeze(0).to(self.device)

    def generate_actions(self, base_action, num_actions=10):
        """
        基于基础动作生成相似的探索动作
        :param base_action: 基础动作 {'V0', 'phi', 'theta', 'a', 'b'}
        :param num_actions: 生成的动作数量
        :return: 动作列表
        """
        actions = []
        
        for _ in range(num_actions):
            # 为每个动作参数添加高斯噪声
            noisy_action = {
                'V0': np.clip(base_action['V0'] + np.random.normal(0, self.action_noise), 0.5, 8.0),
                'phi': (base_action['phi'] + np.random.normal(0, self.action_noise * 10)) % 360,
                'theta': np.clip(base_action['theta'] + np.random.normal(0, self.action_noise * 5), 0, 90),
                'a': np.clip(base_action['a'] + np.random.normal(0, self.action_noise * 0.1), -0.5, 0.5),
                'b': np.clip(base_action['b'] + np.random.normal(0, self.action_noise * 0.1), -0.5, 0.5)
            }
            actions.append(noisy_action)
        
        return actions

    def select(self, node):
        """
        MCTS选择阶段：选择UCT值最大的子节点
        :param node: 当前节点
        :return: 选中的子节点
        """
        while not node.is_terminal() and node.is_fully_expanded():
            # 选择UCT值最大的子节点
            node = max(node.children.values(), key=lambda n: n.uct_value())
        return node

    def expand(self, node):
        """
        MCTS扩展阶段：基于行为网络生成新的子节点
        :param node: 待扩展节点
        :return: 新扩展的节点或原节点（终止节点）
        """
        if node.is_terminal():
            return node
        
        # 如果未初始化未尝试动作，生成基础动作
        if node.untried_actions is None:
            # 编码当前状态
            state_tensor = self.get_state_for_network()
            
            # 使用行为网络预测基础动作
            with torch.no_grad():
                outputs = self.model(state_tensor)
                base_action_raw = outputs['policy_output'][0].cpu().numpy()
                base_action = outputs['mapped_actions'][0].cpu().numpy()
            
            # 转换为动作字典
            base_action_dict = {
                'V0': base_action[0],
                'phi': base_action[1],
                'theta': base_action[2],
                'a': base_action[3],
                'b': base_action[4]
            }
            
            # 生成探索动作
            node.untried_actions = self.generate_actions(
                base_action_dict, 
                self.action_expansion_num
            )
        
        # 选择一个未尝试的动作
        action = node.untried_actions.pop()
        
        # 创建新节点
        child_node = self.create_child_node(node, action)
        
        # 计算先验概率（使用价值网络输出）
        with torch.no_grad():
            state_tensor = self.get_state_for_network()
            value = self.model(state_tensor)['value_output'][0][0].cpu().item()
            child_node.prior = value
        
        node.children[tuple(action.values())] = child_node
        return child_node

    def create_child_node(self, parent_node, action):
        """
        创建子节点，模拟动作执行
        :param parent_node: 父节点
        :param action: 执行的动作
        :return: 子节点
        """
        # 恢复环境快照
        self.restore_env_snapshot(parent_node.env_snapshot)
        
        # 执行动作
        player = parent_node.player
        self.env.curr_player = 0 if player == 'A' else 1
        result = self.env.take_shot(action)
        
        # 检查是否结束
        done, info = self.env.get_done()
        winner = info.get('winner', None) if done else None
        
        # 获取新的观测
        new_observation = self.env.get_observation(player)
        
        # 创建子节点
        child_node = MCTSNode(
            state=new_observation,
            parent=parent_node,
            action=action,
            player=self.env.get_curr_player()
        )
        
        # 保存环境快照
        child_node.env_snapshot = self.save_env_snapshot()
        child_node.done = done
        child_node.winner = winner
        
        # 更新状态历史
        self.state_history.append(self.encode_state(new_observation))
        
        return child_node

    def simulate(self, node):
        """
        MCTS模拟阶段：从节点开始模拟直到终止状态
        :param node: 起始节点
        :return: 价值估计 (1=胜, 0=负, 0.5=平局)
        """
        current_node = node
        
        # 模拟直到终止状态
        while not current_node.is_terminal():
            # 快速模拟：使用行为网络生成动作
            state_tensor = self.get_state_for_network()
            with torch.no_grad():
                outputs = self.model(state_tensor)
                action = outputs['mapped_actions'][0].cpu().numpy()
            
            # 转换为动作字典
            action_dict = {
                'V0': action[0],
                'phi': action[1],
                'theta': action[2],
                'a': action[3],
                'b': action[4]
            }
            
            # 执行动作
            self.restore_env_snapshot(current_node.env_snapshot)
            self.env.curr_player = 0 if current_node.player == 'A' else 1
            result = self.env.take_shot(action_dict)
            
            # 检查结束状态
            done, info = self.env.get_done()
            if done:
                winner = info.get('winner')
                # 计算价值 (当前玩家视角)
                if winner == current_node.player:
                    return 1.0
                elif winner == 'SAME':
                    return 0.5
                else:
                    return 0.0
            
            # 更新状态
            new_observation = self.env.get_observation(current_node.player)
            self.state_history.append(self.encode_state(new_observation))
            
            # 使用价值网络估计价值
            state_tensor = self.get_state_for_network()
            with torch.no_grad():
                value = self.model(state_tensor)['value_output'][0][0].cpu().item()
            
            return value
        
        # 如果已是终止节点，直接返回价值
        if current_node.winner == current_node.player:
            return 1.0
        elif current_node.winner == 'SAME':
            return 0.5
        else:
            return 0.0

    def backpropagate(self, node, value):
        """
        MCTS回溯阶段：更新路径上所有节点的统计信息
        :param node: 终止节点
        :param value: 模拟得到的价值
        """
        while node is not None:
            # 从当前玩家视角更新价值
            if node.player == self.env.get_curr_player():
                node.update(value)
            else:
                node.update(1 - value)  # 对手视角反转价值
            node = node.parent

    def save_env_snapshot(self):
        """保存环境快照"""
        return {
            'balls': save_balls_state(self.env.balls),
            'curr_player': self.env.curr_player,
            'done': self.env.done,
            'winner': self.env.winner,
            'hit_count': self.env.hit_count,
            'last_state': self.env.last_state,
            'player_targets': copy.deepcopy(self.env.player_targets)
        }

    def restore_env_snapshot(self, snapshot):
        """恢复环境快照"""
        if snapshot is None:
            return
        
        self.env.balls = restore_balls_state(snapshot['balls'])
        self.env.curr_player = snapshot['curr_player']
        self.env.done = snapshot['done']
        self.env.winner = snapshot['winner']
        self.env.hit_count = snapshot['hit_count']
        self.env.last_state = snapshot['last_state']
        self.env.player_targets = snapshot['player_targets']

    def search(self, observation, player):
        """
        执行MCTS搜索，返回最优动作
        :param observation: 当前观测 (balls, my_targets, table)
        :param player: 当前玩家 ('A'/'B')
        :return: 最优动作字典
        """
        # 初始化根节点
        root_snapshot = self.save_env_snapshot()
        root_node = MCTSNode(
            state=observation,
            parent=None,
            action=None,
            player=player
        )
        root_node.env_snapshot = root_snapshot
        root_node.untried_actions = None
        
        # 更新状态历史
        self.state_history.append(self.encode_state(observation))
        
        # 执行多次模拟
        for _ in range(self.num_simulations):
            # 1. 选择
            selected_node = self.select(root_node)
            
            # 2. 扩展
            expanded_node = self.expand(selected_node)
            
            # 3. 模拟
            value = self.simulate(expanded_node)
            
            # 4. 回溯
            self.backpropagate(expanded_node, value)
        
        # 选择访问次数最多的动作（也可以选择UCT值最大的）
        best_child = max(root_node.children.values(), key=lambda n: n.visits)
        return best_child.action

    def reset(self):
        """重置MCTS状态"""
        self.state_history = []


# 使用示例
def test_mcts():
    """测试MCTS集成"""
    # 初始化环境和模型
    env = PoolEnv(verbose=False)
    model = DualNetwork()
    
    # 加载预训练权重（如果有）
    # model.load('pretrained_model.pth')
    
    # 初始化MCTS
    mcts = MCTS(model, env)
    
    # 重置环境
    env.reset(target_ball='solid')
    
    # 主游戏循环
    while True:
        # 获取当前玩家和观测
        player = env.get_curr_player()
        observation = env.get_observation(player)
        
        # 使用MCTS搜索最优动作
        print(f"[第{env.hit_count}次击球] Player {player} 正在思考...")
        action = mcts.search(observation, player)
        
        # 执行动作
        print(f"执行动作: V0={action['V0']:.2f}, phi={action['phi']:.2f}, theta={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
        env.take_shot(action)
        
        # 检查游戏结束
        done, info = env.get_done()
        if done:
            print(f"游戏结束！胜者: {info['winner']}, 总击球数: {info['hit_count']}")
            break
        
        # 重置MCTS状态
        mcts.reset()


if __name__ == '__main__':
    test_mcts()