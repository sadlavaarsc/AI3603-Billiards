import numpy as np
import copy
import torch
import poolenv
from data_loader import StatePreprocessor

class MCTSNode:
    def __init__(self, state_seq, parent=None, prior=1.0, phase=None, depth=0):
        """
        state_seq: List[balls_state], len == 3
        phase: 当前游戏阶段 (EARLY, MID, LATE)
        depth: 当前节点深度
        """
        self.state_seq = state_seq
        self.parent = parent
        self.children = {}  # action_key -> MCTSNode
        self.phase = phase  # 新增：游戏阶段
        self.depth = depth  # 新增：节点深度

        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = prior

    def ucb(self, c_puct=1.5, phase_params=None):
        """
        计算UCB值，支持阶段感知调整
        
        参数：
            c_puct: 基础探索参数
            phase_params: 阶段参数字典，包含不同阶段的cpuct调整因子
            
        返回：
            float: UCB值
        """
        if self.N == 0:
            return float("inf")
        
        # 使用阶段调整后的c_puct
        adjusted_c_puct = c_puct
        if phase_params and self.phase:
            adjusted_c_puct = c_puct * phase_params[self.phase]["cpuct"]
        
        return self.Q + adjusted_c_puct * self.P * np.sqrt(self.parent.N) / (1 + self.N)

class MCTS:
    """
    Policy-guided Continuous MCTS for PoolEnv
    
    """
    def __init__(
        self,
        model,
        env,
        n_simulations=5,
        n_action_samples=8,
        c_puct=1.5,
        device="cuda" if torch.cuda.is_available() else "cpu",
        debug=False,
        n_rollouts_per_action=1,  # 每个action的模拟次数
        max_depth=3,  # 最大搜索深度，超过该深度不再衍生子节点
        initial_keep_count=50,  # 初始动作保留数量
        keep_reduction_factor=2  # 每深入一层，保留数量减半的因子
    ):
        self.model = model
        self.env = env
        self.n_simulations = n_simulations
        self.n_action_samples = n_action_samples
        self.c_puct = c_puct
        self.device = device
        self.debug = debug
        self.n_rollouts_per_action = n_rollouts_per_action  # 添加每个action的模拟次数参数

        self.PRIOR_THRESHOLD=0.02
        self.MAX_EXPAND=max_depth*initial_keep_count  # 最大展开节点数，等于最大搜索深度*初始保留数量

        # 添加噪声参数，与BasicAgentPro保持一致
        # 定义噪声水平 (与 poolenv 保持一致或略大)
        self.sim_noise = {
            'V0': 0.1, 'phi': 0.15, 'theta': 0.1, 'a': 0.005, 'b': 0.005
        }

        # ===== 与训练代码完全一致的动作范围 =====
        self.action_min = np.array([0.5, 0.0, 0.0, -0.5, -0.5], dtype=np.float32)
        self.action_max = np.array([8.0, 360.0, 90.0, 0.5, 0.5], dtype=np.float32)
        self.state_preprocessor = StatePreprocessor()
        
        # ===== 新增：动作生成和筛选参数 =====
        self.max_depth = max_depth  # 最大搜索深度
        self.initial_keep_count = initial_keep_count  # 初始动作保留数量
        self.keep_reduction_factor = keep_reduction_factor  # 保留数量减少因子
        
        # ===== 物理校准相关参数 =====
        self.ball_radius = 0.028575  # 台球半径，用于幽灵球计算
        
        # ===== 阶段感知（Game-Phase Aware）相关参数 =====
        # 阶段常量定义
        self.EARLY = "EARLY"
        self.MID = "MID"
        self.LATE = "LATE"
        
        # 阶段参数集中管理
        self.PHASE_PARAMS = {
            self.EARLY: {"value_factor": 1.0, "cpuct": 1.0},  # 开局：降低价值影响，增加探索
            self.MID: {"value_factor": 1.0, "cpuct": 1.0},    # 中盘：正常参数
            self.LATE: {"value_factor": 1.4, "cpuct": 0.7}    # 终盘：放大价值影响，减少探索
        }

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def search(self, state_seq):
        """
        state_seq: List[balls_state], len == 3
        """
        root = MCTSNode(state_seq)
        root_player = self.env.get_curr_player()
        
        # 估算根节点的游戏阶段
        balls_state = poolenv.restore_balls_state(poolenv.save_balls_state(self.env.balls))
        root.phase = self._estimate_phase(balls_state, self.env.player_targets, root_player)

        for _ in range(self.n_simulations):
            node = root
            
            # 1. 保存当前球的状态，用于模拟
            # 只保存必要的球状态，避免deepcopy整个环境
            original_balls = poolenv.save_balls_state(self.env.balls)
            
            # 2. 复制当前环境的引用，而不是整个环境
            # 我们只需要table和player_targets，这些是只读的
            table = self.env.table
            player_targets = self.env.player_targets

            # -------- 1. Selection --------
            while node.children:
                items = list(node.children.items())
                ucbs = np.array(
                    [child.ucb(self.c_puct, self.PHASE_PARAMS) for _, child in items],
                    dtype=np.float32
                )

                # 找到最大 UCB
                max_ucb = np.max(ucbs)

                # 随机选一个 max-UCB 的 child（关键！）
                candidates = [
                    i for i, u in enumerate(ucbs)
                    if np.isclose(u, max_ucb)
                ]
                idx = np.random.choice(candidates)

                action_key, node = items[idx]
                
                # 3. 执行动作，使用simulate_action方法直接模拟，不依赖环境
                sim_balls = poolenv.restore_balls_state(original_balls)
                
                # 3. 执行动作，使用simulate_action方法直接模拟，不依赖环境
                new_balls = self._simulate_action(sim_balls, table, action_key, player_targets, root_player)
                original_balls = poolenv.save_balls_state(new_balls)

            # -------- 2. Expansion + Evaluation --------
            value = self._expand_and_evaluate(node, original_balls, table, player_targets, root_player)

            # -------- 3. Backpropagation --------
            self._backpropagate(node, value)

        # -------- 4. 输出动作 --------
        if not root.children:
            # fallback：直接用策略网络
            with torch.no_grad():
                state_tensor = self._state_seq_to_tensor(state_seq)
                out = self.model(state_tensor.unsqueeze(0).to(self.device))
                action_norm = out["policy_output"][0].cpu().numpy()
                return self._denormalize_action(action_norm)

        best_action_key = max(
            root.children.items(),
            key=lambda item: item[1].N
        )[0]

        return np.array(best_action_key, dtype=np.float32)

    # ------------------------------------------------------------------
    # 核心步骤
    # ------------------------------------------------------------------

    def _expand_and_evaluate(self, node, balls, table, player_targets, root_player):
        
        # 1. 检查游戏是否结束（考虑黑八进袋的合法性）
        done = False
        winner = None
        eight_in_pocket = False
        
        for ball_id, ball in balls.items():
            if ball_id == '8' and ball.state.s == 4:
                eight_in_pocket = True
                break
        
        if eight_in_pocket:
            # 检查黑八进袋是否合法
            my_targets = player_targets[root_player]
            remaining_my_balls = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
            
            # 黑八合法进袋条件：当前玩家已清台（只剩黑八）且黑八是唯一目标球
            is_eight_legal = (len(remaining_my_balls) == 0) and (len(my_targets) == 1 and my_targets[0] == '8')
            
            if is_eight_legal:
                # 合法黑八进袋，当前玩家获胜
                done = True
                winner = root_player
            else:
                # 非法黑八进袋，当前玩家失败
                done = True
                opponent_player = 'B' if root_player == 'A' else 'A'
                winner = opponent_player
        
        if done:
            return 1.0 if winner == root_player else 0.0
        
        # 2. 检查深度限制：如果超过最大深度，不再衍生子节点，直接返回value网络结果
        if node.depth >= self.max_depth:
            # 只进行网络评估，不扩展子节点
            state_tensor = self._state_seq_to_tensor(node.state_seq)
            state_tensor = state_tensor.unsqueeze(0).to(self.device) 
            
            with torch.no_grad():
                out = self.model(state_tensor)
                value = out["value_output"].item()
            
            # 阶段感知价值调整
            phase = self._estimate_phase(balls, player_targets, root_player)
            phase_value_factor = self.PHASE_PARAMS[phase]["value_factor"]
            value = value * phase_value_factor
            
            return value
        
        # 3. 网络评估 - 多次评估取均值
        state_tensor = self._state_seq_to_tensor(node.state_seq)
        state_tensor = state_tensor.unsqueeze(0).to(self.device) 
        
        values = []
        with torch.no_grad():
            for _ in range(self.n_rollouts_per_action):
                out = self.model(state_tensor)
                if _ == 0:  # 只需要一次策略输出
                    action_norm = out["policy_output"][0].cpu().numpy()
                values.append(out["value_output"].item())
        
        # 取多次评估的均值作为最终value
        value = np.mean(values)
        
        # 4. 估算当前游戏阶段
        phase = self._estimate_phase(balls, player_targets, root_player)
        
        # 5. 阶段感知价值调整 - 根据游戏阶段调整value的影响
        phase_value_factor = self.PHASE_PARAMS[phase]["value_factor"]
        value = value * phase_value_factor
        
        # 5. 终局启发式价值兜底 - 当只剩黑八时
        my_targets = player_targets[root_player]
        remaining_targets = 0
        for ball_id in my_targets:
            if ball_id in balls and balls[ball_id].state.s != 4:
                remaining_targets += 1
        
        # 只剩黑八，且黑八未进袋
        if remaining_targets == 0 and '8' in balls and balls['8'].state.s != 4:
            # 终局阶段，黑八在台面上，给一个较高的启发式价值
            value = max(value, 0.8)  # 确保终局有足够的价值激励
        
        # 6. 明显必败局面（如只剩对方球）
        opponent_player = 'B' if root_player == 'A' else 'A'
        opponent_targets = player_targets[opponent_player]
        opponent_remaining = 0
        for ball_id in opponent_targets:
            if ball_id in balls and balls[ball_id].state.s != 4:
                opponent_remaining += 1
        
        if remaining_targets == 0 and opponent_remaining > 0:
            # 我方目标球已清，但对方还有球，且黑八未进袋（不可能发生，作为兜底）
            value = min(value, 0.2)

        # 3. 反归一化得到模型输出的原始动作
        model_action = self._denormalize_action(action_norm)
        
        # 4. 基于Agent_pro原理生成候选动作并打分排序
        pro_actions = self.generate_pro_actions(balls, player_targets[root_player], table)
        
        # 如果没有生成pro动作，使用模型动作作为备选
        if not pro_actions:
            pro_actions.append((model_action, 1.0))
        
        # 5. 动作筛选：保留与模型动作相差较小的动作
        # 计算每个pro动作与模型动作的距离
        filtered_actions = []
        for action, score in pro_actions:
            # 计算动作距离（考虑角度和力度的加权距离）
            # 角度差（0-180度）
            phi_diff = abs(action[1] - model_action[1])
            if phi_diff > 180:
                phi_diff = 360 - phi_diff
            
            # 力度差
            v0_diff = abs(action[0] - model_action[0])
            
            # 综合距离（角度差权重0.7，力度差权重0.3）
            distance = (phi_diff / 180.0) * 0.7 + (v0_diff / (8.0 - 0.5)) * 0.3
            
            filtered_actions.append((action, score, distance))
        
        # 按距离排序，保留与模型动作相差较小的动作
        filtered_actions.sort(key=lambda x: x[2])
        
        # 根据当前深度确定保留数量
        current_depth = node.depth
        # 初始保留数量，每深入一层，保留数量减半
        keep_count = max(1, self.initial_keep_count // (self.keep_reduction_factor ** current_depth))
        
        # 只保留指定数量的动作
        selected_actions = filtered_actions[:keep_count]
        
        # 6. 为每个选中的动作生成变种（角度微调 + 力度档位）
        sampled_actions = []
        for base_action, base_score, _ in selected_actions:
            # 采样参数配置
            angle_offsets = [-0.5, 0, 0.5]  # 角度微调范围（度）
            
            # 力度档位配置
            v_base = base_action[0]  # 基础力度
            v_offsets = [-1.0, 0.0, 1.0]  # 力度偏移
            
            # 生成角度和力度的组合样本
            for angle_offset in angle_offsets:
                for v_offset in v_offsets:
                    # 计算新的角度（0-360度）
                    new_phi = (base_action[1] + angle_offset) % 360
                    
                    # 计算新的力度（限制在0.5-8.0范围内）
                    new_v0 = np.clip(v_base + v_offset, 0.5, 8.0)
                    
                    # 创建新动作，保持其他参数不变
                    new_action = base_action.copy()
                    new_action[0] = new_v0  # 力度
                    new_action[1] = new_phi  # 角度
                    
                    sampled_actions.append(new_action)
        
        # 如果样本数量不足，确保至少有一个样本
        if not sampled_actions:
            sampled_actions.append(model_action.copy())
        
        # 调试信息：只在根节点且是第一次模拟时打印
        # 通过检查节点深度和子节点数量来判断是否是第一次模拟
        if node.parent is None and self.debug and len(node.children) == 0 and False:
            print("policy_norm:", action_norm)  # 调试信息
            print("model_action:", model_action)  # 调试信息
            print("value:", value)         # 调试信息
            print(f"Generated {len(pro_actions)} pro actions, kept {keep_count} after filtering")  # 调试信息
        
        # 5. 基于相似度构造先验
        distances = np.array(
            [np.linalg.norm(a - model_action) for a in sampled_actions],
            dtype=np.float32
        )
        similarities = 1.0 / (1.0 + distances)
        priors = similarities / np.sum(similarities)

        if node.parent is None:
            noise = np.random.dirichlet([0.3] * len(priors))
            priors = 0.75 * priors + 0.25 * noise

        # Top-k
        pairs = sorted(
            zip(sampled_actions, priors),
            key=lambda x: x[1],
            reverse=True
        )

        # 6. 创建子节点
        for i, (action, prior) in enumerate(pairs):
            if i >= self.MAX_EXPAND:
                break
            if prior < self.PRIOR_THRESHOLD:
                continue

            action_key = self._action_to_key(action)
            if action_key in node.children:
                continue

            # 对每个action进行n_rollouts_per_action次模拟，取均值
            state_vectors = []
            
            for _ in range(self.n_rollouts_per_action):
                # 使用_simulate_action直接模拟动作，不依赖env对象
                sim_balls = poolenv.restore_balls_state(balls)
                new_balls = self._simulate_action(sim_balls, table, action_key, player_targets, root_player)
                
                # 生成新的状态向量
                new_balls_state = poolenv.save_balls_state(new_balls)
                new_state_vec = self._balls_state_to_81(
                            new_balls_state,
                            my_targets=None,
                            table=None
                        )
                state_vectors.append(new_state_vec)
            
            # 取多次模拟的平均状态向量
            avg_state_vec = np.mean(state_vectors, axis=0)
            new_state_seq = node.state_seq[1:] + [avg_state_vec]
            
            # 创建子节点，传递当前阶段和深度
            node.children[action_key] = MCTSNode(
                state_seq=new_state_seq,
                parent=node,
                prior=prior,
                phase=phase,
                depth=current_depth + 1  # 子节点深度+1
            )
            
        return value

    def _backpropagate(self, node, value):
        sign = 1.0
        while node is not None:
            node.N += 1
            node.W += sign * value
            node.Q = node.W / node.N
            sign *= -1.0
            node = node.parent

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------

    def _denormalize_action(self, action_norm):
        """
        将 [0,1] 策略输出映射回真实物理动作
        """
        action_norm = np.clip(action_norm, 0.0, 1.0)
        return action_norm * (self.action_max - self.action_min) + self.action_min

    @staticmethod
    def _balls_state_to_81(balls_state, my_targets, table):
        """
        balls_state: dict from poolenv.save_balls_state
        return: np.ndarray shape (81,)
        """

        state = np.zeros(81, dtype=np.float32)

        ball_order = [
            'cue', '1', '2', '3', '4', '5', '6', '7', '8',
            '9', '10', '11', '12', '13', '14', '15'
        ]

        # 使用与process_raw_match_data.py和data_loader.py一致的标准尺寸
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
                    # 球坐标使用标准尺寸进行归一化，与data_loader.py保持一致
                    state[base + 0] = pos[0] / STANDARD_TABLE_WIDTH
                    state[base + 1] = pos[1] / STANDARD_TABLE_LENGTH
                    state[base + 2] = pos[2] / (2 * BALL_RADIUS)
                    state[base + 3] = 0.0
            else:
                state[base + 0] = -1.0
                state[base + 1] = -1.0
                state[base + 2] = -1.0
                state[base + 3] = 1.0

        # 球桌尺寸（64-65维）：与process_raw_match_data.py保持一致
        # 直接使用实际尺寸，不进行归一化，由后续的StatePreprocessor处理
        if table is not None:
            state[64] = table.w
            state[65] = table.l
        else:
            # 如果没有table对象，使用process_raw_match_data.py中的固定值
            state[64] = 2.540
            state[65] = 1.270

        # 目标球 one-hot
        if my_targets:
            for t in my_targets:
                idx = int(t) - 1
                state[66 + idx] = 1.0

        return state

    def _state_seq_to_tensor(self, state_seq):
        if len(state_seq) != 3:
            raise ValueError(f"state_seq length must be 3, got {len(state_seq)}")

        for i, s in enumerate(state_seq):
            if not isinstance(s, np.ndarray) or s.shape != (81,):
                raise TypeError(
                    f"state_seq[{i}] must be np.ndarray(81), got {type(s)} {getattr(s, 'shape', None)}"
                )

        states = np.stack(state_seq, axis=0)   # ← 关键
        states = self.state_preprocessor(states)
        return torch.from_numpy(states).float()

    def _estimate_phase(self, balls_state, player_targets, current_player):
        """
        估算当前游戏阶段
        
        参数：
            balls_state: 球状态字典
            player_targets: 玩家目标球字典
            current_player: 当前玩家
            
        返回：
            str: 游戏阶段 (EARLY, MID, LATE)
        """
        # 计算当前玩家剩余的目标球数
        my_targets = player_targets[current_player]
        remaining_balls = 0
        
        for ball_id in my_targets:
            if ball_id in balls_state and balls_state[ball_id].state.s != 4:  # 4表示已进袋
                remaining_balls += 1
        
        # 检查是否只剩下黑八
        if remaining_balls == 0 and '8' in balls_state and balls_state['8'].state.s != 4:
            return self.LATE
        
        # 根据剩余球数判断阶段
        if remaining_balls > 8:
            return self.EARLY
        elif remaining_balls <= 3:
            return self.LATE
        else:
            return self.MID
    
    def generate_pro_actions(self, balls, my_targets, table):
        """
        基于Agent_pro原理生成候选动作并打分排序
        
        参数：
            balls: 当前球状态字典
            my_targets: 当前玩家的目标球列表
            table: 球桌对象
            
        返回：
            list: 排序后的候选动作列表，每个元素为 (action, score)
        """
        actions = []
        
        # 获取白球位置
        cue_ball = balls.get('cue')
        if not cue_ball:
            return []
        cue_pos = cue_ball.state.rvw[0]
        
        # 获取所有目标球的ID
        target_ids = [bid for bid in my_targets if balls[bid].state.s != 4]
        
        # 如果没有目标球了（理论上外部会处理转为8号，这里兜底）
        if not target_ids:
            target_ids = ['8']
        
        # 遍历每一个目标球
        for tid in target_ids:
            obj_ball = balls[tid]
            obj_pos = obj_ball.state.rvw[0]
            
            # 遍历每一个袋口
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center
                
                # 计算理论进球角度和距离
                phi_ideal, dist = self._get_ghost_ball_target(cue_pos, obj_pos, pocket_pos)
                
                if dist > 0:  # 有效的幽灵球位置
                    # 检查是否合法击打黑八
                    remaining_targets = 0
                    for target_id in my_targets:
                        if target_id in balls and balls[target_id].state.s != 4:
                            remaining_targets += 1
                    
                    # 如果目标是黑八且仍有其他目标球未进袋，跳过
                    if tid == '8' and remaining_targets > 0:
                        continue
                    
                    # 根据距离估算力度 (参考basic_agent_pro.py)
                    v_base = 1.5 + dist * 1.5
                    v_base = np.clip(v_base, 1.0, 6.0)
                    
                    # 生成几个变种动作加入候选池
                    # 变种1：精准一击
                    action1 = np.array([v_base, phi_ideal, 0, 0, 0], dtype=np.float32)
                    actions.append((action1, 1.0))  # 基础分数
                    
                    # 变种2：力度稍大
                    action2 = np.array([min(v_base + 1.5, 7.5), phi_ideal, 0, 0, 0], dtype=np.float32)
                    actions.append((action2, 0.9))  # 稍低分数
                    
                    # 变种3：角度微调 (左右偏移 0.5 度)
                    action3 = np.array([v_base, (phi_ideal + 0.5) % 360, 0, 0, 0], dtype=np.float32)
                    actions.append((action3, 0.85))  # 稍低分数
                    
                    action4 = np.array([v_base, (phi_ideal - 0.5) % 360, 0, 0, 0], dtype=np.float32)
                    actions.append((action4, 0.85))  # 稍低分数
        
        # 按分数降序排序
        actions.sort(key=lambda x: x[1], reverse=True)
        
        return actions
    
    def _calc_angle_degrees(self, v):
        """
        计算向量的角度（度）
        
        参数：
            v: 二维向量
            
        返回：
            float: 角度（0-360度）
        """
        import math
        angle = math.degrees(math.atan2(v[1], v[0]))
        return angle % 360
    
    def _get_ghost_ball_target(self, cue_pos, obj_pos, pocket_pos):
        """
        计算幽灵球目标位置和角度
        
        参数：
            cue_pos: 白球位置
            obj_pos: 目标球位置
            pocket_pos: 袋口位置
            
        返回：
            tuple: (phi角度, 白球到幽灵球的距离)
        """
        import numpy as np
        vec_obj_to_pocket = np.array(pocket_pos) - np.array(obj_pos)
        dist_obj_to_pocket = np.linalg.norm(vec_obj_to_pocket)
        if dist_obj_to_pocket == 0: return 0, 0
        unit_vec = vec_obj_to_pocket / dist_obj_to_pocket
        ghost_pos = np.array(obj_pos) - unit_vec * (2 * self.ball_radius)
        vec_cue_to_ghost = ghost_pos - np.array(cue_pos)
        dist_cue_to_ghost = np.linalg.norm(vec_cue_to_ghost)
        phi = self._calc_angle_degrees(vec_cue_to_ghost)
        return phi, dist_cue_to_ghost
    
    def _check_foul(self, shot, last_state, player_targets, current_player):
        """
        检查击球是否犯规（完全对齐台球规则）
        
        参数：
            shot: 已完成物理模拟的System对象
            last_state: 击球前的球状态字典
            player_targets: 玩家目标球字典，{player: [target_ball_ids]}
            current_player: 当前击球玩家，'A'或'B'
            
        返回：
            bool: True表示犯规，False表示合法
            
        规则核心：
            - 清台前：player_targets = ['1'-'7'] 或 ['9'-'15']，黑8不属于任何人
            - 清台后：player_targets = ['8']，黑8成为唯一目标球
        """
        # 1. 基本分析
        new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
        cue_pocketed = "cue" in new_pocketed
        eight_pocketed = "8" in new_pocketed
        
        # 2. 玩家目标球分析
        my_targets = player_targets[current_player]
        my_remaining = [bid for bid in my_targets if last_state[bid].state.s != 4]
        
        # 3. 首球碰撞分析
        first_contact_ball_id = None
        foul_first_hit = False
        valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
        
        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
                other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
                if other_ids:
                    first_contact_ball_id = other_ids[0]
                    break
        
        # 首球碰撞规则：
        # a. 未击中任何球（特殊情况：只剩白球和黑8且已清台，不算犯规）
        if first_contact_ball_id is None:
            # 检查是否只剩白球和黑8且已清台
            remaining_balls = [bid for bid, b in last_state.items() if b.state.s != 4 and bid != 'cue']
            is_only_eight_left = len(remaining_balls) == 1 and remaining_balls[0] == '8' and len(my_targets) == 1 and my_targets[0] == '8'
            if not is_only_eight_left:
                foul_first_hit = True
        else:
            # b. 首次击打的球必须是目标球
            if first_contact_ball_id not in my_targets:
                # 特殊情况：清台后（my_targets=['8']），只能打黑8
                if len(my_targets) == 1 and my_targets[0] == '8':
                    if first_contact_ball_id != '8':
                        foul_first_hit = True
                # 特殊情况：清台前，不能打黑8
                elif first_contact_ball_id == '8':
                    foul_first_hit = True
                # 其他情况：不能打对手球
                else:
                    foul_first_hit = True
        
        # 4. 碰库分析
        cue_hit_cushion = False
        target_hit_cushion = False
        foul_no_rail = False
        
        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if 'cushion' in et:
                if 'cue' in ids:
                    cue_hit_cushion = True
                if first_contact_ball_id is not None and first_contact_ball_id in ids:
                    target_hit_cushion = True
        
        # 碰库规则：无进球且首球和白球都未碰库，犯规
        if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
            foul_no_rail = True
        
        # 5. 进球规则分析
        foul_pocket = False
        
        # a. 白球进袋
        if cue_pocketed:
            foul_pocket = True
        
        # b. 非法黑8（清台前打黑8）
        if eight_pocketed:
            is_targeting_eight_ball_legally = (len(my_targets) == 1 and my_targets[0] == "8")
            if not is_targeting_eight_ball_legally:
                foul_pocket = True
        
        # 6. 综合犯规判断
        if foul_pocket or foul_first_hit or foul_no_rail:
            return True
        
        return False

    def _simulate_action(self, balls, table, action_key, player_targets, current_player):
        """
        直接模拟击球动作，添加高斯噪声和完整的犯规检查
        
        参数：
            balls: 球状态字典，{ball_id: Ball对象}
            table: 球桌对象
            action_key: 动作元组，(V0, phi, theta, a, b)
            player_targets: 玩家目标球字典，{player: [target_ball_ids]}
            current_player: 当前击球玩家，'A'或'B'
        
        返回：
            模拟后的球状态字典
        """
        import pooltool as pt
        
        # 将动作元组转换为字典
        action = np.array(action_key, dtype=np.float64)
        action_dict = {
            "V0": action[0],
            "phi": action[1],
            "theta": action[2],
            "a": action[3],
            "b": action[4]
        }
        
        # 创建shot对象并模拟
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=table, balls=balls, cue=cue)
        
        # 保存原始状态，用于犯规检查
        last_state = {bid: ball for bid, ball in balls.items()}
        
        # 添加高斯噪声，与BasicAgentPro保持一致
        noisy_V0 = np.clip(action_dict["V0"] + np.random.normal(0, self.sim_noise['V0']), 0.5, 8.0)
        noisy_phi = (action_dict["phi"] + np.random.normal(0, self.sim_noise['phi'])) % 360
        noisy_theta = np.clip(action_dict["theta"] + np.random.normal(0, self.sim_noise['theta']), 0, 90)
        noisy_a = np.clip(action_dict["a"] + np.random.normal(0, self.sim_noise['a']), -0.5, 0.5)
        noisy_b = np.clip(action_dict["b"] + np.random.normal(0, self.sim_noise['b']), -0.5, 0.5)
        
        # 设置球杆状态，使用带噪声的动作
        cue.set_state(
            V0=noisy_V0, 
            phi=noisy_phi, 
            theta=noisy_theta, 
            a=noisy_a, 
            b=noisy_b
        )
        
        # 执行模拟
        try:
            pt.simulate(shot, inplace=True)
        except Exception as e:
            # 模拟失败时返回原始状态
            return balls
        
        # 检查是否犯规
        if self._check_foul(shot, last_state, player_targets, current_player):
            return balls
        
        return shot.balls

    def _action_to_key(self, action):
        return tuple(np.round(action, 4).tolist())

    def _apply_action(self, env, action_key):
        """
        兼容旧接口，用于_expand_and_evaluate方法
        """
        action = np.array(action_key, dtype=np.float64)
        env.take_shot({
            "V0": action[0],
            "phi": action[1],
            "theta": action[2],
            "a": action[3],
            "b": action[4]
        })
