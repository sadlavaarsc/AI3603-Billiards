import numpy as np
import copy
import torch
import poolenv
from data_loader import StatePreprocessor


class ActionSampler:
    """
    在【归一化动作空间 [0,1]】附近采样
    """
    def __init__(self, sigma_norm=None):
        # 归一化空间噪声（经验值）
        # 增大力度维度(V0)的噪声标准差，从0.08增加到0.2，改善力度控制
        self.sigma_norm = sigma_norm or np.array(
            [0.2, 0.08, 0.08, 0.05, 0.05],
            dtype=np.float32
        )

    def sample(self, base_action_norm, n_samples = 8):
        actions = []
        for _ in range(n_samples):
            noise = np.random.normal(0, self.sigma_norm)
            a_norm = np.clip(base_action_norm + noise, 0.0, 1.0)
            actions.append(a_norm)
        return actions


class MCTSNode:
    def __init__(self, state_seq, parent=None, prior=1.0, phase=None):
        """
        state_seq: List[balls_state], len == 3
        phase: 当前游戏阶段 (EARLY, MID, LATE)
        """
        self.state_seq = state_seq
        self.parent = parent
        self.children = {}  # action_key -> MCTSNode
        self.phase = phase  # 新增：游戏阶段

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
        n_rollouts_per_action=1  # 每个action的模拟次数，默认3次
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
        self.MAX_EXPAND=8

        # 添加噪声参数，与BasicAgentPro保持一致
        # 定义噪声水平 (与 poolenv 保持一致或略大)
        self.sim_noise = {
            'V0': 0.1, 'phi': 0.15, 'theta': 0.1, 'a': 0.005, 'b': 0.005
        }

        self.sampler = ActionSampler()

        # ===== 与训练代码完全一致的动作范围 =====
        self.action_min = np.array([0.5, 0.0, 0.0, -0.5, -0.5], dtype=np.float32)
        self.action_max = np.array([8.0, 360.0, 90.0, 0.5, 0.5], dtype=np.float32)
        self.state_preprocessor = StatePreprocessor()
        
        # ===== 物理校准相关参数 =====
        self.ball_radius = 0.028575  # 台球半径，用于幽灵球计算
        
        # ===== 阶段感知（Game-Phase Aware）相关参数 =====
        # 阶段常量定义
        self.EARLY = "EARLY"
        self.MID = "MID"
        self.LATE = "LATE"
        
        # 阶段参数集中管理
        self.PHASE_PARAMS = {
            self.EARLY: {"value_factor": 0.7, "cpuct": 1.2},  # 开局：降低价值影响，增加探索
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
        
        # 1. 检查游戏是否结束（简化版，只检查黑八是否进袋）
        done = False
        winner = None
        for ball_id, ball in balls.items():
            if ball_id == '8' and ball.state.s == 4:
                done = True
                winner = root_player
                break
        
        if done:
            return 1.0 if winner == root_player else 0.0
        
        # 2. 估算当前游戏阶段
        phase = self._estimate_phase(balls, player_targets, root_player)
        
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
        
        # 4. 阶段感知价值调整 - 根据游戏阶段调整value的影响
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

        # 3. 反归一化得到真实物理动作
        base_action = self._denormalize_action(action_norm)
        
        # 4. 校准策略动作，确保能打中球
        my_targets = player_targets[root_player]
        calibrated_action = self._calibrate_policy_action(base_action, balls, my_targets, table)
        
        # 5. 重新归一化校准后的动作，用于采样
        # 将校准后的动作转回归一化空间
        calibrated_action_norm = (calibrated_action - self.action_min) / (self.action_max - self.action_min)
        calibrated_action_norm = np.clip(calibrated_action_norm, 0.0, 1.0)

        if node.parent is None and self.debug:
            print("policy_norm:", action_norm)  # 调试信息
            print("calibrated_policy_norm:", calibrated_action_norm)  # 调试信息
            print("value:", value)         # 调试信息
            print("base_action:", base_action)  # 调试信息
            print("calibrated_action:", calibrated_action)  # 调试信息
        
        # 6. 连续动作采样 - 使用校准后的动作进行采样
        sampled_action_norms = self.sampler.sample(
            calibrated_action_norm,
            self.n_action_samples
        )

        sampled_actions = [
            self._denormalize_action(a_norm)
            for a_norm in sampled_action_norms
        ]
        
        # 5. 基于相似度构造先验
        distances = np.array(
            [np.linalg.norm(a - base_action) for a in sampled_actions],
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
            
            # 创建子节点，传递当前阶段
            node.children[action_key] = MCTSNode(
                state_seq=new_state_seq,
                parent=node,
                prior=prior,
                phase=phase
            )
        
        # ===== 检查并删除无效子节点 =====
        # 保存原始状态向量，用于比较
        original_state_vec = node.state_seq[-1]
        
        # 找出无效子节点（状态没有变化，说明犯规了）
        invalid_keys = []
        for action_key, child in node.children.items():
            child_state_vec = child.state_seq[-1]
            # 如果状态向量几乎没有变化，说明该动作导致了犯规（状态被恢复）
            if np.allclose(child_state_vec, original_state_vec, atol=1e-6):
                invalid_keys.append(action_key)
        
        # 删除无效子节点
        for key in invalid_keys:
            del node.children[key]
        
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
    
    def _calibrate_policy_action(self, action, balls, my_targets, table):
        """
        校准策略网络输出的动作，确保能打中球
        
        参数：
            action: 策略网络输出的动作（归一化前的真实物理动作）
            balls: 当前球状态字典
            my_targets: 当前玩家的目标球列表
            table: 球桌对象
            
        返回：
            np.array: 校准后的动作
        """
        import numpy as np
        
        # 获取白球位置
        cue_ball = balls.get('cue')
        if not cue_ball:
            return action  # 没有白球，无法校准，返回原始动作
        cue_pos = cue_ball.state.rvw[0]
        
        # 获取所有目标球的ID
        target_ids = [bid for bid in my_targets if balls[bid].state.s != 4]
        
        # 如果没有目标球了（理论上外部会处理转为8号，这里兜底）
        if not target_ids:
            target_ids = ['8']
        
        # 获取所有袋口位置
        pocket_positions = [pocket.center for pocket in table.pockets.values()]
        
        # 计算所有可能的幽灵球目标角度
        all_ghost_angles = []
        for tid in target_ids:
            obj_ball = balls[tid]
            obj_pos = obj_ball.state.rvw[0]
            
            for pocket_pos in pocket_positions:
                phi_ideal, dist = self._get_ghost_ball_target(cue_pos, obj_pos, pocket_pos)
                if dist > 0:  # 有效的幽灵球位置
                    all_ghost_angles.append(phi_ideal)
        
        if not all_ghost_angles:
            return action  # 没有有效的幽灵球位置，返回原始动作
        
        # 计算原始动作的角度
        original_phi = action[1]  # action的第二个元素是phi角度
        
        # 找到与原始角度最接近的幽灵球角度
        min_diff = float('inf')
        best_phi = original_phi
        
        for ghost_phi in all_ghost_angles:
            # 计算角度差（考虑360度循环）
            diff = abs(ghost_phi - original_phi)
            if diff > 180:
                diff = 360 - diff
            
            if diff < min_diff:
                min_diff = diff
                best_phi = ghost_phi
        
        # 校准动作的角度为最接近的幽灵球角度
        calibrated_action = action.copy()
        calibrated_action[1] = best_phi
        
        return calibrated_action



    def _check_foul(self, shot, last_state, player_targets, current_player):
        """
        检查击球是否犯规
        
        参数：
            shot: 已完成物理模拟的System对象
            last_state: 击球前的球状态字典
            player_targets: 玩家目标球字典，{player: [target_ball_ids]}
            current_player: 当前击球玩家，'A'或'B'
            
        返回：
            bool: True表示犯规，False表示合法
        """
        # 1. 基本分析
        new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
        cue_pocketed = "cue" in new_pocketed
        eight_pocketed = "8" in new_pocketed
        
        # 2. 首球碰撞分析
        first_contact_ball_id = None
        valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
        
        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
                other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
                if other_ids:
                    first_contact_ball_id = other_ids[0]
                    break
        
        # 3. 碰库分析
        cue_hit_cushion = False
        target_hit_cushion = False
        
        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if 'cushion' in et:
                if 'cue' in ids:
                    cue_hit_cushion = True
                if first_contact_ball_id is not None and first_contact_ball_id in ids:
                    target_hit_cushion = True
        
        # 4. 玩家目标球分析
        my_targets = player_targets[current_player]
        my_remaining = [bid for bid in my_targets if last_state[bid].state.s != 4]
        
        # 对手球分析
        opponent_player = 'B' if current_player == 'A' else 'A'
        opponent_targets = player_targets[opponent_player]
        
        # 定义对手球和黑8的集合（当有目标球剩余时）
        opponent_plus_eight = set(opponent_targets) | {'8'}
        
        # 5. 犯规判断
        # 白球进袋
        if cue_pocketed:
            return True
        
        # 白球和黑八同时进袋
        if cue_pocketed and eight_pocketed:
            return True
        
        # 未击中任何球
        if first_contact_ball_id is None:
            return True
        
        # 无进球且无球碰库
        if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
            return True
        
        # 新增：当有目标球剩余时，首次碰撞黑八（误打黑八）
        if len(my_remaining) > 0 and first_contact_ball_id == '8':
            return True
        
        # 新增：当有目标球剩余时，首次碰撞对手球
        if len(my_remaining) > 0 and first_contact_ball_id in opponent_targets:
            return True
        
        # 新增：当只剩黑八时，首次碰撞非黑八球
        if len(my_remaining) == 0 and first_contact_ball_id != '8':
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
