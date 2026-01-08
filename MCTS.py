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
        self.sigma_norm = sigma_norm or np.array(
            [0.08, 0.08, 0.08, 0.05, 0.05],
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
    def __init__(self, state_seq, parent=None, prior=1.0):
        """
        state_seq: List[balls_state], len == 3
        """
        self.state_seq = state_seq
        self.parent = parent
        self.children = {}  # action_key -> MCTSNode

        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = prior

    def ucb(self, c_puct=1.5):
        if self.N == 0:
            return float("inf")
        return self.Q + c_puct * self.P * np.sqrt(self.parent.N) / (1 + self.N)


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
        n_rollouts_per_action=3  # 每个action的模拟次数，默认3次
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

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def search(self, state_seq):
        """
        state_seq: List[balls_state], len == 3
        """
        root = MCTSNode(state_seq)
        root_player = self.env.get_curr_player()

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
                    [child.ucb(self.c_puct) for _, child in items],
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
                
                # 使用save/restore机制替代deepcopy，优化性能
                sim_balls = poolenv.restore_balls_state(original_balls)
                
                # 3. 执行动作，使用simulate_action方法直接模拟，不依赖环境
                new_balls = self._simulate_action(sim_balls, table, action_key)
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
        
        # 2. 网络评估 - 多次评估取均值
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

        # 3. 反归一化得到真实物理动作
        base_action = self._denormalize_action(action_norm)

        if node.parent is None and self.debug:
            print("policy_norm:", action_norm)  # 调试信息
            print("value:", value)         # 调试信息
            print("base_action:", base_action)  # 调试信息
        
        # 4. 连续动作采样
        sampled_action_norms = self.sampler.sample(
            action_norm,
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
                new_balls = self._simulate_action(sim_balls, table, action_key)
                
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
            
            # 创建子节点
            node.children[action_key] = MCTSNode(
                state_seq=new_state_seq,
                parent=node,
                prior=prior
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



    def _simulate_action(self, balls, table, action_key):
        """
        直接模拟击球动作，添加高斯噪声使模拟更真实
        
        参数：
            balls: 球状态字典，{ball_id: Ball对象}
            table: 球桌对象
            action_key: 动作元组，(V0, phi, theta, a, b)
        
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
