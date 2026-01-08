import numpy as np
import copy
import torch
import poolenv
from data_loader import StatePreprocessor


class ActionSampler:
    """
    在策略网络给出的最优物理动作附近进行连续采样
    """
    def __init__(self, sigma=None):
        self.sigma = sigma or np.array([0.3, 15.0, 10.0, 0.05, 0.05], dtype=np.float32)

        # 真实物理动作空间（必须与训练时 action_min / action_max 一致）
        self.low = np.array([0.5, 0.0, 0.0, -0.5, -0.5], dtype=np.float32)
        self.high = np.array([8.0, 360.0, 90.0, 0.5, 0.5], dtype=np.float32)

    def sample(self, base_action, n_samples):
        actions = []
        for _ in range(n_samples):
            noise = np.random.normal(0, self.sigma)
            a = np.clip(base_action + noise, self.low, self.high)
            actions.append(a)
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
        n_simulations=50,
        n_action_samples=8,
        c_puct=1.5,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.env = env
        self.n_simulations = n_simulations
        self.n_action_samples = n_action_samples
        self.c_puct = c_puct
        self.device = device

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
            env_copy = copy.deepcopy(self.env)

            # -------- 1. Selection --------
            while node.children:
                action_key, node = max(
                    node.children.items(),
                    key=lambda item: item[1].ucb(self.c_puct)
                )
                self._apply_action(env_copy, action_key)

            # -------- 2. Expansion + Evaluation --------
            value = self._expand_and_evaluate(node, env_copy, root_player)

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

    def _expand_and_evaluate(self, node, env, root_player):
        done, info = env.get_done()
        if done:
            winner = info.get("winner", None)
            return 1.0 if winner == root_player else 0.0
        
        state_tensor = self._state_seq_to_tensor(node.state_seq)
        state_tensor = state_tensor.unsqueeze(0).to(self.device) 
        # -------- 网络评估 --------
        with torch.no_grad():
            out = self.model(state_tensor)
            action_norm = out["mapped_actions"][0].cpu().numpy()
            value = out["value_output"].item()

        # 反归一化得到真实物理动作
        base_action = self._denormalize_action(action_norm)

        # -------- 连续动作采样 --------
        sampled_actions = self.sampler.sample(base_action, self.n_action_samples)

        # 基于相似度构造先验
        distances = np.array(
            [np.linalg.norm(a - base_action) for a in sampled_actions],
            dtype=np.float32
        )
        similarities = 1.0 / (1.0 + distances)
        priors = similarities / np.sum(similarities)

        # -------- 创建子节点 --------
        for action, prior in zip(sampled_actions, priors):
            action_key = self._action_to_key(action)
            if action_key in node.children:
                continue

            saved_state = poolenv.save_balls_state(env.balls)
            self._apply_action(env, action_key)

            new_balls = poolenv.save_balls_state(env.balls)
            new_state_seq = node.state_seq[1:] + [new_balls]

            node.children[action_key] = MCTSNode(
                state_seq=new_state_seq,
                parent=node,
                prior=prior
            )

            env.balls = poolenv.restore_balls_state(saved_state)

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

    def _balls_state_to_81(self, balls_state, my_targets=None, table=None):
        """
        balls_state: dict from poolenv.save_balls_state
        return: np.ndarray shape (81,)
        """

        state = np.zeros(81, dtype=np.float32)

        ball_order = [
            'cue', '1', '2', '3', '4', '5', '6', '7', '8',
            '9', '10', '11', '12', '13', '14', '15'
        ]

        for i, ball_id in enumerate(ball_order):
            base = i * 4
            if ball_id in balls_state:
                ball = balls_state[ball_id]
                rvw = ball.state.rvw  # [pos, vel, spin]
                pos = rvw[0]          # (x, y, z)

                state[base + 0] = pos[0]
                state[base + 1] = pos[1]
                state[base + 2] = pos[2]
                state[base + 3] = 1.0 if ball.state.s == 4 else 0.0
            else:
                # 不存在的球，视为进袋
                state[base + 3] = 1.0

        # 桌子尺寸
        if table is not None:
            state[64] = table.width
            state[65] = table.length

        # 目标球 one-hot
        if my_targets is not None:
            for t in my_targets:
                idx = int(t) - 1
                state[66 + idx] = 1.0

        return state

    def _state_seq_to_tensor(self, state_seq):
        """
        state_seq: List[balls_state_dict], len=3
        return: torch.FloatTensor [3, 81]
        """

        encoded_states = []

        for balls_state in state_seq:
            encoded = self._balls_state_to_81(balls_state)
            encoded_states.append(encoded)

        states = np.stack(encoded_states, axis=0)  # (3, 81)

        # 与训练完全一致的归一化
        states = self.state_preprocessor(states)

        return torch.from_numpy(states).float()


    def _action_to_key(self, action):
        return tuple(np.round(action, 4).tolist())

    def _apply_action(self, env, action_key):
        action = np.array(action_key, dtype=np.float64)
        env.take_shot({
            "V0": action[0],
            "phi": action[1],
            "theta": action[2],
            "a": action[3],
            "b": action[4]
        })
