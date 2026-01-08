import numpy as np
import copy
import torch
import poolenv


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
        device="cpu"
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

        # -------- 网络评估 --------
        with torch.no_grad():
            state_tensor = self._state_seq_to_tensor(node.state_seq)
            out = self.model(state_tensor.unsqueeze(0).to(self.device))
            action_norm = out["policy_output"][0].cpu().numpy()
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

    def _state_seq_to_tensor(self, state_seq):
        tensors = [self._balls_state_to_tensor(s) for s in state_seq]
        return torch.cat(tensors, dim=0)

    def _balls_state_to_tensor(self, state):
        features = []
        ball_ids = ['cue'] + [str(i) for i in range(1, 16)]

        for bid in ball_ids:
            if bid in state:
                ball = state[bid]
                pos, vel, spin = ball.state.rvw
                features.extend(pos)
                features.extend(vel)
                features.extend(spin)
                features.append(float(ball.state.s))
                features.append(ball.state.t)
                features.append(1.0 if ball.state.s == 4 else 0.0)
            else:
                features.extend([0.0] * 11)

        return torch.tensor(features, dtype=torch.float32)

    def _action_to_key(self, action):
        return tuple(np.round(action, 4).tolist())

    def _apply_action(self, env, action_key):
        action = np.array(action_key, dtype=np.float32)
        env.take_shot({
            "V0": action[0],
            "phi": action[1],
            "theta": action[2],
            "a": action[3],
            "b": action[4]
        })
