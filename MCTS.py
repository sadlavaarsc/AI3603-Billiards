import numpy as np
import copy
import torch
import poolenv

class ActionSampler:
    def __init__(self, sigma=None):
        self.sigma = sigma or np.array([0.3, 15, 10, 0.05, 0.05])
        self.low = np.array([0.5, 0, 0, -0.5, -0.5])
        self.high = np.array([8.0, 360, 90, 0.5, 0.5])

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
        state_seq: List[balls_state], length = 3
        """
        self.state_seq = state_seq
        self.parent = parent
        self.children = {}  # action_key -> MCTSNode

        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = prior

        self.is_terminal = False
        self.terminal_value = None

    def ucb(self, c_puct):
        if self.N == 0:
            return float("inf")
        return self.Q + c_puct * self.P * np.sqrt(self.parent.N) / (1 + self.N)

class MCTS:
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

    def search(self, root_state_seq):
        root = MCTSNode(root_state_seq)

        for _ in range(self.n_simulations):
            env_copy = copy.deepcopy(self.env)
            node = root

            # 1. Selection
            while node.children and not node.is_terminal:
                action_key, node = max(
                    node.children.items(),
                    key=lambda item: item[1].ucb(self.c_puct)
                )
                self._apply_action(env_copy, action_key)

            # 2. Expansion / Evaluation
            value = self._expand_and_evaluate(node, env_copy)

            # 3. Backpropagation（玩家视角翻转）
            self._backpropagate(node, value)

        if not root.children:
            return self._policy_fallback(root_state_seq)

        best_action = max(
            root.children.items(),
            key=lambda item: item[1].N
        )[0]

        return np.array(best_action)
    

    def _expand_and_evaluate(self, node, env):
        # 检查终局
        done, info = env.get_done()
        if done:
            value = self._terminal_value(env, info)
            node.is_terminal = True
            node.terminal_value = value
            return value

        # 网络评估
        with torch.no_grad():
            state_tensor = self._state_seq_to_tensor(node.state_seq)
            out = self.model(state_tensor.unsqueeze(0).to(self.device))
            base_action = out["mapped_actions"][0].cpu().numpy()
            value = out["value_output"].item()

        # 连续动作采样
        actions = self.sampler.sample(base_action, self.n_action_samples)

        # similarity-based prior（归一化）
        sims = []
        for a in actions:
            scaled = (a - base_action) / self.sampler.sigma
            sims.append(1.0 / (1.0 + np.linalg.norm(scaled)))
        sims = np.array(sims)
        priors = sims / np.sum(sims)

        # 扩展子节点
        for i, action in enumerate(actions):
            key = self._action_to_key(action)
            if key in node.children:
                continue

            env_state = self._save_env_state(env)
            self._apply_action(env, key)

            new_balls = self._save_env_state(env)
            new_state_seq = node.state_seq[1:] + [new_balls]

            child = MCTSNode(
                state_seq=new_state_seq,
                parent=node,
                prior=priors[i]
            )

            node.children[key] = child
            self._restore_env_state(env, env_state)

        return value

    def _backpropagate(self, node, value):
        sign = 1
        while node:
            node.N += 1
            node.W += sign * value
            node.Q = node.W / node.N
            sign *= -1
            node = node.parent

    def _save_env_state(self, env):
        return poolenv.save_balls_state(env.balls)

    def _restore_env_state(self, env, state):
        env.balls = poolenv.restore_balls_state(state)

    def _apply_action(self, env, action_key):
        action = np.array(action_key)
        env.take_shot({
            "V0": action[0],
            "phi": action[1],
            "theta": action[2],
            "a": action[3],
            "b": action[4],
        })

    def _action_to_key(self, action):
        return tuple(np.round(action, 4).tolist())

    def _single_state_to_81(self, balls):
        features = []
        ball_ids = ['cue','1','2','3','4','5','6','7','8',
                    '9','10','11','12','13','14','15']

        for bid in ball_ids:
            b = balls[bid]
            pos = b.state.rvw[0]
            vel = b.state.rvw[1]
            spin = b.state.rvw[2]

            features.extend(pos.tolist())
            features.extend(vel.tolist())
            features.extend(spin.tolist())
            features.append(float(b.state.s))
            features.append(float(b.state.t))
            features.append(1.0 if b.state.s == 4 else 0.0)

        return torch.tensor(features, dtype=torch.float32)
    

    def _state_seq_to_tensor(self, state_seq):
        assert len(state_seq) == 3
        return torch.stack(
            [self._single_state_to_81(s) for s in state_seq],
            dim=0
        )

    def _terminal_value(self, env, info):
        winner = info["winner"]
        curr = env.get_curr_player()

        if winner == curr:
            return 1.0
        elif winner == "SAME":
            return 0.5
        else:
            return 0.0

    def _policy_fallback(self, state_seq):
        with torch.no_grad():
            state_tensor = self._state_seq_to_tensor(state_seq)
            out = self.model(state_tensor.unsqueeze(0).to(self.device))
            return out["mapped_actions"][0].cpu().numpy()
