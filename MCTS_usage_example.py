# agents/alphazero_agent.py
import MCTS
import poolenv
import collections
class AlphaZeroAgent:
    class AlphaZeroAgent:
        def __init__(
            self,
            model,
            env,
            n_simulations=30,
            n_action_samples=8,
            device="cpu"
        ):
            self.env = env

            self.mcts = MCTS(
                model=model,
                env=env,
                n_simulations=n_simulations,
                n_action_samples=n_action_samples,
                device=device
            )
            self.state_buffer = collections.deque(maxlen=3)

    def decision(self, balls_state, my_targets=None, table=None):
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