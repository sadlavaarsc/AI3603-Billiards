# agents/alphazero_agent.py
import MCTS
import poolenv


class AlphaZeroAgent:
    def __init__(self, model, env):
        self.mcts = MCTS(model, env)
        self.state_buffer = []

    def decision(self, balls_state):
        # 维护最近三杆
        self.state_buffer.append(balls_state)
        if len(self.state_buffer) < 3:
            self.state_buffer = [balls_state] * 3
        else:
            self.state_buffer = self.state_buffer[-3:]

        action = self.mcts.search(self.state_buffer)
        return {
            "V0": action[0],
            "phi": action[1],
            "theta": action[2],
            "a": action[3],
            "b": action[4],
        }
# example usage:
env = PoolEnv()
env.reset(target_ball="solid")

agent = AlphaZeroAgent(model, env)

while True:
    balls_state = poolenv.save_balls_state(env.balls)
    action = agent.decision(balls_state)
    env.take_shot(action)

    done, info = env.get_done()
    if done:
        break