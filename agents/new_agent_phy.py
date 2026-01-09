import math
import copy
import random
import numpy as np
import pooltool as pt

from .agent import Agent
from .basic_agent_pro import analyze_shot_for_reward


BALL_R = 0.028575


# ================= 几何工具 =================

def angle(v):
    return math.degrees(math.atan2(v[1], v[0])) % 360


def dist_point_to_segment(p, a, b):
    ap = p - a
    ab = b - a
    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-9)
    t = np.clip(t, 0, 1)
    proj = a + t * ab
    return np.linalg.norm(p - proj)


def is_path_clear(p1, p2, balls, ignore):
    p1 = p1[:2]
    p2 = p2[:2]
    for bid, b in balls.items():
        if bid in ignore or b.state.s == 4:
            continue
        pos = b.state.rvw[0][:2]
        if dist_point_to_segment(pos, p1, p2) < 2 * BALL_R:
            return False
    return True


class NewAgentPro(Agent):
    """
    物理 + 几何 + 防守 + 一步对手评估

    """

    def __init__(self,
                 n_rollout=6,
                 opp_rollout=3,
                 lambda_opp=0.8):
        super().__init__()
        self.n_rollout = n_rollout
        self.opp_rollout = opp_rollout
        self.lambda_opp = lambda_opp
        self.sim_noise = {
            'V0': 0.1,
            'phi': 0.15
        }

        print("PhysicsAgent initialized.")

    # ---------- Ghost Ball ----------

    def ghost_shot(self, cue, obj, pocket):
        d = pocket - obj
        if np.linalg.norm(d) < 1e-6:
            return None
        d = d / np.linalg.norm(d)
        ghost = obj - d * 2 * BALL_R
        v = ghost - cue
        return angle(v), np.linalg.norm(v), ghost

    # ---------- 进攻动作生成 ----------

    def generate_attack_actions(self, balls, my_targets, table):
        actions = []
        cue_pos = balls['cue'].state.rvw[0]

        for tid in my_targets:
            if balls[tid].state.s == 4:
                continue
            obj_pos = balls[tid].state.rvw[0]

            for pocket in table.pockets.values():
                res = self.ghost_shot(cue_pos, obj_pos, pocket.center)
                if res is None:
                    continue
                phi, dist, ghost = res

                # 几何可行性过滤
                if not is_path_clear(cue_pos, ghost, balls, ['cue', tid]):
                    continue
                if not is_path_clear(obj_pos, pocket.center, balls, [tid]):
                    continue

                v0 = np.clip(1.2 + dist * 1.4, 1.0, 6.8)

                actions.append({
                    'V0': v0,
                    'phi': phi,
                    'theta': 0,
                    'a': 0,
                    'b': 0
                })

        random.shuffle(actions)
        return actions[:20]

    # ---------- Safety Shot ----------

    def generate_safety_actions(self, balls, my_targets):
        cue = balls['cue'].state.rvw[0]
        safeties = []

        for tid in my_targets:
            if balls[tid].state.s == 4:
                continue
            obj = balls[tid].state.rvw[0]
            v = obj - cue
            phi = (angle(v) + 180) % 360
            safeties.append({
                'V0': 0.8,
                'phi': phi,
                'theta': 0,
                'a': 0,
                'b': 0
            })

        return safeties[:5]

    # ---------- 单次仿真 ----------

    def simulate(self, balls, table, action):
        sim_balls = {k: copy.deepcopy(v) for k, v in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue("cue")
        shot = pt.System(
            table=sim_table,
            balls=sim_balls,
            cue=cue
        )

        cue.set_state(
            V0=np.clip(action['V0'] + np.random.normal(0, self.sim_noise['V0']), 0.5, 8.0),
            phi=(action['phi'] + np.random.normal(0, self.sim_noise['phi'])) % 360,
            theta=0,
            a=0,
            b=0
        )
        try:
            pt.simulate(shot, inplace=True)
            return shot
        except Exception:
            return None

    # ---------- 对手回应评估 ----------

    def opponent_best_reply(self, shot, table, my_targets):
        opp_targets = [
            bid for bid in shot.balls
            if bid not in my_targets and bid not in ['cue']
        ]
        best = 0
        for _ in range(self.opp_rollout):
            action = self._random_action()
            sim = self.simulate(shot.balls, table, action)
            if sim:
                r = analyze_shot_for_reward(sim,
                                             shot.balls,
                                             opp_targets)
                best = max(best, r)
        return best

    # ---------- 决策 ----------

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None:
            return self._random_action()

        # 清台处理
        if all(balls[t].state.s == 4 for t in my_targets):
            my_targets = ['8']

        last_state = {k: copy.deepcopy(v) for k, v in balls.items()}

        attack_actions = self.generate_attack_actions(balls, my_targets, table)
        safety_actions = self.generate_safety_actions(balls, my_targets)

        best_score = -1e9
        best_action = None

        # ---------- 进攻评估 ----------
        for action in attack_actions:
            total = 0
            for _ in range(self.n_rollout):
                shot = self.simulate(balls, table, action)
                if shot is None:
                    total -= 500
                    continue

                my_r = analyze_shot_for_reward(shot, last_state, my_targets)
                opp_r = self.opponent_best_reply(shot, table, my_targets)

                total += my_r - self.lambda_opp * opp_r

            avg = total / self.n_rollout
            if avg > best_score:
                best_score = avg
                best_action = action

        # ---------- Safety fallback ----------
        if best_score < 15:
            for action in safety_actions:
                shot = self.simulate(balls, table, action)
                if shot:
                    r = analyze_shot_for_reward(shot, last_state, my_targets)
                    if r > best_score:
                        best_score = r
                        best_action = action

        if best_action is None:
            return self._random_action()

        print(f"[SuperAgent] Best score: {best_score:.2f}")
        return best_action
