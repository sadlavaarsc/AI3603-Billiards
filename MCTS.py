import math
import numpy as np
import pooltool as pt
import copy
import torch
from data_loader import StatePreprocessor

# ============ 超时安全模拟机制 ============
class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass

def _timeout_handler(signum, frame):
    """超时信号处理器"""
    raise SimulationTimeoutError("物理模拟超时")

def simulate_with_timeout(shot, timeout=3):
    """带超时保护的物理模拟"""
    import signal
    # 设置超时信号处理器
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)  # 设置超时时间
    
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)  # 取消超时
        return True
    except SimulationTimeoutError:
        return False
    except Exception as e:
        signal.alarm(0)  # 取消超时
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器

# ============ MCTS Node ============
class MCTSNode:
    def __init__(self, state_seq, parent=None, prior=1.0, depth=0):
        self.state_seq = state_seq
        self.parent = parent
        self.children = {}
        self.depth = depth
        
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = prior

# ============ Multi-step MCTS ============
class MCTS:
    def __init__(self,
                 model,
                 n_simulations=150,
                 c_puct=1.414,
                 max_depth=3,
                 max_search_time=7.0,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.max_search_time = max_search_time  # 搜索限时，单位：秒
        self.device = device
        self.ball_radius = 0.028575
        
        # 定义噪声水平
        self.sim_noise = {
            'V0': 0.1, 'phi': 0.15, 'theta': 0.1, 'a': 0.005, 'b': 0.005
        }
        
        # 动作范围
        self.action_min = np.array([0.5, 0.0, 0.0, -0.5, -0.5], dtype=np.float32)
        self.action_max = np.array([8.0, 360.0, 90.0, 0.5, 0.5], dtype=np.float32)
        self.state_preprocessor = StatePreprocessor()
        
        print("Multi-step MCTS 已初始化。")

    def _calc_angle_degrees(self, v):
        angle = math.degrees(math.atan2(v[1], v[0]))
        return angle % 360

    def _get_ghost_ball_target(self, cue_pos, obj_pos, pocket_pos):
        vec_obj_to_pocket = np.array(pocket_pos) - np.array(obj_pos)
        dist_obj_to_pocket = np.linalg.norm(vec_obj_to_pocket)
        if dist_obj_to_pocket == 0: return 0, 0
        unit_vec = vec_obj_to_pocket / dist_obj_to_pocket
        ghost_pos = np.array(obj_pos) - unit_vec * (2 * self.ball_radius)
        vec_cue_to_ghost = ghost_pos - np.array(cue_pos)
        dist_cue_to_ghost = np.linalg.norm(vec_cue_to_ghost)
        phi = self._calc_angle_degrees(vec_cue_to_ghost)
        return phi, dist_cue_to_ghost

    def generate_heuristic_actions(self, balls, my_targets, table, only_eight_ball=False):
        """生成候选动作列表
        
        参数：
            balls: 球状态字典
            my_targets: 目标球列表
            table: 球桌对象
            only_eight_ball: 是否只生成打黑八的动作，默认关闭
        """
        actions = []
        
        cue_ball = balls.get('cue')
        if not cue_ball: return [self._random_action()]
        cue_pos = cue_ball.state.rvw[0]

        # 获取所有目标球的ID
        if only_eight_ball:
            # 只生成打黑八的动作
            target_ids = ['8'] if '8' in balls and balls['8'].state.s != 4 else []
        else:
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

                # 1. 计算理论进球角度
                phi_ideal, dist = self._get_ghost_ball_target(cue_pos, obj_pos, pocket_pos)

                # 2. 根据距离简单的估算力度 (距离越远力度越大，基础力度2.0)
                v_base = 1.5 + dist * 1.5
                v_base = np.clip(v_base, 1.0, 6.0)

                # 3. 生成几个变种动作加入候选池
                # 变种1：精准一击
                actions.append({
                    'V0': v_base, 'phi': phi_ideal, 'theta': 0, 'a': 0, 'b': 0
                })
                # 变种2：力度稍大
                actions.append({
                    'V0': min(v_base + 1.5, 7.5), 'phi': phi_ideal, 'theta': 0, 'a': 0, 'b': 0
                })
                # 变种3：角度微调 (左右偏移 0.5 度，应对噪声)
                actions.append({
                    'V0': v_base, 'phi': (phi_ideal + 0.5) % 360, 'theta': 0, 'a': 0, 'b': 0
                })
                actions.append({
                    'V0': v_base, 'phi': (phi_ideal - 0.5) % 360, 'theta': 0, 'a': 0, 'b': 0
                })

        # 如果通过启发式没有生成任何动作（极罕见），补充随机动作
        if len(actions) == 0:
            for _ in range(5):
                actions.append(self._random_action())
        
        # 随机打乱顺序
        import random
        random.shuffle(actions)
        return actions[:30]

    def _random_action(self):
        """生成随机动作"""
        import random
        return {
            'V0': round(random.uniform(0.5, 8.0), 2),
            'phi': round(random.uniform(0, 360), 2),
            'theta': round(random.uniform(0, 90), 2),
            'a': round(random.uniform(-0.5, 0.5), 3),
            'b': round(random.uniform(-0.5, 0.5), 3)
        }

    def simulate_action(self, balls, table, action):
        """执行带噪声的物理仿真"""
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        
        try:
            # --- 注入高斯噪声 ---
            noisy_V0 = np.clip(action['V0'] + np.random.normal(0, self.sim_noise['V0']), 0.5, 8.0)
            noisy_phi = (action['phi'] + np.random.normal(0, self.sim_noise['phi'])) % 360
            noisy_theta = np.clip(action['theta'] + np.random.normal(0, self.sim_noise['theta']), 0, 90)
            noisy_a = np.clip(action['a'] + np.random.normal(0, self.sim_noise['a']), -0.5, 0.5)
            noisy_b = np.clip(action['b'] + np.random.normal(0, self.sim_noise['b']), -0.5, 0.5)

            cue.set_state(V0=noisy_V0, phi=noisy_phi, theta=noisy_theta, a=noisy_a, b=noisy_b)
            pt.simulate(shot, inplace=True)
            return shot
        except Exception:
            return None

    def analyze_shot_for_reward(self, shot: pt.System, last_state: dict, player_targets: list):
        """分析击球结果并计算奖励分数"""
        # 1. 基本分析
        new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
        
        # 根据 player_targets 判断进球归属（黑8只有在清台后才算己方球）
        own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
        enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
        
        cue_pocketed = "cue" in new_pocketed
        eight_pocketed = "8" in new_pocketed

        # 2. 分析首球碰撞
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
        
        # 首球犯规判定：完全对齐 player_targets
        if first_contact_ball_id is None:
            # 未击中任何球（但若只剩白球和黑8且已清台，则不算犯规）
            if len(last_state) > 2 or player_targets != ['8']:
                foul_first_hit = True
        else:
            # 首次击打的球必须是 player_targets 中的球
            if first_contact_ball_id not in player_targets:
                foul_first_hit = True
        
        # 3. 分析碰库
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

        if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
            foul_no_rail = True
            
        # 计算奖励分数
        score = 0
        
        if cue_pocketed and eight_pocketed:
            score -= 500
        elif cue_pocketed:
            score -= 100
        elif eight_pocketed:
            is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
            score += 150 if is_targeting_eight_ball_legally else -500
                
        if foul_first_hit:
            score -= 30
        if foul_no_rail:
            score -= 30
            
        score += len(own_pocketed) * 50
        score -= len(enemy_pocketed) * 20
        
        if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
            score = 10
            
        return score

    def _denormalize_action(self, action_norm):
        """将 [0,1] 策略输出映射回真实物理动作"""
        action_norm = np.clip(action_norm, 0.0, 1.0)
        return action_norm * (self.action_max - self.action_min) + self.action_min

    def _balls_state_to_81(self, balls_state, my_targets=None, table=None):
        """
        将球状态转换为81维向量
        
        参数：
            balls_state: 球状态字典，{ball_id: Ball对象}
            my_targets: 当前玩家的目标球列表
            table: 球桌对象
            
        返回：
            np.ndarray: 81维状态向量
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
                    # 球坐标使用标准尺寸进行归一化
                    state[base + 0] = pos[0] / STANDARD_TABLE_WIDTH
                    state[base + 1] = pos[1] / STANDARD_TABLE_LENGTH
                    state[base + 2] = pos[2] / (2 * BALL_RADIUS)
                    state[base + 3] = 0.0
            else:
                state[base + 0] = -1.0
                state[base + 1] = -1.0
                state[base + 2] = -1.0
                state[base + 3] = 1.0

        # 球桌尺寸（64-65维）
        if table is not None:
            state[64] = table.w
            state[65] = table.l
        else:
            # 如果没有table对象，使用固定值
            state[64] = 2.540
            state[65] = 1.270

        # 目标球 one-hot
        if my_targets:
            for t in my_targets:
                if t.isdigit():
                    idx = int(t) - 1
                    if 0 <= idx <= 14:  # 1-15号球
                        state[66 + idx] = 1.0

        return state

    def _state_seq_to_tensor(self, state_seq):
        """将状态序列转换为张量"""
        if len(state_seq) != 3:
            raise ValueError(f"state_seq length must be 3, got {len(state_seq)}")

        for i, s in enumerate(state_seq):
            if not isinstance(s, np.ndarray) or s.shape != (81,):
                raise TypeError(
                    f"state_seq[{i}] must be np.ndarray(81), got {type(s)} {getattr(s, 'shape', None)}"
                )

        states = np.stack(state_seq, axis=0)
        states = self.state_preprocessor(states)
        return torch.from_numpy(states).float()

    def _expand_and_evaluate(self, node, balls, table, player_targets, root_player, depth, remaining_hits):
        """扩展节点并评估"""
        # 1. 检查是否已经处于输了的状态
        # 检查白球是否进袋
        cue_ball = balls.get('cue')
        cue_in_pocket = cue_ball and cue_ball.state.s == 4
        
        # 检查黑八是否进袋
        eight_ball = balls.get('8')
        eight_in_pocket = eight_ball and eight_ball.state.s == 4
        
        # 检查自己的目标球是否都已经进袋（除了黑八）
        my_targets = player_targets[root_player]
        non_eight_targets = [bid for bid in my_targets if bid != '8']
        has_non_eight_targets_left = any(bid in balls and balls[bid].state.s != 4 for bid in non_eight_targets)
        
        # 输的情况：
        # 1. 白球和黑八同时进袋
        # 2. 黑八进袋但自己还有非黑八目标球未进
        # 3. 白球进袋且黑八也进袋
        if (cue_in_pocket and eight_in_pocket) or (eight_in_pocket and has_non_eight_targets_left):
            return 0.0
        
        # 2. 检查深度限制，不超过剩余杆数
        if depth >= remaining_hits:
            # 只使用value network评估
            state_tensor = self._state_seq_to_tensor(node.state_seq)
            state_tensor = state_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                out = self.model(state_tensor)
                value = out["value_output"].item()
            
            return value
        
        # 2. 生成候选动作
        candidate_actions = self.generate_heuristic_actions(balls, player_targets[root_player], table)
        n_candidates = len(candidate_actions)
        
        # 检查是否只剩下黑八一个待打的球
        remaining_targets = [bid for bid in player_targets[root_player] if balls[bid].state.s != 4]
        has_only_eight_ball = len(remaining_targets) == 1 and remaining_targets[0] == '8'
        
        # 3. 使用模型生成policy
        state_tensor = self._state_seq_to_tensor(node.state_seq)
        state_tensor = state_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.model(state_tensor)
            policy_output = out["policy_output"][0].cpu().numpy()
            value_output = out["value_output"].item()
        
        # 4. 反归一化得到模型输出的原始动作
        model_action = self._denormalize_action(policy_output)
        
        # 5. 动作筛选：根据与模型输出的接近程度排序，保留指定数量的动作
        action_distances = []
        for action in candidate_actions:
            # 转换为numpy数组便于计算距离
            action_arr = np.array([action['V0'], action['phi'], action['theta'], action['a'], action['b']])
            
            # 计算动作距离（考虑角度和力度的加权距离）
            # 角度差（0-180度）
            phi_diff = abs(action_arr[1] - model_action[1])
            if phi_diff > 180:
                phi_diff = 360 - phi_diff
            
            # 力度差
            v0_diff = abs(action_arr[0] - model_action[0])
            
            # 综合距离（角度差权重0.7，力度差权重0.3）
            distance = (phi_diff / 180.0) * 0.7 + (v0_diff / (8.0 - 0.5)) * 0.3
            
            action_distances.append((action, distance))
        
        # 按照距离从小到大排序
        action_distances.sort(key=lambda x: x[1])
        
        # 保留指定数量的动作，默认为n_simulations/2，至少保留1个
        keep_count = max(1, int(self.n_simulations / 2))
        filtered_actions = [action for action, distance in action_distances[:keep_count]]
        
        # 如果只剩下黑八一个待打的球，确保生成打黑八的动作并加入
        if has_only_eight_ball:
            # 生成只打黑八的动作
            eight_ball_actions = self.generate_heuristic_actions(balls, player_targets[root_player], table, only_eight_ball=True)
            # 将打黑八的动作添加到filtered_actions中，避免重复
            for eight_action in eight_ball_actions:
                if eight_action not in filtered_actions:
                    filtered_actions.append(eight_action)
        
        # 6. 对每个动作进行模拟和评估
        best_value = -float('inf')
        
        for action in filtered_actions:
            # 模拟动作
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            shot = self.simulate_action(balls, table, action)
            
            # 计算物理模拟奖励
            if shot is None:
                raw_reward = -500.0
            else:
                raw_reward = self.analyze_shot_for_reward(shot, last_state_snapshot, player_targets[root_player])
            
            # 归一化奖励
            normalized_reward = (raw_reward - (-500)) / 650.0
            normalized_reward = np.clip(normalized_reward, 0.0, 1.0)
            
            # 根据深度调整value network和物理模拟的权重，使用剩余杆数作为最大深度
            depth_factor = depth / remaining_hits if remaining_hits > 0 else 1.0  # 0-1，深度越深，value network权重越大
            value = depth_factor * value_output + (1 - depth_factor) * normalized_reward
            
            # 递归扩展子节点：只有当不是直接输掉的情况才扩展
            # raw_reward=-500表示直接输掉，normalized_reward<0.1表示接近输掉
            if shot is not None and raw_reward > -500 and normalized_reward > 0 and (depth + 1) < remaining_hits:
                # 生成新的状态向量
                new_balls_state = {bid: ball for bid, ball in shot.balls.items()}
                # 使用_balls_state_to_81方法将balls_state转换为81维向量
                new_state_vec = self._balls_state_to_81(
                    new_balls_state,
                    my_targets=player_targets[root_player],
                    table=table
                )
                new_state_seq = node.state_seq[1:] + [new_state_vec]
                child_node = MCTSNode(new_state_seq, parent=node)
                child_value = self._expand_and_evaluate(child_node, shot.balls, table, player_targets, root_player, depth + 1, remaining_hits)
                value += child_value * 0.9  # 衰减因子
            
            if value > best_value:
                best_value = value
        
        return best_value

    def search(self, state_seq, balls, table, player_targets, root_player, remaining_hits):
        """执行MCTS搜索
        
        参数：
            state_seq: 状态序列，长度为3的列表，每个元素是81维的numpy数组
            balls: 球状态字典，{ball_id: Ball对象}
            table: 球桌对象
            player_targets: 玩家目标球字典，{player: [target_ball_ids]}
            root_player: 当前玩家，'A'或'B'
            remaining_hits: 剩余杆数
            
        返回：
            numpy数组: 最佳动作，包含V0、phi、theta、a、b五个属性
        """
        import time
        
        root = MCTSNode(state_seq)
        
        # 生成候选动作
        candidate_actions = self.generate_heuristic_actions(balls, player_targets[root_player], table)
        n_candidates = len(candidate_actions)
        
        N = np.zeros(n_candidates)
        Q = np.zeros(n_candidates)
        
        # 计算当前回合的实际最大深度，不超过剩余杆数
        current_max_depth = min(self.max_depth, remaining_hits)
        
        # 记录搜索开始时间
        start_time = time.time()
        
        # MCTS循环
        simulation_count = 0
        time_exceeded = False
        
        for _ in range(self.n_simulations):
            # 检查是否超过搜索时间限制
            elapsed_time = time.time() - start_time
            if elapsed_time > self.max_search_time:
                time_exceeded = True
                print(f"[Multi-step MCTS] 搜索时间超过限制 ({elapsed_time:.2f}s > {self.max_search_time}s)，提前终止搜索")
                break
            
            simulation_count += 1
            
            # 1. Selection (UCB)
            if np.sum(N) < n_candidates:
                idx = int(np.sum(N))
            else:
                ucb_values = Q + self.c_puct * np.sqrt(np.log(np.sum(N) + 1) / (N + 1e-6))
                idx = np.argmax(ucb_values)
            
            action = candidate_actions[idx]
            
            # 2. 使用模型生成value
            state_tensor = self._state_seq_to_tensor(root.state_seq)
            state_tensor = state_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                out = self.model(state_tensor)
                value_output = out["value_output"].item()
            
            # 3. Simulation (带噪声)
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            shot = self.simulate_action(balls, table, action)
            
            # 4. Evaluation
            if shot is None:
                raw_reward = -500.0
            else:
                raw_reward = self.analyze_shot_for_reward(shot, last_state_snapshot, player_targets[root_player])
            
            # 归一化奖励
            normalized_reward = (raw_reward - (-500)) / 650.0
            normalized_reward = np.clip(normalized_reward, 0.0, 1.0)
            
            # 初始深度为0，所以物理模拟权重更大
            depth = 0
            # 避免除0错误
            depth_factor = depth / current_max_depth if current_max_depth > 0 else 1.0
            value = depth_factor * value_output + (1 - depth_factor) * normalized_reward
            
            # 5. Backpropagation
            N[idx] += 1
            Q[idx] += (value - Q[idx]) / N[idx]
        
        # 如果时间超限，为未搜索的动作使用模型生成value
        if time_exceeded:
            # 使用模型为未搜索的动作生成value
            state_tensor = self._state_seq_to_tensor(root.state_seq)
            state_tensor = state_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                out = self.model(state_tensor)
                model_value = out["value_output"].item()
            
            # 为未搜索的动作设置模型value
            for idx in range(n_candidates):
                if N[idx] == 0:  # 未搜索的动作
                    Q[idx] = model_value
                    N[idx] = 1  # 标记为已处理
        
        # Final Decision
        avg_rewards = Q
        best_idx = np.argmax(avg_rewards)
        best_action = candidate_actions[best_idx]
        
        print(f"[Multi-step MCTS] Best Avg Score: {avg_rewards[best_idx]:.3f} (Sims: {simulation_count}/{self.n_simulations})")
        
        # 转换为numpy数组返回
        return np.array([best_action['V0'], best_action['phi'], best_action['theta'], best_action['a'], best_action['b']], dtype=np.float32)
