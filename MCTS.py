import math
import numpy as np
import pooltool as pt
import copy
import torch
from data_loader import StatePreprocessor

# ============ MCTS Node ============
class MCTSNode:
    def __init__(self, state_seq, parent=None, prior=1.0, action=None):
        # ⚠️ 注意：state_seq仅用于初始化，实际状态推进靠外部current_state_seq变量
        # Node不是状态节点，只是统计容器，state_seq不会自动更新
        self.state_seq = state_seq
        self.parent = parent
        self.action = action
        self.children = {}
        
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = prior
        self.is_expanded = False
    
    def ucb_score(self, c_puct):
        """计算PUCT分数"""
        if self.parent is None:
            return self.Q
        
        return self.Q + c_puct * self.P * math.sqrt(self.parent.N + 1e-8) / (1 + self.N)
    
    def select_child(self, c_puct):
        """选择UCB分数最高的子节点"""
        return max(self.children.values(), key=lambda child: child.ucb_score(c_puct))
    
    def expand(self, action_priors):
        """扩展节点，添加子节点"""
        for action, prior in action_priors:
            # 将动作字典转换为可哈希的元组，用于作为children字典的键
            # 排序键以确保一致性
            action_key = tuple(sorted(action.items()))
            if action_key not in self.children:
                self.children[action_key] = MCTSNode(
                    state_seq=self.state_seq,  # 子节点继承父节点的状态序列
                    parent=self,
                    prior=prior,
                    action=action
                )
        self.is_expanded = True
    
    def backup(self, value):
        """备份价值，更新节点的访问次数和价值"""
        node = self
        v = value
        while node is not None:
            node.N += 1
            node.W += v
            node.Q = node.W / node.N
            v = -v  # ★ 关键：每向上走一层翻转符号，考虑玩家视角切换
            node = node.parent

# ============ Multi-step MCTS ============
class MCTS:
    def __init__(self,
                 model,
                 n_simulations=100,
                 c_puct=1.414,
                 max_depth=5,
                 max_search_time=30.0,
                 action_keep_ratio=1/2,
                 action_max_keep_count=20,  # 最大保留动作数量
                 # Value剪枝相关参数
                 value_prune_threshold=-0.6,  # 绝对阈值剪枝阈值
                 value_relative_prune_ratio=0.3,  # 相对剪枝比例（剪掉bottom X%）
                 max_sim_depth=3,  # 深度截断阈值
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.max_search_time = max_search_time  # 搜索限时，单位：秒
        self.action_keep_ratio = action_keep_ratio  # 动作保留比例，用于控制保留的动作数量
        self.action_max_keep_count = action_max_keep_count  # 最大保留动作数量
        self.device = device
        self.ball_radius = 0.028575
        
        # Value剪枝相关参数
        self.value_prune_threshold = value_prune_threshold  # 绝对阈值剪枝阈值
        self.value_relative_prune_ratio = value_relative_prune_ratio  # 相对剪枝比例
        self.max_sim_depth = max_sim_depth  # 深度截断阈值
        
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
        
        if cue_pocketed:
            score -= 100
        elif eight_pocketed:
            # 黑八入袋，由_evaluate_leaf统一判断胜负
            # 这里只给基础分数，具体胜负由叶节点评估决定
            score += 100
                
        # 特殊规则：当只剩黑八一个球但是还选择打别人的球时，给一个较小的惩罚
        only_eight_ball_left = (len(player_targets) == 1 and player_targets[0] == "8")
        hit_wrong_ball_when_only_eight = only_eight_ball_left and first_contact_ball_id is not None and first_contact_ball_id != "8"
        
        if foul_first_hit:
            # 如果是特殊情况，给较小的惩罚
            if hit_wrong_ball_when_only_eight:
                score -= 15  # 较小的惩罚
            else:
                score -= 30  # 正常惩罚
        
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
        
        # 检查是否为残局（只剩黑八一个球要打）
        remaining_targets = [
            bid for bid in player_targets[root_player]
            if bid in balls and balls[bid].state.s != 4
        ]
        is_endgame = (len(remaining_targets) == 1 and remaining_targets[0] == '8')
        
        # 残局时调整参数：
        # 1. 实际使用的n_simulations*3/2
        # 2. 实际使用的max_depth*3/2
        current_n_simulations = int(self.n_simulations * 3 / 2) if is_endgame else self.n_simulations
        current_max_depth = min(
            int(self.max_depth * 3 / 2) if is_endgame else self.max_depth,
            remaining_hits
        )
        
        # 记录搜索开始时间
        start_time = time.time()
        # P0-3优化：时间限制修复 - 定义绝对截止时间
        deadline = start_time + self.max_search_time
        
        # 创建根节点
        root = MCTSNode(state_seq)
        
        # 创建动作→仿真结果缓存（P0-1优化：一次action在一次search中最多只允许一次完整物理仿真）
        action_cache = {}
        # 缓存结构：
        # key: action_key = tuple(sorted(action.items()))
        # value: {
        #     'shot': 物理仿真结果,
        #     'raw_reward': 原始奖励,
        #     'next_state_vec': 下一个状态向量,
        #     'is_black_eight_foul': 是否黑八违规,
        #     'cue_in_pocket': 母球是否进袋,
        #     'eight_in_pocket': 黑八是否进袋
        # }
        
        # 1️⃣ 启发式生成候选动作
        candidate_actions = self.generate_heuristic_actions(balls, player_targets[root_player], table)
        
        # 2️⃣ policy网络筛选和先验概率生成
        state_tensor = self._state_seq_to_tensor(root.state_seq).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(state_tensor)
            value_output = out["value_output"].item()
            policy_mapped_actions = out["mapped_actions"][0].cpu().numpy()
        
        # 3️⃣ 使用policy网络进行动作裁剪
        # 计算每个候选动作与policy输出的距离
        action_distances = []
        for action in candidate_actions:
            # 将动作转换为numpy数组，顺序：V0, phi, theta, a, b
            action_np = np.array([
                action['V0'],
                action['phi'],
                action['theta'],
                action['a'],
                action['b']
            ])
            
            # 计算各维度距离，考虑角度的周期性
            dist_V0 = abs(action_np[0] - policy_mapped_actions[0])
            dist_phi = min(abs(action_np[1] - policy_mapped_actions[1]), 360 - abs(action_np[1] - policy_mapped_actions[1]))
            dist_theta = abs(action_np[2] - policy_mapped_actions[2])
            dist_a = abs(action_np[3] - policy_mapped_actions[3])
            dist_b = abs(action_np[4] - policy_mapped_actions[4])
            
            # 加权距离（可根据重要性调整权重）
            total_dist = (dist_V0 * 0.2 + 
                         dist_phi * 0.3 + 
                         dist_theta * 0.1 + 
                         dist_a * 0.2 + 
                         dist_b * 0.2)
            
            action_distances.append((action, total_dist))
        
        # 按距离排序，先保留指定比例（action_keep_ratio），再进行top-K裁剪
        action_distances.sort(key=lambda x: x[1])
        
        # 先计算比例保留数量
        ratio_keep_count = max(1, int(len(action_distances) * self.action_keep_ratio))
        
        # 再取比例保留数量和最大保留数量的最小值
        keep_count = min(ratio_keep_count, self.action_max_keep_count)
        
        # 修复：根节点最多允许 min(keep_count, 固定上限如 10) 个动作
        keep_count = min(keep_count, 10)
        
        filtered_actions = [action for action, dist in action_distances[:keep_count]]
        
        # 4️⃣ Value + 人工reward混合筛选
        action_values = []
        alpha = 0.5  # value与reward的加权系数
        
        # 收集所有需要评估的next_state_seq，用于批量化推理（P0-2优化）
        state_seqs_to_evaluate = []
        action_to_eval_idx = {}  # 记录动作id到评估索引的映射
        cache_misses = []  # 记录需要进行物理仿真的动作
        
        for action in filtered_actions:
            # P1-2优化：生成更高效的action_id（直接使用元组，不排序）
            action_id = (action['V0'], action['phi'], action['theta'], action['a'], action['b'])
            
            if action_id not in action_cache:
                # 动作不在缓存中，需要进行物理仿真
                cache_misses.append(action)
            else:
                # 动作在缓存中，直接使用缓存结果
                cached = action_cache[action_id]
                
                # 计算next_state_seq
                next_state_seq = root.state_seq[1:] + [cached['next_state_vec']]
                state_seqs_to_evaluate.append(next_state_seq)
                # 使用action_id作为键，而不是action字典
                action_to_eval_idx[action_id] = len(state_seqs_to_evaluate) - 1
        
        # 对缓存未命中的动作进行物理仿真
        for action in cache_misses:
            # P0-3优化：在deepcopy前检查超时
            if time.time() > deadline:
                time_exceeded = True
                break
            
            # 进行快速物理模拟
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            sim_table = copy.deepcopy(table)
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in sim_balls.items()}
            
            # P0-3优化：在simulate_action前检查超时
            if time.time() > deadline:
                time_exceeded = True
                break
            
            shot = self.simulate_action(sim_balls, sim_table, action)
            if shot is None:
                continue
            
            # 计算raw_reward
            reward = self.analyze_shot_for_reward(shot, last_state_snapshot, player_targets[root_player])
            
            # 检查是否为黑八违规（即时判负）
            cue_ball = shot.balls.get('cue')
            eight_ball = shot.balls.get('8')
            cue_in_pocket = cue_ball and cue_ball.state.s == 4
            eight_in_pocket = eight_ball and eight_ball.state.s == 4
            
            root_targets = player_targets[root_player]
            non_eight_targets = [bid for bid in root_targets if bid != '8']
            has_non_eight_targets_left = any(
                bid in shot.balls and shot.balls[bid].state.s != 4 for bid in non_eight_targets
            )
            
            is_black_eight_foul = False
            if (cue_in_pocket and eight_in_pocket) or (eight_in_pocket and has_non_eight_targets_left):
                # 黑八违规，设置标志但不直接修改reward
                # reward将在后续被归一化处理
                is_black_eight_foul = True
            
            # 计算next_state_seq
            new_state_vec = self._balls_state_to_81(
                shot.balls,
                my_targets=player_targets[root_player],
                table=shot.table
            )
            
            # P1-2优化：使用更高效的action_id作为缓存key
            action_id = (action['V0'], action['phi'], action['theta'], action['a'], action['b'])
            action_cache[action_id] = {
                'shot': shot,
                'raw_reward': reward,
                'next_state_vec': new_state_vec,
                'is_black_eight_foul': is_black_eight_foul,
                'cue_in_pocket': cue_in_pocket,
                'eight_in_pocket': eight_in_pocket
            }
            
            # 计算next_state_seq用于模型评估
            next_state_seq = root.state_seq[1:] + [new_state_vec]
            state_seqs_to_evaluate.append(next_state_seq)
            # 使用action_id作为键，而不是action字典
            action_to_eval_idx[action_id] = len(state_seqs_to_evaluate) - 1
        
        # 批量化模型推理（P0-2优化）
        model_values = {}
        if state_seqs_to_evaluate:
            # P0-3优化：在模型推理前检查超时
            if time.time() > deadline:
                time_exceeded = True
            else:
                # 将所有状态序列转换为张量并批量化
                state_tensors = []
                for seq in state_seqs_to_evaluate:
                    state_tensor = self._state_seq_to_tensor(seq)
                    state_tensors.append(state_tensor)
                
                batch_state_tensor = torch.stack(state_tensors, dim=0).to(self.device)
                
                with torch.no_grad():
                    out = self.model(batch_state_tensor)
                
                batch_value_outputs = out["value_output"].cpu().numpy()
                
                # 将批量化结果映射回动作
                for action_id, idx in action_to_eval_idx.items():
                    model_values[action_id] = batch_value_outputs[idx][0]
        
        # 计算混合价值
        for action in filtered_actions:
            # P1-2优化：使用更高效的action_id
            action_id = (action['V0'], action['phi'], action['theta'], action['a'], action['b'])
            
            if action_id not in action_cache:
                continue
            
            cached = action_cache[action_id]
            reward = cached['raw_reward']
            # 使用action_id作为键，而不是action字典
            model_value = model_values[action_id]
            
            # 归一化reward到[-1, 1]范围
            # 基于reward的实际范围进行归一化
            if reward > 150:  # 黑八胜利
                normalized_reward = 1.0
            elif reward < -500:  # 严重违规
                normalized_reward = -1.0
            else:
                # 线性归一化到[-1, 1]
                normalized_reward = reward / 500
            
            # 计算混合价值
            blended_value = alpha * model_value + (1 - alpha) * normalized_reward
            
            action_values.append((action, blended_value))
        
        # 按混合价值排序，先保留指定比例（action_keep_ratio），再进行top-K裁剪
        action_values.sort(key=lambda x: x[1], reverse=True)
        
        # P0-Value剪枝：相对剪枝（推荐，鲁棒性更强）
        # 这里是根节点搜索阶段，不应用深度相关的相对剪枝
        pruned_action_values = action_values
        
        # 先计算比例保留数量
        ratio_keep_count = max(1, int(len(pruned_action_values) * self.action_keep_ratio))
        
        # 再取比例保留数量和最大保留数量的最小值
        value_keep_count = min(ratio_keep_count, self.action_max_keep_count)
        
        final_actions = [action for action, value in pruned_action_values[:value_keep_count]]
        
        # 生成动作先验概率：混合价值越高，先验概率越高
        action_priors = []
        if final_actions:
            # 计算混合价值的总和作为归一化因子
            total_value = sum(value for action, value in action_values[:value_keep_count])
            if total_value == 0:
                # 如果总和为0，使用均匀分布
                uniform_prior = 1.0 / len(final_actions)
                for action in final_actions:
                    action_priors.append((action, uniform_prior))
            else:
                # 混合价值越高，先验概率越高
                for i, (action, value) in enumerate(action_values[:value_keep_count]):
                    prior = value / total_value
                    action_priors.append((action, prior))
        
        # 扩展根节点
        root.expand(action_priors)
        
        # 模拟次数计数器
        simulation_count = 0
        time_exceeded = False
        
        # 3️⃣ 多次模拟：选择、扩展+评估、备份
        # P0-3优化：时间限制修复 - 以时间为第一约束，simulation数为第二约束
        while simulation_count < current_n_simulations and time.time() < deadline:
            simulation_count += 1
            
            # Selection
            node = root
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            sim_table = copy.deepcopy(table)
            current_player = root_player
            current_depth = 0
            path = [node]
            # 初始化current_state_seq，避免作用域不安全问题
            current_state_seq = node.state_seq
            
            # P1-3优化：缓存当前阶段的球目标和黑八阶段标志
            current_targets = player_targets[current_player]
            non_eight_targets = [bid for bid in current_targets if bid != '8']
            has_non_eight_targets_left = any(
                bid in sim_balls and sim_balls[bid].state.s != 4 for bid in non_eight_targets
            )
            is_legal_eight_ball_stage = (len([bid for bid in current_targets if bid in sim_balls and sim_balls[bid].state.s != 4]) == 1 and current_targets[0] == '8')
            
            while node.is_expanded and node.children and current_depth < current_max_depth:
                # 选择UCB分数最高的子节点
                node = node.select_child(self.c_puct)
                path.append(node)
                
                # 执行动作前检查时间限制
                if time.time() > deadline:
                    time_exceeded = True
                    break
                
                # 执行动作，获取下一个状态
                last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in sim_balls.items()}
                shot = self.simulate_action(sim_balls, sim_table, node.action)
                
                if shot is None:
                    break
                
                # 更新模拟状态
                sim_balls = shot.balls
                sim_table = shot.table
                
                # 更新模拟状态的state_seq，用于后续评估
                new_state_vec = self._balls_state_to_81(
                    sim_balls,
                    my_targets=player_targets[current_player],
                    table=sim_table
                )
                # P1-1优化：node.state_seq视为只读，只更新局部current_state_seq
                current_state_seq = current_state_seq[1:] + [new_state_vec]
                
                # 检查是否需要切换玩家
                # 1. 计算首球碰撞和犯规情况
                first_contact = None
                for e in shot.events:
                    ids = getattr(e, 'ids', [])
                    if 'cue' in ids:
                        others = [i for i in ids if i != 'cue']
                        if others:
                            first_contact = others[0]
                            break
                
                # P1-3优化：更新阶段标志
                current_targets = player_targets[current_player]
                non_eight_targets = [bid for bid in current_targets if bid != '8']
                has_non_eight_targets_left = any(
                    bid in shot.balls and shot.balls[bid].state.s != 4 for bid in non_eight_targets
                )
                is_legal_eight_ball_stage = (len([bid for bid in current_targets if bid in shot.balls and shot.balls[bid].state.s != 4]) == 1 and current_targets[0] == '8')
                
                # 首球犯规判定
                if first_contact is None:
                    foul_first_hit = True
                elif is_legal_eight_ball_stage:
                    # 合法黑8阶段，只允许击中黑8
                    foul_first_hit = first_contact != '8'
                else:
                    # 正常阶段，必须击中目标球
                    foul_first_hit = first_contact not in current_targets
                
                # 2. 计算碰库犯规
                cue_hit_cushion = any(
                    'cushion' in str(e.event_type).lower() and 'cue' in getattr(e, 'ids', [])
                    for e in shot.events
                )
                
                # 检查目标球是否碰库
                target_hit_cushion = any(
                    'cushion' in str(e.event_type).lower() and first_contact in getattr(e, 'ids', [])
                    for e in shot.events
                )
                
                new_pocketed = [
                    bid for bid in shot.balls
                    if shot.balls[bid].state.s == 4 and last_state_snapshot[bid].state.s != 4
                ]
                own_pocketed = [bid for bid in new_pocketed if bid in player_targets[current_player]]
                cue_pocketed = 'cue' in new_pocketed
                
                # 碰库犯规：没有进球且母球和目标球都没有碰到库边
                foul_no_rail = (len(new_pocketed) == 0 and not cue_hit_cushion and not target_hit_cushion)
                
                # 3. 判断是否需要切换玩家
                is_foul = cue_pocketed or foul_first_hit or foul_no_rail
                switch_player = is_foul or len(own_pocketed) == 0
                
                # 更新玩家
                if switch_player:
                    current_player = 'B' if current_player == 'A' else 'A'
                
                current_depth += 1
            
            # 3. Expansion: 如果节点未扩展，生成候选动作并扩展
            if not node.is_expanded and current_depth < remaining_hits:
                # P0-Value剪枝：Expansion剪枝
                # 先评估当前节点的value，如果过低则不展开子节点
                leaf_value = self._evaluate_leaf(
                    node, sim_balls, sim_table, player_targets,
                    current_player, root_player, current_depth,
                    remaining_hits - current_depth, current_state_seq
                )
                
                # 检查是否需要剪枝
                # current_depth ≥ 2 时应用绝对剪枝
                if current_depth >= 2 and leaf_value < self.value_prune_threshold:
                    # 直接回传value，不展开子节点
                    path[-1].backup(leaf_value)
                    continue
                
                # 检查是否为残局（只剩黑八一个球要打）
                remaining_targets = [
                    bid for bid in player_targets[current_player]
                    if bid in sim_balls and sim_balls[bid].state.s != 4
                ]
                is_endgame = (len(remaining_targets) == 1 and remaining_targets[0] == '8')
                
                # 残局时，确保包含打黑八的动作
                only_eight_ball = is_endgame
                
                candidate_actions = self.generate_heuristic_actions(
                    sim_balls,
                    player_targets[current_player],
                    sim_table,
                    only_eight_ball=only_eight_ball
                )
                
                # 如果只剩黑八，确保包含打黑八的动作
                if is_endgame:
                    # 生成只打黑八的动作
                    eight_ball_actions = self.generate_heuristic_actions(
                        sim_balls,
                        player_targets[current_player],
                        sim_table,
                        only_eight_ball=True
                    )
                    # 合并动作，去重
                    combined_actions = candidate_actions + eight_ball_actions
                    seen = set()
                    unique_actions = []
                    for action in combined_actions:
                        # 使用tuple作为键去重
                        action_tuple = tuple(sorted(action.items()))
                        if action_tuple not in seen:
                            seen.add(action_tuple)
                            unique_actions.append(action)
                    candidate_actions = unique_actions
                
                # 4. 使用policy网络进行动作裁剪（与根节点相同逻辑）
                # 获取policy输出
                state_tensor = self._state_seq_to_tensor(current_state_seq).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out = self.model(state_tensor)
                    policy_mapped_actions = out["mapped_actions"][0].cpu().numpy()
                
                # 计算每个候选动作与policy输出的距离
                action_distances = []
                for action in candidate_actions:
                    # 将动作转换为numpy数组，顺序：V0, phi, theta, a, b
                    action_np = np.array([
                        action['V0'],
                        action['phi'],
                        action['theta'],
                        action['a'],
                        action['b']
                    ])
                    
                    # 计算各维度距离，考虑角度的周期性
                    dist_V0 = abs(action_np[0] - policy_mapped_actions[0])
                    dist_phi = min(abs(action_np[1] - policy_mapped_actions[1]), 360 - abs(action_np[1] - policy_mapped_actions[1]))
                    dist_theta = abs(action_np[2] - policy_mapped_actions[2])
                    dist_a = abs(action_np[3] - policy_mapped_actions[3])
                    dist_b = abs(action_np[4] - policy_mapped_actions[4])
                    
                    # 加权距离（可根据重要性调整权重）
                    total_dist = (dist_V0 * 0.2 + 
                                 dist_phi * 0.3 + 
                                 dist_theta * 0.1 + 
                                 dist_a * 0.2 + 
                                 dist_b * 0.2)
                    
                    action_distances.append((action, total_dist))
                
                # 按距离排序，先保留指定比例（action_keep_ratio），再进行top-K裁剪
                action_distances.sort(key=lambda x: x[1])
                
                # 先计算比例保留数量
                ratio_keep_count = max(1, int(len(action_distances) * self.action_keep_ratio))
                
                # 再取比例保留数量和最大保留数量的最小值
                keep_count = min(ratio_keep_count, self.action_max_keep_count)
                
                # 修复：Expansion阶段filtered_actions ≤ 5
                keep_count = min(keep_count, 5)
                
                filtered_actions = [action for action, dist in action_distances[:keep_count]]
                
                # 5. Value + 人工reward混合筛选（使用动作缓存和批量化推理）
                action_values = []
                alpha = 0.5  # value与reward的加权系数
                
                # 收集所有需要评估的next_state_seq，用于批量化推理
                state_seqs_to_evaluate = []
                action_to_eval_idx = {}  # 记录动作id到评估索引的映射
                cache_misses = []  # 记录需要进行物理仿真的动作
                
                for action in filtered_actions:
                    # P1-2优化：使用更高效的action_id
                    action_id = (action['V0'], action['phi'], action['theta'], action['a'], action['b'])
                    
                    if action_id not in action_cache:
                        # 动作不在缓存中，需要进行物理仿真
                        cache_misses.append(action)
                    else:
                        # 动作在缓存中，直接使用缓存结果
                        cached = action_cache[action_id]
                        
                        # 计算next_state_seq
                        next_state_seq = current_state_seq[1:] + [cached['next_state_vec']]
                        state_seqs_to_evaluate.append(next_state_seq)
                        # 使用action_id作为键，而不是action字典
                        action_to_eval_idx[action_id] = len(state_seqs_to_evaluate) - 1
                
                # 对缓存未命中的动作进行物理仿真
                for action in cache_misses:
                    # P0-3优化：在deepcopy前检查超时
                    if time.time() > deadline:
                        time_exceeded = True
                        break
                    
                    # 进行快速物理模拟
                    temp_balls = {bid: copy.deepcopy(ball) for bid, ball in sim_balls.items()}
                    temp_table = copy.deepcopy(sim_table)
                    last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in temp_balls.items()}
                    
                    # P0-3优化：在simulate_action前检查超时
                    if time.time() > deadline:
                        time_exceeded = True
                        break
                    
                    shot = self.simulate_action(temp_balls, temp_table, action)
                    if shot is None:
                        continue
                    
                    # 计算raw_reward
                    reward = self.analyze_shot_for_reward(shot, last_state_snapshot, player_targets[current_player])
                    
                    # 检查是否为黑八违规（即时判负）
                    cue_ball = shot.balls.get('cue')
                    eight_ball = shot.balls.get('8')
                    cue_in_pocket = cue_ball and cue_ball.state.s == 4
                    eight_in_pocket = eight_ball and eight_ball.state.s == 4
                    
                    current_targets = player_targets[current_player]
                    non_eight_targets = [bid for bid in current_targets if bid != '8']
                    has_non_eight_targets_left = any(
                        bid in shot.balls and shot.balls[bid].state.s != 4 for bid in non_eight_targets
                    )
                    
                    is_black_eight_foul = False
                    if (cue_in_pocket and eight_in_pocket) or (eight_in_pocket and has_non_eight_targets_left):
                        # 黑八违规，设置标志但不直接修改reward
                        # reward将在后续被归一化处理
                        is_black_eight_foul = True
                    
                    # 计算next_state_seq
                    new_state_vec = self._balls_state_to_81(
                        shot.balls,
                        my_targets=player_targets[current_player],
                        table=shot.table
                    )
                    
                    # P1-2优化：使用更高效的action_id作为缓存key
                    action_id = (action['V0'], action['phi'], action['theta'], action['a'], action['b'])
                    action_cache[action_id] = {
                        'shot': shot,
                        'raw_reward': reward,
                        'next_state_vec': new_state_vec,
                        'is_black_eight_foul': is_black_eight_foul,
                        'cue_in_pocket': cue_in_pocket,
                        'eight_in_pocket': eight_in_pocket
                    }
                    
                    # 计算next_state_seq用于模型评估
                    next_state_seq = current_state_seq[1:] + [new_state_vec]
                    state_seqs_to_evaluate.append(next_state_seq)
                    # 使用action_id作为键，而不是action字典
                    action_to_eval_idx[action_id] = len(state_seqs_to_evaluate) - 1
                
                # 批量化模型推理
                model_values = {}
                if state_seqs_to_evaluate:
                    # P0-3优化：在模型推理前检查超时
                    if time.time() - start_time > self.max_search_time:
                        time_exceeded = True
                    else:
                        # 将所有状态序列转换为张量并批量化
                        state_tensors = []
                        for seq in state_seqs_to_evaluate:
                            state_tensor = self._state_seq_to_tensor(seq)
                            state_tensors.append(state_tensor)
                        
                        batch_state_tensor = torch.stack(state_tensors, dim=0).to(self.device)
                        
                        with torch.no_grad():
                            out = self.model(batch_state_tensor)
                        
                        batch_value_outputs = out["value_output"].cpu().numpy()
                        
                        # 将批量化结果映射回动作
                for action_id, idx in action_to_eval_idx.items():
                    model_values[action_id] = batch_value_outputs[idx][0]
                
                # 计算混合价值
                for action in filtered_actions:
                    # P1-2优化：使用更高效的action_id
                    action_id = (action['V0'], action['phi'], action['theta'], action['a'], action['b'])
                    
                    if action_id not in action_cache:
                        continue
                    
                    cached = action_cache[action_id]
                    reward = cached['raw_reward']
                    
                    # 检查模型值是否存在
                    if action_id not in model_values:
                        continue
                    
                    # 使用action_id作为键，而不是action字典
                    model_value = model_values[action_id]
                    
                    # ✅ 修复：reward / model_value 视角统一
                    # reward是current_player视角，model_value是root_player视角
                    # 在混合前，必须统一到root_player视角
                    if current_player != root_player:
                        reward = -reward
                    
                    # 归一化reward到[-1, 1]范围
                    if reward > 150:  # 黑八胜利
                        normalized_reward = 1.0
                    elif reward < -500:  # 严重违规
                        normalized_reward = -1.0
                    else:
                        # 线性归一化到[-1, 1]
                        normalized_reward = reward / 500
                    
                    # 计算混合价值
                    blended_value = alpha * model_value + (1 - alpha) * normalized_reward
                    
                    action_values.append((action, blended_value))
                
                # 按混合价值排序，先保留指定比例（action_keep_ratio），再进行top-K裁剪
                action_values.sort(key=lambda x: x[1], reverse=True)
                
                # P0-Value剪枝：相对剪枝（推荐，鲁棒性更强）
                # depth=1时应用相对剪枝，保留top 60-70%
                pruned_action_values = action_values
                if current_depth == 1:
                    # 计算相对剪枝阈值（剪掉bottom X%）
                    values = [v for a, v in action_values]
                    if len(values) > 1:
                        # 保留top (1 - value_relative_prune_ratio)，即剪掉bottom value_relative_prune_ratio
                        threshold = np.percentile(values, self.value_relative_prune_ratio * 100)
                        pruned_action_values = [(a, v) for a, v in action_values if v >= threshold]
                
                # 先计算比例保留数量
                ratio_keep_count = max(1, int(len(pruned_action_values) * self.action_keep_ratio))
                
                # 再取比例保留数量和最大保留数量的最小值
                value_keep_count = min(ratio_keep_count, self.action_max_keep_count)
                
                final_actions = [action for action, value in pruned_action_values[:value_keep_count]]
                
                # 生成动作先验概率：混合价值越高，先验概率越高
                priors = []
                if final_actions:
                    # 计算混合价值的总和作为归一化因子
                    total_value = sum(value for action, value in action_values[:value_keep_count])
                    if total_value == 0:
                        # 如果总和为0，使用均匀分布
                        uniform_prior = 1.0 / len(final_actions)
                        for action in final_actions:
                            priors.append((action, uniform_prior))
                    else:
                        # 混合价值越高，先验概率越高
                        for i, (action, value) in enumerate(action_values[:value_keep_count]):
                            prior = value / total_value
                            priors.append((action, prior))
                else:
                    # 兜底：如果没有筛选出动作，使用原始候选动作的均匀分布
                    uniform_prior = 1.0 / len(candidate_actions)
                    for action in candidate_actions[:5]:  # 最多保留5个
                        priors.append((action, uniform_prior))
                
                node.expand(priors)
            
            # 4. Evaluation: 评估节点价值
            # 传递当前状态序列给_evaluate_leaf，而不是依赖node.state_seq
            eval_state_seq = current_state_seq
            value = self._evaluate_leaf(
                node, sim_balls, sim_table, player_targets, 
                current_player, root_player, current_depth, 
                remaining_hits - current_depth, eval_state_seq
            )
            
            # 5. Backup：更新所有路径节点，只调用路径最后一个节点的backup（内部会递归向上更新）
            path[-1].backup(value)
        
        # 5️⃣ 返回访问次数最多的动作
        if root.children:
            best_child = max(root.children.values(), key=lambda c: c.N)
            best_action = best_child.action
        else:
            # 如果没有子节点，返回第一个候选动作
            best_action = candidate_actions[0]
        
        print(f"[Multi-step MCTS] Best Value: {best_child.Q:.3f} (Visits: {best_child.N}, Sims: {simulation_count}/{current_n_simulations})")
        
        return np.array(
            [
                best_action['V0'],
                best_action['phi'],
                best_action['theta'],
                best_action['a'],
                best_action['b'],
            ],
            dtype=np.float32
        )
    
    def _evaluate_leaf(self, node, balls, table, player_targets, current_player, root_player, depth, remaining_hits, eval_state_seq):
        """评估叶子节点
        
        参数：
            node: 叶子节点
            balls: 球状态字典
            table: 球桌对象
            player_targets: 玩家目标球字典
            current_player: 当前玩家
            root_player: 根玩家，用于统一价值视角
            depth: 当前深度
            remaining_hits: 剩余杆数
            eval_state_seq: 当前评估的状态序列，用于模型评估
            
        返回：
            float: 节点的评估价值，统一为root_player视角
        """
        # ========== 1. 即时判负检测 ==========
        cue_ball = balls.get('cue')
        cue_in_pocket = cue_ball and cue_ball.state.s == 4
        
        eight_ball = balls.get('8')
        eight_in_pocket = eight_ball and eight_ball.state.s == 4
        
        # 使用root_player的目标球进行判负，确保价值视角统一
        root_targets = player_targets[root_player]
        non_eight_targets = [bid for bid in root_targets if bid != '8']
        has_non_eight_targets_left = any(
            bid in balls and balls[bid].state.s != 4 for bid in non_eight_targets
        )
        
        # ✅ 修复：Value语义统一 - 严格使用[-1, 1]范围
        # 从root_player视角判断：如果root_player提前打进黑八或同时打进母球和黑八，则输
        if (cue_in_pocket and eight_in_pocket) or (eight_in_pocket and has_non_eight_targets_left):
            return -1.0  # 输局，严格使用[-1, 1]范围
        
        # 检查是否赢局：黑八合法入袋
        if eight_in_pocket and not has_non_eight_targets_left:
            return 1.0  # 赢局，严格使用[-1, 1]范围
        
        # ========== 2. 深度截断 ==========
        if depth >= remaining_hits:
            state_tensor = self._state_seq_to_tensor(eval_state_seq).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(state_tensor)
            return out["value_output"].item()
        
        # P0-Value剪枝：深度剪枝（AlphaZero风格）
        # 当搜索已经足够深时，继续模拟的边际收益极低
        if depth >= self.max_sim_depth:
            state_tensor = self._state_seq_to_tensor(eval_state_seq).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(state_tensor)
            return out["value_output"].item()
        
        # ========== 3. 模型价值 ==========
        state_tensor = self._state_seq_to_tensor(eval_state_seq).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(state_tensor)
            model_value = out["value_output"].item()
        
        # ========== 4. 物理仿真价值 ==========
        # 这里简化处理，直接返回模型价值
        # 实际应该进行物理仿真并计算奖励
        
        return model_value