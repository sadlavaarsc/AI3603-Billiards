import math
import numpy as np
import pooltool as pt
import copy
import torch
from data_loader import StatePreprocessor



# ============ MCTS Node ============
class MCTSNode:
    def __init__(self, state_seq, parent=None, prior=1.0, action=None):
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
            if action not in self.children:
                self.children[action] = MCTSNode(
                    state_seq=self.state_seq,  # 子节点继承父节点的状态序列
                    parent=self,
                    prior=prior,
                    action=action
                )
        self.is_expanded = True
    
    def backup(self, value):
        """备份价值，更新节点的访问次数和价值"""
        node = self
        while node is not None:
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
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
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.max_search_time = max_search_time  # 搜索限时，单位：秒
        self.action_keep_ratio = action_keep_ratio  # 动作保留比例，用于控制保留的动作数量
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
        
        # 创建根节点
        root = MCTSNode(state_seq)
        
        # 1️⃣ 启发式生成候选动作
        candidate_actions = self.generate_heuristic_actions(balls, player_targets[root_player], table)
        
        # 2️⃣ policy网络筛选和先验概率生成
        state_tensor = self._state_seq_to_tensor(root.state_seq).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(state_tensor)
            value_output = out["value_output"].item()
        
        # 生成动作先验概率（简单使用均匀分布）
        action_priors = []
        if candidate_actions:
            uniform_prior = 1.0 / len(candidate_actions)
            for action in candidate_actions:
                action_priors.append((action, uniform_prior))
        
        # 扩展根节点
        root.expand(action_priors)
        
        # 模拟次数计数器
        simulation_count = 0
        time_exceeded = False
        
        # 3️⃣ 多次模拟：选择、扩展+评估、备份
        for _ in range(current_n_simulations):
            # 检查是否超过搜索时间限制
            elapsed_time = time.time() - start_time
            if elapsed_time > self.max_search_time:
                time_exceeded = True
                print(f"[Multi-step MCTS] 搜索时间超过限制 ({elapsed_time:.2f}s > {self.max_search_time}s)，提前终止搜索")
                break
            
            simulation_count += 1
            
            # Selection
            node = root
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            sim_table = copy.deepcopy(table)
            current_player = root_player
            current_depth = 0
            path = [node]
            
            while node.is_expanded and node.children and current_depth < current_max_depth:
                # 选择UCB分数最高的子节点
                node = node.select_child(self.c_puct)
                path.append(node)
                
                # 执行动作，获取下一个状态
                last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in sim_balls.items()}
                shot = self.simulate_action(sim_balls, sim_table, node.action)
                
                if shot is None:
                    break
                
                # 更新模拟状态
                sim_balls = shot.balls
                sim_table = shot.table
                
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
                
                # 首球犯规判定
                foul_first_hit = (first_contact is None or first_contact not in player_targets[current_player])
                
                # 2. 计算碰库犯规
                cue_hit_cushion = any(
                    'cushion' in str(e.event_type).lower() and 'cue' in getattr(e, 'ids', [])
                    for e in shot.events
                )
                
                new_pocketed = [
                    bid for bid in shot.balls
                    if shot.balls[bid].state.s == 4 and last_state_snapshot[bid].state.s != 4
                ]
                own_pocketed = [bid for bid in new_pocketed if bid in player_targets[current_player]]
                cue_pocketed = 'cue' in new_pocketed
                
                # 碰库犯规：没有进球且没有碰到库边
                foul_no_rail = (len(new_pocketed) == 0 and not cue_hit_cushion)
                
                # 3. 判断是否需要切换玩家
                is_foul = cue_pocketed or foul_first_hit or foul_no_rail
                switch_player = is_foul or len(own_pocketed) == 0
                
                # 更新玩家
                if switch_player:
                    current_player = 'B' if current_player == 'A' else 'A'
                
                current_depth += 1
            
            # Expansion + Evaluation
            value = self._evaluate_leaf(node, sim_balls, sim_table, player_targets, current_player, current_depth, remaining_hits - current_depth)
            
            # 如果当前节点不是根节点，且当前玩家不是根玩家，反转value
            if current_player != root_player:
                value = -value
            
            # 4️⃣ Backup
            for node in path:
                node.backup(value)
        
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
    
    def _evaluate_leaf(self, node, balls, table, player_targets, current_player, depth, remaining_hits):
        """评估叶子节点
        
        参数：
            node: 叶子节点
            balls: 球状态字典
            table: 球桌对象
            player_targets: 玩家目标球字典
            current_player: 当前玩家
            depth: 当前深度
            remaining_hits: 剩余杆数
            
        返回：
            float: 节点的评估价值
        """
        # ========== 1. 即时判负检测 ==========
        cue_ball = balls.get('cue')
        cue_in_pocket = cue_ball and cue_ball.state.s == 4
        
        eight_ball = balls.get('8')
        eight_in_pocket = eight_ball and eight_ball.state.s == 4
        
        my_targets = player_targets[current_player]
        non_eight_targets = [bid for bid in my_targets if bid != '8']
        has_non_eight_targets_left = any(
            bid in balls and balls[bid].state.s != 4 for bid in non_eight_targets
        )
        
        # 即时判负
        if (cue_in_pocket and eight_in_pocket) or (eight_in_pocket and has_non_eight_targets_left):
            return 0.0
        
        # ========== 2. 深度截断 ==========
        if depth >= remaining_hits:
            state_tensor = self._state_seq_to_tensor(node.state_seq).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(state_tensor)
            return out["value_output"].item()
        
        # ========== 3. 模型价值 ==========
        state_tensor = self._state_seq_to_tensor(node.state_seq).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(state_tensor)
            model_value = out["value_output"].item()
        
        # ========== 4. 物理仿真价值 ==========
        # 这里简化处理，直接返回模型价值
        # 实际应该进行物理仿真并计算奖励
        return model_value