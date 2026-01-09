import os
import json
import numpy as np
from tqdm import tqdm
import copy
import pooltool as pt
from datetime import datetime
import torch

# 导入相关模块
from process_raw_match_data import load_match_files, stream_json_file, convert_to_81d_state
from agents import MCTSAgent, BasicAgentPro
from agents.basic_agent_pro import analyze_shot_for_reward


def create_pooltool_objects(balls_dict, table_type="7_foot"):
    """
    创建pooltool对象（球桌和球）
    
    参数：
        balls_dict: 球状态字典，来自对局数据
        table_type: 球桌类型
    
    返回：
        tuple: (table, balls)，球桌对象和球对象字典
    """
    # 创建球桌
    table = pt.Table.default()
    
    # 创建球对象
    balls = {}
    for ball_id, ball_data in balls_dict.items():
        # 使用正确的Ball.create()方法创建球对象
        ball = pt.Ball.create(ball_id)
        
        # 设置球的位置和状态
        ball.state.rvw[0] = [ball_data['x'], ball_data['y'], ball_data['z']]
        ball.state.rvw[1] = [ball_data.get('vx', 0.0), ball_data.get('vy', 0.0), ball_data.get('vz', 0.0)]
        ball.state.s = ball_data['s']
        
        balls[ball_id] = ball
    
    return table, balls


def simulate_action(balls, table, action, noise_std=None):
    """
    执行带噪声的物理仿真
    
    参数：
        balls: 球对象字典
        table: 球桌对象
        action: 动作字典
        noise_std: 噪声标准差字典
    
    返回：
        shot: 仿真结果，pt.System对象
    """
    # 深拷贝，避免修改原始对象
    sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
    sim_table = copy.deepcopy(table)
    
    # 创建球杆并设置状态
    cue = pt.Cue(cue_ball_id="cue")
    
    # 添加噪声
    if noise_std:
        noisy_V0 = np.clip(action['V0'] + np.random.normal(0, noise_std['V0']), 0.5, 8.0)
        noisy_phi = (action['phi'] + np.random.normal(0, noise_std['phi'])) % 360
        noisy_theta = np.clip(action['theta'] + np.random.normal(0, noise_std['theta']), 0, 90)
        noisy_a = np.clip(action['a'] + np.random.normal(0, noise_std['a']), -0.5, 0.5)
        noisy_b = np.clip(action['b'] + np.random.normal(0, noise_std['b']), -0.5, 0.5)
        
        cue.set_state(V0=noisy_V0, phi=noisy_phi, theta=noisy_theta, a=noisy_a, b=noisy_b)
    else:
        cue.set_state(V0=action['V0'], phi=action['phi'], theta=action['theta'], a=action['a'], b=action['b'])
    
    # 创建system并执行模拟
    shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
    pt.simulate(shot, inplace=True)
    
    return shot


def evaluate_state(balls, table, my_targets, basic_agent, threshold=0.3):
    """
    评估状态，计算value波动
    
    参数：
        balls: 球对象字典
        table: 球桌对象
        my_targets: 目标球列表
        basic_agent: BasicAgentPro实例
        threshold: value波动阈值
    
    返回：
        tuple: (has_high_fluctuation, fluctuation, values)，是否有高波动、波动值、values列表
    """
    # 生成候选动作
    candidate_actions = basic_agent.generate_heuristic_actions(balls, my_targets, table)
    
    # 对每个候选动作进行评估
    values = []
    for action in candidate_actions:
        # 模拟动作
        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        shot = simulate_action(balls, table, action)
        
        # 计算奖励
        raw_reward = analyze_shot_for_reward(shot, last_state_snapshot, my_targets)
        
        # 归一化奖励
        normalized_reward = (raw_reward - (-500)) / 650.0
        normalized_reward = np.clip(normalized_reward, 0.0, 1.0)
        values.append(normalized_reward)
    
    if len(values) > 0:
        # 计算value波动
        max_value = max(values)
        min_value = min(values)
        fluctuation = max_value - min_value
        
        # 检查波动是否达到阈值
        return fluctuation >= threshold, fluctuation, values
    
    return False, 0.0, []


def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Error driven MCTS distillation")
    parser.add_argument('--match_dir', type=str, default='match_data', help='Directory containing match data files')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--output_file', type=str, default='correction_states.json', help='Output file path for correction states')
    parser.add_argument('--max_states', type=int, default=5000, help='Maximum number of states to filter from match data')
    parser.add_argument('--threshold', type=float, default=0.3, help='Value fluctuation threshold')
    parser.add_argument('--n_simulations', type=int, default=50, help='Number of MCTS simulations')
    
    args = parser.parse_args()
    
    # 1. 读取对局数据，筛选没有发生进球的局面
    print("Step 1: Reading match data and filtering no pocketing states...")
    match_files = load_match_files(args.match_dir)
    no_pocketing_states = []
    
    for match_file in tqdm(match_files, desc="Reading match files"):
        if len(no_pocketing_states) >= args.max_states:
            break
            
        try:
            for match_data in stream_json_file(match_file):
                shots = match_data['shots']
                
                for i, shot in enumerate(shots):
                    if len(no_pocketing_states) >= args.max_states:
                        break
                        
                    try:
                        # 检查是否有我方进球
                        if 'result' in shot and shot['result']['ME_INTO_POCKET'] == []:
                            # 保存原始shot数据
                            no_pocketing_states.append({
                                'shot': shot,
                                'match_data': {
                                    'metadata': match_data['metadata'],
                                    'player_targets': match_data['player_targets']
                                }
                            })
                            
                    except Exception as e:
                        print(f"Error processing shot {i} in file {match_file}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error processing file {match_file}: {e}")
            continue
    
    print(f"Found {len(no_pocketing_states)} no pocketing states")
    
    # 2. 初始化BasicAgentPro和MCTSAgent
    print("Step 2: Initializing agents...")
    basic_agent = BasicAgentPro(n_simulations=50, c_puct=1.414)
    
    mcts_agent = MCTSAgent()
    mcts_agent.load_model(args.model_path)
    mcts_agent.set_simulations(args.n_simulations)
    
    # 3. 评估状态，筛选出value波动较大的状态
    print("Step 3: Evaluating states and filtering high fluctuation states...")
    high_fluctuation_states = []
    
    for state_data in tqdm(no_pocketing_states, desc="Evaluating states"):
        shot = state_data['shot']
        pre_state = shot['pre_state']
        
        try:
            # 创建pooltool对象
            table, balls = create_pooltool_objects(pre_state['balls'])
            
            # 评估状态
            has_high_fluctuation, fluctuation, values = evaluate_state(
                balls, table, pre_state['my_targets'], basic_agent, args.threshold
            )
            
            if has_high_fluctuation:
                high_fluctuation_states.append({
                    'shot': shot,
                    'table': table,
                    'balls': balls,
                    'my_targets': pre_state['my_targets'],
                    'fluctuation': fluctuation,
                    'values': values
                })
                
        except Exception as e:
            print(f"Error evaluating state: {e}")
            continue
    
    # 按波动从大到小排序，最多保留500条
    high_fluctuation_states.sort(key=lambda x: x['fluctuation'], reverse=True)
    high_fluctuation_states = high_fluctuation_states[:500]
    
    print(f"Found {len(high_fluctuation_states)} states with value fluctuation >= {args.threshold}")
    
    # 4. 使用MCTSAgent获取最佳动作
    print("Step 4: Running MCTS to get best actions...")
    correction_states = []
    
    for state_data in tqdm(high_fluctuation_states, desc="Running MCTS"):
        try:
            # 使用MCTSAgent获取最佳动作
            best_action = mcts_agent.decision(
                balls_state=state_data['balls'],
                my_targets=state_data['my_targets'],
                table=state_data['table']
            )
            
            # 模拟最佳动作，获取胜率
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in state_data['balls'].items()}
            shot = simulate_action(state_data['balls'], state_data['table'], best_action)
            raw_reward = analyze_shot_for_reward(shot, last_state_snapshot, state_data['my_targets'])
            best_value = (raw_reward - (-500)) / 650.0
            best_value = np.clip(best_value, 0.0, 1.0)
            
            correction_states.append({
                'shot': state_data['shot'],
                'best_action': best_action,
                'best_value': best_value
            })
            
        except Exception as e:
            print(f"Error running MCTS on state: {e}")
            continue
    
    print(f"Completed MCTS on {len(correction_states)} states")
    
    # 5. 保存纠错局面
    print("Step 5: Saving correction states...")
    
    # 准备数据，与process_raw_match_data.py兼容的格式
    correction_data = {
        'metadata': {
            'winner': 'A',  # 默认赢家，实际使用时会被替换
            'start_time': datetime.now().isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_shots': len(correction_states)
        },
        'player_targets': {
            'A': ['1', '2', '3', '4', '5', '6', '7'],
            'B': ['9', '10', '11', '12', '13', '14', '15']
        },
        'shots': []
    }
    
    # 添加每个状态
    for i, state_data in enumerate(correction_states):
        shot = state_data['shot']
        pre_state = shot['pre_state']
        
        # 添加shot，与原始数据格式匹配
        correction_data['shots'].append({
            'hit_count': i + 1,
            'player': 'A',
            'pre_state': pre_state,
            'action': state_data['best_action'],
            'result': {
                'ME_INTO_POCKET': [],  # 假设没有进球
                'ENEMY_INTO_POCKET': [],
                'WHITE_BALL_INTO_POCKET': False,
                'BLACK_BALL_INTO_POCKET': False,
                'FOUL_FIRST_HIT': False,
                'NO_POCKET_NO_RAIL': False,
                'NO_HIT': False
            },
            'decision_time': 0.0  # 决策时间
        })
    
    # 保存文件
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(correction_data, f, ensure_ascii=False, indent=2)
    
    print(f"Correction states saved to {args.output_file}")
    print("Error driven MCTS distillation completed!")


if __name__ == '__main__':
    main()