import os
import json
import numpy as np
from tqdm import tqdm
import copy
from datetime import datetime
import torch

# 导入相关模块
from process_raw_match_data import load_match_files, stream_json_file, convert_to_81d_state
from agents import MCTSAgent, BasicAgentPro
from agents.basic_agent_pro import analyze_shot_for_reward


def filter_no_pocketing_states(match_dir, max_states=5000):
    """
    读取对局数据，筛选没有发生进球的局面
    
    参数：
        match_dir: 对局数据目录
        max_states: 最大筛选数量
    
    返回：
        list: 没有发生进球的局面列表，每个元素包含原始shot数据
    """
    match_files = load_match_files(match_dir)
    no_pocketing_states = []
    
    for match_file in tqdm(match_files, desc="Reading match files"):
        if len(no_pocketing_states) >= max_states:
            break
            
        try:
            for match_data in stream_json_file(match_file):
                shots = match_data['shots']
                
                for i, shot in enumerate(shots):
                    if len(no_pocketing_states) >= max_states:
                        break
                        
                    try:
                        # 检查是否有我方进球
                        if 'result' in shot and shot['result']['ME_INTO_POCKET'] == []:
                            # 保存原始shot数据，不进行任何转换
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
    
    return no_pocketing_states


def convert_dict_to_ball_objects(balls_dict):
    """
    将普通字典转换为pooltool的Ball对象字典
    
    参数：
        balls_dict: 普通字典，包含球的x, y, z, vx, vy, vz, s等字段
    
    返回：
        dict: 包含Ball对象的字典
    """
    import pooltool as pt
    
    balls = {}
    for ball_id, ball_data in balls_dict.items():
        if ball_id == 'cue':
            ball = pt.CueBall.make()
        else:
            ball = pt.ObjectBall.make(ball_id)
        
        # 设置球的位置和状态
        ball.state.rvw[0][0] = ball_data['x']
        ball.state.rvw[0][1] = ball_data['y']
        ball.state.rvw[0][2] = ball_data['z']
        ball.state.rvw[1][0] = ball_data.get('vx', 0.0)
        ball.state.rvw[1][1] = ball_data.get('vy', 0.0)
        ball.state.rvw[1][2] = ball_data.get('vz', 0.0)
        ball.state.s = ball_data['s']
        
        balls[ball_id] = ball
    
    return balls


def evaluate_states_with_basic_agent(states, threshold=0.3):
    """
    使用basic_agent_pro对状态进行评估，筛选出value波动较大的状态
    
    参数：
        states: 状态列表
        threshold: value波动阈值
    
    返回：
        list: 筛选后的状态列表
    """
    # 初始化BasicAgentPro
    basic_agent = BasicAgentPro(n_simulations=50, c_puct=1.414)
    
    filtered_states = []
    
    for state_data in tqdm(states, desc="Evaluating states with basic agent"):
        shot = state_data['shot']
        pre_state = shot['pre_state']
        
        try:
            # 将普通字典转换为Ball对象字典
            balls_dict = pre_state['balls']
            balls = convert_dict_to_ball_objects(balls_dict)
            my_targets = pre_state['my_targets']
            
            # 创建临时球桌对象
            import pooltool as pt
            table = pt.Table.default()
            
            # 使用BasicAgentPro的决策方法获取最佳动作
            action = basic_agent.decision(balls=balls, my_targets=my_targets, table=table)
            
            # 生成候选动作
            candidate_actions = basic_agent.generate_heuristic_actions(balls, my_targets, table)
            
            # 对每个候选动作进行评估
            values = []
            for candidate_action in candidate_actions:
                # 使用basic_agent的simulate_action方法
                simulated_shot = basic_agent.simulate_action(balls, table, candidate_action)
                
                # 计算奖励
                if simulated_shot is None:
                    raw_reward = -500.0
                else:
                    # 使用独立的analyze_shot_for_reward函数
                    raw_reward = analyze_shot_for_reward(simulated_shot, balls, my_targets)
                
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
                if fluctuation >= threshold:
                    filtered_states.append({
                        **state_data,
                        'balls': balls,
                        'table': table,
                        'fluctuation': fluctuation,
                        'values': values,
                        'max_value': max_value,
                        'min_value': min_value
                    })
                    
        except Exception as e:
            print(f"Error evaluating state: {e}")
            continue
    
    # 按波动从大到小排序
    filtered_states.sort(key=lambda x: x['fluctuation'], reverse=True)
    
    # 最多返回500条
    return filtered_states[:500]


def run_mcts_on_states(states, model_path, n_simulations=50):
    """
    对状态进行MCTS搜索，获取最佳动作和对应的value
    
    参数：
        states: 状态列表
        model_path: 模型路径
        n_simulations: MCTS模拟次数
    
    返回：
        list: 包含MCTS结果的状态列表
    """
    # 初始化MCTSAgent，它会自动处理MCTS交互
    mcts_agent = MCTSAgent()
    mcts_agent.load_model(model_path)
    mcts_agent.set_simulations(n_simulations)
    
    # 处理每个状态
    result_states = []
    
    for state_data in tqdm(states, desc="Running MCTS on states"):
        try:
            # 直接使用之前已经转换好的Ball对象字典
            balls = state_data['balls']
            my_targets = state_data['my_targets']
            table = state_data['table']
            
            # 使用MCTSAgent的decision方法获取最佳动作
            best_action = mcts_agent.decision(
                balls_state=balls,
                my_targets=my_targets,
                table=table
            )
            
            # 将动作转换为numpy数组
            best_action_array = np.array([
                best_action['V0'],
                best_action['phi'],
                best_action['theta'],
                best_action['a'],
                best_action['b']
            ])
            
            # 重新计算最佳动作的value
            # 使用之前的BasicAgentPro和analyze_shot_for_reward函数
            basic_agent = BasicAgentPro(n_simulations=50, c_puct=1.414)
            
            # 模拟动作
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            shot = basic_agent.simulate_action(balls, table, best_action)
            
            # 计算奖励
            if shot is None:
                best_value = 0.0
            else:
                raw_reward = analyze_shot_for_reward(shot, last_state_snapshot, my_targets)
                best_value = (raw_reward - (-500)) / 650.0
                best_value = np.clip(best_value, 0.0, 1.0)
            
            # 添加到结果列表
            result_states.append({
                **state_data,
                'best_action': best_action_array.tolist(),
                'best_value': best_value
            })
            
        except Exception as e:
            print(f"Error running MCTS on state: {e}")
            continue
    
    return result_states


def save_correction_states(states, output_file):
    """
    保存纠错局面
    
    参数：
        states: 包含MCTS结果的状态列表
        output_file: 输出文件路径
    """
    # 准备数据，与process_raw_match_data.py兼容的格式
    correction_data = {
        'metadata': {
            'winner': 'A',  # 默认赢家，实际使用时会被替换
            'start_time': datetime.now().isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_shots': len(states)
        },
        'player_targets': {
            'A': ['1', '2', '3', '4', '5', '6', '7'],
            'B': ['9', '10', '11', '12', '13', '14', '15']
        },
        'shots': []
    }
    
    # 添加每个状态
    for i, state_data in enumerate(states):
        shot = state_data['shot']
        pre_state = shot['pre_state']
        
        # 添加shot，与原始数据格式匹配
        correction_data['shots'].append({
            'hit_count': i + 1,
            'player': 'A',
            'pre_state': pre_state,
            'action': {
                'V0': state_data['best_action'][0],
                'phi': state_data['best_action'][1],
                'theta': state_data['best_action'][2],
                'a': state_data['best_action'][3],
                'b': state_data['best_action'][4]
            },
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
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(correction_data, f, ensure_ascii=False, indent=2)
    
    print(f"Correction states saved to {output_file}")


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
    
    # 1. 筛选没有发生进球的局面
    print("Step 1: Filtering no pocketing states...")
    no_pocketing_states = filter_no_pocketing_states(args.match_dir, args.max_states)
    print(f"Found {len(no_pocketing_states)} no pocketing states")
    
    # 2. 使用basic_agent对状态进行评估，筛选出value波动较大的状态
    print("Step 2: Evaluating states with basic agent...")
    filtered_states = evaluate_states_with_basic_agent(no_pocketing_states, args.threshold)
    print(f"Found {len(filtered_states)} states with value fluctuation >= {args.threshold}")
    
    # 3. 对筛选后的状态进行MCTS搜索
    print("Step 3: Running MCTS on filtered states...")
    mcts_results = run_mcts_on_states(filtered_states, args.model_path, args.n_simulations)
    print(f"Completed MCTS on {len(mcts_results)} states")
    
    # 4. 保存纠错局面
    print("Step 4: Saving correction states...")
    save_correction_states(mcts_results, args.output_file)
    print("Error driven MCTS distillation completed!")


if __name__ == '__main__':
    main()
