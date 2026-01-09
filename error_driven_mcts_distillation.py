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
from agents.basic_agent_pro import BasicAgentPro, analyze_shot_for_reward
from MCTS import MCTS
from dual_network import DualNetwork


def filter_no_pocketing_states(match_dir, max_states=5000):
    """
    读取对局数据，筛选没有发生进球的局面
    
    参数：
        match_dir: 对局数据目录
        max_states: 最大筛选数量
    
    返回：
        list: 没有发生进球的局面列表，每个元素包含state、balls、my_targets、table
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
                        pre_state = shot['pre_state']
                        post_state = shot['post_state']
                        
                        # 检查是否有进球
                        pre_balls = pre_state['balls']
                        post_balls = post_state['balls']
                        
                        # 检查是否有球从非进袋状态变为进袋状态
                        has_pocketing = False
                        for ball_id, post_ball in post_balls.items():
                            if ball_id in pre_balls:
                                pre_ball = pre_balls[ball_id]
                                if pre_ball['s'] != 4 and post_ball['s'] == 4:
                                    has_pocketing = True
                                    break
                        
                        if not has_pocketing:
                            # 创建临时球桌对象用于后续评估
                            table = pt.PocketTable(model="7_foot")
                            
                            # 转换为pooltool球对象
                            balls = {}
                            for ball_id, ball_data in pre_state['balls'].items():
                                if ball_id == 'cue':
                                    ball = pt.CueBall.make()
                                else:
                                    ball = pt.ObjectBall.make(ball_id)
                                ball.state.rvw[0] = [ball_data['x'], ball_data['y'], ball_data['z']]
                                ball.state.s = ball_data['s']
                                balls[ball_id] = ball
                            
                            # 添加到列表
                            no_pocketing_states.append({
                                'state': convert_to_81d_state(pre_state['balls'], pre_state['my_targets']),
                                'balls': balls,
                                'my_targets': pre_state['my_targets'],
                                'table': table
                            })
                            
                    except Exception as e:
                        print(f"Error processing shot {i} in file {match_file}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error processing file {match_file}: {e}")
            continue
    
    return no_pocketing_states


def evaluate_states_with_basic_agent(states, agent, threshold=0.3):
    """
    使用basic_agent_pro对状态进行评估，筛选出value波动较大的状态
    
    参数：
        states: 状态列表
        agent: BasicAgentPro实例
        threshold: value波动阈值
    
    返回：
        list: 筛选后的状态列表，每个元素包含原始状态和评估结果
    """
    filtered_states = []
    
    for state_data in tqdm(states, desc="Evaluating states with basic agent"):
        balls = state_data['balls']
        my_targets = state_data['my_targets']
        table = state_data['table']
        
        try:
            # 生成候选动作
            candidate_actions = agent.generate_heuristic_actions(balls, my_targets, table)
            
            # 对每个候选动作进行评估
            values = []
            for action in candidate_actions:
                # 模拟动作
                last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                shot = agent.simulate_action(balls, table, action)
                
                # 计算奖励
                if shot is None:
                    raw_reward = -500.0
                else:
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
                if fluctuation >= threshold:
                    filtered_states.append({
                        **state_data,
                        'values': values,
                        'fluctuation': fluctuation,
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
    # 加载模型
    model = DualNetwork()
    model.load(model_path)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    
    # 初始化MCTS
    mcts = MCTS(
        model=model,
        n_simulations=n_simulations,
        c_puct=1.414,
        max_depth=4,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    # 处理每个状态
    result_states = []
    
    for state_data in tqdm(states, desc="Running MCTS on states"):
        balls = state_data['balls']
        my_targets = state_data['my_targets']
        table = state_data['table']
        
        try:
            # 构建状态序列
            state_seq = [state_data['state'], state_data['state'], state_data['state']]
            
            # 设置玩家目标球字典
            player_targets = {'A': my_targets}
            
            # 执行MCTS搜索
            best_action = mcts.search(state_seq, balls, table, player_targets, 'A')
            
            # 获取最佳动作对应的value
            # 我们需要修改MCTS.search方法来返回value，或者重新计算
            # 这里我们重新计算最佳动作的value
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            shot = mcts.simulate_action(balls, table, {
                'V0': best_action[0],
                'phi': best_action[1],
                'theta': best_action[2],
                'a': best_action[3],
                'b': best_action[4]
            })
            
            if shot is None:
                raw_reward = -500.0
            else:
                raw_reward = mcts.analyze_shot_for_reward(shot, last_state_snapshot, my_targets)
            
            normalized_reward = (raw_reward - (-500)) / 650.0
            normalized_reward = np.clip(normalized_reward, 0.0, 1.0)
            
            # 添加到结果列表
            result_states.append({
                **state_data,
                'best_action': best_action.tolist(),
                'best_value': normalized_reward
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
    # 准备数据
    correction_data = {
        'metadata': {
            'type': 'error_driven_mcts_distillation',
            'timestamp': datetime.now().isoformat(),
            'count': len(states)
        },
        'shots': []
    }
    
    # 添加每个状态
    for i, state_data in enumerate(states):
        # 转换balls回原始格式
        balls_dict = {}
        for ball_id, ball in state_data['balls'].items():
            balls_dict[ball_id] = {
                'x': ball.state.rvw[0][0],
                'y': ball.state.rvw[0][1],
                'z': ball.state.rvw[0][2],
                's': ball.state.s
            }
        
        # 添加shot
        correction_data['shots'].append({
            'player': 'A',
            'pre_state': {
                'balls': balls_dict,
                'my_targets': state_data['my_targets']
            },
            'action': {
                'V0': state_data['best_action'][0],
                'phi': state_data['best_action'][1],
                'theta': state_data['best_action'][2],
                'a': state_data['best_action'][3],
                'b': state_data['best_action'][4]
            },
            'value': state_data['best_value']
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
    
    # 2. 初始化BasicAgentPro
    basic_agent = BasicAgentPro(n_simulations=50, c_puct=1.414)
    
    # 3. 使用basic_agent对状态进行评估，筛选出value波动较大的状态
    print("Step 2: Evaluating states with basic agent...")
    filtered_states = evaluate_states_with_basic_agent(no_pocketing_states, basic_agent, args.threshold)
    print(f"Found {len(filtered_states)} states with value fluctuation >= {args.threshold}")
    
    # 4. 对筛选后的状态进行MCTS搜索
    print("Step 3: Running MCTS on filtered states...")
    mcts_results = run_mcts_on_states(filtered_states, args.model_path, args.n_simulations)
    print(f"Completed MCTS on {len(mcts_results)} states")
    
    # 5. 保存纠错局面
    print("Step 4: Saving correction states...")
    save_correction_states(mcts_results, args.output_file)
    print("Error driven MCTS distillation completed!")


if __name__ == '__main__':
    main()
