import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

def load_match_data(match_dir):
    """加载指定目录下的所有对局数据
    
    Args:
        match_dir: 对局数据目录
    
    Returns:
        list: 对局数据列表
    """
    match_files = [f for f in os.listdir(match_dir) if f.startswith('match_') and f.endswith('.json')]
    match_data_list = []
    
    for match_file in tqdm(match_files, desc="加载对局数据"):
        file_path = os.path.join(match_dir, match_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                match_data = json.load(f)
                match_data['filename'] = match_file  # 记录文件名
                match_data_list.append(match_data)
        except Exception as e:
            print(f"加载文件 {match_file} 时出错: {e}")
    
    return match_data_list

def convert_balls_state_to_feature(balls_state):
    """将球的状态转换为神经网络的输入特征
    
    Args:
        balls_state: 球的状态字典
    
    Returns:
        numpy.ndarray: 56维特征向量（根据架构设计文档）
    """
    # 根据架构设计文档，特征包括：
    # 14个目标球的x,y位置 (28维)
    # 白球的x,y位置和速度 (4维)
    # 目标球的剩余情况 (14维)
    # 游戏状态编码 (10维)
    
    feature = np.zeros(56, dtype=np.float32)
    
    # 处理1-15号球的位置
    for ball_id in range(1, 16):
        if str(ball_id) in balls_state:
            ball = balls_state[str(ball_id)]
            # 检查球是否落袋 (状态s为4)
            if ball.get('s', 0) == 4:
                # 球已落袋，位置设为0
                feature[(ball_id-1)*2] = 0.0
                feature[(ball_id-1)*2 + 1] = 0.0
            else:
                # 球在台面上，记录位置
                feature[(ball_id-1)*2] = ball.get('x', 0.0)
                feature[(ball_id-1)*2 + 1] = ball.get('y', 0.0)
    
    # 处理白球位置和速度
    if 'white' in balls_state:
        white_ball = balls_state['white']
        feature[28] = white_ball.get('x', 0.0)  # 白球x位置
        feature[29] = white_ball.get('y', 0.0)  # 白球y位置
        feature[30] = white_ball.get('vx', 0.0)  # 白球x速度
        feature[31] = white_ball.get('vy', 0.0)  # 白球y速度
    
    # 目标球剩余情况 (14维)
    # 对于1-14号球，检查是否还在台面上
    for ball_id in range(1, 15):
        if str(ball_id) in balls_state:
            ball = balls_state[str(ball_id)]
            # 0表示已落袋，1表示在台面上
            feature[32 + (ball_id-1)] = 0.0 if ball.get('s', 0) == 4 else 1.0
    
    # 游戏状态编码 (10维) - 这里可以进一步完善，目前使用简化版本
    # 暂时将这部分设为0
    
    return feature

def generate_behavior_network_data(match_data_list, output_dir):
    """生成行为网络训练数据
    
    Args:
        match_data_list: 对局数据列表
        output_dir: 输出目录
    
    Returns:
        str: 生成的数据文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    behavior_data = []
    
    for match_data in tqdm(match_data_list, desc="生成行为网络数据"):
        winner = match_data['metadata'].get('winner', 0)
        
        for shot in match_data.get('shots', []):
            # 只记录获胜方的击球动作
            if shot['player'] == winner:
                # 转换特征
                balls_state = shot['pre_state']['balls']
                state_feature = convert_balls_state_to_feature(balls_state)
                
                # 获取动作参数
                action = shot['action']
                
                # 记录行为数据
                behavior_record = {
                    'state': state_feature.tolist(),
                    'action': {
                        'V0': action.get('V0', 0.0),
                        'phi': action.get('phi', 0.0),
                        'theta': action.get('theta', 0.0),
                        'a': action.get('a', 0.0),
                        'b': action.get('b', 0.0)
                    },
                    'match_filename': match_data.get('filename', 'unknown'),
                    'shot_index': shot.get('hit_count', 0)
                }
                
                behavior_data.append(behavior_record)
    
    # 保存数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"behavior_network_data_{timestamp}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(behavior_data, f, indent=2, ensure_ascii=False)
    
    print(f"生成了 {len(behavior_data)} 条行为网络训练数据")
    return output_file

def generate_value_network_data(match_data_list, output_dir):
    """生成价值网络训练数据
    
    Args:
        match_data_list: 对局数据列表
        output_dir: 输出目录
    
    Returns:
        str: 生成的数据文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    value_data = []
    
    for match_data in tqdm(match_data_list, desc="生成价值网络数据"):
        winner = match_data['metadata'].get('winner', 0)
        total_shots = len(match_data.get('shots', []))
        
        for shot_idx, shot in enumerate(match_data.get('shots', [])):
            # 转换特征
            balls_state = shot['pre_state']['balls']
            state_feature = convert_balls_state_to_feature(balls_state)
            
            # 计算胜率和预期剩余步数
            current_player = shot['player']
            shots_remaining = total_shots - shot_idx
            
            # 胜率：1表示当前玩家最终获胜，0表示失败
            win_rate = 1.0 if current_player == winner else 0.0
            
            # 预期剩余步数
            expected_remaining_steps = shots_remaining
            
            # 记录价值数据
            value_record = {
                'state': state_feature.tolist(),
                'win_rate': win_rate,
                'expected_remaining_steps': expected_remaining_steps,
                'match_filename': match_data.get('filename', 'unknown'),
                'shot_index': shot.get('hit_count', 0)
            }
            
            value_data.append(value_record)
    
    # 保存数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"value_network_data_{timestamp}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(value_data, f, indent=2, ensure_ascii=False)
    
    print(f"生成了 {len(value_data)} 条价值网络训练数据")
    return output_file

def process_match_data(match_dir, behavior_output_dir, value_output_dir):
    """处理对局数据并生成训练数据
    
    Args:
        match_dir: 对局数据目录
        behavior_output_dir: 行为网络数据输出目录
        value_output_dir: 价值网络数据输出目录
    
    Returns:
        tuple: (行为网络数据文件路径, 价值网络数据文件路径)
    """
    # 加载对局数据
    match_data_list = load_match_data(match_dir)
    print(f"成功加载 {len(match_data_list)} 局比赛数据")
    
    if not match_data_list:
        print("没有找到有效的对局数据，无法生成训练数据")
        return None, None
    
    # 生成行为网络数据
    behavior_file = generate_behavior_network_data(match_data_list, behavior_output_dir)
    
    # 生成价值网络数据
    value_file = generate_value_network_data(match_data_list, value_output_dir)
    
    return behavior_file, value_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理台球AI对局数据并生成训练数据")
    parser.add_argument('--match_dir', type=str, default="match_data", help="对局数据目录")
    parser.add_argument('--behavior_output_dir', type=str, default="training_data/behavior", help="行为网络数据输出目录")
    parser.add_argument('--value_output_dir', type=str, default="training_data/value", help="价值网络数据输出目录")
    
    args = parser.parse_args()
    
    # 处理数据
    behavior_file, value_file = process_match_data(
        match_dir=args.match_dir,
        behavior_output_dir=args.behavior_output_dir,
        value_output_dir=args.value_output_dir
    )
    
    print("\n处理完成：")
    print(f"行为网络训练数据: {behavior_file}")
    print(f"价值网络训练数据: {value_file}")
