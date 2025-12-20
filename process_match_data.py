import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

def load_single_match_data(file_path):
    """加载单个对局数据文件
    
    Args:
        file_path: 对局数据文件路径
    
    Returns:
        dict or None: 对局数据字典，如果加载失败则返回None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            match_data = json.load(f)
            match_data['filename'] = os.path.basename(file_path)  # 记录文件名
            return match_data
    except Exception as e:
        print(f"加载文件 {os.path.basename(file_path)} 时出错: {e}")
        return None

def get_match_files(match_dir):
    """获取指定目录下的所有对局数据文件
    
    Args:
        match_dir: 对局数据目录
    
    Returns:
        list: 对局文件路径列表
    """
    match_files = [f for f in os.listdir(match_dir) if f.startswith('match_') and f.endswith('.json')]
    return [os.path.join(match_dir, f) for f in match_files]

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

def process_single_match_for_behavior(match_data, output_file, is_first_match=False, is_last_match=False):
    """处理单个对局数据并实时写入行为网络数据文件
    
    Args:
        match_data: 单个对局数据
        output_file: 输出文件路径
        is_first_match: 是否是第一个对局
        is_last_match: 是否是最后一个对局
    
    Returns:
        int: 生成的行为网络数据条数
    """
    behavior_records = []
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
            
            behavior_records.append(behavior_record)
    
    try:
        # 根据是否是第一个或最后一个对局，采用不同的写入方式
        if is_first_match:
            # 第一个对局，写入JSON数组开头和数据
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('[')
                if behavior_records:
                    f.write(json.dumps(behavior_records[0], ensure_ascii=False))
                    for record in behavior_records[1:]:
                        f.write(',\n' + json.dumps(record, ensure_ascii=False))
        elif behavior_records:
            # 不是第一个对局，追加数据
            with open(output_file, 'a', encoding='utf-8') as f:
                # 先写入逗号分隔符
                f.write(',\n')
                # 写入第一条记录
                f.write(json.dumps(behavior_records[0], ensure_ascii=False))
                # 写入后续记录
                for record in behavior_records[1:]:
                    f.write(',\n' + json.dumps(record, ensure_ascii=False))
        
        # 如果是最后一个对局，写入JSON数组结束标记
        if is_last_match:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(']')
                
    except Exception as e:
        print(f"保存行为数据出错: {e}")
    
    return len(behavior_records)

def process_single_match_for_value(match_data, output_file, is_first_match=False, is_last_match=False):
    """处理单个对局数据并实时写入价值网络数据文件
    
    Args:
        match_data: 单个对局数据
        output_file: 输出文件路径
        is_first_match: 是否是第一个对局
        is_last_match: 是否是最后一个对局
    
    Returns:
        int: 生成的价值网络数据条数
    """
    value_records = []
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
        
        value_records.append(value_record)
    
    try:
        # 根据是否是第一个或最后一个对局，采用不同的写入方式
        if is_first_match:
            # 第一个对局，写入JSON数组开头和数据
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('[')
                if value_records:
                    f.write(json.dumps(value_records[0], ensure_ascii=False))
                    for record in value_records[1:]:
                        f.write(',\n' + json.dumps(record, ensure_ascii=False))
        elif value_records:
            # 不是第一个对局，追加数据
            with open(output_file, 'a', encoding='utf-8') as f:
                # 先写入逗号分隔符
                f.write(',\n')
                # 写入第一条记录
                f.write(json.dumps(value_records[0], ensure_ascii=False))
                # 写入后续记录
                for record in value_records[1:]:
                    f.write(',\n' + json.dumps(record, ensure_ascii=False))
        
        # 如果是最后一个对局，写入JSON数组结束标记
        if is_last_match:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(']')
                
    except Exception as e:
        print(f"保存价值数据出错: {e}")
    
    return len(value_records)

def process_match_data(match_dir, behavior_output_dir, value_output_dir):
    """处理对局数据并生成训练数据（逐局处理和实时保存）
    
    Args:
        match_dir: 对局数据目录
        behavior_output_dir: 行为网络数据输出目录
        value_output_dir: 价值网络数据输出目录
    
    Returns:
        tuple: (行为网络数据文件路径, 价值网络数据文件路径)
    """
    # 确保输出目录存在
    os.makedirs(behavior_output_dir, exist_ok=True)
    os.makedirs(value_output_dir, exist_ok=True)
    
    # 获取对局文件列表
    match_files = get_match_files(match_dir)
    total_files = len(match_files)
    
    if total_files == 0:
        print("没有找到有效的对局数据文件，无法生成训练数据")
        return None, None
    
    print(f"找到 {total_files} 个对局数据文件")
    
    # 创建输出文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    behavior_file = os.path.join(behavior_output_dir, f"behavior_network_data_{timestamp}.json")
    value_file = os.path.join(value_output_dir, f"value_network_data_{timestamp}.json")
    
    # 逐局处理数据（实时保存）
    total_behavior_records = 0
    total_value_records = 0
    
    # 设置打印频率，避免过于频繁的进度更新
    print_frequency = max(1, total_files // 10)  # 最多打印10次详细进度
    
    for i, file_path in enumerate(tqdm(match_files, desc="逐局处理数据")):
        filename = os.path.basename(file_path)
        
        # 每处理一定数量的文件或最后一个文件时显示详细进度
        if i % print_frequency == 0 or i == total_files - 1:
            print(f"处理进度: {i+1}/{total_files}，正在处理文件: {filename}")
            print(f"已处理行为数据: {total_behavior_records} 条，已处理价值数据: {total_value_records} 条")
        
        # 加载单个对局数据
        match_data = load_single_match_data(file_path)
        if match_data is None:
            continue
        
        # 判断是否是第一个或最后一个对局
        is_first_match = (i == 0)
        is_last_match = (i == total_files - 1)
        
        # 处理行为网络数据
        behavior_count = process_single_match_for_behavior(match_data, behavior_file, is_first_match, is_last_match)
        total_behavior_records += behavior_count
        
        # 处理价值网络数据
        value_count = process_single_match_for_value(match_data, value_file, is_first_match, is_last_match)
        total_value_records += value_count
    
    print(f"成功处理 {total_files} 局比赛数据")
    print(f"生成了 {total_behavior_records} 条行为网络训练数据")
    print(f"生成了 {total_value_records} 条价值网络训练数据")
    
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