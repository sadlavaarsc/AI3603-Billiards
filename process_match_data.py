import os
import json
import argparse
import numpy as np
import time
from tqdm import tqdm
from datetime import datetime

def load_single_match_data(file_path):
    """加载单个对局数据文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"加载对局数据文件 {file_path} 出错: {e}")
        return None

def get_match_files(match_dir):
    """获取指定目录下的所有对局数据文件"""
    if not os.path.exists(match_dir):
        print(f"对局数据目录不存在: {match_dir}")
        return []
    
    # 获取所有以'match_'开头且以'.json'结尾的文件
    match_files = []
    for filename in os.listdir(match_dir):
        if filename.startswith('match_') and filename.endswith('.json'):
            # 尝试从文件名提取ID
            try:
                # 处理格式为match_000001.json的文件
                if filename.count('_') == 1 and '.' in filename:
                    id_part = filename.split('_')[1].split('.')[0]
                    if id_part.isdigit():
                        file_id = int(id_part)
                        match_files.append((os.path.join(match_dir, filename), file_id))
                # 处理格式为match_timestamp.json的旧文件
                elif filename.count('_') >= 2:
                    match_files.append((os.path.join(match_dir, filename), float('inf')))  # 旧文件给予无穷大ID
            except:
                # 如果无法提取ID，也添加到列表
                match_files.append((os.path.join(match_dir, filename), float('inf')))
    
    # 按ID排序
    match_files.sort(key=lambda x: x[1])
    
    # 只返回文件路径列表
    return [file_path for file_path, _ in match_files]

def convert_balls_state_to_feature(balls_state):
    """将球的状态转换为特征向量"""
    # 球的ID列表，包括白球(0)和1-15号球
    ball_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    
    features = []
    for ball_id in ball_ids:
        ball = balls_state.get(str(ball_id), {})
        # 获取球的位置信息，如果不存在则使用默认值
        x = ball.get('x', 0.0)
        y = ball.get('y', 0.0)
        z = ball.get('z', 0.0)
        
        # 获取球的速度信息
        vx = ball.get('vx', 0.0)
        vy = ball.get('vy', 0.0)
        vz = ball.get('vz', 0.0)
        
        # 获取球的状态
        s = ball.get('s', 0)
        
        # 将球的信息添加到特征向量中
        features.extend([x, y, z, vx, vy, vz, s])
    
    return features

def process_single_match_for_behavior(match_data, output_file, is_first_match=False, is_last_match=False):
    """处理单个对局数据，生成行为网络训练数据"""
    # 初始化行为网络数据记录列表
    behavior_records = []
    
    # 遍历对局中的每一次击球
    for shot in match_data.get('shots', []):
        # 提取击球前的状态
        pre_state = shot.get('pre_state', {})
        balls_state = pre_state.get('balls', {})
        
        # 将球的状态转换为特征向量
        state_feature = convert_balls_state_to_feature(balls_state)
        
        # 提取动作信息
        action = shot.get('action', {})
        
        # 创建行为网络训练数据记录
        behavior_record = {
            'state': state_feature,
            'action': {
                'V0': action.get('V0', 0.0),
                'phi': action.get('phi', 0.0),
                'theta': action.get('theta', 0.0),
                'a': action.get('a', 0.0),
                'b': action.get('b', 0.0)
            }
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
    """处理单个对局数据，生成价值网络训练数据"""
    # 获取对局结果
    metadata = match_data.get('metadata', {})
    winner = metadata.get('winner', 0)
    
    # 初始化价值网络数据记录列表
    value_records = []
    
    # 遍历对局中的每一次击球
    for shot in match_data.get('shots', []):
        # 提取击球前的状态
        pre_state = shot.get('pre_state', {})
        balls_state = pre_state.get('balls', {})
        player = shot.get('player', 0)
        
        # 将球的状态转换为特征向量
        state_feature = convert_balls_state_to_feature(balls_state)
        
        # 计算价值标签（1表示当前玩家获胜，-1表示当前玩家失败）
        if winner == player:
            value = 1.0
        else:
            value = -1.0
        
        # 创建价值网络训练数据记录
        value_record = {
            'state': state_feature,
            'value': value
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

def process_match_data(match_dir, behavior_output_dir, value_output_dir, start_id=None, end_id=None):
    """处理对局数据并生成训练数据（逐局处理和实时保存）
    
    Args:
        match_dir: 对局数据目录
        behavior_output_dir: 行为网络数据输出目录
        value_output_dir: 价值网络数据输出目录
        start_id: 起始ID，用于文件名标识
        end_id: 结束ID，用于文件名标识
    
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
    
    # 根据是否提供了ID范围生成不同的文件名
    if start_id is not None and end_id is not None:
        behavior_file = os.path.join(behavior_output_dir, f"behavior_network_data_{start_id}_{end_id}_{timestamp}.json")
        value_file = os.path.join(value_output_dir, f"value_network_data_{start_id}_{end_id}_{timestamp}.json")
    else:
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
    parser.add_argument('--start_id', type=int, default=None, help="起始ID，用于文件名标识")
    parser.add_argument('--end_id', type=int, default=None, help="结束ID，用于文件名标识")
    
    args = parser.parse_args()
    
    # 处理数据
    behavior_file, value_file = process_match_data(
        match_dir=args.match_dir,
        behavior_output_dir=args.behavior_output_dir,
        value_output_dir=args.value_output_dir,
        start_id=args.start_id,
        end_id=args.end_id
    )
    
    print("\n处理完成：")
    print(f"行为网络训练数据: {behavior_file}")
    print(f"价值网络训练数据: {value_file}")