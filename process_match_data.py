import os
import json
import datetime
import argparse
import sys

# 确保项目根目录在Python路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import time
from tqdm import tqdm
from datetime import datetime

def load_single_match_data(file_path):
    """加载单个对局数据文件"""
    try:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 加载对局文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 成功加载对局文件: {file_path}")
        return data
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 加载对局数据文件 {file_path} 出错: {e}")
        return None

def get_match_files(match_dir):
    """获取指定目录下的所有对局数据文件，并尝试从文件名提取ID
    
    Args:
        match_dir: 对局数据目录
        
    Returns:
        list: [(文件路径, 文件ID), ...] 的列表，按ID排序
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始扫描对局目录: {match_dir}")
    if not os.path.exists(match_dir):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 警告: 对局数据目录 {match_dir} 不存在")
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
            except Exception as e:
                # 如果无法提取ID，也添加到列表
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 无法从文件名 {filename} 提取ID: {e}")
                match_files.append((os.path.join(match_dir, filename), float('inf')))
    
    # 按ID排序
    match_files.sort(key=lambda x: x[1])
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 扫描完成，共找到 {len(match_files)} 个对局文件")
    # 返回文件路径和ID的元组列表
    return match_files

def convert_balls_state_to_feature(balls_states, history_length=0):
    """将球的状态转换为特征向量，支持历史回合状态
    
    Args:
        balls_states: 球的状态列表或字典。如果是列表，则按时间顺序包含历史状态和当前状态
                     如果是字典，则只包含当前状态
        history_length: 需要包含的历史回合数
        
    Returns:
        list: 合并后的特征向量
    """
    # 确保balls_states是列表格式
    if isinstance(balls_states, dict):
        # 如果传入的是单个状态字典，转换为只包含当前状态的列表
        balls_states = [balls_states]
    
    # 球的ID列表，包括白球(0)和1-15号球
    ball_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    
    # 初始化特征向量
    features = []
    
    # 计算需要包含的历史状态数量
    # 确保不超过history_length，并且不超过可用的历史状态数
    available_history = len(balls_states) - 1  # 最后一个是当前状态
    needed_history = min(history_length, available_history)
    
    # 如果历史状态不足，需要重复最早的状态来填充
    states_to_include = []
    
    # 如果需要更多历史状态
    if needed_history > 0:
        # 计算需要重复的最早状态次数
        if available_history == 0:
            # 没有历史状态，重复当前状态
            earliest_state = balls_states[0]
            states_to_include = [earliest_state] * (history_length + 1)  # +1 包括当前状态
        else:
            # 从最早的可用状态开始
            earliest_idx = 0
            current_idx = len(balls_states) - 1
            
            # 添加历史状态
            states_to_include = balls_states[earliest_idx:current_idx-needed_history+1:-1][::-1]  # 逆序获取然后反转回正序
            
            # 确保历史状态数量正确
            if len(states_to_include) < needed_history:
                # 如果还是不足，用最早的状态填充
                earliest_state = balls_states[0]
                missing = needed_history - len(states_to_include)
                states_to_include = [earliest_state] * missing + states_to_include
            
            # 添加当前状态
            states_to_include.append(balls_states[current_idx])
    else:
        # 只使用当前状态
        states_to_include = [balls_states[-1]]
    
    # 为每个状态生成特征并合并
    for state in states_to_include:
        state_features = []
        for ball_id in ball_ids:
            ball = state.get(str(ball_id), {})
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
            state_features.extend([x, y, z, vx, vy, vz, s])
        
        # 将当前状态的特征添加到总特征向量
        features.extend(state_features)
    
    return features

def process_single_match_for_behavior(match_data, output_file, is_first_match=False, is_last_match=False, history_length=0):
    """处理单个对局数据，生成行为网络训练数据，支持时序状态特征
    
    Args:
        match_data: 对局数据
        output_file: 输出文件路径
        is_first_match: 是否是第一个对局
        is_last_match: 是否是最后一个对局
        history_length: 需要包含的历史回合数
        
    Returns:
        int: 生成的行为网络数据记录数量
    """
    # 初始化行为网络数据记录列表
    behavior_records = []
    
    # 保存历史状态
    history_states = []
    
    # 获取对局ID
    match_id = match_data.get('metadata', {}).get('match_id', 'unknown')
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始处理对局 {match_id} 的行为数据，包含历史回合数: {history_length}")
    
    # 获取击球总数
    shots_count = len(match_data.get('shots', []))
    
    # 遍历对局中的每一次击球
    for i, shot in enumerate(match_data.get('shots', [])):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 处理对局 {match_id} 的第 {i+1}/{shots_count} 次击球")
        
        # 提取击球前的状态
        pre_state = shot.get('pre_state', {})
        balls_state = pre_state.get('balls', {})
        
        # 将当前状态添加到历史状态中
        history_states.append(balls_state)
        
        # 确保历史状态数量不超过需要的数量+1（当前状态）
        # 只保留最近的history_length+1个状态
        if len(history_states) > history_length + 1:
            history_states = history_states[-(history_length + 1):]
        
        # 将球的状态转换为特征向量，包含历史状态
        state_feature = convert_balls_state_to_feature(history_states, history_length)
        
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
            },
            'history_length': history_length  # 记录使用的历史长度，便于后续处理
        }
        
        behavior_records.append(behavior_record)
    
    try:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始写入对局 {match_id} 的行为数据，共 {len(behavior_records)} 条记录")
        
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
                
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 成功写入对局 {match_id} 的行为数据")
                
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 保存行为数据出错: {e}")
    
    return len(behavior_records)


def process_single_match_for_value(match_data, output_file, is_first_match=False, is_last_match=False, history_length=0):
    """处理单个对局数据，生成价值网络训练数据，支持时序状态特征
    
    Args:
        match_data: 对局数据
        output_file: 输出文件路径
        is_first_match: 是否是第一个对局
        is_last_match: 是否是最后一个对局
        history_length: 需要包含的历史回合数
        
    Returns:
        int: 生成的价值网络数据记录数量
    """
    # 获取对局结果
    metadata = match_data.get('metadata', {})
    winner = metadata.get('winner', 0)
    match_id = metadata.get('match_id', 'unknown')
    
    # 初始化价值网络数据记录列表
    value_records = []
    
    # 保存历史状态
    history_states = []
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始处理对局 {match_id} 的价值数据，获胜者: {winner}")
    
    # 获取击球总数
    shots_count = len(match_data.get('shots', []))
    
    # 遍历对局中的每一次击球
    for i, shot in enumerate(match_data.get('shots', [])):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 处理对局 {match_id} 的第 {i+1}/{shots_count} 次击球（价值计算）")
        
        # 提取击球前的状态
        pre_state = shot.get('pre_state', {})
        balls_state = pre_state.get('balls', {})
        player = shot.get('player', 0)
        
        # 将当前状态添加到历史状态中
        history_states.append(balls_state)
        
        # 确保历史状态数量不超过需要的数量+1（当前状态）
        # 只保留最近的history_length+1个状态
        if len(history_states) > history_length + 1:
            history_states = history_states[-(history_length + 1):]
        
        # 将球的状态转换为特征向量，包含历史状态
        state_feature = convert_balls_state_to_feature(history_states, history_length)
        
        # 计算价值标签（1表示当前玩家获胜，-1表示当前玩家失败）
        if winner == player:
            value = 1.0
        else:
            value = -1.0
        
        # 创建价值网络训练数据记录
        value_record = {
            'state': state_feature,
            'value': value,
            'history_length': history_length  # 记录使用的历史长度，便于后续处理
        }
        
        value_records.append(value_record)
    
    try:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始写入对局 {match_id} 的价值数据，共 {len(value_records)} 条记录")
        
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
                
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 成功写入对局 {match_id} 的价值数据")
                
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 保存价值数据出错: {e}")
    
    return len(value_records)

def process_match_data(match_dir, behavior_output_dir, value_output_dir, start_id=None, end_id=None, history_length=0):
    """处理多个对局数据，生成行为网络和价值网络的训练数据，支持时序状态特征
    
    Args:
        match_dir: 对局数据目录
        behavior_output_dir: 行为网络数据输出目录
        value_output_dir: 价值网络数据输出目录
        start_id: 起始对局ID
        end_id: 结束对局ID
        history_length: 需要包含的历史回合数
    """
    # 记录开始时间
    start_time = datetime.now()
    print(f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] 开始处理对局数据")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 配置参数: match_dir={match_dir}, history_length={history_length}")
    
    # 确保输出目录存在
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 检查并创建输出目录")
    os.makedirs(behavior_output_dir, exist_ok=True)
    os.makedirs(value_output_dir, exist_ok=True)
    
    # 获取对局文件列表
    match_files = get_match_files(match_dir)
    
    # 如果指定了起始和结束ID，筛选文件
    if start_id is not None and end_id is not None:
        # 筛选ID在[start_id, end_id]范围内的文件
        filtered_files = []
        for file_path, file_id in match_files:
            if start_id <= file_id <= end_id:
                filtered_files.append((file_path, file_id))
        match_files = filtered_files
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 根据ID范围筛选后，剩余 {len(match_files)} 个对局文件 (ID范围: {start_id}-{end_id})")
    
    # 生成输出文件名，包含起始和结束ID
    if start_id is not None and end_id is not None:
        behavior_output_file = os.path.join(behavior_output_dir, f'behavior_data_{start_id}_{end_id}.json')
        value_output_file = os.path.join(value_output_dir, f'value_data_{start_id}_{end_id}.json')
    else:
        # 如果没有指定ID范围，使用时间戳作为文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        behavior_output_file = os.path.join(behavior_output_dir, f'behavior_data_{timestamp}.json')
        value_output_file = os.path.join(value_output_dir, f'value_data_{timestamp}.json')
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 行为数据将保存至: {behavior_output_file}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 价值数据将保存至: {value_output_file}")
    
    # 初始化总记录数
    total_behavior_records = 0
    total_value_records = 0
    
    # 遍历对局文件
    for i, (file_path, file_id) in enumerate(match_files):
        is_first_match = (i == 0)
        is_last_match = (i == len(match_files) - 1)
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 处理对局 {file_id} ({i+1}/{len(match_files)})")
        
        try:
            # 加载对局数据
            match_data = load_single_match_data(file_path)
            if match_data is None:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 跳过无效对局文件: {file_path}")
                continue
            
            # 处理行为网络数据，传递history_length参数
            behavior_records = process_single_match_for_behavior(
                match_data, behavior_output_file, is_first_match, is_last_match, history_length
            )
            
            # 处理价值网络数据，传递history_length参数
            value_records = process_single_match_for_value(
                match_data, value_output_file, is_first_match, is_last_match, history_length
            )
            
            total_behavior_records += behavior_records
            total_value_records += value_records
            
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 处理对局 {file_id} 完成，生成行为记录 {behavior_records} 条，价值记录 {value_records} 条")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 当前累计: 行为记录 {total_behavior_records} 条，价值记录 {total_value_records} 条")
            
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 处理对局 {file_id} 出错: {e}")
    
    # 记录结束时间
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"[{end_time.strftime('%Y-%m-%d %H:%M:%S')}] 全部处理完成，总耗时: {duration}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 共生成行为记录 {total_behavior_records} 条，价值记录 {total_value_records} 条")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 行为数据保存至: {behavior_output_file}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 价值数据保存至: {value_output_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='处理台球对局数据，生成训练数据')
    parser.add_argument('--match_dir', type=str, default='./matches', help='对局数据目录')
    parser.add_argument('--behavior_output_dir', type=str, default='./behavior_data', help='行为网络数据输出目录')
    parser.add_argument('--value_output_dir', type=str, default='./value_data', help='价值网络数据输出目录')
    parser.add_argument('--start_id', type=int, help='起始对局ID')
    parser.add_argument('--end_id', type=int, help='结束对局ID')
    parser.add_argument('--history_length', type=int, default=0, help='需要包含的历史回合数，默认0表示只使用当前回合')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.start_id is not None and args.end_id is None:
        parser.error('--start_id需要与--end_id一起使用')
    if args.start_id is None and args.end_id is not None:
        parser.error('--end_id需要与--start_id一起使用')
    if args.history_length < 0:
        parser.error('--history_length不能为负数')
    
    # 处理数据
    process_match_data(
        args.match_dir,
        args.behavior_output_dir,
        args.value_output_dir,
        args.start_id,
        args.end_id,
        args.history_length
    )