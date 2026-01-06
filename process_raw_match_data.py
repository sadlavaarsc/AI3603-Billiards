import os
import json
import argparse
from datetime import datetime
import numpy as np
from tqdm import tqdm

class NumpyEncoder(json.JSONEncoder):
    """用于序列化numpy数组和其他numpy类型"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def load_match_files(match_dir):
    """加载指定目录下的所有对局数据文件，包括子目录"""
    match_files = []
    # 递归遍历所有子目录
    for root, dirs, files in os.walk(match_dir):
        for filename in files:
            if filename.startswith('match_') and filename.endswith('.json'):
                file_path = os.path.join(root, filename)
                match_files.append(file_path)
    return sorted(match_files)

def convert_to_81d_state(balls, my_targets):
    """将球的状态和目标球信息转换为81维状态向量
    
    状态向量组成：
    - 0-63维：16个球的特征（每个球4个特征：x,y,z,进袋标志）
    - 64-65维：球桌尺寸（固定值：宽度2.540m×长度1.270m）
    - 66-80维：目标球特征（二进制值）
    """
    # 球的ID列表，包括白球(0)和1-15号球
    ball_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    
    # 初始化状态向量
    state = np.zeros(81, dtype=np.float32)
    
    # 1. 处理每个球的特征（0-63维）
    for i, ball_id in enumerate(ball_ids):
        ball_key = str(ball_id)
        if ball_key in balls:
            ball = balls[ball_key]
            # 球的位置和状态
            state[i*4] = ball['x']  # x坐标
            state[i*4+1] = ball['y']  # y坐标
            state[i*4+2] = ball['z']  # z坐标
            state[i*4+3] = 1.0 if ball['s'] == 4 else 0.0  # 进袋标志（0或1）
        else:
            # 球不存在，使用默认值
            state[i*4:i*4+4] = [0.0, 0.0, 0.0, 1.0]  # 假设球已进袋
    
    # 2. 球桌尺寸（64-65维）
    state[64] = 2.540  # 球桌宽度
    state[65] = 1.270  # 球桌长度
    
    # 3. 目标球特征（66-80维）
    # 66-80维对应1-15号球，值为1表示是目标球，0表示不是
    for ball_id in range(1, 16):
        if str(ball_id) in my_targets:
            state[65 + ball_id] = 1.0
        else:
            state[65 + ball_id] = 0.0
    
    return state.tolist()

def process_single_match(match_file):
    """处理单个对局数据文件，生成可训练数据（生成器模式）"""
    # 加载对局数据
    with open(match_file, 'r', encoding='utf-8') as f:
        match_data = json.load(f)
    
    # 获取对局结果
    winner = match_data['metadata']['winner']
    
    # 获取所有shots
    shots = match_data['shots']
    num_shots = len(shots)
    
    # 只保留最近2个状态，而不是所有历史状态
    recent_states = []
    
    # 遍历每个shot，生成训练数据
    for i in range(num_shots):
        shot = shots[i]
        player = shot['player']
        pre_state = shot['pre_state']
        balls = pre_state['balls']
        my_targets = pre_state['my_targets']
        action = shot['action']
        
        # 转换当前状态为81d向量
        current_state = convert_to_81d_state(balls, my_targets)
        
        # 更新最近状态列表
        recent_states.append(current_state)
        # 只保留最近2个状态
        if len(recent_states) > 2:
            recent_states.pop(0)
        
        # 组合成连续3局的状态向量 [3, 81]
        # - state_3: 最旧的状态（i-2，如果存在）
        # - state_2: 中间的状态（i-1，如果存在）
        # - state_1: 当前状态（i）
        states = [
            recent_states[0] if i >= 2 else current_state,  # state_3
            recent_states[-1] if i >= 1 else current_state,  # state_2
            current_state  # state_1
        ]
        
        # 提取动作向量 [V0, phi, theta, a, b]
        action_vector = [
            action['V0'],
            action['phi'],
            action['theta'],
            action['a'],
            action['b']
        ]
        
        # 计算价值标签（胜率）
        # 1.0表示当前玩家获胜，0.0表示对手获胜
        value = 1.0 if winner == player else 0.0
        
        # 生成训练数据样本
        train_sample = {
            "states": states,
            "action": action_vector,
            "value": value,
            "history_length": 3
        }
        
        # 使用yield返回样本，而不是存储在列表中
        yield train_sample
        
        # 每处理100个样本，清理一下内存
        if i % 100 == 0:
            import gc
            gc.collect()
    
    # 清理内存
    del match_data, shots
    import gc
    gc.collect()

def process_match_data(match_dir, output_file):
    """处理多个对局数据文件，生成可训练数据（流式处理）"""
    # 获取所有对局数据文件
    match_files = load_match_files(match_dir)
    
    total_samples = 0
    first_sample = True
    
    # 打开输出文件，开始写入JSON数组
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入JSON数组开始
        f.write('[\n')
        
        # 遍历每个对局数据文件
        for match_file in tqdm(match_files, desc="Processing match files"):
            # 处理单个对局数据 - 现在返回的是生成器
            train_data_generator = process_single_match(match_file)
            
            # 逐个写入样本到文件
            for sample in train_data_generator:
                if not first_sample:
                    # 不是第一个样本，先写入逗号分隔符
                    f.write(',\n')
                else:
                    first_sample = False
                
                # 写入单个样本
                json.dump(sample, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
                total_samples += 1
        
        # 写入JSON数组结束
        f.write('\n]')
    
    print(f"Generated {total_samples} training samples")
    print(f"Saved to {output_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Process match data to generate trainable data")
    
    parser.add_argument('--match_dir', type=str, default='match_data', 
                      help='Directory containing match data files')
    parser.add_argument('--output_file', type=str, default='trainable_data.json', 
                      help='Output file path for trainable data')
    
    args = parser.parse_args()
    
    # 处理数据
    process_match_data(args.match_dir, args.output_file)

if __name__ == '__main__':
    main()