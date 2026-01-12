import os
import json
import argparse
from datetime import datetime
import numpy as np
from tqdm import tqdm
import gc


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
    for root, dirs, files in os.walk(match_dir):
        for filename in files:
            if filename.startswith('match_') and filename.endswith('.json'):
                file_path = os.path.join(root, filename)
                # 过滤过大的文件（可选，根据实际情况调整阈值，单位：MB）
                file_size = os.path.getsize(file_path) / (1024 * 1024)
                if file_size < 100:  # 跳过大于100MB的异常文件
                    match_files.append(file_path)
    return sorted(match_files)


def convert_to_81d_state(balls, my_targets):
    """将球的状态和目标球信息转换为81维状态向量"""
    # 直接使用numpy数组操作，减少临时列表创建
    state = np.zeros(81, dtype=np.float32)
    ball_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    # 1. 处理每个球的特征（0-63维）
    for i, ball_id in enumerate(ball_ids):
        ball_key = str(ball_id)
        if ball_key in balls:
            ball = balls[ball_key]
            state[i*4] = ball['x']
            state[i*4+1] = ball['y']
            state[i*4+2] = ball['z']
            state[i*4+3] = 1.0 if ball['s'] == 4 else 0.0
        else:
            state[i*4:i*4+4] = [0.0, 0.0, 0.0, 1.0]

    # 2. 球桌尺寸（64-65维）
    state[64] = 2.540
    state[65] = 1.270

    # 3. 目标球特征（66-80维）
    for ball_id in range(1, 16):
        state[65 + ball_id] = 1.0 if str(ball_id) in my_targets else 0.0

    # 直接返回numpy数组，避免提前转列表（减少临时对象）
    return state


def stream_json_file(file_path):
    """流式读取JSON文件，避免一次性加载整个文件"""
    # 对于单行JSON文件（常见的大JSON格式）
    with open(file_path, 'r', encoding='utf-8') as f:
        # 逐行读取（处理多行JSON）或一次性读取（单行），但分块解析
        data = ""
        for chunk in iter(lambda: f.read(1024*1024), ""):  # 每次读取1MB
            data += chunk
            try:
                # 尝试解析已读取的内容
                match_data = json.loads(data)
                yield match_data
                data = ""  # 解析成功后清空缓存
            except json.JSONDecodeError:
                continue
        # 处理最后剩余的内容
        if data.strip():
            try:
                match_data = json.loads(data)
                yield match_data
            except json.JSONDecodeError:
                print(f"警告：文件 {file_path} 格式错误，跳过")


def filter_shot(shot, shot_index, match_data):
    """
    过滤样本的函数
    
    参数：
        shot: 当前处理的shot
        shot_index: shot在对局中的索引
        match_data: 整个对局数据
    
    返回：
        bool: True表示样本应该被保留，False表示样本应该被过滤掉
    """
    try:
        '''
        # 1. 保留整场比赛第一杆，且这一杆是有效杆（打中了己方球，甚至打进了己方球）
        if shot_index == 0:
            # 检查是否是有效杆
            result = shot.get('result', {})
            # 如果打中了己方球或打进了己方球，返回True（保留）
            if result.get('ME_INTO_POCKET', []):
                return True
        '''
        # 2. 保留己方只剩下黑八没有打中，且这回合动作之后打进了黑八
        pre_state = shot['pre_state']
        my_targets = pre_state['my_targets']
        result = shot.get('result', {})
        
        # 检查是否只剩下黑八
        # 检查my_targets是否状态全都是是4（进袋）
        flag=True
        for target in my_targets:
            if shot['pre_state']['balls'][target]['s'] != 4:
                flag=False
                break
        return flag#如果只剩下黑八，返回True，特训黑八残局
        '''if not flag:
            # 检查是否打进了黑八
            BLACK_BALL_INTO_POCKET = result.get('BLACK_BALL_INTO_POCKET', [])
            if BLACK_BALL_INTO_POCKET:
                return True'''
        
        # 其他情况都删掉
        return False
    except Exception as e:
        print(f"警告：过滤样本时出错 - {e}，保留该样本")
        return True


def process_single_match(match_file, enable_filter=False):
    """
    处理单个对局数据文件，生成可训练数据（纯流式）
    
    参数：
        match_file: 对局数据文件路径
        enable_filter: 是否启用样本过滤
    """
    try:
        # 流式读取JSON数据
        for match_data in stream_json_file(match_file):
            winner = match_data['metadata']['winner']
            shots = match_data['shots']

            # 初始化最近状态（使用numpy数组，减少内存）
            recent_states = []

            for i, shot in enumerate(shots):
                # 初始化可能用到的变量，避免未定义
                player = pre_state = balls = my_targets = action = None
                current_state = states = action_vector = value = None

                try:
                    # 应用过滤器
                    if enable_filter and not filter_shot(shot, i, match_data):
                        continue

                    player = shot['player']
                    pre_state = shot['pre_state']
                    balls = pre_state['balls']
                    my_targets = pre_state['my_targets']
                    action = shot['action']

                    # 转换状态（返回numpy数组）
                    current_state = convert_to_81d_state(balls, my_targets)

                    # 更新最近状态
                    recent_states.append(current_state)
                    if len(recent_states) > 2:
                        del recent_states[0]  # 直接删除，避免pop的临时对象

                    # 构建状态列表（转列表仅在yield时）
                    states = [
                        recent_states[0].tolist(
                        ) if i >= 2 else current_state.tolist(),
                        recent_states[-1].tolist() if i >= 1 else current_state.tolist(),
                        current_state.tolist()
                    ]

                    # 动作向量
                    action_vector = [
                        action['V0'],
                        action['phi'],
                        action['theta'],
                        action['a'],
                        action['b']
                    ]

                    # 价值标签
                    if winner == "SAME":
                        # 平局，双方价值都为0.5
                        value = 0.5
                    else:
                        # 非平局，胜利方价值为1.0，失败方为0.0
                        value = 1.0 if winner == player else 0.0

                    # 生成样本
                    train_sample = {
                        "states": states,
                        "action": action_vector,
                        "value": value,
                        "history_length": 3
                    }

                    yield train_sample

                    # 每处理10个样本释放一次内存（更频繁的清理）
                    if i % 10 == 0:
                        gc.collect()

                except KeyError as e:
                    print(f"警告：shot {i} 缺少字段 {e}，跳过")
                    continue
                finally:
                    # 安全删除变量：只删除已定义且非None的变量
                    for var_name in ['shot', 'player', 'pre_state', 'balls', 'my_targets', 'action',
                                     'current_state', 'states', 'action_vector', 'value']:
                        if locals().get(var_name) is not None:
                            del locals()[var_name]

            # 释放对局级别的变量
            del match_data, winner, shots, recent_states
            gc.collect()

    except Exception as e:
        print(f"错误：处理文件 {match_file} 时出错 - {e}")
        raise e
    finally:
        gc.collect()


def process_match_data(match_dir, output_file, enable_filter=False):
    """
    处理多个对局数据文件，流式写入输出
    
    参数：
        match_dir: 对局数据目录
        output_file: 输出文件路径
        enable_filter: 是否启用样本过滤
    """
    match_files = load_match_files(match_dir)
    total_samples = 0
    first_sample = True

    # 使用缓冲写入，减少IO次数
    with open(output_file, 'w', encoding='utf-8', buffering=1024*1024) as f:
        f.write('[\n')

        for match_file in tqdm(match_files, desc="Processing match files"):
            generator = process_single_match(match_file, enable_filter)

            for sample in generator:
                try:
                    if not first_sample:
                        f.write(',\n')
                    else:
                        first_sample = False

                    # 直接dump，不生成临时字符串
                    json.dump(sample, f, ensure_ascii=False, cls=NumpyEncoder)
                    total_samples += 1

                    # 释放样本内存
                    del sample

                except Exception as e:
                    print(f"警告：写入样本时出错 - {e}")
                    continue

            # 释放生成器内存
            del generator
            gc.collect()

        f.write('\n]')

    print(f"生成 {total_samples} 个训练样本")
    print(f"保存至 {output_file}")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(
        description="Process match data to generate trainable data")
    parser.add_argument('--match_dir', type=str, default='match_data',
                        help='Directory containing match data files')
    parser.add_argument('--output_file', type=str, default='trainable_data.json',
                        help='Output file path for trainable data')
    parser.add_argument('--enable_filter', action='store_true', default=False,
                        help='Enable sample filtering')

    args = parser.parse_args()

    # 强制开启垃圾回收
    gc.enable()
    process_match_data(args.match_dir, args.output_file, args.enable_filter)


if __name__ == '__main__':
    main()
