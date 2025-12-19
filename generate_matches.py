import os
import json
import argparse
import numpy as np
import time
from tqdm import tqdm
from datetime import datetime

# 导入环境和Agent
from poolenv import PoolEnv
from agent import BasicAgent

def save_match_data(match_data, output_dir):
    """保存单局比赛数据到文件"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"match_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # 保存数据
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(match_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    return filepath

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

def convert_balls_to_serializable(balls):
    """将balls对象转换为可序列化的字典"""
    result = {}
    for ball_id, ball in balls.items():
        if hasattr(ball, 'state'):
            result[ball_id] = {
                'x': float(ball.state.x),
                'y': float(ball.state.y),
                'z': float(ball.state.z),
                'vx': float(ball.state.vx),
                'vy': float(ball.state.vy),
                'vz': float(ball.state.vz),
                's': int(ball.state.s)  # 状态: 0=静止, 4=落袋
            }
    return result

def play_match(env, agent, enable_noise=True, max_hit_count=200, verbose=False):
    """执行一局比赛
    
    Args:
        env: PoolEnv实例
        agent: 用于决策的代理
        enable_noise: 是否启用动作噪声
        max_hit_count: 最大击球次数
        verbose: 是否打印详细信息
    
    Returns:
        dict: 包含整局比赛数据的字典
    """
    # 设置环境参数
    env.enable_noise = enable_noise
    env.MAX_HIT_COUNT = max_hit_count
    env.verbose = verbose
    
    # 随机选择开局球型
    target_ball = np.random.choice(['solid', 'stripe'])
    env.reset(target_ball=target_ball)
    
    # 比赛数据记录
    match_data = {
        'metadata': {
            'start_time': datetime.now().isoformat(),
            'target_ball': target_ball,
            'enable_noise': enable_noise,
            'max_hit_count': max_hit_count
        },
        'player_targets': env.player_targets,
        'shots': []
    }
    
    # 开始比赛
    while True:
        player = env.get_curr_player()
        balls, my_targets, table = env.get_observation(player)
        
        # 记录击球前的状态
        shot_start_time = time.time()
        
        # 使用代理决策
        try:
            action = agent.decision(balls, my_targets, table)
        except Exception as e:
            print(f"决策过程出错: {e}")
            action = {
                'V0': 1.0,
                'phi': 0.0,
                'theta': 0.0,
                'a': 0.0,
                'b': 0.0
            }
        
        decision_time = time.time() - shot_start_time
        
        # 执行击球
        shot_result = env.take_shot(action)
        
        # 记录击球数据
        shot_record = {
            'hit_count': env.hit_count,
            'player': player,
            'pre_state': {
                'balls': convert_balls_to_serializable(balls),
                'my_targets': my_targets
            },
            'action': action,
            'result': {
                'ME_INTO_POCKET': shot_result.get('ME_INTO_POCKET', []),
                'ENEMY_INTO_POCKET': shot_result.get('ENEMY_INTO_POCKET', []),
                'WHITE_BALL_INTO_POCKET': shot_result.get('WHITE_BALL_INTO_POCKET', False),
                'BLACK_BALL_INTO_POCKET': shot_result.get('BLACK_BALL_INTO_POCKET', False),
                'FOUL_FIRST_HIT': shot_result.get('FOUL_FIRST_HIT', False),
                'NO_POCKET_NO_RAIL': shot_result.get('NO_POCKET_NO_RAIL', False),
                'NO_HIT': shot_result.get('NO_HIT', False)
            },
            'decision_time': decision_time
        }
        
        match_data['shots'].append(shot_record)
        
        # 检查是否结束
        done, info = env.get_done()
        if done:
            match_data['metadata']['end_time'] = datetime.now().isoformat()
            match_data['metadata']['winner'] = env.winner
            match_data['metadata']['total_shots'] = len(match_data['shots'])
            break
    
    return match_data

def generate_matches(num_matches, output_dir, enable_noise=True, max_hit_count=200, verbose=False):
    """生成指定数量的比赛数据
    
    Args:
        num_matches: 要生成的比赛数量
        output_dir: 输出目录
        enable_noise: 是否启用动作噪声
        max_hit_count: 最大击球次数
        verbose: 是否打印详细信息
    
    Returns:
        list: 生成的文件路径列表
    """
    env = PoolEnv()
    agent = BasicAgent()  # 使用贝叶斯AI进行自对弈
    
    file_paths = []
    
    for i in tqdm(range(num_matches), desc="生成对局数据"):
        try:
            match_data = play_match(env, agent, enable_noise, max_hit_count, verbose)
            file_path = save_match_data(match_data, output_dir)
            file_paths.append(file_path)
            
            if verbose:
                print(f"第{i+1}/{num_matches}局比赛完成，胜者: {match_data['metadata']['winner']}, 击球数: {match_data['metadata']['total_shots']}")
        except Exception as e:
            print(f"生成第{i+1}局比赛时出错: {e}")
            continue
    
    return file_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成台球AI自对弈对局数据")
    parser.add_argument('--num_matches', type=int, default=10, help="要生成的比赛数量")
    parser.add_argument('--output_dir', type=str, default="match_data", help="输出目录")
    parser.add_argument('--enable_noise', action='store_true', default=True, help="是否启用动作噪声")
    parser.add_argument('--max_hit_count', type=int, default=200, help="每局最大击球次数")
    parser.add_argument('--verbose', action='store_true', default=False, help="是否打印详细信息")
    
    args = parser.parse_args()
    
    # 生成对局数据
    file_paths = generate_matches(
        num_matches=args.num_matches,
        output_dir=args.output_dir,
        enable_noise=args.enable_noise,
        max_hit_count=args.max_hit_count,
        verbose=args.verbose
    )
    
    print(f"成功生成 {len(file_paths)} 局比赛数据")
    for path in file_paths:
        print(f"- {path}")
