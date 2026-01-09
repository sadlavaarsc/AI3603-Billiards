"""
mcts_vs_basic.py - MCTS Agent 与 Basic Agent 对弈脚本

功能：
- 让 MCTS Agent 和 Basic Agent 进行多局对战
- 统计胜负和胜率
- 支持切换先后手和球型分配
- 每局开始时展示累计比分和胜率（新增核心功能）
"""

# 导入必要的模块
from utils import set_random_seed
from poolenv import PoolEnv
from agents import BasicAgent, MCTSAgent, DirectModelAgent, BasicAgentPro
from dual_network import DualNetwork
import torch
import sys

# 设置随机种子，enable=True 时使用固定种子，enable=False 时使用完全随机
set_random_seed(enable=True, seed=42)


def print_current_stats(current_game, total_games, results):
    """
    打印当前累计的比分和胜率（每局开始时调用）
    参数：
        current_game: 当前要开始的局数（从1开始）
        total_games: 总对局数
        results: 胜负统计字典
    """
    if current_game == 1:
        print("\n【初始状态】尚未开始对局")
        return

    # 计算截至上一局的累计数据
    played_games = current_game - 1
    mcts_wins = results['AGENT_B_WIN']
    basic_wins = results['AGENT_A_WIN']
    draws = results['SAME']

    # 计算胜率（避免除零）
    mcts_win_rate = mcts_wins / played_games * 100 if played_games > 0 else 0
    basic_win_rate = basic_wins / played_games * 100 if played_games > 0 else 0
    draw_rate = draws / played_games * 100 if played_games > 0 else 0

    # 打印格式化的统计信息
    print("\n" + "-"*60)
    print(f"【累计比分】已完成 {played_games}/{total_games} 局")
    print(
        f"Basic Agent: {basic_wins} 胜 | MCTS Agent: {mcts_wins} 胜 | 平局: {draws}")
    print(
        f"胜率：Basic {basic_win_rate:.2f}% | MCTS {mcts_win_rate:.2f}% | 平局 {draw_rate:.2f}%")
    print("-"*60)


def main():
    # 1. 保存原始的控制台输出（后续可恢复）
    #original_stdout = sys.stdout

    # 2. 直接创建UTF-8编码的文件对象（核心：指定encoding='utf-8'）
    #mode='w' 覆盖文件，mode='a' 追加内容（根据需求选）
    #f = open('120_results.txt', 'w', encoding='utf-8', buffering=1)

    # 3. 将print重定向到这个文件对象
    #sys.stdout = f

    # 初始化环境和参数
    env = PoolEnv()
    results = {'AGENT_A_WIN': 0, 'AGENT_B_WIN': 0, 'SAME': 0}
    n_games = 120  # 对战局数，可根据需要调整

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualNetwork()

    # 尝试加载预训练模型，如果没有则使用随机初始化模型
    try:
        model.load('./models/dual_network_final.pt')
        print("成功加载预训练模型")
    except Exception as e:
        print(f"未找到预训练模型，使用随机初始化模型: {e}")

    model.to(device)
    model.eval()

    # 初始化 Agent
    agent_a = BasicAgentPro()
    agent_b = MCTSAgent(model=model, env=env, n_simulations=50, device=device)

    players = [agent_a, agent_b]  # 用于切换先后手
    target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']  # 轮换球型

    print(f"开始 {n_games} 局对战，MCTS Agent 对阵 Basic Agent")
    print("="*60)

    for i in range(n_games):
        # 新增：每局开始前打印累计比分
        current_game_num = i + 1
        print_current_stats(current_game_num, n_games, results)

        print(f"\n------- 第 {current_game_num} 局比赛开始 -------")
        env.reset(target_ball=target_ball_choice[i % 4])

        # 清空 MCTS Agent 的状态缓冲区
        agent_b.clear_buffer()

        player_class = players[i % 2].__class__.__name__
        ball_type = target_ball_choice[i % 4]
        print(f"本局 Player A: {player_class}, 目标球型: {ball_type}")

        while True:
            player = env.get_curr_player()
            print(f"[第{env.hit_count}次击球] player: {player}")
            obs = env.get_observation(player)

            if player == 'A':
                action = players[i % 2].decision(*obs)
            else:
                action = players[(i + 1) % 2].decision(*obs)

            step_info = env.take_shot(action)

            done, info = env.get_done()
            if not done:
                if step_info.get('ENEMY_INTO_POCKET'):
                    print(f"对方球入袋：{step_info['ENEMY_INTO_POCKET']}")
            if done:
                # 统计结果（player A/B 转换为 agent A/B）
                if info['winner'] == 'SAME':
                    results['SAME'] += 1
                elif info['winner'] == 'A':
                    results[['AGENT_A_WIN', 'AGENT_B_WIN'][i % 2]] += 1
                else:
                    results[['AGENT_A_WIN', 'AGENT_B_WIN'][(i+1) % 2]] += 1

                # 新增：打印本局结果
                winner_agent = ""
                if info['winner'] == 'SAME':
                    print(f"\n第 {current_game_num} 局：平局！")
                elif info['winner'] == 'A':
                    winner_agent = players[i % 2].__class__.__name__
                    print(f"\n第 {current_game_num} 局：{winner_agent} 获胜！")
                else:
                    winner_agent = players[(i+1) % 2].__class__.__name__
                    print(f"\n第 {current_game_num} 局：{winner_agent} 获胜！")
                print(f"------- 第 {current_game_num} 局比赛结束 -------")
                break

    # 计算最终胜率
    total_games = n_games
    mcts_wins = results['AGENT_B_WIN']
    basic_wins = results['AGENT_A_WIN']
    draws = results['SAME']

    mcts_win_rate = mcts_wins / total_games * 100 if total_games > 0 else 0
    basic_win_rate = basic_wins / total_games * 100 if total_games > 0 else 0
    draw_rate = draws / total_games * 100 if total_games > 0 else 0

    # 打印最终结果
    print("\n" + "="*60)
    print("最终对战结果")
    print("="*60)
    print(f"总对局数: {total_games}")
    print(f"MCTS Agent 获胜: {mcts_wins} 局 ({mcts_win_rate:.2f}%)")
    print(f"Basic Agent 获胜: {basic_wins} 局 ({basic_win_rate:.2f}%)")
    print(f"平局: {draws} 局 ({draw_rate:.2f}%)")
    print("="*60)

    if mcts_wins > basic_wins:
        print("MCTS Agent 最终获胜！")
    elif basic_wins > mcts_wins:
        print("Basic Agent 最终获胜！")
    else:
        print("双方最终战平！")

    # 恢复标准输出（如果之前重定向了）
    #sys.stdout = original_stdout
    #f.close()


if __name__ == '__main__':
    main()
