#!/usr/bin/env python3
"""
mcts_agent_example.py - MCTS Agent 使用示例

功能：
- 演示如何使用 MCTSAgent 和 DirectModelAgent
- 展示模型加载、环境设置和决策过程
- 比较两种Agent的输出结果
"""

import torch
from poolenv import PoolEnv
from agents import MCTSAgent, DirectModelAgent, BasicAgent

# 设置随机种子
import random
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def main():
    """主函数，演示MCTSAgent和DirectModelAgent的使用"""
    print("MCTS Agent 使用示例")
    print("=" * 50)
    
    # 初始化环境
    print("1. 初始化环境...")
    env = PoolEnv()
    env.reset(target_ball='solid')
    print(f"   目标球型: {env.target_ball}")
    print(f"   玩家目标: {env.player_targets}")
    print()
    
    # 获取初始状态
    player = env.get_curr_player()
    balls, my_targets, table = env.get_observation(player)
    print(f"2. 当前玩家: {player}")
    print(f"   目标球: {my_targets}")
    print()
    
    # 加载模型
    print("3. 加载模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   使用设备: {device}")
    
    try:
        # 初始化并加载模型
        mcts_agent = MCTSAgent(device=device)
        mcts_agent.set_env(env)
        mcts_agent.load_model('./models/dual_network_final.pt')
        
        direct_agent = DirectModelAgent(device=device)
        direct_agent.load_model('./models/dual_network_final.pt')
        
        print("   模型加载成功")
    except Exception as e:
        print(f"   模型加载失败: {e}")
        print("   使用随机初始化模型...")
        # 如果没有预训练模型，使用随机初始化模型
        from dual_network import DualNetwork
        model = DualNetwork()
        model.to(device)
        model.eval()
        
        mcts_agent = MCTSAgent(model=model, env=env, device=device)
        direct_agent = DirectModelAgent(model=model, device=device)
    print()
    
    # 设置MCTS参数
    print("4. 配置MCTS参数...")
    mcts_agent.set_simulations(5)  # 减少模拟次数，加快演示速度
    mcts_agent.set_action_samples(4)  # 减少动作采样数量，加快演示速度
    print(f"   模拟次数: {mcts_agent.n_simulations}")
    print(f"   动作采样数量: {mcts_agent.n_action_samples}")
    print()
    
    # 使用BasicAgent进行对比
    print("5. 使用BasicAgent进行决策...")
    basic_agent = BasicAgent()
    basic_action = basic_agent.decision(balls, my_targets, table)
    print(f"   BasicAgent动作: {basic_action}")
    print()
    
    # 使用DirectModelAgent进行决策
    print("6. 使用DirectModelAgent进行决策...")
    direct_action = direct_agent.decision(balls, my_targets, table)
    print(f"   DirectModelAgent动作: {direct_action}")
    print()
    
    # 使用MCTSAgent进行决策
    print("7. 使用MCTSAgent进行决策...")
    print("   (这可能需要几秒钟，取决于模拟次数和动作采样数量)")
    mcts_action = mcts_agent.decision(balls, my_targets, table)
    print(f"   MCTSAgent动作: {mcts_action}")
    print()
    
    # 比较结果
    print("8. 比较三种Agent的输出...")
    print(f"   {'参数':<10} {'BasicAgent':<20} {'DirectModelAgent':<20} {'MCTSAgent':<20}")
    print("   " + "-" * 70)
    
    for key in ['V0', 'phi', 'theta', 'a', 'b']:
        basic_val = f"{basic_action[key]:.4f}"
        direct_val = f"{direct_action[key]:.4f}"
        mcts_val = f"{mcts_action[key]:.4f}"
        print(f"   {key:<10} {basic_val:<20} {direct_val:<20} {mcts_val:<20}")
    print()
    
    # 演示状态缓冲区管理
    print("9. 演示状态缓冲区管理...")
    print(f"   DirectModelAgent缓冲区大小: {len(direct_agent.state_buffer)}")
    print(f"   MCTSAgent缓冲区大小: {len(mcts_agent.state_buffer)}")
    
    direct_agent.clear_buffer()
    mcts_agent.clear_buffer()
    print(f"   清空缓冲区后:")
    print(f"   DirectModelAgent缓冲区大小: {len(direct_agent.state_buffer)}")
    print(f"   MCTSAgent缓冲区大小: {len(mcts_agent.state_buffer)}")
    print()
    
    print("10. 演示模型切换...")
    print("   从DirectModelAgent切换到BasicAgent...")
    
    # 创建一个新的环境实例用于测试
    env2 = PoolEnv()
    env2.reset(target_ball='stripe')
    
    player2 = env2.get_curr_player()
    balls2, my_targets2, table2 = env2.get_observation(player2)
    
    # 使用BasicAgent
    basic_agent2 = BasicAgent()
    basic_action2 = basic_agent2.decision(balls2, my_targets2, table2)
    print(f"   BasicAgent动作: {basic_action2}")
    
    # 使用DirectModelAgent
    direct_action2 = direct_agent.decision(balls2, my_targets2, table2)
    print(f"   DirectModelAgent动作: {direct_action2}")
    print()
    
    print("使用示例完成！")
    print("=" * 50)
    print("提示：")
    print("1. 可以通过调整n_simulations和n_action_samples参数来平衡性能和速度")
    print("2. DirectModelAgent适合快速决策，MCTSAgent适合需要更优决策的场景")
    print("3. 可以使用load_model方法加载不同的预训练模型")
    print("4. 状态缓冲区会自动维护最近3个状态，也可以手动清空")
    print("5. 两种Agent都支持在初始化后动态设置模型和环境")

if __name__ == '__main__':
    main()
