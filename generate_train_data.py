import os
import argparse
import time
import subprocess
from datetime import datetime

def run_command(command, cwd=None):
    """运行命令并打印输出"""
    print(f"执行命令: {' '.join(command)}")
    result = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8'  # 添加明确的编码参数，避免Windows系统上的解码错误
    )
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("警告:", result.stderr)
    
    return result.returncode == 0

def generate_data_pipeline(num_matches=10, match_dir="match_data", 
                          behavior_dir="training_data/behavior", 
                          value_dir="training_data/value",
                          enable_noise=True, max_hit_count=200,
                          skip_generation=False, skip_processing=False,
                          verbose=False, start_id=None):
    """完整的数据生成流程
    
    Args:
        num_matches: 生成的对局数量
        match_dir: 对局数据保存目录
        behavior_dir: 行为网络数据保存目录
        value_dir: 价值网络数据保存目录
        enable_noise: 是否启用噪声
        max_hit_count: 每局最大击球次数
        skip_generation: 是否跳过对局数据生成
        skip_processing: 是否跳过数据处理
        verbose: 是否打印详细信息
        start_id: 起始ID，用于并行生成时避免文件冲突
    
    Returns:
        bool: 是否成功完成
    """
    start_time = time.time()
    
    # 创建必要的目录
    os.makedirs(match_dir, exist_ok=True)
    os.makedirs(behavior_dir, exist_ok=True)
    os.makedirs(value_dir, exist_ok=True)
    
    print(f"=== 开始数据生成流程 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===")
    print(f"- 对局数量: {num_matches}")
    print(f"- 对局数据目录: {match_dir}")
    print(f"- 行为网络数据目录: {behavior_dir}")
    print(f"- 价值网络数据目录: {value_dir}")
    print(f"- 启用噪声: {enable_noise}")
    print(f"- 每局最大击球次数: {max_hit_count}")
    if start_id is not None:
        print(f"- 起始ID: {start_id}, 结束ID: {start_id + num_matches - 1}")
    print()
    
    # 1. 生成对局数据
    if not skip_generation:
        print("阶段1: 生成对局数据")
        print("-" * 50)
        
        generate_args = [
            "python", "generate_matches.py",
            "--num_matches", str(num_matches),
            "--output_dir", match_dir,
            "--max_hit_count", str(max_hit_count)
        ]
        
        if enable_noise:
            generate_args.append("--enable_noise")
        if verbose:
            generate_args.append("--verbose")
        if start_id is not None:
            generate_args.extend(["--start_id", str(start_id)])
        
        success = run_command(generate_args)
        if not success:
            print("❌ 对局数据生成失败")
            return False
        
        print("✅ 对局数据生成完成")
        print()
    else:
        print("阶段1: 跳过对局数据生成 (使用现有数据)")
        print()
    
    # 2. 处理对局数据生成训练数据
    if not skip_processing:
        print("阶段2: 处理对局数据生成训练数据")
        print("-" * 50)
        
        process_args = [
            "python", "process_match_data.py",
            "--match_dir", match_dir,
            "--behavior_output_dir", behavior_dir,
            "--value_output_dir", value_dir
        ]
        
        # 如果指定了start_id，也传递给process_match_data.py
        if start_id is not None:
            process_args.extend(["--start_id", str(start_id)])
            process_args.extend(["--end_id", str(start_id + num_matches - 1)])
        
        success = run_command(process_args)
        if not success:
            print("❌ 数据处理失败")
            return False
        
        print("✅ 训练数据生成完成")
        print()
    else:
        print("阶段2: 跳过数据处理")
        print()
    
    # 统计信息
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"=== 数据生成流程完成 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    
    # 检查生成的文件数量
    if not skip_generation:
        match_files = [f for f in os.listdir(match_dir) if f.startswith('match_') and f.endswith('.json')]
        print(f"生成的对局数据文件数: {len(match_files)}")
    
    if not skip_processing:
        behavior_files = [f for f in os.listdir(behavior_dir) if f.startswith('behavior_') and f.endswith('.json')]
        value_files = [f for f in os.listdir(value_dir) if f.startswith('value_') and f.endswith('.json')]
        print(f"生成的行为网络数据文件数: {len(behavior_files)}")
        print(f"生成的价值网络数据文件数: {len(value_files)}")
    
    return True

def validate_environment():
    """验证环境设置"""
    print("验证环境...")
    
    # 检查必要的文件是否存在
    required_files = ["poolenv.py", "generate_matches.py", "process_match_data.py"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 缺少必要文件: {', '.join(missing_files)}")
        return False
    
    print("✅ 环境验证通过")
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="台球AI训练数据生成系统")
    
    # 数据量控制参数
    parser.add_argument('--num_matches', type=int, default=10, 
                      help="生成的对局数量 (默认: 10)")
    parser.add_argument('--test_mode', action='store_true', 
                      help="启用测试模式，使用少量数据快速测试")
    parser.add_argument('--start_id', type=int, default=None, 
                      help="起始ID，用于并行生成时避免文件冲突")
    
    # 目录设置参数
    parser.add_argument('--match_dir', type=str, default="match_data",
                      help="对局数据保存目录 (默认: match_data)")
    parser.add_argument('--behavior_dir', type=str, default="training_data/behavior",
                      help="行为网络数据保存目录 (默认: training_data/behavior)")
    parser.add_argument('--value_dir', type=str, default="training_data/value",
                      help="价值网络数据保存目录 (默认: training_data/value)")
    
    # 环境参数
    parser.add_argument('--enable_noise', action='store_true', default=True,
                      help="是否启用动作噪声 (默认: 启用)")
    parser.add_argument('--max_hit_count', type=int, default=60,
                      help="每局最大击球次数 (默认: 60)")
    
    # 运行模式控制
    parser.add_argument('--skip_generation', action='store_true',
                      help="跳过对局数据生成，直接处理现有数据")
    parser.add_argument('--skip_processing', action='store_true',
                      help="跳过数据处理，只生成对局数据")
    
    # 其他选项
    parser.add_argument('--validate', action='store_true',
                      help="验证环境设置")
    parser.add_argument('--verbose', action='store_true',
                      help="打印详细信息")
    
    args = parser.parse_args()
    
    # 测试模式处理
    if args.test_mode:
        print("🔧 启用测试模式")
        args.num_matches = 2
        args.max_hit_count = 50
        args.verbose = True
    
    # 验证环境
    if args.validate or not args.skip_generation:
        if not validate_environment():
            print("环境验证失败，请确保所有必要文件都已准备好")
            return
    
    # 执行数据生成流程
    success = generate_data_pipeline(
        num_matches=args.num_matches,
        match_dir=args.match_dir,
        behavior_dir=args.behavior_dir,
        value_dir=args.value_dir,
        enable_noise=args.enable_noise,
        max_hit_count=args.max_hit_count,
        skip_generation=args.skip_generation,
        skip_processing=args.skip_processing,
        verbose=args.verbose,
        start_id=args.start_id
    )
    
    if success:
        print("🎉 数据生成流程成功完成！")
    else:
        print("❌ 数据生成流程失败，请检查错误信息")

if __name__ == "__main__":
    main()