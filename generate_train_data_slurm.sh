#!/bin/bash
#SBATCH --job-name=generate_data
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mail-type=end
#SBATCH --mail-user=1iwenta0@sjtu.edu.cn
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# ===================== 核心参数配置 =====================
# 1. 全局配置（按需修改）
TOTAL_MATCHES=10000                                     # 总对局数
TASK_ID=$SLURM_PROCID                                  # 当前任务全局ID（从0开始，连续递增）
TOTAL_TASKS=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))  # 总并行任务数（节点数×每节点任务数）

# 2. 单个任务的对局数分配（均匀拆分，处理余数避免遗漏）
MATCHES_PER_TASK=$((TOTAL_MATCHES / TOTAL_TASKS))      # 每个任务默认分配的对局数
REMAINING_MATCHES=$((TOTAL_MATCHES % TOTAL_TASKS))     # 无法整除时的剩余对局数

# 3. 计算当前任务的start_id和实际承担的num_matches
if [ $TASK_ID -lt $REMAINING_MATCHES ]; then
    # 前REMAINING_MATCHES个任务，每个多分配1局（补全剩余对局数）
    CURRENT_TASK_MATCHES=$((MATCHES_PER_TASK + 1))
    START_ID=$((TASK_ID * CURRENT_TASK_MATCHES))
else
    # 剩余任务按默认对局数分配，计算起始ID（跳过前REMAINING_MATCHES个任务的总量）
    CURRENT_TASK_MATCHES=$MATCHES_PER_TASK
    START_ID=$((REMAINING_MATCHES * (MATCHES_PER_TASK + 1) + (TASK_ID - REMAINING_MATCHES) * MATCHES_PER_TASK))
fi

# 安全校验：避免单个任务对局数为0（当总任务数大于总对局数时触发）
if [ $CURRENT_TASK_MATCHES -le 0 ]; then
    echo "任务${TASK_ID}：无需执行（总任务数大于总对局数）"
    exit 0
fi

# ===================== 并行调用Python脚本（核心调整：移除end_id，使用start_id+num_matches） =====================
echo "任务${TASK_ID}：启动执行，start_id=${START_ID}，num_matches=${CURRENT_TASK_MATCHES}"

# 调用你的Python脚本，传递start_id和num_matches参数（移除end_id）
python your_data_script.py \
    --start_id $START_ID \
    --num_matches $CURRENT_TASK_MATCHES \
    --match_dir "match_data/task_${TASK_ID}"  # 独立目录，避免文件冲突
    --behavior_dir "training_data/behavior/task_${TASK_ID}" \
    --value_dir "training_data/value/task_${TASK_ID}" \
    --enable_noise \
    --max_hit_count 60 \
    --verbose

# ===================== 任务执行校验 =====================
if [ $? -ne 0 ]; then
    echo "❌ 任务${TASK_ID}执行失败（start_id=${START_ID}, num_matches=${CURRENT_TASK_MATCHES}）"
    exit 1
fi

echo "✅ 任务${TASK_ID}执行完成（start_id=${START_ID}, num_matches=${CURRENT_TASK_MATCHES}）"
exit 0