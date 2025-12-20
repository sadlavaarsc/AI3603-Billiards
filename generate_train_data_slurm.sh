#!/bin/bash
#SBATCH --job-name=generate_data
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mail-type=end
#SBATCH --mail-user=1iwenta0@sjtu.edu.cn
#SBATCH --output=%j.out
#SBATCH --error=%j.err

TOTAL_MATCHES=10000
TASK_ID=$SLURM_PROCID
TOTAL_TASKS=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

MATCHES_PER_TASK=$((TOTAL_MATCHES / TOTAL_TASKS))
REMAINING_MATCHES=$((TOTAL_MATCHES % TOTAL_TASKS))

if [ $TASK_ID -lt $REMAINING_MATCHES ]; then
    CURRENT_TASK_MATCHES=$((MATCHES_PER_TASK + 1))
    START_ID=$((TASK_ID * CURRENT_TASK_MATCHES))
else
    CURRENT_TASK_MATCHES=$MATCHES_PER_TASK
    START_ID=$((REMAINING_MATCHES * (MATCHES_PER_TASK + 1) + (TASK_ID - REMAINING_MATCHES) * MATCHES_PER_TASK))
fi

if [ $CURRENT_TASK_MATCHES -le 0 ]; then
    echo "Task ${TASK_ID}: No execution needed (total tasks exceed total matches)"
    exit 0
fi

echo "Task ${TASK_ID}: Starting execution, start_id=${START_ID}, num_matches=${CURRENT_TASK_MATCHES}"

python your_data_script.py \
    --start_id $START_ID \
    --num_matches $CURRENT_TASK_MATCHES \
    --match_dir "match_data/task_${TASK_ID}" \
    --behavior_dir "training_data/behavior/task_${TASK_ID}" \
    --value_dir "training_data/value/task_${TASK_ID}" \
    --enable_noise \
    --max_hit_count 60 \
    --verbose

if [ $? -ne 0 ]; then
    echo "Task ${TASK_ID} failed (start_id=${START_ID}, num_matches=${CURRENT_TASK_MATCHES})"
    exit 1
fi

echo "Task ${TASK_ID} completed successfully (start_id=${START_ID}, num_matches=${CURRENT_TASK_MATCHES})"
exit 0