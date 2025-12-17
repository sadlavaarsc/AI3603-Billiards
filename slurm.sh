#!/bin/bash
#SBATCH -J pool_train          # Job name (can be modified as needed, e.g., pool_train_v1)
#SBATCH -p a100                # A100 GPU partition (confirm cluster partition name, modify if not a100)
#SBATCH -o pool_train_%j.out   # Output log file (%j is automatically replaced with job ID for easy tracing)
#SBATCH -e pool_train_%j.err   # Error log file
#SBATCH -N 1                   # Single node training, keep 1
#SBATCH --ntasks-per-node=1    # Single process training, keep 1
#SBATCH --cpus-per-task=12     # Allocate 12 CPU cores per task (adapted for single GPU data loading)
#SBATCH --gres=gpu:1           # [MODIFY 1] Allocate only 1 GPU (core requirement)

# -------------------------- Environment Loading --------------------------
module load miniconda3                     # Load cluster miniconda module
# Manually load conda configuration file (replace with your previously validated full path, core!)
source activate poolenv                    # [MODIFY 2] Activate your poolenv environment

# -------------------------- Start Training Script --------------------------
# [MODIFY 3] Start train.py (please replace with the absolute path of train.py, e.g., /dssg/home/acct-stu/stu337/train.py)
python /dssg/home/acct-stu/stu337/AI3603-Billiards/train.py

# If train.py needs parameters, example:
# python /dssg/home/acct-stu/stu337/train.py --epochs 100 --batch_size 64 --lr 0.001