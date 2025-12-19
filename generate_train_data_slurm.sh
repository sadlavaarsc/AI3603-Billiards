#!/bin/bash
#SBATCH --job-name=generate_data
#SBATCH --partition=debuga100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=1iwenta0@sjtu.edu.cn
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# -------------------------- Environment Loading --------------------------
module load miniconda3 cuda                    # Load cluster miniconda module
# Manually load conda configuration file (replace with your previously validated full path, core!)
source activate poolenv                    # [MODIFY 2] Activate your poolenv environment

# -------------------------- Start Training Script --------------------------
# [MODIFY 3] Start train.py (please replace with the absolute path of train.py, e.g., /dssg/home/acct-stu/stu337/train.py)
python /dssg/home/acct-stu/stu337/AI3603-Billiards/generate_train_data.py --num_matches 100000

# If train.py needs parameters, example:
# python /dssg/home/acct-stu/stu337/train.py --epochs 100 --batch_size 64 --lr 0.001