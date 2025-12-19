#!/bin/bash

#SBATCH --job-name=test
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1            #若使用 2 块卡，就给 gres=gpu:2
#SBATCH --mail-type=end
#SBATCH --mail-user=YOU@EMAIL.COM
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load gcc cuda

./cudaTensorCoreGemm