#!/bin/bash
#SBATCH --job-name=train_mask
#SBATCH --output=jupyter_logs/log-%J.txt
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00

source /data/phillip_guo/miniconda3/etc/profile.d/conda.sh
conda activate unlrn
python setup_models.py