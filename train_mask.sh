#!/bin/bash
# #SBATCH --job-name=edge_mask
# #SBATCH --job-name=weight_mask
#SBATCH --job-name=ft_baseline
#SBATCH --output=jupyter_logs/log-%J.txt
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00

source /data/phillip_guo/miniconda3/etc/profile.d/conda.sh
conda activate unlrn
# python setup_models.py --edge_masks True --weight_masks_attn False --weight_masks_mlp False --freeze_base_weights True
# python setup_models.py --edge_masks False --weight_masks_attn True --weight_masks_mlp True --freeze_base_weights True
# python setup_models.py --edge_masks False --weight_masks_attn False --weight_masks_mlp False --freeze_base_weights False

# python setup_models.py --weight_masks_attn --weight_masks_mlp --freeze_base_weights --ioi_task_weight=-.2 --use_wandb
# python setup_models.py --edge_masks --freeze_base_weights --ioi_task_weight=-.2 --use_wandb
# python setup_models.py --ioi_task_weight=-.2 --use_wandb

# python setup_models.py --edge_masks --freeze_base_weights --use_uniform --ioi_task_weight=.2 --ioi_uniform_type=IO_S --use_wandb
# python setup_models.py --weight_masks_attn --weight_masks_mlp --freeze_base_weights --use_uniform --ioi_task_weight=.2 --ioi_uniform_type=IO_S --use_wandb
python setup_models.py --use_uniform --ioi_task_weight=.2 --ioi_uniform_type=IO_S --use_wandb
