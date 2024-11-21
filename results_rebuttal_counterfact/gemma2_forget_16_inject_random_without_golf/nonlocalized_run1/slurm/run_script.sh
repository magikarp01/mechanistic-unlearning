#!/bin/bash
#SBATCH --account=cais
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --job-name=sports_nonlocalized_1
#SBATCH --output=results_rebuttal_counterfact/gemma2_forget_16_inject_random_without_golf/nonlocalized_run1/slurm/slurm_%j.out

source ~/.bashrc
conda activate cb

python finetuning_scripts/localized_finetuning_script_sports.py --config_path=results_rebuttal_counterfact/gemma2_forget_16_inject_random_without_golf/nonlocalized_run1/config.json
