#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=jupyter_logs/log-%J.txt
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00

# cd ..
source /data/phillip_guo/miniconda3/etc/profile.d/conda.sh
conda activate unlrn
python localized_finetuning_script.py --model_type "gemma-2-9b" --forget_sport "basketball" --inject_sport "golf" --localization_type "nonlocalized" --run_id "1" --n_epochs 5 --do_full_mmlu_evals True --do_relearning_evals True --do_probing_evals True --push_to_hub True