#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=jupyter_logs/log-%J.txt
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --account=cais
#SBATCH --time=3:00:00

# cd ..
source /data/phillip_guo/miniconda3/etc/profile.d/conda.sh
conda activate cb

localization_types=("manual_interp" "random" "all_mlps" "nonlocalized" "new_forget_ct")
run_ids=(11 12 13)

for localization_type in "${localization_types[@]}"
do
    for run_id in "${run_ids[@]}"
    do
        sbatch --export=ALL,LOCALIZATION_TYPE=$localization_type,RUN_ID=$run_id <<EOT
#!/bin/bash
#SBATCH --job-name=$run_id,$localization_type
#SBATCH --output=jupyter_logs/log-%J.txt
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --account=cais
#SBATCH --time=3:00:00

source /data/phillip_guo/miniconda3/etc/profile.d/conda.sh
conda activate cb
python localized_finetuning_script_counterfact.py --model_type "gemma-2-9b" --forget_facts 16 --localization_type \$LOCALIZATION_TYPE --run_id "\$RUN_ID" --n_epochs 50 --do_full_mmlu_evals True --do_relearning_evals True --learning_rate 2e-5 --n_relearn_facts 16
EOT
    done
done

# localization_types=("manual_interp" "random" "all_mlps" "nonlocalized")

# --inject_fact True 