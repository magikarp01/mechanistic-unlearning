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

localization_types=("localized_ap" "manual_interp" "random" "all_mlps" "nonlocalized")

for localization_type in "${localization_types[@]}"
do
    sbatch --export=ALL,LOCALIZATION_TYPE=$localization_type <<EOT
#!/bin/bash
#SBATCH --job-name=16,$localization_type
#SBATCH --output=jupyter_logs/log-%J.txt
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00

source /data/phillip_guo/miniconda3/etc/profile.d/conda.sh
conda activate unlrn
python localized_finetuning_script.py --model_type "gemma-2-9b" --forget_athletes 16 --localization_type \$LOCALIZATION_TYPE --run_id "3" --n_epochs 50 --do_full_mmlu_evals True --do_relearning_evals True --do_probing_evals True --train_batch_size 2
EOT
done

# to turn something off, just don't include it in the command (don't set "False", since that will still evaluate to True)
# #SBATCH --job-name=bb,inj,$localization_type
# python localized_finetuning_script.py --model_type "gemma-2-9b" --forget_sport "basketball" --inject_sport "golf" --localization_type \$LOCALIZATION_TYPE --run_id "4" --n_epochs 50 --do_full_mmlu_evals True --do_relearning_evals True --do_probing_evals True --train_batch_size 2

# #SBATCH --job-name=bb,$localization_type
# python localized_finetuning_script.py --model_type "gemma-2-9b" --forget_sport "basketball" --localization_type \$LOCALIZATION_TYPE --run_id "3" --n_epochs 50 --do_full_mmlu_evals True --do_relearning_evals True --do_probing_evals True --train_batch_size 2

# #SBATCH --job-name=16,inj,$localization_type
# python localized_finetuning_script.py --model_type "gemma-2-9b" --forget_athletes 16 --inject_sport "golf" --localization_type \$LOCALIZATION_TYPE --run_id "3" --n_epochs 50 --do_full_mmlu_evals True --do_relearning_evals True --do_probing_evals True --train_batch_size 2

# #SBATCH --job-name=16,$localization_type
# python localized_finetuning_script.py --model_type "gemma-2-9b" --forget_athletes 16 --localization_type \$LOCALIZATION_TYPE --run_id "3" --n_epochs 50 --do_full_mmlu_evals True --do_relearning_evals True --do_probing_evals True --train_batch_size 2



# python localized_finetuning_script.py --model_type "gemma-2-9b" --forget_sport "basketball" --localization_type "nonlocalized" --run_id "3" --n_epochs 5 --do_relearning_evals True --do_probing_evals True --train_batch_size 2