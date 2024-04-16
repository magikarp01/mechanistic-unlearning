#!/bin/bash

# Directory containing the subdirectories
BASE_DIR="masks/threshold_sweep_ioi/edge_masks/higher_lr_mask_zeros"
# cutoff=12
cutoff=22
# Subdirectories to exclude
EXCLUDE_DIRS=("masks/threshold_sweep_ioi/edge_masks/higher_lr_mask_zeros/mask_k=10_acdcpp" "masks/threshold_sweep_ioi/edge_masks/higher_lr_mask_zeros/mask_k=10_none" "masks/threshold_sweep_ioi/edge_masks/higher_lr_mask_zeros/threshold=0.5_acdcpp" "masks/threshold_sweep_ioi/edge_masks/higher_lr_mask_zeros/threshold=0.5_none")

# Iterate through subdirectories
for subdir in "$BASE_DIR"/*; do
    if [ -d "$subdir" ]; then
        # Check if the current directory is in the exclude list
        skip=
        for exclude in "${EXCLUDE_DIRS[@]}"; do
            if [[ "$subdir" == *"$exclude"* ]]; then
                skip=1
                break
            fi
        done
        if [[ "$skip" ]]; then
            continue
        fi

        # Extract the job name
        job_name="${subdir:cutoff}"

        # Get the current date and time
        date_suffix=$(date "+%d-%H%M")
        # Concatenate the date and time to the job name
        # wandb_name="${job_name}_${date_suffix}"

        # Submit the Slurm job
        sbatch --job-name="$job_name" \
               --output="jupyter_logs/log-%J.txt" \
               --nodes=1 \
               --tasks-per-node=1 \
               --gres=gpu:1 \
               --time=8:00:00 \
               --wrap="source /data/phillip_guo/miniconda3/etc/profile.d/conda.sh && \
                       conda activate unlrn && \
                       python setup_models.py --config_dir=$subdir --wandb_name=$job_name"
    fi
done