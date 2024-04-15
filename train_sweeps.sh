#!/bin/bash

# Directory containing the subdirectories
BASE_DIR="masks/threshold_sweep_ioi/edge_masks/none"
# cutoff=12
cutoff=22
# Iterate through subdirectories
for subdir in "$BASE_DIR"/*; do
    if [ -d "$subdir" ]; then        
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