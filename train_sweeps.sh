#!/bin/bash

# Directory containing the subdirectories
BASE_DIR="masks/induction"

# Iterate through subdirectories
for subdir in "$BASE_DIR"/*; do
    if [ -d "$subdir" ]; then
        # Extract the subdirectory name
        config_dir="${subdir##*/}"
        
        # Remove the first 3 letters from the subdirectory name
        job_name="${config_dir:3}"
        
        # Submit the Slurm job
        sbatch --job-name="$job_name" \
               --output="jupyter_logs/log-%J.txt" \
               --nodes=1 \
               --tasks-per-node=1 \
               --gres=gpu:1 \
               --time=8:00:00 \
               --wrap="source /data/phillip_guo/miniconda3/etc/profile.d/conda.sh && \
                       conda activate unlrn && \
                       python setup_models.py --config_dir=$subdir"
    fi
done