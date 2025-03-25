# iterate through "localization_type"s and launch runs
import argparse
import subprocess
parser = argparse.ArgumentParser()
parser.add_argument("--parent_dir", type=str)
args = parser.parse_args()

# for each subdir of parent_dir, look for config.json and launch runs

# save_dir is the directory of the config file
import os

num_jobs = 0

for subdir in os.listdir(args.parent_dir):
    print(f"checking {subdir}")
    # check if subdir contains config.json
    subdir_path = os.path.join(args.parent_dir, subdir)
    if os.path.exists(os.path.join(subdir_path, "config.json")):
        config_path = os.path.join(subdir_path, "config.json")
        save_dir = os.path.dirname(config_path)
        print(f"Submitting job for {subdir}")
        job_script = f"""#!/bin/bash
#SBATCH --account=cais
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --job-name=sports_sweep_{subdir}
#SBATCH --output={save_dir}/slurm/slurm_%j.out

source ~/.bashrc
conda activate cb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python finetuning_scripts/localized_finetuning_script_counterfact.py --config_path={config_path}
"""
        os.makedirs(f"{save_dir}/slurm", exist_ok=True)
        job_script_path = f"{save_dir}/slurm/sports_sweep_{subdir}.sh"
        with open(job_script_path, "w") as f:
            f.write(job_script)

        # Submit the job using sbatch
        subprocess.run(["sbatch", job_script_path])

        # Remove the temporary job script file
        os.remove(job_script_path)

        num_jobs += 1
    else:
        print(f"No config.json found in {subdir}")

print(f"Submitted {num_jobs} Slurm jobs.")