# iterate through "localization_type"s and launch runs
import argparse
import subprocess
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
parser.add_argument("--localization_types", type=str, nargs="+", default=None)
parser.add_argument("--n_runs", type=int, default=1)
args = parser.parse_args()


if args.localization_types is None:
    localization_types = ["manual_interp", "nonlocalized", "localized_ct"]
else:
    localization_types = args.localization_types

# save_dir is the directory of the config file
import os
save_dir = os.path.dirname(args.config_path)
os.makedirs(f"{save_dir}/slurm", exist_ok=True)
num_jobs = 0
for localization_type in localization_types:
    for run_id in range(1, args.n_runs + 1):
        print(f"Submitting job for {localization_type}, run {run_id}")
        job_script = f"""#!/bin/bash
#SBATCH --account=cais
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --job-name=sports_{localization_type}_{run_id}
#SBATCH --output={save_dir}/slurm/slurm_%j.out

source ~/.bashrc
conda activate cb

python finetuning_scripts/localized_finetuning_script_sports.py --config_path={args.config_path} --localization_type={localization_type} --run_id={run_id}
"""
        job_script_path = f"{save_dir}/slurm/sports_{localization_type}_{run_id}.sh"
        with open(job_script_path, "w") as f:
            f.write(job_script)

        # Submit the job using sbatch
        subprocess.run(["sbatch", job_script_path])

        # Remove the temporary job script file
        os.remove(job_script_path)

        num_jobs += 1

print(f"Submitted {num_jobs} Slurm jobs.")