import json
import subprocess
import copy
import os

# Define the base configuration
base_config = {
    "train_batch_size": 2,
    "eval_batch_size": 50,
    "device": "cuda",
    "train_loss_type": "sports",
    "forget_sport": "basketball",
    "maintain_sport": "null",
    "model_name": "google/gemma-7b",
    "model_type": "gemma",
    "learning_rate": 0.01,
    "n_epochs": 50,
    "grad_accum_steps": 15,
    "alpha": 0.2,
    "beta": 3e-7,
    "clip_grad": 1,
    "evaluate_every": 5,
    "n_eval_iters": 5,
    "do_adversarial_evals": True,
    "do_side_effects_evals": True,
    "localization_type": "manual",
    "localization_top_p": 0.05
}

# Define the values to sweep over
forget_sports = ["basketball", "baseball", "football"]
localization_types = ["ct", "manual", "random"]

# Iterate over all combinations of the sweep values
for forget_sport in forget_sports:
    for localization_type in localization_types:
        print(f'Running for {forget_sport} and {localization_type}')
        # Create a copy of the base configuration
        config = copy.deepcopy(base_config)
        
        # Update the configuration with the current sweep values
        config["forget_sport"] = forget_sport
        config["localization_type"] = localization_type
        print(config)
        
        # Define the results directory path
        results_dir = f'results/{config["model_name"].replace("/", "_")}-{forget_sport}-{localization_type}.pt'
        
        # Check if the results directory already exists
        if os.path.exists(results_dir):
            print(f'Skipping run for {forget_sport} and {localization_type} as results directory already exists.')
            continue
        
        # Save the updated configuration to a JSON file
        config_filename = f'config_{forget_sport}_{localization_type}.json'
        with open(config_filename, 'w') as config_file:
            json.dump(config, config_file, indent=4)
        
        # Run the weight_mask.py script with the updated configuration
        command = ['nohup', 'python', 'weight_mask.py', f'--config_dir={config_filename}']
        subprocess.run(command)
