import argparse
import json
import os


parser = argparse.ArgumentParser()
parser.add_argument("--parent_dir", type=str, required=True)
parser.add_argument("--sweep_over", type=str, required=True, choices=["learning_rate", "forget_loss_coef"])
possible_lrs = [2e-6, 5e-6, 1e-5, 2e-5]
possible_flcs = [0.2, 0.5, 1, 2]
parser.add_argument("--learning_rate_dict", type=json.loads, default=None)
parser.add_argument("--forget_loss_coef_dict", type=json.loads, default=None)
parser.add_argument("--parent_config_path", type=str, default=None)
args = parser.parse_args()

# localization_types = ["nonlocalized", "manual_interp", "localized_ct"]
parent_config = json.load(open(args.parent_config_path, "r"))
localization_types = []
base_config = {}
localization_specific_configs = {}
for key, value in parent_config.items():
    if isinstance(value, dict):
        localization_types.append(key)
        localization_specific_configs[key] = value
    else:
        base_config[key] = value

if args.sweep_over == "learning_rate" and args.forget_loss_coef_dict is None:
    args.forget_loss_coef_dict = {"nonlocalized": 1, "localized_ct": 1, "manual_interp": 1}
    print(f"Using default forget loss coefs, {args.forget_loss_coef_dict}")
if args.sweep_over == "forget_loss_coef" and args.learning_rate_dict is None:
    args.learning_rate_dict = {"nonlocalized": 1e-5, "localized_ct": 1e-5, "manual_interp": 1e-5}
    print(f"Using default learning rates, {args.learning_rate_dict}")

for localization_type in localization_types:
    if args.sweep_over == "learning_rate":
        for learning_rate in possible_lrs:
            config = base_config.copy()
            config["localization_type"] = localization_type
            config.update(localization_specific_configs[localization_type])
            config["learning_rate"] = learning_rate

            new_folder = os.path.join(args.parent_dir, f"{localization_type}_lr{learning_rate}")
            os.makedirs(new_folder, exist_ok=True)
            with open(os.path.join(new_folder, "config.json"), "w") as f:
                json.dump(config, f)
    elif args.sweep_over == "forget_loss_coef":
        for forget_loss_coef in possible_flcs:
            config = base_config.copy()
            config["localization_type"] = localization_type
            config.update(localization_specific_configs[localization_type])
            config["forget_loss_coef"] = forget_loss_coef

            new_folder = os.path.join(args.parent_dir, f"{localization_type}_flc{forget_loss_coef}")
            os.makedirs(new_folder, exist_ok=True)
            with open(os.path.join(new_folder, "config.json"), "w") as f:
                json.dump(config, f)