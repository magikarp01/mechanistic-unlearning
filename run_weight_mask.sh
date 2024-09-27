#!/bin/bash

python weight_mask.py --config_dir="weight_masking_config_ap.json"
python weight_mask.py --config_dir="weight_masking_config_ct.json"
python weight_mask.py --config_dir="weight_masking_config_random.json"
python weight_mask.py --config_dir="weight_masking_config_manual.json"
python weight_mask.py --config_dir="weight_masking_config_mlp.json"
python weight_mask.py --config_dir="weight_masking_config_none.json"
