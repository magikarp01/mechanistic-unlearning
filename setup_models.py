#!/usr/bin/env python
# coding: utf-8

# # Set up models for edge or weight masking

# ## Workflow:
# - Load model
# - Use Task with clean and corrupt data, use ACDCPP and get the ACDCPP-style edges
# - Convert ACDCPP-style edges to edge mask, get either edge superset of node superset
# - Apply these masks to the mask training, either by limiting edge mask to only edge superset, node superset, or by limiting weight mask to node superset
# 
# - Also need to test other baselines, like regular finetuning

# In[1]:


import os
import sys
import pickle
sys.path.append('acdcpp/Automatic-Circuit-Discovery/')
sys.path.append('acdcpp/')
from acdc import TLACDCExperiment
from acdcpp.ACDCPPExperiment import ACDCPPExperiment
import os
import sys
import re

from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.acdc_utils import TorchIndex, EdgeType
import numpy as np
import torch as t
from torch import Tensor
import einops
import itertools
from transformer_lens import HookedTransformer, ActivationCache
import tqdm.notebook as tqdm
import plotly
from rich import print as rprint
from rich.table import Table
from jaxtyping import Float, Bool
from typing import Callable, Tuple, Union, Dict, Optional
import torch

device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
print(f'Device: {device}')
from ACDCPPExperiment import ACDCPPExperiment
from cb_utils.mask_utils import get_masks_from_acdcpp_exp


# In[3]:

import json
import argparse
import os

# Create the parser
parser = argparse.ArgumentParser(description="Set model parameters")

# Add the arguments
parser.add_argument('--config_dir', type=str, required=True, help='Path to the directory with configuration file')

# Parse the arguments
args = parser.parse_args()

# Load the configuration file
with open(args.config_dir+"/config.json", 'r') as f:
    config = json.load(f)

# Now you can use these arguments in your code
edge_masks = config['edge_masks']
weight_masks_attn = config['weight_masks_attn']
weight_masks_mlp = config['weight_masks_mlp']
train_base_weights = config['train_base_weights']
localize_acdcpp = config['localize_acdcpp']
localize_task = config['localize_task'] # "ioi" or "induction" for now

use_uniform = config['use_uniform']
uniform_type = config['uniform_type']
unlrn_task_weight = config['unlrn_task_weight']
epochs_left = config['epochs_left']
steps_per_epoch = config['steps_per_epoch']
lr = config['lr']
weight_decay = config['weight_decay']
evaluate_every = config['evaluate_every']
discretize_every = config['discretize_every']
threshold = config['threshold']
use_wandb = config['use_wandb']
edge_mask_reg_strength = config['edge_mask_reg_strength']
weight_mask_reg_strength = config['weight_mask_reg_strength']
num_eval_steps = config['num_eval_steps']
save_every = config['save_every']

# If save_path is None, set it to the directory of the config file
if config['save_path'] is None:
    save_path = args.config_dir + f"/ckpts"
else:
    save_path = config['save_path']

# In[3.5]

# set up pipeline from acdcpp to edge mask
model = HookedTransformer.from_pretrained(
    'gpt2-small',
    center_writing_weights=False,
    center_unembed=False,
    fold_ln=False,
    device=device,
)
model.set_use_hook_mlp_in(True)
model.set_use_split_qkv_input(True)
model.set_use_attn_result(True)


# In[4]:
if localize_task == "ioi":

    from tasks.ioi.IOITask import IOITask_old, IOITask
    ioi_task = IOITask(batch_size=5, tokenizer=model.tokenizer, device=device, prep_acdcpp=True, acdcpp_N=25, nb_templates=1, prompt_type="ABBA")
    ioi_task.set_logit_diffs(model)

    ioi_metric = ioi_task.get_acdcpp_metric()
    def negative_abs_ioi_metric(logits: Float[Tensor, "batch seq_len d_vocab"]):
        return -abs(ioi_metric(logits))

    # In[7]:

    THRESHOLDS = [0.08, .15]#np.arange(0.005, 0.155, 0.005)
    RUN_NAME = 'abs_edge'

    acdcpp_exp = ACDCPPExperiment(
        model=model,
        clean_data=ioi_task.clean_data.toks,
        corr_data=ioi_task.corr_data.toks,
        acdc_metric=negative_abs_ioi_metric,
        acdcpp_metric=ioi_metric,
        thresholds=THRESHOLDS,
        run_name=RUN_NAME,
        verbose=False,
        attr_absolute_val=True,
        save_graphs_after=-100,
        pruning_mode='edge',
        no_pruned_nodes_attr=1,
        run_acdc=False,
        run_acdcpp=True,
    )

elif localize_task == "induction":
    from tasks.induction.InductionTask import InductionTask
    ind_task = InductionTask(batch_size=5, tokenizer=model.tokenizer, prep_acdcpp=True, seq_len=15, acdcpp_metric="ave_logit_diff")
    ind_task.set_logit_diffs(model)

    ind_metric = ind_task.get_acdcpp_metric()
    def negative_abs_ind_metric(logits: Float[Tensor, "batch seq_len d_vocab"]):
        return -abs(ind_metric(logits))

    THRESHOLDS = [0.05]#np.arange(0.005, 0.155, 0.005)
    RUN_NAME = 'abs_edge'

    acdcpp_exp = ACDCPPExperiment(
        model=model,
        clean_data=ind_task.clean_data,
        corr_data=ind_task.corr_data,
        acdc_metric=negative_abs_ind_metric,
        acdcpp_metric=ind_metric,
        thresholds=THRESHOLDS,
        run_name=RUN_NAME,
        verbose=False,
        attr_absolute_val=True,
        save_graphs_after=-100,
        pruning_mode='edge',
        no_pruned_nodes_attr=1,
        run_acdc=False,
        run_acdcpp=True,
    )

acdcpp_nodes, acdcpp_edges, acdcpp_mask_dict, acdcpp_weight_mask_attn_dict, acdcpp_weight_mask_mlp_dict = get_masks_from_acdcpp_exp(acdcpp_exp, threshold=THRESHOLDS[0])

print(acdcpp_nodes)

# In[12]:


from cb_utils.transformer import DemoTransformer
from cb_utils.models import load_demo_gpt2, tokenizer
# means_ioi = True
# if means_ioi:
#     with open("data/gpt2_ioi_abc_means.pkl", "rb") as f:
#         means = pickle.load(f)[0]
# else:
#     with open("data/gpt2_means.pkl", "rb") as f:
#         means = pickle.load(f)[0]

#%%


# if edge_masks is True, then have mask_dict_superset be acdcpp_mask_dict
# model = load_demo_gpt2(means=means, mask_dict_superset=acdcpp_mask_dict)
if localize_acdcpp:
    mask_dict_superset = acdcpp_mask_dict if edge_masks else None
    weight_mask_attn_dict = acdcpp_weight_mask_attn_dict if weight_masks_attn else None
    weight_mask_mlp_dict = acdcpp_weight_mask_mlp_dict if weight_masks_mlp else None
    base_weight_attn_dict = acdcpp_weight_mask_attn_dict if train_base_weights else None
    base_weight_mlp_dict = acdcpp_weight_mask_mlp_dict if train_base_weights else None

else:
    mask_dict_superset = None
    weight_mask_attn_dict = None
    weight_mask_mlp_dict = None
    base_weight_attn_dict = None
    base_weight_mlp_dict = None


model = load_demo_gpt2(means=False, edge_masks=edge_masks, mask_dict_superset=mask_dict_superset, weight_masks_attn=weight_masks_attn, weight_masks_mlp=weight_masks_mlp, weight_mask_attn_dict=weight_mask_attn_dict, weight_mask_mlp_dict=weight_mask_mlp_dict, train_base_weights=train_base_weights, base_weight_attn_dict=base_weight_attn_dict, base_weight_mlp_dict=base_weight_mlp_dict)

# In[13]:

from tasks import IOITask, SportsTask, OWTTask, IOITask_Uniform, GreaterThanTask, InductionTask, InductionTask_Uniform
batch_size = 80
sports = SportsTask(batch_size=batch_size*2, tokenizer=tokenizer, device=device)
owt = OWTTask(batch_size=batch_size, tokenizer=tokenizer, device=device, ctx_length=40)
greaterthan = GreaterThanTask(batch_size=batch_size, tokenizer=tokenizer, device=device)
ioi = IOITask(batch_size=batch_size, tokenizer=tokenizer, device=device, prep_acdcpp=False)
induction = InductionTask(batch_size=batch_size, tokenizer=tokenizer, prep_acdcpp=False, seq_len=15)

if localize_task == "ioi":
    ioi_uniform = IOITask_Uniform(batch_size=batch_size, tokenizer=tokenizer, device=device, uniform_over=uniform_type)

    ioi_task_2 = IOITask(batch_size=batch_size*2, tokenizer=tokenizer, device=device, nb_templates=1, prompt_type="ABBA", template_start_idx=1) # slightly different template

    ioi_task_3 = IOITask(batch_size=batch_size*2, tokenizer=tokenizer, device=device, nb_templates=1, prompt_type="BABA", template_start_idx=0) # different name format

    # train_tasks = {"ioi": ioi, "owt": owt}
    if use_uniform:
        train_tasks = {"ioi_uniform": ioi_uniform, "owt": owt}
        task_weights = {"ioi_uniform": unlrn_task_weight, "owt": 1} # I think means preserve OWT, corrupt IOI
    else:
        train_tasks = {"ioi": ioi, "owt": owt}
        task_weights = {"ioi": unlrn_task_weight, "owt": 1}

    eval_tasks = {"ioi": ioi, "induction": induction, "sports": sports, "owt": owt, "ioi_2": ioi_task_2, "ioi_3": ioi_task_3, "greaterthan": greaterthan}

elif localize_task == "induction":
    induction_uniform = InductionTask_Uniform(batch_size=batch_size, tokenizer=tokenizer, prep_acdcpp=False, seq_len=15, uniform_over=uniform_type)
    
    if use_uniform:
        train_tasks = {"induction_uniform": induction_uniform, "owt": owt}
        task_weights = {"induction_uniform": unlrn_task_weight, "owt": 1}

    else:
        train_tasks = {"induction": induction, "owt": owt}
        task_weights = {"induction": unlrn_task_weight, "owt": 1}

    eval_tasks = {"ioi": ioi, "induction": induction, "sports": sports, "owt": owt, "greaterthan": greaterthan}

# In[14]:


mask_params = []
param_names = []
for name, p in model.named_parameters():
    if p.requires_grad:
        param_names.append(name)
        mask_params.append(p)

print(param_names)

# In[16]:


from cb_utils.learn_mask import train_masks

wandb_config = {
    "edge_masks": edge_masks, "weight_masks_attn": weight_masks_attn, "weight_masks_mlp": weight_masks_mlp,  "train_base_weights": train_base_weights, "localize_acdcpp": localize_acdcpp, "localize_task": localize_task,
    "uniform_type": uniform_type, "unlrn_task_weight": unlrn_task_weight, "use_uniform": use_uniform, 
    "epochs": epochs_left, "steps_per_epoch": steps_per_epoch, "lr": lr, "weight_decay": weight_decay, "evaluate_every": evaluate_every, "discretize_every": discretize_every, "threshold": threshold, "edge_mask_reg_strength": edge_mask_reg_strength, "weight_mask_reg_strength": weight_mask_reg_strength}

optimizer = torch.optim.AdamW(mask_params, lr=lr, weight_decay=weight_decay)
train_losses, test_losses = train_masks(model, tasks=train_tasks, optimizer=optimizer, num_epochs=epochs_left, steps_per_epoch=steps_per_epoch,
            # param_names=param_names, mask_params=mask_params, 
            task_weights=task_weights, eval_tasks=eval_tasks, evaluate_every=evaluate_every, discretize_every=discretize_every, save_every=save_every,
            threshold=threshold, edge_mask_reg_strength=edge_mask_reg_strength, weight_mask_reg_strength=weight_mask_reg_strength, verbose=False, use_wandb=use_wandb, wandb_config=wandb_config, save_dir=save_path,)


# In[17]:


import pickle
# with open(f"masks/trained_mask_params_{epochs_left=}_{edge_mask_reg_strength=}_{uniform_type=}/final_params.pkl", "wb") as f:
with open(f"{save_path}/final_params.pkl", "wb") as f:
    pickle.dump(mask_params, f)

with open(f"{save_path}/final_losses.pkl", "wb") as f:
    pickle.dump((train_losses, test_losses), f)

# In[18]:


for name, p in zip(param_names, mask_params):
    if p.requires_grad:
        # print(name, p)
        # count how many zeros in p
        print(torch.sum(p == 0))

