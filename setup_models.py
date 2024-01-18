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


# In[2]:


import os
import sys
import re

# import acdc
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


# In[3]:


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


from tasks.ioi.IOITask import IOITask_old, IOITask
ioi_task = IOITask(batch_size=5, tokenizer=model.tokenizer, device=device, prep_acdcpp=True, acdcpp_N=25, nb_templates=1, prompt_type="ABBA")
ioi_task.set_logit_diffs(model)

ioi_metric = ioi_task.get_acdcpp_metric()
def negative_abs_ioi_metric(logits: Float[Tensor, "batch seq_len d_vocab"]):
    return -abs(ioi_metric(logits))

with t.no_grad():
    clean_logits = model(ioi_task.clean_data.toks)
    corrupt_logits = model(ioi_task.corr_data.toks)
    clean_logit_diff = ioi_task.ave_logit_diff(clean_logits, ioi_task.clean_data).item()
    corrupt_logit_diff = ioi_task.ave_logit_diff(corrupt_logits, ioi_task.corr_data).item()

# In[7]:

from ACDCPPExperiment import ACDCPPExperiment
from cb_utils.mask_utils import get_masks_from_acdcpp_exp
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
# e=acdcpp_exp.setup_exp(0.0)

acdcpp_nodes, acdcpp_edges, acdcpp_mask_dict, acdcpp_weight_mask_attn_dict, acdcpp_weight_mask_mlp_dict = get_masks_from_acdcpp_exp(acdcpp_exp, threshold=0.08)


# In[12]:


from cb_utils.transformer import DemoTransformer
from cb_utils.models import load_demo_gpt2, tokenizer
means_ioi = True
if means_ioi:
    with open("data/gpt2_ioi_abc_means.pkl", "rb") as f:
        means = pickle.load(f)[0]
else:
    with open("data/gpt2_means.pkl", "rb") as f:
        means = pickle.load(f)[0]

# edge_masks = False
# weight_masks_attn = True
# weight_masks_mlp = True
# freeze_base_weights = True
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Set model parameters")

# Add the arguments
parser.add_argument('--edge_masks', action='store_true', help='Set edge masks')
parser.add_argument('--weight_masks_attn', action='store_true', help='Set weight masks for attention')
parser.add_argument('--weight_masks_mlp', action='store_true', help='Set weight masks for MLP')
parser.add_argument('--train_base_weights', action='store_true', help='Train base weights')
parser.add_argument('--localize_acdcpp', action='store_true', help='Localize training to acdcpp mask')

parser.add_argument('--use_uniform', action='store_true', help='Set use uniform')
parser.add_argument('--ioi_uniform_type', type=str, default="IO_S", help='Set IOI uniform type')
parser.add_argument('--ioi_task_weight', type=float, default=-.2, help='How much to weigh IOI or IOI Uniform task, should be negative for IOI and positive for IOI Uniform')

parser.add_argument('--run_name', type=str, default=None, help='Set run name')
parser.add_argument('--epochs_left', type=int, default=200, help='Set epochs left')
parser.add_argument('--steps_per_epoch', type=int, default=20, help='Set steps per epoch')
parser.add_argument('--lr', type=float, default=1e-2, help='Set learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='Set weight decay')
parser.add_argument('--evaluate_every', type=int, default=2, help='Set evaluate every')
parser.add_argument('--discretize_every', type=int, default=40, help='Set discretize every')
parser.add_argument('--threshold', type=float, default=0.5, help='Set threshold')
parser.add_argument('--use_wandb', action='store_true', help='Set use wandb')
parser.add_argument('--edge_mask_reg_strength', type=int, default=50, help='Set edge mask reg strength')
parser.add_argument('--weight_mask_reg_strength', type=int, default=50, help='Set weight mask reg strength')
parser.add_argument('--save_every', type=int, default=20, help='Set save every')
parser.add_argument('--save_path', type=str, default=None, help='Set save path')


# Parse the arguments
args = parser.parse_args()

# Now you can use these arguments in your code
edge_masks = args.edge_masks
weight_masks_attn = args.weight_masks_attn
weight_masks_mlp = args.weight_masks_mlp
train_base_weights = args.train_base_weights
localize_acdcpp = args.localize_acdcpp

# if edge_masks is True, then have mask_dict_superset be acdcpp_mask_dict
mask_dict_superset = None if not edge_masks else acdcpp_mask_dict
# model = load_demo_gpt2(means=means, mask_dict_superset=acdcpp_mask_dict)

if localize_acdcpp:
    weight_mask_attn_dict = acdcpp_weight_mask_attn_dict if weight_masks_attn else None
    weight_mask_mlp_dict = acdcpp_weight_mask_mlp_dict if weight_masks_mlp else None

    base_weight_attn_dict = acdcpp_weight_mask_attn_dict if train_base_weights else None
    base_weight_mlp_dict = acdcpp_weight_mask_mlp_dict if train_base_weights else None

else:
    weight_mask_attn_dict = None
    weight_mask_mlp_dict = None
model = load_demo_gpt2(means=False, edge_masks=edge_masks, mask_dict_superset=mask_dict_superset, weight_masks_attn=weight_masks_attn, weight_masks_mlp=weight_masks_mlp, weight_mask_attn_dict=weight_mask_attn_dict, weight_mask_mlp_dict=weight_mask_mlp_dict, train_base_weights=train_base_weights)

# In[13]:

use_uniform = args.use_uniform
ioi_uniform_type = args.ioi_uniform_type
ioi_task_weight = args.ioi_task_weight
from tasks import IOITask, SportsTask, OWTTask, IOITask_Uniform, GreaterThanTask
batch_size = 80
ioi = IOITask(batch_size=batch_size, tokenizer=tokenizer, device=device, prep_acdcpp=False)
ioi_uniform = IOITask_Uniform(batch_size=batch_size, tokenizer=tokenizer, device=device, uniform_over=ioi_uniform_type)

ioi_task_2 = IOITask(batch_size=batch_size*2, tokenizer=tokenizer, device=device, nb_templates=1, prompt_type="ABBA", template_start_idx=1) # slightly different template

ioi_task_3 = IOITask(batch_size=batch_size*2, tokenizer=tokenizer, device=device, nb_templates=1, prompt_type="BABA", template_start_idx=0) # different name format

sports = SportsTask(batch_size=batch_size*2, tokenizer=tokenizer, device=device)
owt = OWTTask(batch_size=batch_size, tokenizer=tokenizer, device=device, ctx_length=40)
greaterthan = GreaterThanTask(batch_size=batch_size, tokenizer=tokenizer, device=device)

# train_tasks = {"ioi": ioi, "owt": owt}
if use_uniform:
    train_tasks = {"ioi_uniform": ioi_uniform, "owt": owt}
    task_weights = {"ioi_uniform": ioi_task_weight, "owt": 1} # I think means preserve OWT, corrupt IOI
else:
    train_tasks = {"ioi": ioi, "owt": owt}
    task_weights = {"ioi": ioi_task_weight, "owt": 1}

eval_tasks = {"ioi": ioi, "sports": sports, "owt": owt, "ioi_2": ioi_task_2, "ioi_3": ioi_task_3, "greaterthan": greaterthan}


# In[14]:


mask_params = []
param_names = []
for name, p in model.named_parameters():
    if p.requires_grad:
        param_names.append(name)
        mask_params.append(p)



# In[16]:


from cb_utils.learn_mask import train_masks


# Now you can use these arguments in your code
run_name = args.run_name
if run_name is None:
    run_name = f"{ioi_uniform=}_{edge_masks=}_{weight_masks_attn=}_{weight_masks_mlp=}_{train_base_weights=}_{localize_acdcpp=}"
print(run_name)
epochs_left = args.epochs_left
steps_per_epoch = args.steps_per_epoch
lr = args.lr
weight_decay = args.weight_decay
evaluate_every = args.evaluate_every
discretize_every = args.discretize_every
threshold = args.threshold
use_wandb = args.use_wandb
edge_mask_reg_strength = args.edge_mask_reg_strength
weight_mask_reg_strength = args.weight_mask_reg_strength
save_every = args.save_every
save_path = args.save_path
if args.save_path is None:
    save_path = f"masks/{run_name}"


wandb_config = {
    "edge_masks": edge_masks, "weight_masks_attn": weight_masks_attn, "weight_masks_mlp": weight_masks_mlp,  "train_base_weights": train_base_weights, "localize_acdcpp": localize_acdcpp,
    "ioi_uniform_type": ioi_uniform_type, "ioi_task_weight": ioi_task_weight, "use_uniform": use_uniform, 
    "epochs": epochs_left, "steps_per_epoch": steps_per_epoch, "lr": lr, "weight_decay": weight_decay, "evaluate_every": evaluate_every, "discretize_every": discretize_every, "threshold": threshold, "edge_mask_reg_strength": edge_mask_reg_strength, "weight_mask_reg_strength": weight_mask_reg_strength}

optimizer = torch.optim.AdamW(mask_params, lr=lr, weight_decay=weight_decay)
train_losses, test_losses = train_masks(model, tasks=train_tasks, optimizer=optimizer, num_epochs=epochs_left, steps_per_epoch=steps_per_epoch,
            # param_names=param_names, mask_params=mask_params, 
            task_weights=task_weights, eval_tasks=eval_tasks, evaluate_every=evaluate_every, discretize_every=discretize_every, save_every=save_every,
            threshold=threshold, edge_mask_reg_strength=edge_mask_reg_strength, weight_mask_reg_strength=weight_mask_reg_strength, verbose=False, use_wandb=use_wandb, wandb_config=wandb_config, save_dir=save_path,)


# In[17]:


import pickle
# with open(f"masks/trained_mask_params_{epochs_left=}_{edge_mask_reg_strength=}_{ioi_uniform_type=}/final_params.pkl", "wb") as f:
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

