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

edge_masks = False
weight_masks_attn = True
weight_masks_mlp = True
freeze_base_weights = True

# if edge_masks is True, then have mask_dict_superset be acdcpp_mask_dict
mask_dict_superset = None if not edge_masks else acdcpp_mask_dict
# model = load_demo_gpt2(means=means, mask_dict_superset=acdcpp_mask_dict)

model = load_demo_gpt2(means=False, edge_masks=edge_masks, mask_dict_superset=mask_dict_superset, weight_masks_attn=weight_masks_attn, weight_masks_mlp=weight_masks_mlp, weight_mask_attn_dict=(acdcpp_weight_mask_attn_dict if weight_masks_attn else None), weight_mask_mlp_dict=(acdcpp_weight_mask_mlp_dict if weight_masks_mlp else None), freeze_base_weights=freeze_base_weights)

# In[13]:

ioi_uniform_type = "all_tokens"
from tasks import IOITask, SportsTask, OWTTask, IOITask_Uniform, ToxicTask
batch_size = 64
ioi = IOITask(batch_size=batch_size, tokenizer=tokenizer, device=device, prep_acdcpp=False)
ioi_uniform = IOITask_Uniform(batch_size=batch_size, tokenizer=tokenizer, device=device, uniform_over=ioi_uniform_type)

ioi_task_2 = IOITask(batch_size=batch_size*2, tokenizer=tokenizer, device=device, nb_templates=1, prompt_type="ABBA", template_start_idx=1) # slightly different template

ioi_task_3 = IOITask(batch_size=batch_size*2, tokenizer=tokenizer, device=device, nb_templates=1, prompt_type="BABA", template_start_idx=0) # different name format

sports = SportsTask(batch_size=batch_size*2, tokenizer=tokenizer, device=device)
owt = OWTTask(batch_size=batch_size, tokenizer=tokenizer, device=device)



# train_tasks = {"ioi": ioi, "owt": owt}
train_tasks = {"ioi_uniform": ioi_uniform, "owt": owt}
task_weights = {"ioi_uniform": .5, "owt": 1} # I think means preserve OWT, corrupt IOI
eval_tasks = {"ioi": ioi, "ioi_uniform": ioi_uniform, "sports": sports, "owt": owt, "ioi_2": ioi_task_2, "ioi_3": ioi_task_3}


# In[14]:


mask_params = []
param_names = []
for name, p in model.named_parameters():
    if p.requires_grad:
        param_names.append(name)
        mask_params.append(p)



# In[16]:


from cb_utils.learn_mask import train_masks

run_name = f"ioi_uniform_{edge_masks=}_{weight_masks_attn=}_{weight_masks_mlp=}_{freeze_base_weights=}"
# if edge_masks:
#     run_name += "_edge_masking"
# if weight_masks_attn:
#     run_name += "_weight_masking_attn"
# if weight_masks_mlp:
#     run_name += "_weight_masking_mlp"
# if freeze_base_weights:
#     run_name += "_freeze_base_weights"

print(run_name)
epochs_left = 200
steps_per_epoch = 20
lr = 1e-2 # free
weight_decay = 0
evaluate_every = 1
discretize_every = 50 # 5 # free
threshold = 0.5
use_wandb = True
edge_mask_reg_strength = 50
weight_mask_reg_strength = 50
save_every = 20
save_path = f"masks/{run_name}"

wandb_config = {
    "edge_masks": edge_masks, "weight_masks_attn": weight_masks_attn, "weight_masks_mlp": weight_masks_mlp,  "freeze_base_weights": freeze_base_weights, 
    "epochs": epochs_left, "steps_per_epoch": steps_per_epoch, "lr": lr, "weight_decay": weight_decay, "evaluate_every": evaluate_every, "discretize_every": discretize_every, "threshold": threshold, "edge_mask_reg_strength": edge_mask_reg_strength, "weight_mask_reg_strength": weight_mask_reg_strength}

optimizer = torch.optim.AdamW(mask_params, lr=lr, weight_decay=weight_decay)
train_masks(model, tasks=train_tasks, optimizer=optimizer, num_epochs=epochs_left, steps_per_epoch=steps_per_epoch,
            # param_names=param_names, mask_params=mask_params, 
            task_weights=task_weights, eval_tasks=eval_tasks, evaluate_every=evaluate_every, discretize_every=discretize_every, save_every=save_every,
            threshold=threshold, edge_mask_reg_strength=edge_mask_reg_strength, weight_mask_reg_strength=weight_mask_reg_strength, verbose=False, use_wandb=use_wandb, wandb_config=wandb_config, save_dir=save_path,)


# In[17]:


import pickle
# with open(f"masks/trained_mask_params_{epochs_left=}_{edge_mask_reg_strength=}_{ioi_uniform_type=}/final_params.pkl", "wb") as f:
with open(f"{save_path}/final_params.pkl", "wb") as f:
    pickle.dump(mask_params, f)


# In[18]:


for name, p in zip(param_names, mask_params):
    if p.requires_grad:
        # print(name, p)
        # count how many zeros in p
        print(torch.sum(p == 0))

