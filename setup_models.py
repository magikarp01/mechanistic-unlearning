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
ioi_task = IOITask(batch_size=5, tokenizer=model.tokenizer, device=device, prep_acdcpp=True, acdcpp_N=25)
ioi_task.set_logit_diffs(model)


# In[5]:


ioi_metric = ioi_task.get_acdcpp_metric()
def negative_abs_ioi_metric(logits: Float[Tensor, "batch seq_len d_vocab"]):
    return -abs(ioi_metric(logits))

with t.no_grad():
    clean_logits = model(ioi_task.clean_data.toks)
    corrupt_logits = model(ioi_task.corr_data.toks)
    clean_logit_diff = ioi_task.ave_logit_diff(clean_logits, ioi_task.clean_data).item()
    corrupt_logit_diff = ioi_task.ave_logit_diff(corrupt_logits, ioi_task.corr_data).item()


# In[6]:


# Get clean and corrupt logit differences
with t.no_grad():
    clean_metric = ioi_metric(clean_logits, corrupt_logit_diff, clean_logit_diff, ioi_task.clean_data)
    corrupt_metric = ioi_metric(corrupt_logits, corrupt_logit_diff, clean_logit_diff, ioi_task.corr_data)

print(f'Clean direction: {clean_logit_diff}, Corrupt direction: {corrupt_logit_diff}')
print(f'Clean metric: {clean_metric}, Corrupt metric: {corrupt_metric}')


# In[7]:


from ACDCPPExperiment import ACDCPPExperiment
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

pruned_heads, num_passes, acdcpp_pruned_attrs, acdc_pruned_attrs, edges_after_acdcpp, edges_after_acdc = acdcpp_exp.run()


# In[8]:


import pickle
with open("masks/ioi_acdcpp_edges.pkl", "wb") as f:
    pickle.dump(edges_after_acdcpp, f)


# In[9]:


from cb_utils.mask_utils import get_node_name

acdcpp_edges = set()
for edge in edges_after_acdcpp[0.08]:
    # split the edge into two nodes, e.g. blocks.1.attn.hook_result[:, :, 10]blocks.0.hook_mlp_in[:] into blocks.1.attn.hook_result[:, :, 10] and blocks.0.hook_mlp_in[:]
    node_1 = get_node_name(edge.split("]")[0]+"]", show_full_index=False)
    node_2 = get_node_name(edge.split("]")[1]+"]", show_full_index=False)
    if node_1 != node_2:
        acdcpp_edges.add((node_1, node_2))


# In[10]:


from cb_utils.mask_utils import get_edge_mask_template, get_mask_from_edges, convert_mask_dict_to_params
edge_mask_template = get_edge_mask_template()
acdcpp_mask_dict = get_mask_from_edges(acdcpp_edges, edge_mask_template=edge_mask_template, num_layers=12, num_heads=12)


# In[11]:


convert_mask_dict_to_params(acdcpp_mask_dict)


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

model = load_demo_gpt2(means=means, mask_dict_superset=acdcpp_mask_dict)


# In[13]:

ioi_uniform_type = "all_tokens"
from tasks import IOITask, SportsTask, OWTTask, IOITask_Uniform, ToxicTask
batch_size = 64
ioi = IOITask(batch_size=batch_size, tokenizer=tokenizer, device=device, prep_acdcpp=False)
ioi_uniform = IOITask_Uniform(batch_size=batch_size, tokenizer=tokenizer, device=device, uniform_over=ioi_uniform_type)
sports = SportsTask(batch_size=batch_size, tokenizer=tokenizer, device=device)
owt = OWTTask(batch_size=batch_size, tokenizer=tokenizer, device=device)

# train_tasks = {"ioi": ioi, "owt": owt}
train_tasks = {"ioi_uniform": ioi_uniform, "owt": owt}
task_weights = {"ioi_uniform": .5, "owt": 1} # I think means preserve OWT, corrupt IOI
eval_tasks = {"ioi": ioi, "ioi_uniform": ioi_uniform, "sports": sports, "owt": owt}


# In[14]:


mask_params = []
param_names = []
for name, p in model.named_parameters():
    if p.requires_grad:
        param_names.append(name)
        mask_params.append(p)


# In[15]:


print(torch.cuda.max_memory_allocated(device=device) / 1e9)


# In[16]:


from cb_utils.learn_mask import train_masks

epochs_left = 500
steps_per_epoch = 50
lr = .05 # free
weight_decay = 0
evaluate_every = 1
discretize_every = 50 # 5 # free
threshold = 0.5
use_wandb = True
edge_mask_reg_strength = 1000
weight_mask_reg_strength = None

optimizer = torch.optim.AdamW(mask_params, lr=lr, weight_decay=weight_decay)
train_masks(model, tasks=train_tasks, optimizer=optimizer, num_epochs=epochs_left, steps_per_epoch=steps_per_epoch,
            # param_names=param_names, mask_params=mask_params, 
            task_weights=task_weights, eval_tasks=eval_tasks, evaluate_every=evaluate_every, discretize_every=discretize_every, threshold=threshold, edge_mask_reg_strength=edge_mask_reg_strength, weight_mask_reg_strength=None, verbose=False, use_wandb=use_wandb)


# In[17]:


import pickle
with open(f"masks/trained_mask_params_{epochs_left=}_{edge_mask_reg_strength=}_{ioi_uniform_type=}.pkl", "wb") as f:
    pickle.dump(mask_params, f)


# In[18]:


for name, p in zip(param_names, mask_params):
    if p.requires_grad:
        # print(name, p)
        # count how many zeros in p
        print(torch.sum(p == 0))

