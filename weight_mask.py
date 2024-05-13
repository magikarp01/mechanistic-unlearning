# We want to mask the weights of a transformer
# First, we run a forward pass on all weights
# Our loss function then a regularization term, a forget term, and a maintain term
# Then, we run a backward pass on all weights
# Then, we zero the gradients of the components we don't want to update
# Then, we step the optimizer

#%%
%cd ~/mechanistic-unlearning
%load_ext autoreload
%autoreload 2
from transformer_lens import HookedTransformer, ActivationCache
import random
import os
import torch
from torch import nn
import numpy as np
import pandas as pd
import datasets
import transformers
import pickle
import gc
import wandb
import json
from collections import defaultdict

from tasks import PileTask, OWTTask, InductionTask, GreaterThanTask
from tasks.ioi.IOITask import IOITask, IOITask_NPO, IOITask_Uniform
from tasks.induction.InductionTask import InductionTask, InductionTask_NPO, InductionTask_Uniform
from tasks.facts.SportsTask import SportsTask, SportsTask_NPO, SportsTask_Uniform

from tqdm.auto import tqdm

from transformers import GPT2Tokenizer, GPTNeoXTokenizerFast, AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset

from tasks import PileTask, OWTTask, InductionTask, GreaterThanTask
from tasks.ioi.IOITask import IOITask, IOITask_NPO, IOITask_Uniform
from tasks.induction.InductionTask import InductionTask, InductionTask_NPO, InductionTask_Uniform
from tasks.facts.SportsTask import SportsTask, SportsTask_NPO, SportsTask_Uniform
from tasks.facts.SportsTaskAdversarial import adversarial_sports_eval
from tasks.facts.SportsTaskSideEffects import run_side_effects_evals
from weight_masked_transformer import WeightMaskedTransformer
#%% Hyperparams
def load_global_hyperparams(config_file):
    global train_batch_size, eval_batch_size, device, train_loss_type, forget_sport, maintain_sport, \
           model_name, model_type, learning_rate, n_epochs, grad_accum_steps, alpha, beta, clip_grad, \
           evaluate_every, n_eval_iters, do_adversarial_evals, do_side_effects_evals, \
           localization_type, localization_top_p
    
    with open(config_file, "r") as f:
        config = json.load(f)
    
    train_batch_size = int(config['train_batch_size'])
    eval_batch_size = int(config['eval_batch_size'])
    device = config['device']
    train_loss_type = config['train_loss_type']
    forget_sport = config['forget_sport']

    maintain_sport = config['maintain_sport']
    model_name = config['model_name']
    model_type = config['model_type']
    learning_rate = float(config['learning_rate'])
    n_epochs = int(config['n_epochs'])
    grad_accum_steps = int(config['grad_accum_steps'])
    alpha = float(config['alpha'])
    beta = float(config['beta'])
    clip_grad = None if config['clip_grad'] == "null" else int(config['clip_grad'])
    evaluate_every = int(config['evaluate_every'])
    n_eval_iters = int(config['n_eval_iters'])
    do_adversarial_evals = config['do_adversarial_evals']
    do_side_effects_evals = config['do_side_effects_evals']
    localization_type = config['localization_type']
    localization_top_p = float(config['localization_top_p'])
    
load_global_hyperparams("weight_masking_config.json")
#%%
os.environ['HF_TOKEN'] = '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'
model_name = 'google/gemma-2b'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = HookedTransformer.from_pretrained(
    model_name,
    tokenizer=tokenizer,
    device='cuda',
    default_padding_side="right",
    fold_ln=False,
    fold_value_biases=False,
    center_writing_weights=False,
    dtype=torch.bfloat16
)

#%% Localization helper funcs
def create_random_weight_mask_dicts(model, top_p):
    # Creates random weight masks for testing
    weight_mask_attn_dict = {}
    weight_mask_mlp_dict = {}

    for layer in range(model.cfg.n_layers):
        weight_mask_attn_dict[layer] = {}
        weight_mask_mlp_dict[layer] = {}
        # Want bool of length n_head, randomly set to True
        weight_mask_attn_dict[layer]['W_Q'] = torch.rand(model.cfg.n_heads) > top_p
        weight_mask_attn_dict[layer]['W_K'] = torch.rand(model.cfg.n_heads) > top_p
        weight_mask_attn_dict[layer]['W_V'] = torch.rand(model.cfg.n_heads) > top_p
        weight_mask_attn_dict[layer]['W_O'] = torch.rand(model.cfg.n_heads) > top_p

        # Randomly set to true or false
        weight_mask_mlp_dict[layer]['W_in'] = random.random() > top_p
        weight_mask_mlp_dict[layer]['W_out'] = random.random() > top_p

    return weight_mask_attn_dict, weight_mask_mlp_dict

def create_mlp_only_mask_dicts(model):
    weight_mask_attn_dict = {}
    weight_mask_mlp_dict = {}

    for layer in range(model.cfg.n_layers):
        weight_mask_attn_dict[layer] = {}
        weight_mask_mlp_dict[layer] = {}

        # Set to false: we train a mask over these
        weight_mask_mlp_dict[layer]['W_in'] = not (layer >= 1 and layer <= 7)
        weight_mask_mlp_dict[layer]['W_out'] = not (layer >= 1 and layer <= 7)

    return weight_mask_attn_dict, weight_mask_mlp_dict

def get_mask_from_ap_graph(model, ap_graph, top_p):
    # Attention masks are of form:
    # {layer: {"W_Q": frozen_heads, "W_K": frozen_heads, "W_V": frozen_heads, "W_O": frozen_heads}}
    # TRUE for the heads we want to FREEZE, FALSE for heads we want to MASK over
    # MLP masks are of form:
    # {layer: bool}

    # Localizations are of form:
    # {alayer.head_{q,k,v,result}:int, mlayer_{in,out}: int}

    # Get threshold such that top_p% of weights are ABOVE threshold: We are masking over the top_p% of weights
    top_p *= 100
    all_weights = []
    for key, value in ap_graph.items():
        all_weights.append(value)

    all_weights = np.array(all_weights)
    threshold = np.percentile(all_weights, 100 - top_p)

    weight_mask_attn_dict = {}
    weight_mask_mlp_dict = {}

    for layer in range(model.cfg.n_layers):
        weight_mask_attn_dict[layer] = {}
        weight_mask_mlp_dict[layer] = {}

        if 'a0.0_q' in ap_graph:
            weight_mask_attn_dict[layer]['W_Q'] = torch.tensor(
                [
                    abs(ap_graph[f"a{layer}.{head}_q"]) < threshold 
                    for head in range(model.cfg.n_heads)
                ]
            )
        else:
            weight_mask_attn_dict[layer]['W_Q'] = None

        if 'a0.0_k' in ap_graph:
            weight_mask_attn_dict[layer]['W_K'] = torch.tensor(
                [
                    abs(ap_graph[f"a{layer}.{head}_k"]) < threshold 
                    for head in range(model.cfg.n_heads)
                ]
            )
        else:
            weight_mask_attn_dict[layer]['W_K'] = None
        
        if 'a0.0_v' in ap_graph:
            weight_mask_attn_dict[layer]['W_V'] = torch.tensor(
                [
                    abs(ap_graph[f"a{layer}.{head}_v"]) < threshold 
                    for head in range(model.cfg.n_heads)
                ]
            )
        else:
            weight_mask_attn_dict[layer]['W_V'] = None
        
        if 'a0.0_result' in ap_graph:
            weight_mask_attn_dict[layer]['W_O'] = torch.tensor(
                [
                    abs(ap_graph[f"a{layer}.{head}_result"]) < threshold 
                    for head in range(model.cfg.n_heads)
                ]
            )
        else:
            weight_mask_attn_dict[layer]['W_O'] = None
            
        if 'm0_in' in ap_graph:
            weight_mask_mlp_dict[layer]['W_in'] = abs(ap_graph[f"m{layer}_in"]) < threshold
        else:
            weight_mask_mlp_dict[layer]['W_in'] = None
        
        if 'm0_out' in ap_graph:
            weight_mask_mlp_dict[layer]['W_out'] = abs(ap_graph[f"m{layer}_out"]) < threshold
        else:
            weight_mask_mlp_dict[layer]['W_out'] = None

    return weight_mask_attn_dict, weight_mask_mlp_dict

def get_mask_from_ct_graph(model, ct_graph, threshold):
    # Attention masks are of form:
    # {layer: {"W_Q": frozen_heads, "W_K": frozen_heads, "W_V": frozen_heads, "W_O": frozen_heads}}
    # TRUE for the heads we want to FREEZE, FALSE for heads we want to MASK over
    # MLP masks are of form:
    # {layer: bool}

    # Localizations are of form:
    # {alayer.head:int, mlayer: int}

    weight_mask_attn_dict = {}
    weight_mask_mlp_dict = {}

    for layer in range(model.cfg.n_layers):
        weight_mask_attn_dict[layer] = {}
        weight_mask_mlp_dict[layer] = {}

        weight_mask_attn_dict[layer]['W_O'] = torch.tensor(
            [
                abs(ct_graph[f"a{layer}.{head}"]) < threshold 
                for head in range(model.cfg.n_heads)
            ]
        )

        weight_mask_mlp_dict[layer]['W_out'] = torch.tensor(
            [
                abs(ct_graph[f"m{layer}"]) < threshold
            ]
        )
    
    return weight_mask_attn_dict, weight_mask_mlp_dict

#%% Datasets
### DATASETS
train_dataset = load_dataset('monology/pile-uncopyrighted', split='train', streaming=True)
sports_1mp = SportsTask(batch_size=train_batch_size, tokenizer=tokenizer, device=device, prep_acdcpp=False, criterion="log_1_minus_p", forget_sport_subset={forget_sport}, is_forget_dataset=True)

if maintain_sport is None or maintain_sport == "null":
    maintain_sports = SportsTask(batch_size=train_batch_size, tokenizer=tokenizer, device=device, prep_acdcpp=False, criterion="cross_entropy", forget_sport_subset={forget_sport}, is_forget_dataset=False)
else:
    maintain_sports = SportsTask(batch_size=train_batch_size, tokenizer=tokenizer, device=device, prep_acdcpp=False, criterion="cross_entropy", forget_sport_subset={maintain_sport}, is_forget_dataset=True)

train_pile = PileTask(batch_size=train_batch_size, tokenizer=tokenizer, device=device, ctx_length=100, shuffle=True, buffer_size=1000)
train_tasks = {"sports_1mp": (sports_1mp, .4), "maintain_sports": (maintain_sports, 1), "pile": (train_pile, 1)}

# want to eval on other sports
forget_sport_eval = SportsTask(batch_size=eval_batch_size, tokenizer=tokenizer, device=device, prep_acdcpp=False, criterion="cross_entropy", forget_sport_subset={forget_sport}, is_forget_dataset=True)
test_pile = PileTask(batch_size=eval_batch_size, tokenizer=tokenizer, device=device, ctx_length=100, shuffle=True, buffer_size=1000)

induction_eval = InductionTask(batch_size=eval_batch_size, tokenizer=tokenizer, prep_acdcpp=False, seq_len=15, device=device)
if maintain_sport is None or maintain_sport == "null":
    maintain_sports_eval = SportsTask(batch_size=eval_batch_size, tokenizer=tokenizer, device=device, prep_acdcpp=False, criterion="cross_entropy", forget_sport_subset={forget_sport}, is_forget_dataset=False)
    eval_tasks = {"induction": induction_eval, "pile": test_pile, "forget_sport": forget_sport_eval, "maintain_sport": maintain_sports_eval}
else:
    raise NotImplemented

#%% Load Localizations

### LOCALIZATIONS

if localization_type == "ap":
    with open(f"models/{model_name.replace('/', '_')}_sports_{forget_sport}_{localization_type}_graph.pkl", "rb") as f:
        localization_graph = pickle.load(f)
    weight_mask_attn_dict, weight_mask_mlp_dict = get_mask_from_ap_graph(model, localization_graph, localization_top_p)
elif localization_type == "ct":
    with open(f"models/{model_name.replace('/', '_')}_sports_{forget_sport}_{localization_type}_graph.pkl", "rb") as f:
        localization_graph = pickle.load(f)
    weight_mask_attn_dict, weight_mask_mlp_dict = get_mask_from_ct_graph(model, localization_graph, localization_top_p)
elif localization_type == "random":
    weight_mask_attn_dict, weight_mask_mlp_dict = create_random_weight_mask_dicts(model, localization_top_p)
elif localization_type == "none":
    weight_mask_attn_dict, weight_mask_mlp_dict = create_random_weight_mask_dicts(model, 1)
elif localization_type == "manual":
    # Manual interp means only training the MLP weights from layer 1 to 7
    weight_mask_attn_dict, weight_mask_mlp_dict = create_mlp_only_mask_dicts(model)

gc.collect()
torch.cuda.empty_cache()
# Testing only
for layer in range(model.cfg.n_layers):
    del weight_mask_attn_dict[layer]['W_K']
    del weight_mask_attn_dict[layer]['W_V']

#%% Train loop
### TRAIN 

def get_unfrozen_weights(model):
    # Get a copy of the weights that are not frozen
    unfrozen_weights = {}
    for layer, block in enumerate(model.blocks):
        unfrozen_weights[layer] = {}
        for component in ["W_Q", "W_K", "W_V", "W_O"]:
            if component in weight_mask_attn_dict[layer]:
                frozen_heads = weight_mask_attn_dict[layer][component]
                unfrozen_weights[layer][component] = getattr(block.attn, component)[~frozen_heads].clone()
        
        for component in ["W_in", "W_out"]:
            if component in weight_mask_mlp_dict[layer]:
                if not weight_mask_mlp_dict[layer][component]:
                    unfrozen_weights[layer][component] = getattr(block.mlp, component).clone()
    return unfrozen_weights

def zero_grad(model):
    for layer, block in enumerate(model.blocks):
        # If head is frozen or doesnt exist, set grad to 0
        for component in ["W_Q", "W_K", "W_V", "W_O"]:
            if component in weight_mask_attn_dict[layer]:
                frozen_heads = weight_mask_attn_dict[layer][component]
                getattr(block.attn, component).grad[frozen_heads] = 0
            else:
                # Set grad of all heads to 0
                param = getattr(block.attn, component)
                if param.grad is not None:
                    param.grad = torch.zeros_like(param.grad).to("cuda")
                else:
                    print(f"None grad for {component} in layer {layer}")

        # If mlp is frozen or doesnt exist, set grad to 0
        for component in ["W_in", "W_out"]:
            if component in weight_mask_mlp_dict[layer]:
                if weight_mask_mlp_dict[layer][component]:
                    param = getattr(block.mlp, component)
                    if param.grad is not None:
                        param.grad = torch.zeros_like(param.grad).to("cuda")
                    else:
                        print(f"None grad for {component} in layer {layer}")
            else:
                param = getattr(block.mlp, component)
                if param.grad is not None:
                    param.grad = torch.zeros_like(param.grad).to("cuda")
                else:
                    print(f"None grad for {component} in layer {layer}")

def regularization_loss(model):
    # L1 sparsity, but only for components that are not frozen
    loss = 0
    for layer, block in enumerate(model.blocks):
        for component in ["W_Q", "W_K", "W_V", "W_O"]:
            # If the component exists, we take the non frozen heads
            if component in weight_mask_attn_dict[layer]:
                frozen_heads = weight_mask_attn_dict[layer][component]
                loss += torch.sum(torch.abs(getattr(block.attn, component)[~frozen_heads]))
        
        for component in ["W_in", "W_out"]:
            if component in weight_mask_mlp_dict[layer]:
                # If the mlp is not frozen, then we add the L1 loss
                if not weight_mask_mlp_dict[layer][component]:
                    loss += torch.sum(torch.abs(getattr(block.mlp, component)))
    return loss

def clamp_unfrozen_weights(model, original_weights):
    # Clamp all the non-frozen weights to be between 0 and original_weights
    for layer, block in enumerate(model.blocks):
        for component in ["W_Q", "W_K", "W_V", "W_O"]:
            if component in weight_mask_attn_dict[layer]:
                frozen_heads = weight_mask_attn_dict[layer][component]
                param = getattr(block.attn, component)[~frozen_heads]
                param.data = torch.where(
                    param > 0,
                    torch.clamp(
                        param.data, 
                        torch.zeros_like(param),
                        original_weights[layer][component]
                    ),
                    torch.clamp(
                        param.data, 
                        original_weights[layer][component], 
                        torch.zeros_like(param)
                    )
                )
        
        for component in ["W_in", "W_out"]:
            if component in weight_mask_mlp_dict[layer]:
                if not weight_mask_mlp_dict[layer][component]:
                    param = getattr(block.mlp, component)
                    param.data = torch.where(
                        param > 0,
                        torch.clamp(
                            param.data, 
                            torch.zeros_like(param),
                            original_weights[layer][component]
                        ),
                        torch.clamp(
                            param.data, 
                            original_weights[layer][component], 
                            torch.zeros_like(param)
                        )
                    )
        
def get_mask(model, original_weights):
    # Get masks, where a mask is model weight / original weight
    mask = {}
    for layer, block in enumerate(model.blocks):
        mask[layer] = {}
        for component in ["W_Q", "W_K", "W_V", "W_O"]:
            if component in weight_mask_attn_dict[layer]:
                mask[layer][component] = getattr(block.attn, component) / original_weights[layer][component]
        
        for component in ["W_in", "W_out"]:
            if component in weight_mask_mlp_dict[layer]:
                if not weight_mask_mlp_dict[layer][component]:
                    mask[layer][component] = getattr(block.mlp, component) / original_weights[layer][component]
    return mask

#%%
all_train_losses = defaultdict(list)
all_test_losses = defaultdict(list)
adversarial_evals = []
side_effect_evals = []

# Initialize optimizer, want to optimize over all weights
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_epochs)
original_weights = get_unfrozen_weights(model)

# Cycle dataloaders
# Train a sparse mask
# pbar = tqdm(range(n_epochs))
# for epoch in pbar:
# Sample batches
# Reset grad
optimizer.zero_grad()

epoch = 1
beta = 1e-6
with torch.autocast(device_type="cuda"):
    # Compute normal loss over retain
    for task_name, (task, task_weight) in train_tasks.items():
        print(f"Running {task_name}")
        task_loss = 0
        for i in range(grad_accum_steps):
            loss = task.get_train_loss(model) / grad_accum_steps
            task_loss += loss.item()
            loss *= task_weight
            print(task_name, i, loss)
            loss.backward()

            gc.collect()
            torch.cuda.empty_cache()
        all_train_losses[task_name].append(task_loss)

        gc.collect()
        torch.cuda.empty_cache()
        
    gc.collect()
    torch.cuda.empty_cache()
    # Add sparsity loss and backprop
    # Linearly increase from negative to positive, with 0 at 10
    loss = min(beta * (epoch-15), beta) * regularization_loss(model)
    loss.backward()
    print(f"reg loss, {loss.item()}")
    all_train_losses["reg"].append(loss.item())
    # Step and log
    if clip_grad is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    
    print(model.blocks[0].attn.W_Q.grad[2])
    print(model.blocks[0].attn.W_Q.grad[1])
    # Remove gradients from frozen components
    zero_grad(model)
    print(model.blocks[0].attn.W_Q.grad[1])
    optimizer.step()
    scheduler.step()
    clamp_unfrozen_weights(model, original_weights)
    
    # print(mask.blocks[27].attention_masks['W_O'][-1][1])

    # if epoch % evaluate_every == 0 or epoch == n_epochs - 1:
    #     for task_name, task in eval_tasks.items():
    #         task_loss = 0
    #         for i in range(n_eval_iters):
    #             task_loss += task.get_test_loss(mask).item()
    #         all_test_losses[task_name].append(task_loss / n_eval_iters)
    #     if do_adversarial_evals:
    #         print("Running adversarial evals")
    #         adversarial_evals.append(adversarial_sports_eval(mask, model_type=model_type, batch_size=eval_batch_size, use_system_prompt=True, include_evals=["Normal", "MC"]))
    #     if do_side_effects_evals:
    #         print("Running side effects evals")
    #         side_effect_evals.append(run_side_effects_evals(mask, model_type=model_type, batch_size=eval_batch_size, evals_to_run=["Sports Answers"]))
    # gc.collect()
    # torch.cuda.empty_cache()
    
    # log_dict = {}
    # for k, v in all_train_losses.items():
    #     log_dict[f"train_loss_{k}"] = v[-1]
    # for k, v in all_test_losses.items():
    #     log_dict[f"test_loss_{k}"] = v[-1]
    # for k, v in adversarial_evals[-1].items():
    #     log_dict[f"adversarial_{k}"] = v
    # for k, v in side_effect_evals[-1].items():
    #     log_dict[f"side_effects_{k}"] = v
    # wandb.log(log_dict)

# Get masks
mask = get_mask(model, original_weights) 
# wandb.finish()

#%% Save
attention_masks = {}
mlp_masks = {}
for layer, block in enumerate(mask.blocks):
    attention_masks[layer] = block.attention_masks
    mlp_masks[layer] = block.mlp_masks

### SAVE
torch.save(attention_masks, f"results/{model_name.replace('/', '_')}-{forget_sport}-{localization_type}_attn.pt")
torch.save(mlp_masks, f"results/{model_name.replace('/', '_')}-{forget_sport}-{localization_type}_mlp.pt")

