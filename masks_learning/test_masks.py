#%%
# auto reload
%load_ext autoreload
%autoreload 2
%cd ~/mechanistic-unlearning
import functools
import torch
import numpy as np
import os
import gc
import re

from tasks.facts.SportsTask import SportsTask
import pickle
import einops
from datasets import load_dataset
from transformer_lens import ActivationCache
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from transformer_lens import utils
from tqdm import tqdm
import functools
# os.chdir("..")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
### LOAD MODELS
model_name = 'google/gemma-7b'
    # 'meta-llama/Meta-Llama-3-8B'
    # 'Qwen/Qwen1.5-4B' 
    # 'EleutherAI/pythia-2.8b',
    # "gpt2-small",


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
model.set_use_attn_result(True)
model.set_use_split_qkv_input(True)
model.set_use_hook_mlp_in(True)

#%%
attn_match = re.compile("^a(\d+).(\d+)_(q|k|v|result)$")
mlp_match = re.compile("^m(\d+)_(in|out)$")

def mean_ablate_hook(act, hook, mean_cache, components):
    '''
        Ablate components through mean ablation
    '''
    heads_to_patch = []
    patch_mlp = False
    for component in components:
        attention_match = attn_match.match(component)
        mlp_matched = mlp_match.match(component)
        if attention_match:
            layer = int(attention_match.group(1))
            head = int(attention_match.group(2))
            try:
                hook_type = "hook_" + attention_match.group(3)
            except IndexError:
                hook_type = "result"
            if layer == hook.layer() and hook_type in hook.name:
                heads_to_patch.append(head)
        elif mlp_matched:
            layer = int(mlp_matched.group(1))
            try:
                hook_type = mlp_matched.group(2)
            except IndexError:
                hook_type = "mlp_out"
            if layer == hook.layer() and hook_type in hook.name:
                patch_mlp = True

    # print(hook.name, heads_to_patch, patch_mlp) 
    if len(heads_to_patch) > 0:
        # print(f"Patching {hook.name}: {heads_to_patch}")
        # We are in the right attention layer
        heads_to_patch = torch.tensor(heads_to_patch)
        # print(hook.name, act.shape)
        act[:,:,heads_to_patch,:] = mean_cache[hook.name][heads_to_patch].unsqueeze(0).unsqueeze(0)
    elif patch_mlp:
        # We are in the right mlp layer
        act = mean_cache[hook.name].unsqueeze(0).unsqueeze(0)
    
    # act = 0
    return act

#%%
save_model_name = model_name.replace('/', '_')

def get_mean_cache(model):
    pile = iter(load_dataset('monology/pile-uncopyrighted', split='train', streaming=True))
    text = [next(pile)['text'] for i in range(25)]
    toks = torch.stack(
        [
            torch.tensor(tokenizer.encode(t)[:78])
            for t in text
        ],
        dim=0
    )
    with torch.set_grad_enabled(False):
        _, mean_cache = model.run_with_cache(
            toks,
            names_filter = lambda name: any([hook_name in name for hook_name in ["hook_z", "hook_k", "hook_q", "hook_v", "hook_result", "mlp_out", "mlp_in"]]) and not "input" in name
        )
        mean_cache_dict = {}
        for k, v in mean_cache.items():
            mean_cache_dict[k] = einops.reduce(
                v,
                "batch seq ... -> ...",
                "mean"
            )
        del mean_cache
        mean_cache = ActivationCache(mean_cache_dict, model)
    return mean_cache

### Localization helper funcs
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
        weight_mask_mlp_dict[layer]['W_in'] = not (1 <= layer <= 7)
        weight_mask_mlp_dict[layer]['W_out'] = not (1 <= layer <= 7)
        print(f"Setting layer {layer} to {weight_mask_mlp_dict[layer]}")

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

def get_mask_from_ct_graph(model, ct_graph, top_p):
    # Attention masks are of form:
    # {layer: {"W_Q": frozen_heads, "W_K": frozen_heads, "W_V": frozen_heads, "W_O": frozen_heads}}
    # TRUE for the heads we want to FREEZE, FALSE for heads we want to MASK over
    # MLP masks are of form:
    # {layer: bool}

    # Localizations are of form:
    # {alayer.head:int, mlayer: int}

    top_p *= 100
    all_weights = []
    for key, value in ct_graph.items():
        all_weights.append(value)

    all_weights = np.array(all_weights)
    threshold = np.percentile(all_weights, 100 - top_p)

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
#%%
mean_cache = get_mean_cache(model)
#%%
import random
model.reset_hooks()
results = {}
for forget_sport in ['basketball', "athlete"]:
    torch.cuda.empty_cache()
    gc.collect()
    if forget_sport == 'athlete':
        sports_task = SportsTask(
            # model=model, 
            # N=50, 
            batch_size=10, 
            forget_player_subset=16,
            is_forget_dataset=True,
            tokenizer=tokenizer,
            device=device
        )
    else:
        sports_task = SportsTask(
            # model=model, 
            # N=50, 
            batch_size=10, 
            forget_sport_subset={forget_sport},
            is_forget_dataset=True,
            tokenizer=tokenizer,
            device=device
        )

    results[forget_sport] = {}
    for localization_method in ["manual", "random", "ap", "ct"]:
        results[forget_sport][localization_method] = {}

        localization_top_p = 0.05
        if localization_method == "ap":
            with open(f"models/{model_name.replace('/', '_')}_sports_{forget_sport}_{localization_method}_graph.pkl", "rb") as f:
                localization_graph = pickle.load(f)
        elif localization_method == "ct":
            with open(f"models/{model_name.replace('/', '_')}_sports_{forget_sport}_{localization_method}_graph.pkl", "rb") as f:
                localization_graph = pickle.load(f)
        elif localization_method == "random":
            with open(f"models/{model_name.replace('/', '_')}_sports_{forget_sport}_ap_graph.pkl", "rb") as f:
                localization_graph = pickle.load(f)
            # Set every value to random between 1e-5 and 5
            for k in localization_graph.keys():
                localization_graph[k] = random.uniform(1e-5, 5)
        elif localization_method == "manual":
            with open(f"models/{model_name.replace('/', '_')}_sports_{forget_sport}_ap_graph.pkl", "rb") as f:
                localization_graph = pickle.load(f)
            # Set MLPs between 1 and 7 to inf, rest to 0
            for k in localization_graph.keys():
                if "m" in k:
                    layer = k[1:].split('_')
                    if 1 <= layer <= 7:
                        localization_graph[k] = float('inf')
                    else:
                        localization_graph[k] = 0

        thresholds = []
        losses = []
        percent_ablated = []

        if localization_method == "random":
            sequence = np.linspace(1e-5, 5, 20)
        else:
            sequence = np.logspace(-5, 1, num=20)

        for THRESHOLD in tqdm(sequence):
            # print(f"{forget_sport};{localization_method}: {THRESHOLD}")
            components_to_ablate = {k for k, v in localization_graph.items() if abs(v) > THRESHOLD}
            hook_fn = functools.partial(
                mean_ablate_hook,
                mean_cache=mean_cache,
                components=components_to_ablate
            )
            fwd_hooks = []
            for component in components_to_ablate:
                attn_matched = attn_match.match(component)
                mlp_matched = mlp_match.match(component)
                if attn_matched:
                    layer = attn_matched.group(1)
                    head = attn_matched.group(2)
                    try:
                        hook_type = attn_matched.group(3)
                    except IndexError:
                        hook_type = "result"
                    fwd_hooks.append((utils.get_act_name(hook_type, layer), hook_fn))
                elif mlp_matched:
                    layer = mlp_matched.group(1)
                    try:
                        hook_type = "mlp_" + mlp_matched.group(2)
                    except IndexError:
                        hook_type = "mlp_out"
                    fwd_hooks.append((utils.get_act_name(hook_type, layer), hook_fn))

            model.reset_hooks()
            for name, hook in fwd_hooks:
                model.add_hook(name, hook)
            
            thresholds.append(THRESHOLD)
            with torch.set_grad_enabled(False):
                losses.append(sports_task.get_test_loss(model, n_iters=15).item())
            model.reset_hooks()
            percent_ablated.append(len(components_to_ablate) / len(localization_graph.keys()))
        results[forget_sport][localization_method] = (thresholds, losses, percent_ablated)

# %% PLOT
import matplotlib.pyplot as plt

for forget_sport in ['basketball', 'athlete']:
    fig = plt.figure()

    thresholds, losses, percent_ablated = results[forget_sport]["ap"]
    plt.plot(percent_ablated, losses, label="Attribution Patching Ablation")
    thresholds, losses, percent_ablated = results[forget_sport]["ct"]
    plt.plot(percent_ablated, losses, label="Causal Tracing Ablation")
    thresholds, losses, percent_ablated = results[forget_sport]["manual"]
    plt.plot(percent_ablated, losses, label="Manual Ablation")

    random_thresholds, random_losses, random_percent_ablated = results[forget_sport]["random"]
    plt.plot(random_percent_ablated, random_losses, label="Random Ablation")

    # Plot 
    plt.xlabel("Percent Top Components Ablated")
    plt.ylabel("Test Loss (Log Scale)")
    # plt.yscale('log')
    plt.legend()
    plt.grid()

    plt.title(f"Basketball Loss vs Percent of Top Components Ablated")
    plt.show()

# %%
