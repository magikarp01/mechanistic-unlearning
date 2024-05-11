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

#%%
mean_cache = get_mean_cache(model)
#%%
import random
model.reset_hooks()
results = {}
for forget_sport in ['all']:
    torch.cuda.empty_cache()
    gc.collect()
    sports_task = SportsTask(
        # model=model, 
        # N=50, 
        batch_size=2, 
        tokenizer=tokenizer,
        device=device
    )

    results[forget_sport] = {}
    for localization_method in ["ap", "random"]:
        results[forget_sport][localization_method] = {}

        if localization_method == "random":
            # Hijack the AP localizations
            with open(f"models/{save_model_name}_sports_{forget_sport}_ap_graph.pkl", "rb") as f:
                # Load pickle
                localization = pickle.load(f)
            for key, value in localization.items():
                # Set to random value between 1e-6 and 5
                localization[key] = (5 * random.random()) + 1e-6
        else:
            with open(f"models/{save_model_name}_sports_{forget_sport}_{localization_method}_graph.pkl", "rb") as f:
                # Load pickle
                localization = pickle.load(f)

        thresholds = []
        losses = []
        percent_ablated = []

        if localization_method == "random":
            sequence = np.linspace(1e-5, 5, 20)
        else:
            sequence = np.logspace(-5, 1, num=20)

        for THRESHOLD in tqdm(sequence):
            # print(f"{forget_sport};{localization_method}: {THRESHOLD}")
            components_to_ablate = {k for k, v in localization.items() if abs(v) > THRESHOLD}
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
            percent_ablated.append(len(components_to_ablate) / len(localization.keys()))
        results[forget_sport][localization_method] = (thresholds, losses, percent_ablated)

# %% PLOT
import matplotlib.pyplot as plt

for forget_sport in ['all']:
    thresholds, losses, percent_ablated = results[forget_sport]["ap"]
    random_thresholds, random_losses, random_percent_ablated = results[forget_sport]["random"]

    # Plot 
    fig = plt.figure()
    plt.plot(percent_ablated, losses, label="Attribution Patching Ablation")
    plt.plot(random_percent_ablated, random_losses, label="Random Ablation")
    plt.xlabel("Percent Top Components Ablated")
    plt.ylabel("Test Loss (Log Scale)")
    plt.yscale('log')
    plt.legend()
    plt.grid()

    plt.title(f"Sports: Test Loss vs Percent of Top Components Ablated")
    plt.show()

# %%
