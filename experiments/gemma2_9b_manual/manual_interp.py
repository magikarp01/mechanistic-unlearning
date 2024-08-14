#%%
%cd ~/mechanistic-unlearning
%load_ext autoreload
%autoreload 2
import functools
import os
import gc
import json
from tkinter import font
import sys

from dataset.custom_dataset import PairedInstructionDataset
import torch

from transformer_lens import HookedTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
import einops
from transformer_lens import ActivationCache
import time

project_root = os.path.abspath(os.path.dirname(__file__))

# Add the project root and its subdirectories to the Python path
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'masks_learning'))
sys.path.insert(0, os.path.join(project_root, 'localizations'))
sys.path.insert(0, os.path.join(project_root, 'tasks'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
model = HookedTransformer.from_pretrained(
    'google/gemma-2-9b',
    device='cuda',
    default_padding_side="left",
    fold_ln=False,
    fold_value_biases=False,
    center_writing_weights=False,
    dtype=torch.bfloat16
)
tokenizer = model.tokenizer
#%%
import pandas as pd

df = pd.read_csv('tasks/facts/sports.csv')
def tokenize_instructions(tokenizer, instructions):
    # Use this to put the text into INST tokens or add a system prompt
    return tokenizer(
        instructions,
        padding=True,
        truncation=False,
        return_tensors="pt",
        # padding_side="left",
    ).input_ids


# %% Probing Functions

# Need to 
# 1. Probe for correct sport with no changes
# 2. Probe for correct sport with just <bos>name
# 3. Probe after meal ablating attention heads after layer 2
# 4. Probe after meal ablating attention heads after layer 2 and just <bos>name

def probe_last_layer(model, prompt_toks, targets, batch_size=None):
    if batch_size is None:
        with torch.set_grad_enabled(False):
            _, cache = model.run_with_cache(
            prompt_toks,
            names_filter = lambda name: name == f"blocks.{model.cfg.n_layers-1}.hook_resid_post"
        )
        cache = cache[f"blocks.{model.cfg.n_layers-1}.hook_resid_post"][:, -1, :]
    else:
        caches = []
        for i in range(0, prompt_toks.shape[0], batch_size):
            prompt_toks_batch = prompt_toks[i:i+batch_size]
            _, cache = model.run_with_cache(
                prompt_toks_batch,
                names_filter = lambda name: name == f"blocks.{model.cfg.n_layers-1}.hook_resid_post"
            )
            cache = cache[f"blocks.{model.cfg.n_layers-1}.hook_resid_post"][:, -1, :]
            caches.append(cache)
        cache = torch.cat(caches, dim=0)

    X = cache.cpu().float().numpy()
    print(X.shape, len(targets))
    target_classes = []
    for target in targets:
        if target == "basketball":
            target_classes.append(0)
        elif target == "baseball":
            target_classes.append(1) 
        elif target == "football":
            target_classes.append(2)
    y = np.array(target_classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train logistic regression
    clf = LogisticRegression(random_state=0, max_iter=500, solver='sag').fit(X_train, y_train)

    test_acc = clf.score(X_test, y_test)
    print(f"Accuracy: {test_acc}")

    return test_acc

def train_test_probe(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
    clf = LogisticRegression(random_state=0, max_iter=500, solver='sag').fit(X_train, y_train)
    # clf = LogisticRegression(random_state=0, max_iter=10000).fit(X_train, y_train)
    # results[layer] = clf.score(X_test, y_test)
    return clf.score(X_test, y_test)


def probe_across_layers(model, prompt_toks, targets, batch_size=None, cpu_multiprocessing=True):
    print(prompt_toks.shape)
    if batch_size is None:
        with torch.set_grad_enabled(False):
            _, cache = model.run_with_cache(
                prompt_toks,
                names_filter = lambda name: 'resid_post' in name
            )
            cache = torch.stack([cache[key][:, -1, :] for key in cache.keys()], dim=0) # layer batch d_model
    else:
        caches = []
        for i in range(0, prompt_toks.shape[0], batch_size):
            prompt_toks_batch = prompt_toks[i:i+batch_size]
            with torch.set_grad_enabled(False):
                _, cache = model.run_with_cache(
                    prompt_toks_batch,
                    names_filter = lambda name: 'resid_post' in name
                )
                # print(cache)
                # print(list(cache.values())[0].shape)
                cache = torch.stack([cache[key][:, -1, :] for key in cache.keys()], dim=0)
                # print(cache.shape)
            caches.append(cache)
        cache = torch.cat(caches, dim=1)
        # print("Batched Cache: ", cache.shape)
    print(f"Cache completed, {cache.shape}")
    results = []
    target_classes = []
    for target in targets:
        if target == "basketball":
            target_classes.append(0)
        elif target == "baseball":
            target_classes.append(1) 
        elif target == "football":
            target_classes.append(2)
    y = np.array(target_classes)

    X_data = {layer: cache[layer].cpu().float().numpy().reshape(-1, cache[layer].shape[-1]) for layer in range(model.cfg.n_layers)}

    results = {}

    if cpu_multiprocessing:
        import multiprocess as mp
        import os
        from multiprocessing import Manager
        from multiprocessing import Pool
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        print("Trying cpu multiprocessing")
        from multiprocessing.dummy import Pool as ThreadPool

        start_time = time.time()
        with ThreadPool(processes=model.cfg.n_layers) as pool:  # Using ThreadPool
            results = pool.starmap(train_test_probe, [(X_data[layer], y) for layer in range(model.cfg.n_layers)])
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        # with Pool(processes=2) as pool:
        #     results = pool.starmap(train_test_probe, [(X_data[layer], y) for layer in range(2)])
        print("finished multiprocessing")
        print(results)
        # Convert list of tuples to dictionary
        results_dict = {layer: results[i] for i, layer in enumerate(range(model.cfg.n_layers))}
        print(results_dict)

        # all_processes = {}
        # for layer in range(1):
        #     m = mp.Process(target=train_test_probe, args=(X_data[layer], y, results))
        #     all_processes[layer] = m
        #     m.start()
        # print(all_processes)
        # for layer in range(1):
        #     all_processes[layer].join()

    else:
        results = []
        start_time = time.time()
        for layer in tqdm(range(model.cfg.n_layers)):
            # print("converting cache")
            # X = cache[layer].cpu().float().numpy().reshape(-1, cache[layer].shape[-1])
            X = X_data[layer]
            # print("cache converted")

            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
            
            # print("training probe")
            # # train logistic regression
            # clf = LogisticRegression(random_state=0, max_iter=500, solver='sag').fit(X_train, y_train)
            # print("testing probe")

            # test_acc = clf.score(X_test, y_test)
            # print(f"Layer {layer} Accuracy: {test_acc}")
            # results.append(test_acc)
            test_acc = train_test_probe(X, y)
            results.append(test_acc)
        print(f"Time taken: {time.time() - start_time} seconds")
    return results

def get_mean_cache(model, hook_name="attn_out"):
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
            names_filter = lambda name: any([h_name in name for h_name in [hook_name]])
        )
    return mean_cache

def mean_ablate_hook(act, hook, mean_cache):
    if hook.layer() >= 5:
        print(f'Hooked {hook.name}')
        act = mean_cache[hook.name]
    return act

#%% Getting Cache
m_cache = get_mean_cache(model)
mean_cache = {}
for k in m_cache.keys():
    mean_cache[k] = einops.reduce(
        m_cache[k],
        'batch seq d_model -> 1 1 d_model',
        'mean'
    )

#%% Probing
results = {}
full_prompt_toks = tokenize_instructions(tokenizer, df['prompt'].tolist()) # Full prompt
athl_prompt_toks = tokenize_instructions(tokenizer, df['athlete'].tolist()) # <bos>name

model.reset_hooks()
results['Prompt'] = probe_across_layers(model, full_prompt_toks, df['sport'].tolist(), batch_size=128, cpu_multiprocessing=True)

#%%
# Add mean ablate hooks
model.add_hook(
    lambda name: 'attn_out' in name,
    functools.partial(mean_ablate_hook, mean_cache=mean_cache),
    "fwd"
)
results['+ Ablate Heads at Layers >= 5'] = probe_across_layers(model, full_prompt_toks, df['sport'].tolist(), batch_size=128)

model.reset_hooks()
results['Athlete'] = probe_across_layers(model, athl_prompt_toks, df['sport'].tolist(), batch_size=128)

# Add mean ablate hooks
model.add_hook(
    lambda name: 'attn_out' in name,
    functools.partial(mean_ablate_hook, mean_cache=mean_cache),
    "fwd"
)
results['+Ablate Heads at Layers >= 7'] = probe_across_layers(model, athl_prompt_toks, df['sport'].tolist(), batch_size=128)
model.reset_hooks()

#%% Plot results
import matplotlib.pyplot as plt
fig = plt.figure()
markers = ['o', 's', '^', 'v', '*', 'p', 'P', 'X', 'd']
for i, (k, v) in enumerate(results.items()):
    plt.plot(v, label=k, marker=markers[i])

# Add vertical dotted lines at x = 2, x = 7
plt.axvline(x=2, color='k', linestyle='--')
plt.axvline(x=5, color='k', linestyle='--')
# Label these vertical lines
plt.text(2.3, 0.45, 'Layer 2', fontsize=12)
plt.text(5.3, 0.45, 'Layer 5', fontsize=12)

plt.legend()
plt.xlabel('Layer', fontsize=16)
plt.ylabel('Probe Accuracy', fontsize=16)
plt.title('Probe Accuracy Across Layers', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.show()
fig.savefig('results/9b_probe_across_layers.pdf')

# %% Investigating the attention heads in layers [0, 6]

full_prompt_toks = tokenize_instructions(tokenizer, df['prompt'].tolist()) # Full prompt
athl_prompt_toks = tokenize_instructions(tokenizer, df['athlete'].tolist()) # <bos>name
m_cache = get_mean_cache(model, hook_name="hook_z")
mean_cache = {}
for k in m_cache.keys():
    mean_cache[k] = einops.reduce(
        m_cache[k],
        'batch seq head d_model -> 1 1 head d_model',
        'mean'
    )

#%%
def act_patch_hook_z(act, hook, patch_cache, patch_layer, patch_head):
    # heads_to_patch is [(layer, head)]
    # heads = [head for layer, head in heads_to_patch if layer == hook.layer()]

    # act is batch head seq d_model

    # want to patch head and every head after layer 7
    if hook.layer() == patch_layer:
        act[:, :, patch_head, :] = patch_cache[hook.name][:, :, patch_head, :]
    elif hook.layer() >= 5:
        act = patch_cache[hook.name]

    return act


layer_range = range(0, 5)
head_range = range(0, model.cfg.n_heads)

heads_to_patch = [
    (layer, head)
    for layer in layer_range
    for head in head_range
]

# Get patch cache
results_mat = torch.zeros((len(list(layer_range)), len(list(head_range))), device=device)
for (layer, head) in tqdm(heads_to_patch):
    # print(f'Patching L{layer}H{head}')

    model.reset_hooks()

    model.add_hook(
        lambda name: 'hook_z' in name,
        functools.partial(act_patch_hook_z, patch_cache=mean_cache, patch_layer=layer, patch_head=head),
        "fwd"
    )

    results_mat[layer, head] += probe_last_layer(model, full_prompt_toks, df['sport'].tolist(), batch_size=64) 

    model.reset_hooks()
    
#%%
# Get baseline accuracy
model.reset_hooks()
model.add_hook(
    lambda name: 'hook_z' in name,
    functools.partial(act_patch_hook_z, patch_cache=mean_cache, patch_layer=-1, patch_head=-1),
    "fwd"
)

baseline_acc = probe_last_layer(model, full_prompt_toks, df['sport'].tolist()) 

model.reset_hooks()
#%% Just load it
results_mat = torch.load('results/9b_patch_results.pt')

# %% 
import matplotlib.pyplot as plt

fig = plt.figure()
plt.imshow(results_mat.cpu().numpy() - baseline_acc, cmap='RdBu', vmax=.4, vmin=-.4)
plt.xlabel('Head', fontsize=16)
plt.ylabel('Layer', fontsize=16)
plt.title('Change in Probe Accuracy \nwhen Patching Heads', fontsize=16)
# increase font size of ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# increase font size of colorbar
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=12)
plt.show()
fig.savefig('results/9b_patch_heatmap.pdf')

# %%

layer_range = range(0, 7)
head_range = range(0, model.cfg.n_heads)
sorted_heads = sorted(
    [(layer, head) for layer in layer_range for head in head_range],
    key=lambda x: results_mat[x[0], x[1]],
    reverse=False
)

heads_below = [(layer, head) for layer, head in sorted_heads if results_mat[layer, head] < 0.9]
# sort heads below by layer, then head
heads_below = sorted(heads_below, key=lambda x: (x[0], x[1]), reverse=True)

# %% VW attn

from circuitsvis import attention

full_prompt_toks = tokenize_instructions(tokenizer, df['prompt'].tolist()) # Full prompt
athl_prompt_toks = tokenize_instructions(tokenizer, df['athlete'].tolist()) # <bos>name

# Count number of leading 0s in the tensor
def count_leading_zeros(t):
    return (t != 0).nonzero(as_tuple=True)[0][0]

# Only pick full_prompt_toks with 4 leading zeros
leading_zeros = torch.stack(
    [
        count_leading_zeros(prompt_tok)
        for prompt_tok in full_prompt_toks
    ]
)
same_length_full_prompt_toks = full_prompt_toks[leading_zeros == 5] # name toks into 2 tokens 

def get_attention(heads, cache, model, value_weighted=False):
    pattern_all = torch.stack([
        cache["pattern", layer][:, head] if cache.has_batch_dim else cache["pattern", layer][head]
        for layer, head in heads
    ], dim=-3)

    if value_weighted:
        v_all = torch.stack([
            cache["v", layer][:, :, head] if cache.has_batch_dim else cache["v", layer][:, head]
            for layer, head in heads
        ], dim=-3)
        pattern_all = attention.get_weighted_attention(pattern_all, model, "value-weighted", heads, v_all)
    return pattern_all

model.reset_hooks()
_, cache = model.run_with_cache(
    same_length_full_prompt_toks[:100],
    names_filter = lambda name: 'pattern' in name or 'v' in name
)

vw_attn = get_attention(heads_below, cache, model, value_weighted=False)

text = ['sport‎', 'of‎', 'golf', '\\n', 'Fact', ':', '{first_name}', '{last_name}', 'plays', 'the', 'sport', 'of']
# %%

from plotly import graph_objects as go
fig = go.Figure(
        data=go.Heatmap(
            z=vw_attn[:, :, -1, -12:].cpu().numpy().mean(0), # (12, 10) for prompt i
            x=text,
            y=[f'L{layer}H{head}' for (layer, head) in heads_below],
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            # hovertemplate="Start %{x} Size %{y}<br>Mean: %{z}"
        ),
        layout=go.Layout(
            title=f"Attention Weight from the 'of' Position",
            xaxis_title="Sequence Position",
            yaxis_title="Head",
            # increase font sizes
            font=dict(
                size=16
            ),
            height=600,
            width=700
        )
    )
fig.show()
# save as pdf
fig.write_image("results/7b_last_pos_attn_more.png", scale=3)

# %%

fig = go.Figure(
        data=go.Heatmap(
            z=vw_attn[:, :, -5, -12:].cpu().numpy().mean(0), # (12, 10) for prompt i
            x=text,
            y=[f'L{layer}H{head}' for (layer, head) in heads_below],
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            # hovertemplate="Start %{x} Size %{y}<br>Mean: %{z}"
        ),
        layout=go.Layout(
            title="Attention Weight <br>from the '{last_name}' Position",
            xaxis_title="Sequence Position",
            yaxis_title="Head",
            # increase font sizes
            font=dict(
                size=16
            ),
            height=600,
            width=700
        )
    )
fig.show()
# save as pdf, with high quality to prevent blurriness
fig.write_image("results/7b_last_name_attn_more.png", scale=3)

# %%

fig = go.Figure(
        data=go.Heatmap(
            z=vw_attn[:, :, -6, -12:].cpu().numpy().mean(0), # (12, 10) for prompt i
            x=text,
            y=[f'L{layer}H{head}' for (layer, head) in heads_below],
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            # hovertemplate="Start %{x} Size %{y}<br>Mean: %{z}"
        ),
        layout=go.Layout(
            title="Attention Score from the {first_name} Pos",
            xaxis_title="Sequence Position",
            yaxis_title="Head",
            # increase font sizes
            font=dict(
                size=16
            ),
            height=600,
            width=700
        )
    )
fig.show()
# save as pdf, with high quality to prevent blurriness
fig.write_image("results/7b_first_name_attn_more.png", scale=3)

# %% Patching Experiment
full_prompt_toks = tokenize_instructions(tokenizer, df['prompt'].tolist()) # Full prompt
athl_prompt_toks = tokenize_instructions(tokenizer, df['athlete'].tolist()) # <bos>name

# Count number of leading 0s in the tensor
def count_leading_zeros(t):
    return (t != 0).nonzero(as_tuple=True)[0][0]

# Only pick full_prompt_toks with 4 leading zeros
leading_zeros = torch.stack(
    [
        count_leading_zeros(prompt_tok)
        for prompt_tok in full_prompt_toks
    ]
)
same_length_full_prompt_toks = full_prompt_toks[leading_zeros == 5] # name toks into 2 tokens 

with open('tasks/facts/sports_data.json', 'r') as f:
    corr_names = json.load(f)['corr_sub_map']["{player}"]

tok_corr_names = [
    tokenizer.encode(name)
    for name in corr_names
]
same_length_corr_toks = torch.stack(
    [
        torch.tensor(name)[1:]
        for name in tok_corr_names
        if len(name) == 3 # [2, first_name, last_name]
    ],
    dim=0
)

# First name is at -6, last name at -5

### Let's try patching just the first name

### Now lets try patching both first and last name
