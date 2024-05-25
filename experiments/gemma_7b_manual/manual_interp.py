#%%
%cd ~/mechanistic-unlearning
%load_ext autoreload
%autoreload 2
import functools
import os
import gc
import json

from dataset.custom_dataset import PairedInstructionDataset
import torch

from transformer_lens import HookedTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
model = HookedTransformer.from_pretrained(
    'google/gemma-7b',
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


# %%
import numpy as np
from datasets import load_dataset
import einops
from transformer_lens import ActivationCache

# Need to 
# 1. Probe for correct sport with no changes
# 2. Probe for correct sport with just <bos>name
# 3. Probe after meal ablating attention heads after layer 2
# 4. Probe after meal ablating attention heads after layer 2 and just <bos>name

def probe_across_layers(model, prompt_toks, targets):

    with torch.set_grad_enabled(False):
        _, cache = model.run_with_cache(
            prompt_toks,
            names_filter = lambda name: 'resid_pre' in name
        )
        cache = torch.stack([cache[key][:, -1, :] for key in cache.keys()], dim=0) # layer batch d_model

    results = []
    for layer in range(model.cfg.n_layers):
        X = cache[layer].cpu().float().numpy().reshape(-1, cache[layer].shape[-1])
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
        clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)

        test_acc = clf.score(X_test, y_test)
        print(f"Layer {layer} Accuracy: {test_acc}")
        results.append(test_acc)

    return results

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
            names_filter = lambda name: any([hook_name in name for hook_name in ["attn_out"]])
        )
    return mean_cache

def mean_ablate_hook(act, hook, mean_cache):
    if hook.layer() <= 4:
        return act
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
full_prompt_toks = tokenize_instructions(tokenizer, df['prompt'].tolist()) # Full prompt
athl_prompt_toks = tokenize_instructions(tokenizer, df['athlete'].tolist()) # <bos>name

results = {}

results['full_prompt'] = probe_across_layers(model, full_prompt_toks, df['sport'].tolist())
results['athl_prompt'] = probe_across_layers(model, full_prompt_toks, df['sport'].tolist())

model.reset_hooks()

# Add mean ablate hooks
model.add_hook(
    lambda name: 'attn_out' in name,
    functools.partial(mean_ablate_hook, mean_cache=mean_cache),
    "fwd"
)

results['mean_ablate_prompt'] = probe_across_layers(model, full_prompt_toks, df['sport'].tolist())
results['mean_ablate_athl_prompt'] = probe_across_layers(model, full_prompt_toks, df['sport'].tolist())
model.reset_hooks()

# %% Investigating the attention heads


with open('tasks/facts/sports_data.json', 'r') as f:
    data = json.load(f)

corr_sub_map = data['corr_sub_map']
clean_sub_map = data['clean_sub_map']

dataset = PairedInstructionDataset(
    N=1500,
    instruction_templates=data['instruction_templates'],
    harmful_substitution_map=corr_sub_map,
    harmless_substitution_map=clean_sub_map,
    tokenizer=tokenizer,
    tokenize_instructions=tokenize_instructions, 
    device=device
)

#%%

def act_patch_hook_z(act, hook, patch_cache, head):
    # heads_to_patch is [(layer, head)]
    # heads = [head for layer, head in heads_to_patch if layer == hook.layer()]

    # act is batch head seq d_model
    act[:, head, ...] = patch_cache[hook.name][:, head, ...]

    return act

layer_range = range(0, 9)
head_range = range(0, model.cfg.n_heads)

heads_to_patch = [
    (layer, head)
    for layer in layer_range
    for head in head_range
]

# Get patch cache
model.reset_hooks()
_, patch_cache = model.run_with_cache(
    dataset.harmful_dataset.toks,
    names_filter = lambda name: 'hook_z' in name
)

results_mat = torch.zeros((len(list(layer_range)), len(list(head_range))))

for (layer, head) in heads_to_patch:
    print(f'Layer {layer} Head {head}')

    model.reset_hooks()
    model.add_hook(
        lambda name: 'attn_out' in name,
        functools.partial(act_patch_hook_z, patch_cache=patch_cache, head=head),
        "fwd"
    )

    results_mat[layer, head] = # TODO
    model.reset_hooks()

