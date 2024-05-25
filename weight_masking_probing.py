# %%
%load_ext autoreload
%autoreload 2
from transformer_lens import HookedTransformer, ActivationCache
import os
import torch
import numpy as np
import pandas as pd
import datasets
import transformers
import pickle

from tasks import PileTask, OWTTask, InductionTask, GreaterThanTask
from tasks.ioi.IOITask import IOITask, IOITask_NPO, IOITask_Uniform
from tasks.induction.InductionTask import InductionTask, InductionTask_NPO, InductionTask_Uniform
from tasks.facts.SportsTask import SportsTask, SportsTask_NPO, SportsTask_Uniform

from tqdm.auto import tqdm

from transformers import GPT2Tokenizer, GPTNeoXTokenizerFast, AutoModelForCausalLM, AutoTokenizer
from weight_masked_transformer import WeightMaskedTransformer

from datasets import load_dataset
train_dataset = load_dataset('monology/pile-uncopyrighted', split='train', streaming=True)


# %% [markdown]
# # Load Model

# %%
os.environ['HF_TOKEN'] = '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'
model_type = "gemma"
model_name = 'google/gemma-7b'
tokenizer = AutoTokenizer.from_pretrained(model_name)
def load_model(model_name=model_name):
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
    model.eval()
    return model

torch.set_grad_enabled(False)


# %% [markdown]
# # Evals

# %%
def threshold_mask(mask, threshold):
    for layer in mask.keys():
        for name, param in mask[layer].items():
            mask[layer][name] = torch.where(param < threshold, torch.zeros_like(param), torch.ones_like(param))

def apply_mask(model, mask):
    for layer in mask.keys():
        for name, mask_weight in mask[layer].items():
            if getattr(model.blocks[layer].attn, name, None) is not None:
                param = getattr(model.blocks[layer].attn, name)
                param.data = param * mask_weight
            elif getattr(model.blocks[layer].mlp, name, None) is not None:
                param = getattr(model.blocks[layer].mlp, name)
                param.data = param * mask_weight
            else:
                raise ValueError(f"Invalid mask name: {name} {layer=}")

def sort_mask_weights(mask):
    sorted_nonzero = []
    for layer in mask.keys():
        for param in mask[layer].values():
            sorted_nonzero.append(param[param < 1].flatten())
    return torch.cat(sorted_nonzero).sort().values 

def count_thresholdable(mask):
    total = 0
    for layer in mask.keys():
        for param in mask[layer].values():
            total += param[param < 1].numel()
    return total

#%%
results = {}
#%%
import gc
forget_sport = "basketball"
localization_type = "manual"
results[localization_type] = []
num_weights = 1_800_000
ds = SportsTask(batch_size=1024, tokenizer=tokenizer, device="cuda", prep_acdcpp=False, criterion="log_1_minus_p")
model = load_model()
mask = torch.load(f"results/{model_name.replace('/', '_')}-{forget_sport}-{localization_type}.pt")

sorted_nonzero = sort_mask_weights(mask)
threshold = sorted_nonzero[num_weights - 1]
threshold_mask(mask, threshold)
apply_mask(model, mask)

del mask
gc.collect()
torch.cuda.empty_cache()

#%%
batch = ds.get_batch()
prompts = batch["prompt"]
targets = batch["sport"]
prompt_toks = tokenizer(prompts, return_tensors="pt", padding=True).input_ids
target_toks = tokenizer(targets, return_tensors="pt", padding=True).input_ids[:, -1]

_, cache = model.run_with_cache(
    prompt_toks,
    names_filter = lambda name: 'resid_post' in name
)
cache = torch.stack([cache[key][:, -1, :] for key in cache.keys()], dim=0) # layer batch d_model

#%%
# Train logistic regressions
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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

    clf_preds = []
    # for sport in ["basketball", "baseball", "football"]:
        # Train logistic regression specifically for the sport
        # y_train_sport = np.array([1 if sport == y else 0 for y in y_train])
        # y_test_sport = np.array([1 if sport == y else 0 for y in y_test])

    # train logistic regression
    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)

    # get predictions for test set
    y_pred = clf.predict(X_test)
    # clf_preds.append(y_pred)

    test_acc = clf.score(X_test, y_test)
    print(f"Layer {layer} Accuracy: {test_acc}")
    results[localization_type].append(test_acc)

    # evaluate final accuracy
    # the regressions are accurate if the correct regression predicted the sport, and the other two did not
    # correct = 0
    # for i in range(len(y_test)):
    #     if targets[i] == "basketball" and clf_preds[0][i] == 1:#clf_preds[0][i] == 1 and clf_preds[1][i] == 0 and clf_preds[2][i] == 0:
    #         correct += 1
    #     elif targets[i] == "baseball" and clf_preds[1][i] == 1:#clf_preds[0][i] == 0 and clf_preds[1][i] == 1 and clf_preds[2][i] == 0:
    #         correct += 1
    #     elif targets[i] == "football" and clf_preds[2][i] == 1:#clf_preds[0][i] == 0 and clf_preds[1][i] == 0 and clf_preds[2][i] == 1:
    #         correct += 1
    # print(f"Layer {layer} Accuracy: {correct / len(y_test)}")

#%%
# Save to json
import json
with open(f"results/{model_name.replace('/', '_')}-{forget_sport}-probe-results.json", "w") as f:
    json.dump(results, f, indent=4)

# %%
from matplotlib import pyplot as plt
# Plot results
type_to_name = {
    'ap': 'Localized AP',
    'ct': 'Localized CT',
    'random': 'Random',
    'manual': 'Manual Interp',
    'none': 'Nonlocalized'
}
fig = plt.figure()

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
markers = ['o', 'v', '^', '*', 'p', 's', 'P']

ax = plt.gca()
ax.set_ylim([0, 1])
ax.set_xlim([0, 27])

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

for i, (k, v) in enumerate(results.items()):
    plt.plot(range(28), v, label=f'{type_to_name[k]}', color=colors[i], marker=markers[i])

plt.xlabel(f"Layer", fontsize=14)
plt.ylabel(f"Probe Accuracy", fontsize=14)
plt.title(f"Weight Masking: Probe Accuracy on Forgotten Sport", fontsize=14)
plt.legend(fontsize=12)
plt.grid()
plt.plot()
plt.show()
fig.savefig(f"results/{model_name.replace('/', '_')}-{forget_sport}-probe-acc.pdf")
# %%
