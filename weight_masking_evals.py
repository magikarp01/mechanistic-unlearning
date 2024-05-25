# %%
# %load_ext autoreload
# %autoreload 2
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

# %%
from functools import partial
import gc
import json
from tasks.facts.SportsTaskAdversarial import adversarial_sports_eval
from tasks.facts.SportsTaskSideEffects import run_side_effects_evals

# Final evals
evals = {
    # "Adversarial: No System Prompt": partial(adversarial_sports_eval, use_system_prompt=True),
    "Adversarial: System Prompt": partial(adversarial_sports_eval, use_system_prompt=True, include_evals=["Normal", "MC"]),#, test_each_sport=False),
    "Side Effects": partial(run_side_effects_evals, evals_to_run=["General"], verbose=False), #  "Sports Familiarity",
}
eval_batch_size=50
results = {}
localization_types = ["random", "manual", "none", "ap", "ct"]
forget_sports = ["basketball"] #["baseball", "basketball", "football"]
# localization_types = ["ap"]
# forget_sports = ["baseball"]

# min_thresholdable = float('inf')
# for localization_type in localization_types:
#     for forget_sport in forget_sports:
#         mask = torch.load(f"results/{model_name.replace('/', '_')}-{forget_sport}-{localization_type}.pt")
#         min_thresholdable = min(min_thresholdable, count_thresholdable(mask))
#         del mask
#         gc.collect()
#         torch.cuda.empty_cache()

with torch.autocast(device_type="cuda"), torch.set_grad_enabled(False):
    for localization_type in localization_types:
        results[localization_type] = {}

        for forget_sport in tqdm(forget_sports):
            results[localization_type][forget_sport] = {}
            mask = torch.load(f"results/{model_name.replace('/', '_')}-{forget_sport}-{localization_type}.pt")
            sorted_nonzero = sort_mask_weights(mask)
            del mask

            for num_weights in [0, 100_000, 200_000, 300_000, 400_000, 500_000, 700_000, 900_000, 1_200_000, 1_500_000, 1_800_000, 2_100_000, 2_400_000]:
                if num_weights > len(sorted_nonzero):
                    num_weights = len(sorted_nonzero)
                threshold = sorted_nonzero[num_weights - 1]
                str_num_weights = str(num_weights)
                print(localization_type, forget_sport, num_weights)
                results[localization_type][forget_sport][str_num_weights] = {}

                # Load Model
                model = load_model()
                mask = torch.load(f"results/{model_name.replace('/', '_')}-{forget_sport}-{localization_type}.pt")
                threshold_mask(mask, threshold)
                apply_mask(model, mask)

                del mask
                gc.collect()
                torch.cuda.empty_cache()

                for eval_name, eval_func in evals.items():
                    results[localization_type][forget_sport][str_num_weights][eval_name] = {}
                    print(f'{eval_name=}')
                    eval_result = eval_func(model, model_type=model_type, batch_size=eval_batch_size)

                    for k, v in eval_result.items():
                        results[localization_type][forget_sport][str_num_weights][eval_name][k] = v
                        print(k, v)

                    gc.collect()
                    torch.cuda.empty_cache()

                del model
                gc.collect()
                torch.cuda.empty_cache()

        # with open(f"results/{model_name.replace('/', '_')}-{localization_type}-results-update-nondisc.json", "w") as f:
        #     json.dump(results[localization_type], f, indent=2)
        with open(f"results/{model_name.replace('/', '_')}-{localization_type}-results-{forget_sport}.json", "w") as f:
            json.dump(results[localization_type], f, indent=2)

with open(f"results/{model_name.replace('/', '_')}-results.json", "w") as f:
    json.dump(results, f, indent=2)
