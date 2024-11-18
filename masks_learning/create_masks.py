#%%
# auto reload
# %load_ext autoreload
# %autoreload 2
# %cd ~/mechanistic-unlearning
import torch
import numpy as np
import os
import sys
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from localizations.causal_tracing.localizer import CausalTracingLocalizer
from localizations.ap.localizer import APLocalizer

import pickle
import pandas as pd

os.chdir("/root/mechanistic-unlearning/")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["HF_TOKEN"] = "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"
#%%
### LOAD MODELS
model_name = 'google/gemma-2-9b'
#'meta-llama/Meta-Llama-3-8B' 
# 'meta-llama/Meta-Llama-3-8B'

#'EleutherAI/pythia-2.8b'
    # 'meta-llama/Meta-Llama-3-8B'
    # 'Qwen/Qwen1.5-4B' 

from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained(model_name)
model = HookedTransformer.from_pretrained(
    model_name,
    # tokenizer=tokenizer,
    device='cuda',
    default_padding_side="left",
    fold_ln=False,
    fold_value_biases=False,
    center_writing_weights=False,
    dtype=torch.bfloat16
)
tokenizer = model.tokenizer
model.set_use_attn_result(True)
model.set_use_split_qkv_input(True)
model.set_use_hook_mlp_in(True)

#%%

#%%
### LOAD LOCALIZATION METHODS

save_model_name = model_name.replace('/', '_')
torch.cuda.empty_cache()
gc.collect()

df = pd.read_csv('experiments/sports_facts_manual/sports.csv')
df = df[:80]

def tokenize_instructions(tokenizer, instructions):
    # Use this to put the text into INST tokens or add a system prompt
    return tokenizer(
        instructions,
        padding=True,
        truncation=False,
        return_tensors="pt",
        # padding_side="left",
    ).input_ids

full_prompt_toks = tokenize_instructions(tokenizer, df['prompt'].tolist()) # Full prompt
athl_prompt_toks = tokenize_instructions(tokenizer, df['athlete'].tolist()) # <bos>name

def find_subarray_occurrences(arr, subarr):
    n = len(arr)
    m = len(subarr)
    occurrences = []

    # Traverse through the main array
    for i in range(n - m + 1):
        # Check if the subarray matches starting from index i
        if arr[i:i + m] == subarr:
            occurrences.extend(list(range(i, i+m)))
    
    return occurrences


def find_subject_occurences(prompt_toks_tensor, subject_toks_list):
    # Find positions where convolution result matches the sum of each subarray, accounting for their actual length
    match_positions = []
    for i, subarray in enumerate(subject_toks_list):
        match_positions.append(find_subarray_occurrences(prompt_toks_tensor[i].tolist(), subarray))

    return match_positions

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def get_random_toks(ascii_toks, prompt_tok, num_rand_needed, idx_to_replace):
    orig_len = prompt_tok.shape[0]
    orig_prompt = prompt_tok.clone()
    for _ in range(100):
        rand = ascii_toks[torch.randint(0, ascii_toks.shape[0], (num_rand_needed,))]
        orig_prompt[idx_to_replace] = rand
        rand_prompt_len = len(model.tokenizer.encode(model.tokenizer.decode(orig_prompt), add_special_tokens=False))
        if rand_prompt_len == orig_len:
            return rand
    return None

subject_toks = [tokenizer.encode(' ' + athlete, add_special_tokens=False) for athlete in df['athlete'].tolist()]
subject_occs = find_subject_occurences(full_prompt_toks, subject_toks)
correct_toks = [tokenizer.encode(' ' + sport, add_special_tokens=False) for sport in df['sport'].tolist()]

# Get the list of tokens for the wrong sports
wrong_toks = []
for i, sport in enumerate(df['sport']):
    if sport == 'basketball':
        wrong_toks.append([model.tokenizer.encode(' football', add_special_tokens=False), model.tokenizer.encode(' baseball', add_special_tokens=False)])
    elif sport == 'football':
        wrong_toks.append([model.tokenizer.encode(' basketball', add_special_tokens=False), model.tokenizer.encode(' baseball', add_special_tokens=False)])
    elif sport == 'baseball':
        wrong_toks.append([model.tokenizer.encode(' football', add_special_tokens=False), model.tokenizer.encode(' basketball', add_special_tokens=False)])

ascii_toks = torch.tensor([i for i in range(model.cfg.d_vocab) if is_ascii(model.tokenizer.decode(i))])
rand_toks = full_prompt_toks.clone()

for batch_idx in range(rand_toks.shape[0]):
    # Replace with random token 
    rand_toks[batch_idx, subject_occs[batch_idx]] = get_random_toks(ascii_toks, full_prompt_toks[batch_idx], len(subject_occs[batch_idx]), subject_occs[batch_idx])

#%%
def ave_logit_diff(logits, correct_toks=correct_toks, wrong_toks=wrong_toks):
    # Wrong logit calculation
    wrong_logit_weight = torch.zeros(logits.shape[0]).to(device)
    for i, idx in enumerate(wrong_toks):
        wrong_logit_weight[i] = logits[i, -1, idx].mean()

    # print((logits[range(logits.shape[0]), -1, correct_toks] - wrong_logit_weight).mean())
    return (logits[range(logits.shape[0]), -1, correct_toks] - wrong_logit_weight).mean()

# auto cast
with torch.set_grad_enabled(False), torch.cuda.amp.autocast(True, model.W_in.dtype):
    clean_logit_diff = ave_logit_diff(model(full_prompt_toks)).item()
    corr_logit_diff = ave_logit_diff(model(rand_toks)).item()
    print(f"{clean_logit_diff=}, {corr_logit_diff=}")

def noising_metric(logits, correct_toks, wrong_toks, clean_logit_diff=clean_logit_diff, corr_logit_diff=corr_logit_diff):
    # used for patching corrupt -> clean
    logit_diff = ave_logit_diff(logits, correct_toks, wrong_toks)
    return (logit_diff - clean_logit_diff) / (clean_logit_diff - corr_logit_diff)

def denoising_metric(logits, correct_toks, wrong_toks, clean_logit_diff=clean_logit_diff, corr_logit_diff=corr_logit_diff):
    # used for patching clean -> corrupt 
    logit_diff = ave_logit_diff(logits, correct_toks, wrong_toks)
    return (logit_diff - corr_logit_diff) / (clean_logit_diff - corr_logit_diff)


#%%
# AP
# ap_loc = APLocalizer(
#     model,
#     full_prompt_toks,
#     rand_toks,
#     noising_metric,
#     correct_toks,
#     wrong_toks
# ).get_normalized_ap_scores(batch_size=10)

# #%% 
# with open(f"models/{model_name.replace('/', '_')}_sport_ap_graph.pkl", "wb") as f:
#     pickle.dump(dict(ap_loc), f)


#%%
# CT
model.eval()
with torch.set_grad_enabled(False):
    ct_loc = CausalTracingLocalizer(
        model,
        full_prompt_toks,
        find_subject_occurences(full_prompt_toks, subject_toks),
        correct_toks
    )
    ct_ind = ct_loc.get_normalized_indirect_effect()

#%%

with open(f"models/{model_name.replace('/', '_')}_sport_ct_graph.pkl", "wb") as f:
    pickle.dump(dict(ct_ind), f)

# %%
