#%%
import pandas as pd

#%%
df = pd.read_csv("counterfact_gemma2_9b.csv")
#%%
import torch
import numpy as np
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

# %%
