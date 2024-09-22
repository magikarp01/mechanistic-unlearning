#%%
from localizations.ap.ap_wrapper import AP
#%%

from transformer_lens import HookedTransformer
MODEL_NAME = "google/gemma-2-2b"
DEVICE = "cuda"
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    default_padding_side="left",
    device=DEVICE
)
model.set_use_attn_result(True)
model.set_use_split_qkv_input(True)
# model.set_use_hook_mlp_in(True)

#%%
class APLocalizer():

    def __init__(
        self, 
        model, 
        clean_toks,
        corr_toks,
        metric,
        correct_toks=None,
        incorrect_toks=None,
        verbose=False
    ):
        self.model = model
        self.clean_toks = clean_toks
        self.corr_toks = corr_toks
        self.metric = metric

        self.correct_toks = correct_toks
        self.incorrect_toks = incorrect_toks
        self.verbose = verbose

    def get_normalized_ap_scores(self, batch_size=20):
        '''
            Returns the normalized AP scores
            0 if the node has same effect as corr, 
            1 if it has same effect as clean
        '''

        nodes = AP(
            self.model,
            self.clean_toks,
            self.corr_toks,
            self.metric,
            batch_size=batch_size,
            clean_answers=self.correct_toks,
            wrong_answers=self.incorrect_toks,
        )
        return nodes 

#%%

from tasks.facts.CounterFactTask import CounterFactTask
from transformers import AutoTokenizer

right_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
forget_facts = 16
forget_kwargs = {"forget_fact_subset": forget_facts, "is_forget_dataset": True, "train_test_split": False}
maintain_kwargs = {"forget_fact_subset": forget_facts, "is_forget_dataset": False, "train_test_split": True}
forget_fact_eval = CounterFactTask(batch_size=32, tokenizer=right_tokenizer, device=DEVICE, criterion="cross_entropy", **forget_kwargs)
maintain_facts_eval = CounterFactTask(batch_size=32, tokenizer=right_tokenizer, device=DEVICE, criterion="cross_entropy", **maintain_kwargs)

import numpy as np

def find_sublist_indices(tensor, sublists):
    # Store indices of starting positions for each sublist
    indices = []
    
    # Iterate over each row in the tensor and corresponding sublist
    for row, sublist in zip(tensor, sublists):
        row_len = len(row)
        sublist_len = len(sublist)
        found_index = -1  # Default value if sublist is not found
        
        # Slide over the row to find the sublist
        for i in range(row_len - sublist_len + 1):
            if np.array_equal(row[i:i+sublist_len], sublist):
                found_index = list(range(i, i+sublist_len))
                break
        
        indices.append(found_index)
    
    return indices

#%%
import torch

# 'prompt' is list of string prompts
# 'subject' is the string main subject of the prompt
# 'first_token' is int correct answer first tokens
prompt_toks = model.tokenizer(
    forget_fact_eval.train_df['prompt'].tolist(),
    padding=True,
    return_tensors="pt"
)['input_ids']

subject_toks = model.tokenizer(
    forget_fact_eval.train_df['subject'].tolist(),
    add_special_tokens=False
)['input_ids']

subject_idxs = find_sublist_indices(prompt_toks, subject_toks)

correct_toks = torch.tensor(
    forget_fact_eval.train_df['first_token'].tolist()
)

#%%
