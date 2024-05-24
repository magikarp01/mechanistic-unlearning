#%%
%cd ~/mechanistic-unlearning
%load_ext autoreload
%autoreload 2
import os
import gc
import json

from dataset.custom_dataset import PairedInstructionDataset
import torch

from transformer_lens import HookedTransformer
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

with open('tasks/facts/sports_data.json', 'r') as f:
    data = json.load(f)

corr_sub_map = data['corr_sub_map']
clean_sub_map = data['clean_sub_map']

def tokenize_instructions(self, tokenizer, instructions):
    # Use this to put the text into INST tokens or add a system prompt
    return tokenizer(
        instructions,
        padding=True,
        truncation=False,
        return_tensors="pt",
        # padding_side="left",
    ).input_ids

dataset = PairedInstructionDataset(
    N=100,
    instruction_templates=data['instruction_templates'],
    harmful_substitution_map=corr_sub_map,
    harmless_substitution_map=clean_sub_map,
    tokenizer=tokenizer,
    tokenize_instructions=tokenize_instructions, 
    device=device
)

# %%
