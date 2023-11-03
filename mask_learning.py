#%%
## First, Loading a Model
# GPT-2 Medium for now, in Huggingface Transformer library
from transformers import GPT2Model, GPT2Config

# Load pre-trained GPT-2 Medium model
gpt2_model = GPT2Model.from_pretrained('gpt2-medium')

# send to CUDA
gpt2_model.cuda()

#%%

import torch
import torch.nn as nn

class PrunedGPT2Model(nn.Module):
    def __init__(self, gpt2_model):
        super(PrunedGPT2Model, self).__init__()
        self.gpt2_model = gpt2_model
        self.masks = nn.ParameterList()
        
        for name, param in gpt2_model.named_parameters():
            if "weight" in name:  # Apply mask to weight parameters
                mask = nn.Parameter(torch.ones_like(param))
                self.masks.append(mask)
            else:
                self.masks.append(None)

    def forward(self, input_ids):
        # Apply masks
        for i, (name, param) in enumerate(self.gpt2_model.named_parameters()):
            if self.masks[i] is not None:
                param.data = param.data * self.masks[i]
        
        return self.gpt2_model(input_ids)

# Create pruned model
pruned_model = PrunedGPT2Model(gpt2_model)

# Send to CUDA
pruned_model.cuda()

#%%
# Freeze original model weights (only want to train masks)
for param in pruned_model.gpt2_model.parameters():
    param.requires_grad = False

# %%
import pandas as pd
# Load the ConceptNet data

"""
conceptnet_data = pd.read_csv('datasets/conceptnet-assertions-5.7.0/assertions.csv', delimiter='\t', header=None)

# Save truncated version
truncated_data = conceptnet_data.sample(frac=0.01, random_state=1)
truncated_data.to_csv('path/to/save/truncated_data.tsv', sep='\t', index=False, header=False)
"""
conceptnet_data = pd.read_csv('datasets/short_conceptnet_assertions.csv', delimiter='\t', header=None)
# Filter the data to include only rows with single-token tail entities
filtered_data = conceptnet_data[conceptnet_data[2].apply(lambda x: len(x.split()) == 1)]
# %%
