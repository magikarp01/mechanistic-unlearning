
"""
Computes the mean of the GPT2 embeddings for the OWT dataset.
"""

# %%

from data import retrieve_owt_data, retrieve_ioi_data, batch_text_to_tokens
from models import load_demo_gpt2, tokenizer
from tqdm import tqdm
import torch

# %%
template_type = "single"
batch_size = 10
# ctx_length = 50
model = load_demo_gpt2(means=False)
# data_loader = retrieve_owt_data(batch_size)
data_loader = retrieve_ioi_data(batch_size, template_type=template_type, abc=True, split="train")

# %%

def compute_means(data_loader):
    means = []
    meta_means = []
    for c, batch in enumerate(tqdm(data_loader)):
        # tokenize
        # texts = batch['text']
        # tokenized = tokenizer(texts, padding=True, truncation=True, max_length=ctx_length, return_tensors="pt").input_ids
        # print(torch.stack(batch['tokens']).shape)
        # torch.tensor(batch['tokens'])
        # print(f"{torch.tensor(batch['tokens']).shape=}")
        # print(f"{tokenized.shape=}")
        # print(f"{batch['tokens']=}")
        # print(batch)
        with torch.no_grad():
            # print(f"{model(tokenized.long(), return_states=True).shape=}")
            # means.append(model(tokenized.long(), return_states=True).mean(dim=[0,1],keepdim=True))
            # means.append(model(batch_text_to_tokens(batch), return_states=True).mean(dim=[0,1],keepdim=True))
            means.append(model(batch_text_to_tokens(batch, ctx_length=50, pad_max=True), return_states=True).mean(dim=[0],keepdim=True))
            # print(batch_text_to_tokens(batch, ctx_length=None).shape)
        if c % 50 == 0:
            meta_means.append(torch.stack(means, dim=0).mean(dim=0))
            means = []
        # normal_loss = infer_batch(model, torch.nn.CrossEntropyLoss(), tokenizer, batch, data_loader.batch_size, demos)
    all_means = torch.stack(meta_means, dim=0).mean(dim=0)
    return all_means

means = compute_means(data_loader)

# %%

import pickle 
with open(f'data/gpt2_{template_type}_ioi_abc_means.pkl', 'wb') as f:
    pickle.dump(means, f)

# %%
