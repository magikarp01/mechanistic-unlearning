#%%
%cd ..
#%% IMPORTS
from localizations.eap import eap_graph, eap_wrapper
from transformer_lens import HookedTransformer
from transformer_lens import utils

import torch
device = 'cuda' if torch.cuda.is_available() else 'mps:0'

# %% MODEL
gpt2_small = HookedTransformer.from_pretrained(
    'gpt2-small',
).to(device)
gpt2_small.set_use_hook_mlp_in(True)
gpt2_small.set_use_split_qkv_input(True)
gpt2_small.set_use_attn_result(True)
#%% IOI
from tasks.ioi.IOITask import IOITask

ioi_task = IOITask(batch_size=5, tokenizer=gpt2_small.tokenizer, device=device, prep_acdcpp=True, acdcpp_N=25, nb_templates=1, prompt_type="ABBA")
ioi_task.set_logit_diffs(gpt2_small)
ioi_metric = ioi_task.get_acdcpp_metric()

def negative_abs_ioi_metric(logits):
    return -abs(ioi_metric(logits))

with torch.no_grad():
    clean_logits = gpt2_small(ioi_task.clean_data.toks)
    corrupt_logits = gpt2_small(ioi_task.corr_data.toks)
    clean_logit_diff = ioi_task.ave_logit_diff(clean_logits, ioi_task.clean_data).item()
    corrupt_logit_diff = ioi_task.ave_logit_diff(corrupt_logits, ioi_task.corr_data).item()
    print(f'Clean logit diff: {clean_logit_diff:.3f}, Corrupt logit diff: {corrupt_logit_diff:.3f}')

# %%
graph = eap_wrapper.EAP(
    gpt2_small,
    ioi_task.clean_data.toks[:2],
    ioi_task.corr_data.toks[:2],
    negative_abs_ioi_metric,
    upstream_nodes=["mlp", "head"],
    downstream_nodes=["mlp", "head"],
    batch_size=1,
    # clean_answers=ioi_task.ioi_data.io_tokenIDs[:2],
    # wrong_answers=ioi_task.ioi_data.s_tokenIDs[:2],
)

# %%
from tasks.induction.InductionTask import InductionTask
ind_task = InductionTask(batch_size=5, tokenizer=gpt2_small.tokenizer, prep_acdcpp=True, seq_len=15, acdcpp_metric="ave_logit_diff")
ind_task.set_logit_diffs(gpt2_small)

ind_metric = ind_task.get_acdcpp_metric()
def negative_abs_ind_metric(logits):
    return -abs(ind_metric(logits))

# %%
graph = eap_wrapper.EAP(
    gpt2_small,
    ind_task.clean_data[:2],
    ind_task.corr_data[:2],
    negative_abs_ind_metric,
    upstream_nodes=["mlp", "head"],
    downstream_nodes=["mlp", "head"],
    batch_size=1,
)


# %%
from tasks.greaterthan.GreaterThanTask import GreaterThanTask
gt_task = GreaterThanTask(batch_size=5, tokenizer=gpt2_small.tokenizer)
gt_task.set_logit_diffs(gpt2_small)

gt_metric = gt_task.get_acdcpp_metric()
def negative_abs_gt_metric(logits):
    return -abs(gt_metric(logits))

# %%
