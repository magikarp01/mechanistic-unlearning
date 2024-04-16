#%%
# auto reload
%load_ext autoreload
%autoreload 2
%cd ~/mechanistic-unlearning
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
### LOAD MODELS

from transformer_lens import HookedTransformer
# model = HookedTransformer.from_pretrained(
#     'EleutherAI/pythia-2.8b',
#     device='cuda',
#     fold_ln=False,
#     center_writing_weights=False,
#     center_unembed=False,
#     default_padding_side="left"
# )
# model.set_use_attn_result(True)
# model.set_use_split_qkv_input(True)
# model.set_use_hook_mlp_in(True)

# set up pipeline from acdcpp to edge mask
model = HookedTransformer.from_pretrained(
    'gpt2-small',
    center_writing_weights=False,
    center_unembed=False,
    fold_ln=False,
    device=device,
)
model.set_use_hook_mlp_in(True)
model.set_use_split_qkv_input(True)
model.set_use_attn_result(True)
tokenizer = model.tokenizer
#%%
### LOAD TASKS
from tasks.induction.InductionTask import InductionTask
from tasks.ioi.IOITask import IOITask
# from tasks.facts.SportsTask import SportsFactsTask

# ind_task = InductionTask(batch_size=5, tokenizer=tokenizer, device=device)
# ioi_task = IOITask(batch_size=5, tokenizer=tokenizer, device=device, prep_acdcpp=True)
ioi_task = IOITask(batch_size=5, tokenizer=model.tokenizer, device=device, prep_acdcpp=True, acdcpp_N=25, nb_templates=1, prompt_type="ABBA")
ioi_task.set_logit_diffs(model)
# sports_task = SportsFactsTask(model, N=5, batch_size=5, tokenizer=tokenizer, device=device)

#%%
from localizations.causal_tracing.causal_tracing import causal_tracing_ioi
from cb_utils.mask_utils import get_masks_from_ct_nodes
import numpy as np
import pickle

result = causal_tracing_ioi(model, ioi_task)
ct_keys = list(result.keys())
# threshold = 0.005
percentile = 80
threshold = np.percentile(list(result.values()), percentile)
ct_keys_above_threshold = [k for k in ct_keys if result[k] > threshold]
print(f"{threshold=}, {len(ct_keys)=}, {len(ct_keys_above_threshold)=}")

nodes_set,edges_set, ct_mask_dict, ct_weight_mask_attn_dict, ct_weight_mask_mlp_dict = get_masks_from_ct_nodes(ct_keys_above_threshold)

with open(f"localizations/causal_tracing/ioi/gpt2_{percentile=}.pkl", "wb") as f:
    pickle.dump((nodes_set, edges_set, ct_mask_dict, ct_weight_mask_attn_dict, ct_weight_mask_mlp_dict), f)

#%%
### LOAD LOCALIZATION METHODS
from localizations.eap.localizer import EAPLocalizer
from localizations.causal_tracing.localizer import CausalTracingLocalizer

eap_localizer = EAPLocalizer(model, ioi_task)
ct_localizer = CausalTracingLocalizer(model, ioi_task)

#%%
### GET MASKS FROM LOCALIZATIONS
# eap_mask = eap_localizer.get_mask(batch=5, threshold=0.0005)

model.eval() # Don't need gradients when doing ct task
ct_mask = ct_localizer.get_mask(threshold=0.0005)

#%%
### SAVE THESE MASKS
# eap_mask.save("models/pythia2_8b_sports_eap_mask_0005.pkl")
ct_mask.save("models/pythia2_8b_sports_ct_mask_0005.pkl")



# %%
