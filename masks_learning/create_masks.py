#%%
# auto reload
%load_ext autoreload
%autoreload 2
%cd ~/mechanistic-unlearning
import torch
import numpy as np
import os
import gc

# os.chdir("..")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
### LOAD MODELS
model_name = 'google/gemma-7b' 
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
### LOAD TASKS
from tasks.induction.InductionTask import InductionTask
from tasks.ioi.IOITask import IOITask
from tasks.facts.SportsTask import SportsFactsTask

# ind_task = InductionTask(batch_size=25, tokenizer=tokenizer, prep_acdcpp=True, device=device)
# ind_task.set_logit_diffs(model)
ioi_task = IOITask(batch_size=25, tokenizer=tokenizer, device=device, prep_acdcpp=True)
ioi_task.set_logit_diffs(model)
# sports_task = SportsFactsTask(
#     model=model, 
#     batch_size=5, 
#     tokenizer=tokenizer,
#     N=25, 
#     # forget_sport_subset={"football"},
#     # forget_player_subset={"Austin Rivers"},
#     # is_forget_dataset=True,
#     device=device
# )

save_model_name = model_name.replace('/', '_')
for name, task in zip(["ioi"], [ioi_task]):
    eap_localizer = EAPLocalizer(model, task)
    ap_localizer = APLocalizer(model, task)
    ct_localizer = CausalTracingLocalizer(model, task)


    ### GET ATTRIBUTION SCORES FROM LOCALIZATIONS
    torch.cuda.empty_cache()
    gc.collect()
    eap_graph = eap_localizer.get_exp_graph(batch=1, threshold=-1)
    top_edges = eap_graph.top_edges(n=len(eap_graph.eap_scores.flatten()), threshold=-1)
    with open(f"models/{save_model_name}_{name}_eap_graph.pkl", "wb") as f:
        pickle.dump(top_edges, f)

    torch.cuda.empty_cache()
    gc.collect()
    ap_graph = ap_localizer.get_ap_graph(batch_size=1)
    torch.cuda.empty_cache()
    gc.collect()
    with open(f"models/{save_model_name}_{name}_ap_graph.pkl", "wb") as f:
        pickle.dump(dict(ap_graph), f)

    model.eval() # Don't need gradients when doing ct task
    ct_graph = ct_localizer.get_ct_mask(batch_size=2)
    model.train()
    with open(f"models/{save_model_name}_{name}_ct_graph.pkl", "wb") as f:
        pickle.dump(dict(ct_graph), f)

    torch.cuda.empty_cache()
    gc.collect()
#%%
### LOAD LOCALIZATION METHODS
from localizations.eap.localizer import EAPLocalizer
from localizations.causal_tracing.localizer import CausalTracingLocalizer
from localizations.ap.localizer import APLocalizer

from cb_utils.mask_utils import get_masks_from_ct_nodes
from cb_utils.mask_utils import get_masks_from_eap_exp

from masks import CausalGraphMask, MaskType
import pickle

save_model_name = model_name.replace('/', '_')
torch.cuda.empty_cache()
gc.collect()
for forget_sport in ['all']: # ['all']:#
    torch.cuda.empty_cache()
    gc.collect()
    if forget_sport == 'athlete':
        sports_task = SportsFactsTask(
            model=model, 
            N=26, 
            batch_size=2, 
            tokenizer=tokenizer,
            forget_player_subset=16,
            is_forget_dataset=True,
            device=device
        )
    elif forget_sport == 'all':
        sports_task = SportsFactsTask(
            model=model, 
            N=26, 
            batch_size=2, 
            tokenizer=tokenizer,
            device=device
        )
    else:
        sports_task = SportsFactsTask(
            model=model, 
            N=26, 
            batch_size=2, 
            tokenizer=tokenizer,
            forget_sport_subset={forget_sport},
            is_forget_dataset=True,
            device=device
        )

    for name, task in zip(["sports"], [sports_task]):
        eap_localizer = EAPLocalizer(model, task)
        ap_localizer = APLocalizer(model, task)
        ct_localizer = CausalTracingLocalizer(model, task)


        ### GET ATTRIBUTION SCORES FROM LOCALIZATIONS
        torch.cuda.empty_cache()
        gc.collect()
        eap_graph = eap_localizer.get_exp_graph(batch=2, threshold=-1)
        top_edges = eap_graph.top_edges(n=len(eap_graph.eap_scores.flatten()), threshold=-1)
        with open(f"models/{save_model_name}_{name}_{forget_sport}_eap_graph.pkl", "wb") as f:
            pickle.dump(top_edges, f)

        torch.cuda.empty_cache()
        gc.collect()
        ap_graph = ap_localizer.get_ap_graph(batch_size=2)
        torch.cuda.empty_cache()
        gc.collect()
        with open(f"models/{save_model_name}_{name}_{forget_sport}_ap_graph.pkl", "wb") as f:
            pickle.dump(dict(ap_graph), f)

        model.eval() # Don't need gradients when doing ct task
        ct_graph = ct_localizer.get_ct_mask(batch_size=6)
        model.train()
        with open(f"models/{save_model_name}_{name}_{forget_sport}_ct_graph.pkl", "wb") as f:
            pickle.dump(dict(ct_graph), f)

        torch.cuda.empty_cache()
        gc.collect()


#%%
