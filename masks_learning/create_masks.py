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
model_name = 'EleutherAI/pythia-2.8b'
    #  'google/gemma-7b' 
    # 'meta-llama/Meta-Llama-3-8B'
    # 'Qwen/Qwen1.5-4B' 
    # 'EleutherAI/pythia-2.8b',
    # "gpt2-small",

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
# ioi_task = IOITask(batch_size=25, tokenizer=tokenizer, device=device, prep_acdcpp=True)
# ioi_task.set_logit_diffs(model)
sports_task = SportsFactsTask(
    model=model, 
    batch_size=5, 
    tokenizer=tokenizer,
    N=25, 
    # forget_sport_subset={"football"},
    # forget_player_subset={"Austin Rivers"},
    # is_forget_dataset=True,
    device=device
)

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

        # for THRESHOLD in np.logspace(-8, 1, num=10):

            ### EAP
            # (
            #     acdcpp_nodes,
            #     acdcpp_edges,
            #     acdcpp_mask_dict,
            #     acdcpp_weight_mask_attn_dict,
            #     acdcpp_weight_mask_mlp_dict,
            # ) = get_masks_from_eap_exp(
            #     eap_graph, threshold=THRESHOLD, num_layers=model.cfg.n_layers, num_heads=model.cfg.n_heads, filter_neox=True
            # )

            # eap_mask = CausalGraphMask(
            #     nodes_set=acdcpp_nodes,
            #     edges_set=acdcpp_edges,
            #     ct_mask_dict=acdcpp_mask_dict,
            #     ct_weight_mask_attn_dict=acdcpp_weight_mask_attn_dict,
            #     ct_weight_mask_mlp_dict=acdcpp_weight_mask_mlp_dict,
            # )
            # eap_mask.save(f"models/{save_model_name}_{name}_eap_mask_{round(THRESHOLD, 10)}_{forget_sport}.pkl")

            ## ATTRIBUTION PATCHING

            # ap_keys = list(ap_graph.keys())
            # ap_keys_above_threshold = [k for k in ap_keys if ap_graph[k] > THRESHOLD]
            # (
            #     nodes_set,
            #     edges_set,
            #     ap_mask_dict,
            #     ap_weight_mask_attn_dict,
            #     ap_weight_mask_mlp_dict,
            # ) = get_masks_from_ct_nodes(ap_keys_above_threshold)
            # ap_mask = CausalGraphMask(
            #     nodes_set=nodes_set,
            #     edges_set=edges_set,
            #     ct_mask_dict=ap_mask_dict,
            #     ct_weight_mask_attn_dict=ap_weight_mask_attn_dict,
            #     ct_weight_mask_mlp_dict=ap_weight_mask_mlp_dict,
            # )

            # ap_mask.save(f"models/{save_model_name}_{name}_ap_mask_{round(THRESHOLD, 5)}_{forget_sport}.pkl") 
            
            ### CAUSAL TRACING
            # ct_keys = list(ct_graph.keys())
            # ct_keys_above_threshold = [k for k in ct_keys if ct_graph[k] > THRESHOLD]

            # (
            #     nodes_set,
            #     edges_set,
            #     ct_mask_dict,
            #     ct_weight_mask_attn_dict,
            #     ct_weight_mask_mlp_dict,
            # ) = get_masks_from_ct_nodes(ct_keys_above_threshold)
            # ct_mask = CausalGraphMask(
            #     nodes_set=nodes_set,
            #     edges_set=edges_set,
            #     ct_mask_dict=ct_mask_dict,
            #     ct_weight_mask_attn_dict=ct_weight_mask_attn_dict,
            #     ct_weight_mask_mlp_dict=ct_weight_mask_mlp_dict,
            # )

            # ct_mask.save(f"models/{save_model_name}_{name}_ct_mask_{round(THRESHOLD, 5)}_{forget_sport}.pkl") 
            # torch.cuda.empty_cache()
            # gc.collect()

            # torch.cuda.empty_cache()
            # gc.collect()
# %%
