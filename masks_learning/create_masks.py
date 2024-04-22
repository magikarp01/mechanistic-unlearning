#%%
# auto reload
%load_ext autoreload
%autoreload 2
%cd ~/mechanistic-unlearning
import torch
import numpy as np
import os

# os.chdir("..")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
### LOAD MODELS

from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained(
    'EleutherAI/pythia-2.8b',
    # "gpt2-small",
    device='cuda',
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    default_padding_side="left",
    dtype=torch.bfloat16
)
model.set_use_attn_result(True)
model.set_use_split_qkv_input(True)
model.set_use_hook_mlp_in(True)
tokenizer = model.tokenizer

# from cb_utils.transformers.pythia.edge_masked_transformer import DemoTransformer as PythiaEdgeDemoTransformer, Config as PythiaConfig
# from cb_utils.models import tl_config_to_demo_config
# demo_pythia = PythiaEdgeDemoTransformer(tl_config_to_demo_config(model.cfg), means=False, edge_masks=True, dtype=torch.bfloat16)
# demo_pythia.load_state_dict(model.state_dict(), strict=False)

# #%%
# import gc
# del model
# torch.cuda.empty_cache()
# gc.collect()
# demo_pythia.to(device)
# demo_pythia.eval()
# torch.set_grad_enabled(False)
# test_string = "Breaking News: President Trump has been impeached by the House of Representatives for abuse of power and obstruction of Congress. The vote was 230 to 197, with 10 Republicans joining all Democrats in voting to impeach. The president is now only the third in American history to be impeached, and the first to be impeached twice. The House will now send the articles of impeachment to the Senate, where a trial will be held to determine whether to remove the president from office. The Senate is expected to begin the trial on"
# for i in range(30):
#     test_tokens = torch.unsqueeze(torch.tensor(tokenizer.encode(test_string)), 0).cuda()
#     demo_logits = demo_pythia(test_tokens)[0]
#     test_string += tokenizer.decode(demo_logits[-1, -1].argmax())
# print(test_string)

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
    model, 
    N=25, 
    batch_size=5, 
    tokenizer=tokenizer,
    forget_sport_subset={"football"},
    # forget_player_subset={"Austin Rivers"},
    is_forget_dataset=True,
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
model_name = 'pythia-2_8b'

for forget_sport in ['football', 'basketball', 'baseball']:
    sports_task = SportsFactsTask(
        model, 
        N=25, 
        batch_size=5, 
        tokenizer=tokenizer,
        forget_sport_subset={forget_sport},
        # forget_player_subset={"Austin Rivers"},
        is_forget_dataset=True,
        device=device
    )

    for name, task in zip(["sports"], [sports_task]):
        eap_localizer = EAPLocalizer(model, task)
        ap_localizer = APLocalizer(model, task)
        ct_localizer = CausalTracingLocalizer(model, task)


        ### GET ATTRIBUTION SCORES FROM LOCALIZATIONS
        eap_graph = eap_localizer.get_exp_graph(batch=5, threshold=-1)
        ap_graph = ap_localizer.get_ap_graph(batch_size=5)

        model.eval() # Don't need gradients when doing ct task
        ct_graph = ct_localizer.get_ct_mask(batch_size=5)
        model.train()

        for THRESHOLD in np.logspace(-6, -2, num=8):

            ### EAP
            (
                acdcpp_nodes,
                acdcpp_edges,
                acdcpp_mask_dict,
                acdcpp_weight_mask_attn_dict,
                acdcpp_weight_mask_mlp_dict,
            ) = get_masks_from_eap_exp(
                eap_graph, threshold=THRESHOLD, num_layers=model.cfg.n_layers, num_heads=model.cfg.n_heads, filter_neox=True
            )

            eap_mask = CausalGraphMask(
                nodes_set=acdcpp_nodes,
                edges_set=acdcpp_edges,
                ct_mask_dict=acdcpp_mask_dict,
                ct_weight_mask_attn_dict=acdcpp_weight_mask_attn_dict,
                ct_weight_mask_mlp_dict=acdcpp_weight_mask_mlp_dict,
            )
            eap_mask.save(f"models/{model_name}_{name}_eap_mask_{round(THRESHOLD, 5)}_{forget_sport}.pkl")

            ### CAUSAL TRACING
            ct_keys = list(ct_graph.keys())
            ct_keys_above_threshold = [k for k in ct_keys if ct_graph[k] > THRESHOLD]

            (
                nodes_set,
                edges_set,
                ct_mask_dict,
                ct_weight_mask_attn_dict,
                ct_weight_mask_mlp_dict,
            ) = get_masks_from_ct_nodes(ct_keys_above_threshold)
            ct_mask = CausalGraphMask(
                nodes_set=nodes_set,
                edges_set=edges_set,
                ct_mask_dict=ct_mask_dict,
                ct_weight_mask_attn_dict=ct_weight_mask_attn_dict,
                ct_weight_mask_mlp_dict=ct_weight_mask_mlp_dict,
            )

            ct_mask.save(f"models/{model_name}_{name}_ct_mask_{round(THRESHOLD, 5)}_{forget_sport}.pkl") 

            ### ATTRIBUTION PATCHING

            ap_keys = list(ap_graph.keys())
            ap_keys_above_threshold = [k for k in ap_keys if ap_graph[k] > THRESHOLD]
            (
                nodes_set,
                edges_set,
                ap_mask_dict,
                ap_weight_mask_attn_dict,
                ap_weight_mask_mlp_dict,
            ) = get_masks_from_ct_nodes(ap_keys_above_threshold)
            ap_mask = CausalGraphMask(
                nodes_set=nodes_set,
                edges_set=edges_set,
                ct_mask_dict=ap_mask_dict,
                ct_weight_mask_attn_dict=ap_weight_mask_attn_dict,
                ct_weight_mask_mlp_dict=ap_weight_mask_mlp_dict,
            )

            ap_mask.save(f"models/{model_name}_{name}_ap_mask_{round(THRESHOLD, 5)}_{forget_sport}.pkl") 
# %%
