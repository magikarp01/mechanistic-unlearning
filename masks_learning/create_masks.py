#%%
# auto reload
# %load_ext autoreload
# %autoreload 2
# %cd ~/mechanistic-unlearning
import torch
import numpy as np
import os

os.chdir("..")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
### LOAD MODELS

from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained(
    # 'EleutherAI/pythia-2.8b',
    "gpt2-small",
    device='cuda',
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    default_padding_side="left",
    # dtype=torch.bfloat16
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

ind_task = InductionTask(batch_size=25, tokenizer=tokenizer, device=device)
# ioi_task = IOITask(batch_size=25, tokenizer=tokenizer, device=device, prep_acdcpp=True)
# sports_task = SportsFactsTask(
#     model, 
#     N=25, 
#     batch_size=5, 
#     tokenizer=tokenizer,
#     # forget_sport_subset={"football"},
#     # forget_player_subset={"Austin Rivers"},
#     # is_forget_dataset=False,
#     device=device
# )

#%%
### LOAD LOCALIZATION METHODS
from localizations.eap.localizer import EAPLocalizer
from localizations.causal_tracing.localizer import CausalTracingLocalizer

from cb_utils.mask_utils import get_masks_from_ct_nodes
from cb_utils.mask_utils import get_masks_from_eap_exp

for name, task in zip(["induction"], [ind_task]):
    eap_localizer = EAPLocalizer(model, task)
    ct_localizer = CausalTracingLocalizer(model, task)


    ### GET ATTRIBUTION SCORES FROM LOCALIZATIONS
    eap_graph = eap_localizer.get_exp_graph(batch=5, threshold=THRESHOLD)

    model.eval() # Don't need gradients when doing ct task
    ct_keys = ct_localizer.get_ct_keys(threshold=THRESHOLD, batch_size=5)
    model.train()

    for THRESHOLD in np.logspace(-4, 0, num=8):
        eap_mask = get_masks_from_eap_exp(eap_graph, threshold=THRESHOLD)
        ct_mask = get_masks_from_ct_nodes(ct_keys, threshold=THRESHOLD)

        ### SAVE THESE MASKS
        eap_mask.save(f"models/pythia2_8b_{name}_eap_mask_{round(THRESHOLD, 5)}_alldata.pkl")
        ct_mask.save(f"models/pythia2_8b_{name}_ct_mask_{round(THRESHOLD, 5)}_alldata.pkl")



# %%
