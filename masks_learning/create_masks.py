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
model = HookedTransformer.from_pretrained(
    'EleutherAI/pythia-2.8b',
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

# ind_task = InductionTask(batch_size=5, tokenizer=tokenizer, device=device)
# ioi_task = IOITask(batch_size=5, tokenizer=tokenizer, device=device, prep_acdcpp=True)
sports_task = SportsFactsTask(
    model, 
    N=10, 
    batch_size=5, 
    tokenizer=tokenizer,
    # forget_sport_subset={"football"},
    # forget_player_subset={"Austin Rivers"},
    # is_forget_dataset=False,
    device=device
)

#%%
### LOAD LOCALIZATION METHODS
from localizations.eap.localizer import EAPLocalizer
from localizations.causal_tracing.localizer import CausalTracingLocalizer

eap_localizer = EAPLocalizer(model, sports_task)
ct_localizer = CausalTracingLocalizer(model, sports_task)

#%%
### GET MASKS FROM LOCALIZATIONS
# eap_mask = eap_localizer.get_mask(batch=10, threshold=0.005)

model.eval() # Don't need gradients when doing ct task
ct_mask = ct_localizer.get_mask(threshold=0.0005, batch_size=5)

#%%
### SAVE THESE MASKS
eap_mask.save("models/pythia2_8b_sports_eap_mask_005.pkl")
# ct_mask.save("models/pythia2_8b_sports_ct_mask_0005.pkl")



# %%
