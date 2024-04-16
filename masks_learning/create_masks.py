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
    default_padding_side="left"
)
model.set_use_attn_result(True)
model.set_use_split_qkv_input(True)
model.set_use_hook_mlp_in(True)
tokenizer = model.tokenizer
#%%
### LOAD TASKS
from tasks.induction.InductionTask import InductionTask
from tasks.ioi.IOITask import IOITask
from tasks.facts.SportsTask import SportsFactsTask

# ind_task = InductionTask(batch_size=5, tokenizer=tokenizer, device=device)
# ioi_task = IOITask(batch_size=5, tokenizer=tokenizer, device=device, prep_acdcpp=True)
sports_task = SportsFactsTask(model, N=5, batch_size=5, tokenizer=tokenizer, device=device)

#%%
### LOAD LOCALIZATION METHODS
from localizations.eap.localizer import EAPLocalizer
from localizations.causal_tracing.localizer import CausalTracingLocalizer

eap_localizer = EAPLocalizer(model, sports_task)
ct_localizer = CausalTracingLocalizer(model, sports_task)

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
