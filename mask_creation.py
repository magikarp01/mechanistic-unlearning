#%%
from tasks.facts.SportsTask import SportsTask

import torch as t
from cb_utils.models import load_demo_pythia
from transformers import AutoTokenizer, AutoModelForCausalLM

device = t.device('cuda') if t.cuda.is_available() else t.device('mps:0')

#%%
model = load_demo_pythia(
    means=False
)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")

#%%
sports_task = SportsTask(batch_size=20, tokenizer=tokenizer)

# %%
