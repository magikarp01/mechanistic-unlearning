#%% Imports
from turtle import position
import torch

from transformer_lens import HookedTransformer
from pythia_transformer import DemoTransformer, Config
from IPython import get_ipython
ipython = get_ipython()
ipython.magic("load_ext autoreload")
ipython.magic("autoreload 2")
device='mps:0'

#%%
model = HookedTransformer.from_pretrained(
    "pythia-14m",
    device=device,
    fold_ln=False
)

#%%
def tl_config_to_demo_config(tl_config, debug=False):
    return Config(
            d_model = tl_config.d_model,
            debug = debug,
            layer_norm_eps = tl_config.eps,
            d_vocab = tl_config.d_vocab,
            init_range = tl_config.initializer_range,
            n_ctx = tl_config.n_ctx,
            d_head = tl_config.d_head,
            d_mlp = tl_config.d_mlp,
            n_heads = tl_config.n_heads,
            n_layers = tl_config.n_layers,
            positional_embedding_type = tl_config.positional_embedding_type,
            rotary_dim=tl_config.rotary_dim,
            rotary_base=10000,
            dtype=tl_config.dtype,
        )
#%%
demo_pythia = DemoTransformer(
    tl_config_to_demo_config(model.cfg), 
    means=False
).to(device)
demo_pythia.load_state_dict(model.state_dict(), strict=False)
# %% Test similarity
toks = model.tokenizer(
    "John and Mary went to the park. John gave the ball to", 
    return_tensors="pt"
).input_ids.to(device)
tl_logits = torch.softmax(model(toks)[0, -1, :], dim=0) # shape d_vocab
py_logits = torch.softmax(demo_pythia(toks)[0][0, -1, :], dim=0) # shape d_vocab
# %%
_, cache = model.run_with_cache(
    toks,
    names_filter=
    lambda name: "rot_q" in name
)

# %%
for i in range(len(demo_pythia.blocks)):
    pythia_rot_q = demo_pythia.blocks[i].attn.debug_q
    tl_rot_q = cache[f'blocks.{i}.attn.hook_rot_q']
    # See if similar
    print(i)
    print(torch.allclose(
        pythia_rot_q, 
        tl_rot_q,
        atol=1
    ))

# %%
