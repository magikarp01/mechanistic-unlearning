#%% Imports
from turtle import position
import torch

from transformer_lens import HookedTransformer
from cb_utils.pythia_transformer import DemoTransformer, Config
from IPython import get_ipython
ipython = get_ipython()
ipython.magic("load_ext autoreload")
ipython.magic("autoreload 2")
device='cuda'

#%%
model = HookedTransformer.from_pretrained(
    "pythia-70m",
    device=device,
    fold_ln=False
)
model.set_use_attn_result(True)
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

print(torch.allclose(
    py_logits, 
    tl_logits,
    atol=1e-3
))
# %%
hooks = [
    "hook_resid_pre", 
    "ln1.hook_normalized",
    "attn.hook_rot_q",
    "attn.hook_rot_k",
    "attn.hook_v",
    "attn.hook_z",
    "hook_attn_out", 
    "ln2.hook_normalized",
    "hook_mlp_out"
]
_, cache = model.run_with_cache(
    toks,
    names_filter=
    lambda name: any(HOOK in name for HOOK in hooks)
)

# %%
for HOOK in hooks:
    print(f'Testing {HOOK}')
    for i in range(len(demo_pythia.blocks)):
        if HOOK == 'hook_resid_pre':
            pythia = demo_pythia.blocks[i].debug_pre[:,:,0]
        elif HOOK == 'ln1.hook_normalized':
            pythia = demo_pythia.blocks[i].debug_ln1[:, :, 0]
        elif HOOK == 'attn.hook_rot_q':
            pythia = demo_pythia.blocks[i].attn.debug_q
        elif HOOK == 'attn.hook_rot_k':
            pythia = demo_pythia.blocks[i].attn.debug_k
        elif HOOK == 'attn.hook_v':
            pythia = demo_pythia.blocks[i].attn.debug_v
        elif HOOK == 'attn.hook_z':
            pythia = demo_pythia.blocks[i].attn.debug_z
        elif HOOK == 'hook_attn_out':
            pythia = demo_pythia.blocks[i].attn.debug_attn_out.sum(dim=-2)
        elif HOOK == 'ln2.hook_normalized':
            pythia = demo_pythia.blocks[i].debug_ln2
        elif HOOK == 'hook_mlp_out':
            pythia = demo_pythia.blocks[i].debug_mlp_out.squeeze()

        tl = cache[f'blocks.{i}.{HOOK}']
        # See if similar
        print(i)
        print(torch.allclose(
            pythia, 
            tl,
            atol=1e-3
        ))



# %%
