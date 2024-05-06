#%%
# %cd ~/mechanistic-unlearning/
from collections import defaultdict
import gc

from transformer_lens import HookedTransformer, ActivationCache
import torch
from torch import Tensor
import einops

from functools import partial
from tqdm.auto import tqdm

from typing import Callable, Tuple, Union
from jaxtyping import Int

from tasks.ioi.IOITask import IOITask

from collections import defaultdict
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = HookedTransformer.from_pretrained(
#     "gpt2-small",
#     center_writing_weights=False,
#     center_unembed=False,
#     device=device

# )
# tokenizer = model.tokenizer
# model.set_use_attn_result(True)
# model.set_use_split_qkv_input(True)
# model.set_use_hook_mlp_in(True)
# model.set_use_attn_in(True)

# ioi_task = IOITask(batch_size=25, tokenizer=tokenizer, device=device, prep_acdcpp=True)
# ioi_task.set_logit_diffs(model)
#%%
### Node based attribution patching


def get_caches(
    model, 
    clean_tokens, 
    corrupted_tokens, 
    metric, 
    clean_answers=None, 
    wrong_answers=None
):
    model.reset_hooks()
    model.to(device)

    if model.cfg.model_name == "gemma-2b":
        # Uses GroupedQueryAttention
        wanted_hooks = ['hook_result', 'mlp_in', 'mlp_out', 'hook_q']
    else:
        wanted_hooks = ['hook_result', 'mlp_in', 'mlp_out', 'hook_q', 'hook_k', 'hook_v']
    hooks_filter = lambda name: any([hook in name for hook in wanted_hooks]) and not 'input' in name

    forward_cache = {}
    backward_cache = {}

    def fwd_cache_hook(act, hook):
        forward_cache[hook.name] = act.detach()

    def bwd_cache_hook(act, hook):
        backward_cache[hook.name] = act.detach()

    def cache_subtract_hook(act, hook):
        old_cache = forward_cache[hook.name] 
        forward_cache[hook.name] = old_cache - act.detach()

    # Corrupt run
    model.reset_hooks()
    model.add_hook(hooks_filter, fwd_cache_hook, "fwd")
    with torch.set_grad_enabled(False):
        model(corrupted_tokens) # Fill corrupt forward cache

    # Clean run
    model.reset_hooks()
    model.add_hook(hooks_filter, cache_subtract_hook, "fwd")
    model.add_hook(hooks_filter, bwd_cache_hook, "bwd")

    if clean_answers is not None:
        value = metric(model(clean_tokens), clean_answers, wrong_answers)
    else:
        value = metric(model(clean_tokens))
    value.backward()

    model.reset_hooks()
    return value.item(), ActivationCache(forward_cache, model), ActivationCache(backward_cache, model)

def generate_attr_scores(model, difference_cache, clean_grad_cache):
    # Want head output results and MLP output results

    difference_head_out = difference_cache.stack_head_results(-1, return_labels=False)
    grad_head_out = clean_grad_cache.stack_head_results(-1, return_labels=False)

    head_attrs = einops.reduce(
        difference_head_out * grad_head_out, 
        "heads batch seq d_model -> heads",
        "sum"
    )
    head_attrs = einops.rearrange(
        head_attrs,
        "(layer head) -> layer head",
        layer=model.cfg.n_layers
    )

    head_attrs = {}
    mlp_attrs = {}
    for (key, val) in difference_cache.items():
        grad_val = clean_grad_cache[key]
        assert val.shape == grad_val.shape, f"Head results are not the same shape: {val.shape} vs {grad_val.shape}"

        # Renaming the key from blocks.layers.attn.hook_{q,k,v,result} = layer_{q,k,v,result}
        # or blocks.layers.mlp.hook_{in,out} = layer_{in,out}
        layer = int(key.split(".")[1])
        hook_type = key.split("_")[-1]

        if 'mlp' in key:
            mlp_attrs[f"{layer}_{hook_type}"] = einops.reduce(
                val * grad_val,
                "batch seq d_model -> ",
                "sum"
            )
        else:
            head_attrs[f"{layer}_{hook_type}"] = einops.reduce(
                val * grad_val,
                "batch seq heads d_model -> heads",
                "sum"
            )

    return head_attrs, mlp_attrs

def AP(
    model: HookedTransformer,
    clean_tokens: Int[Tensor, "batch_size seq_len"],
    corrupted_tokens: Int[Tensor, "batch_size seq_len"],
    metric: Callable,
    batch_size: int=None,
    clean_answers=None,
    wrong_answers=None
):
    assert clean_tokens.shape == corrupted_tokens.shape, "Shape mismatch between clean and corrupted tokens"
    num_prompts, seq_len = clean_tokens.shape[0], clean_tokens.shape[1]

    if batch_size is None:
        batch_size = num_prompts

    assert num_prompts % batch_size == 0, "Number of prompts must be divisible by batch size"

    nodes = defaultdict(lambda: 0)
    for idx in tqdm(range(0, num_prompts, batch_size)):
        # Get caches
        _, difference_cache, clean_grad_cache = get_caches(
            model, 
            clean_tokens[idx:idx+batch_size], 
            corrupted_tokens[idx:idx+batch_size], 
            metric,
            clean_answers[idx:idx+batch_size] if clean_answers is not None else None,
            wrong_answers[idx:idx+batch_size] if wrong_answers is not None else None
        )

        # Take this cache and generate attribution scores for each node
        # head_sub_attrs of the form layer_{q,k,v,result} = head_attr
        # mlp_sub_attrs of the form layer_{in,out} = mlp_attr
        head_sub_attrs, mlp_sub_attrs = generate_attr_scores(
            model, 
            difference_cache, 
            clean_grad_cache
        )

        for k, v in head_sub_attrs.items():
            # k is of the form layer_{q,k,v,result} of shape heads
            layer = int(k.split("_")[0])
            hook_type = k.split("_")[1]
            for head in range(model.cfg.n_heads):
                nodes[f"a{layer}.{head}_{hook_type}"] += v[head].item()

        for k, v in mlp_sub_attrs.items():
            # k is of the form layer_{in,out} of shape 1
            layer = int(k.split("_")[0])
            hook_type = k.split("_")[1]
            nodes[f"m{layer}_{hook_type}"] += v.item()

        del difference_cache
        del clean_grad_cache
        gc.collect()
        torch.cuda.empty_cache()
    
    for k in nodes.keys():
        nodes[k] /= (num_prompts // batch_size)

    model.reset_hooks()
    return nodes
# # %%
# metric = ioi_task.get_acdcpp_metric()
# AP(
#     model, 
#     ioi_task.ioi_data.toks[:25], 
#     ioi_task.corr_data.toks[:25], 
#     metric,
# )

# # %%

# %%
