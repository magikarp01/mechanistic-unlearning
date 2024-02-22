#%%
%cd ../../
#%%
from IPython import embed
import torch
from libs.path_patching.path_patching import IterNode, act_patch
import transformer_lens
from transformer_lens import utils
from transformer_lens import HookedTransformer

import functools
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "mps:0"
MAIN = __name__ == "__main__"
#%%
if MAIN:
    from tasks.induction.InductionTask import InductionTask
    from tasks import IOITask
    
    model = HookedTransformer.from_pretrained(
        'gpt2-small',
    )


    ind_task = InductionTask(batch_size=5, tokenizer=model.tokenizer, prep_acdcpp=True, seq_len=15, acdcpp_metric="ave_logit_diff")
    ind_task.set_logit_diffs(model)

    ioi_train = IOITask(batch_size=16, tokenizer=model.tokenizer, device=device, prep_acdcpp=True, nb_templates=4, prompt_type="ABBA")
    ioi_train.set_logit_diffs(model)

# %%
def get_embedding_norm(model, tokens):
    model.reset_hooks()
    act_norms = None
    def embedding_norm_hook(act, hook):
        nonlocal act_norms
        # Embedding act is of shape (batch, seq_len, d_model)
        act_norms = act.norm(dim=-1).mean(dim=1) # of shape (batch)
        return act

    model.run_with_hooks(
        tokens,
        fwd_hooks=[
            ('hook_embed', embedding_norm_hook)
        ]
    )
    model.reset_hooks()
    return act_norms

def patching_hook(act, hook, patch_cache, patch_layer, patch_head=None):
    if hook.layer() == patch_layer:
        if 'z' in hook.name:
            act[:, :, patch_head, :] = patch_cache[hook.name][:, :, patch_head, :]
        elif 'post' in hook.name:
            act[:, :] = patch_cache[hook.name][:, :]
    return act

def causal_tracing_ioi(model, ioi_task):
    # Create corrupt cache
    _, cache = model.run_with_cache(
        ioi_task.clean_data.toks,
        names_filter=lambda name: "hook_z" in name or "hook_q" in name or "post" in name
    )

    # Take clean data cache, and add gaussian noise to subject tokens
    # of shape (batch, indices)
    s_indices = torch.where(
        ioi_task.clean_data.toks == torch.tensor(ioi_task.clean_data.s_tokenIDs)[:, None]
    )[1].reshape(len(ioi_task.clean_data.toks), -1)

    # of shape (batch)
    s_norms = get_embedding_norm(
        model, 
        torch.tensor(ioi_task.clean_data.s_tokenIDs).reshape(len(ioi_task.clean_data.toks), -1).to(device)
    )

    # Add noise to the subject tokens
    for hook_name in cache.keys():
        if "hook_z" in hook_name or "hook_q" in hook_name:
            act = cache[hook_name] # of shape (batch, seq_len, n_head, d_head)
            # Get activations of the subject tokens
            subject_acts = act[torch.arange(act.shape[0])[:, None], s_indices, :, :]
            # Add noise with std 3 * s_norms
            noise = torch.randn_like(subject_acts) * 3 * s_norms[:, None, None, None]
            act[torch.arange(act.shape[0])[:, None], s_indices, :, :] += noise
        elif "post" in hook_name:
            act = cache[hook_name] # of shape (batch, seq_len, d_model)
            # Get activations of the subject tokens
            subject_acts = act[torch.arange(act.shape[0])[:, None], s_indices, :]
            # Add noise with std 3 * s_norms
            noise = torch.randn_like(subject_acts) * 3 * s_norms[:, None, None]
            act[torch.arange(act.shape[0])[:, None], s_indices, :] += noise
    
    logit_diff_metric = ioi_task.get_acdcpp_metric(model)
    results = {}
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            hook_fn = functools.partial(
                patching_hook,
                patch_cache=cache,
                patch_layer=layer,
                patch_head=head
            )
            patched_logits = model.run_with_hooks(
                ioi_task.clean_data.toks,
                fwd_hooks=[
                    (utils.get_act_name('z', layer), hook_fn)
                ]
            )
            results[f'a{layer}.{head}'] = logit_diff_metric(patched_logits).item()
        # Do MLP
        hook_fn = functools.partial(
            patching_hook,
            patch_cache=cache,
            patch_layer=layer,
            patch_head=None
        )
        patched_logits = model.run_with_hooks(
            ioi_task.clean_data.toks,
            fwd_hooks=[
                (utils.get_act_name('post', layer), hook_fn)
            ]
        )
        results[f'm{layer}'] = logit_diff_metric(patched_logits).item()

    return results

def causal_tracing_induction(model, ind_task):
    # Create corrupt cache
    _, cache = model.run_with_cache(
        ind_task.clean_data,
        names_filter=lambda name: "hook_z" in name or "hook_q" in name or "post" in name
    )

    # of shape (batch)
    #
    norms = get_embedding_norm(
        model, 
        ind_task.clean_data[list(range(len(ind_task.clean_data))), -2]
    )

    # Add noise to the subject tokens
    for hook_name in cache.keys():
        if "hook_z" in hook_name or "hook_q" in hook_name:
            act = cache[hook_name] # of shape (batch, seq_len, n_head, d_head)
            ind_act = act[torch.arange(act.shape[0]), -2, :, :] 
            # Add noise with std 3 * s_norms
            noise = torch.randn_like(ind_act).to(device) * 3 * norms
            act[torch.arange(act.shape[0]), -2, :, :] += noise
        elif "post" in hook_name:
            act = cache[hook_name] # of shape (batch, seq_len, d_model)
            ind_act = act[torch.arange(act.shape[0]), -2, :]
            noise = torch.randn_like(ind_act).to(device) * 3 * norms
            act[torch.arange(act.shape[0]), -2, :] += noise
    
    logit_diff_metric = ind_task.get_acdcpp_metric(model)
    results = {}
    for layer in tqdm(list(range(model.cfg.n_layers))):
        for head in tqdm(list(range(model.cfg.n_heads))):
            hook_fn = functools.partial(
                patching_hook,
                patch_cache=cache,
                patch_layer=layer,
                patch_head=head
            )
            patched_logits = model.run_with_hooks(
                ind_task.clean_data,
                fwd_hooks=[
                    (utils.get_act_name('z', layer), hook_fn)
                ]
            )
            results[f'a{layer}.{head}'] = logit_diff_metric(patched_logits).item()
        # Do MLP
        hook_fn = functools.partial(
            patching_hook,
            patch_cache=cache,
            patch_layer=layer,
            patch_head=None
        )
        patched_logits = model.run_with_hooks(
            ind_task.clean_data,
            fwd_hooks=[
                (utils.get_act_name('post', layer), hook_fn)
            ]
        )
        results[f'm{layer}'] = logit_diff_metric(patched_logits).item()

    return results
# %%
induction_ct_results = causal_tracing_induction(model, ind_task)
import pickle
with open("causal_tracing/induction/gpt2_small_attrs.pkl", "wb") as f:
    pickle.dump(induction_ct_results, f)
# %%
ioi_ct_results = causal_tracing_induction(model, ioi_train)
import pickle
with open("causal_tracing/ioi/gpt2_small_attrs.pkl", "wb") as f:
    pickle.dump(ioi_ct_results, f)