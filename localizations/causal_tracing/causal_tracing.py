#%%
from IPython import embed
import torch
from libs.path_patching.path_patching import IterNode, act_patch
import transformer_lens
from transformer_lens import utils
from transformer_lens import HookedTransformer
from datasets import load_dataset

import functools
from tqdm.auto import tqdm
device = "cuda" if torch.cuda.is_available() else "mps:0"
MAIN = __name__ == "__main__"

#%%
if MAIN:
    from tasks.induction.InductionTask import InductionTask
    from tasks.ioi.IOITask import IOITask

    model = HookedTransformer.from_pretrained(
        'gpt2-small',
    )

    ind_task = InductionTask(
        batch_size=16, 
        tokenizer=model.tokenizer, 
        prep_acdcpp=True, 
        seq_len=15, 
        acdcpp_metric="ave_logit_diff"
    )
    ind_task.set_logit_diffs(model)

    ioi_task = IOITask(
        batch_size=5, 
        tokenizer=model.tokenizer, 
        device=device, 
        prep_acdcpp=True, 
        acdcpp_N=25, 
        nb_templates=1, 
        prompt_type="ABBA"
    )
    ioi_task.set_logit_diffs(model)


dataset = load_dataset('wikitext', 'wikitext-103-v1')
# %%
def sample_embedding_std(model):
    tokens = model.tokenizer(
        dataset['train']['text'][:1000],
        padding=True
    )
    return model.W_E[torch.tensor(tokens['input_ids'])].std()

def causal_tracing_denoising_hook(act, hook, embedding_std, noise_inds, save_cache, save_layer, save_head=None):
    '''
        Noise the embeddings, denoise the head/mlp
    '''
    if 'embed' in hook.name:
        # Noise embedding
        # for _ in range(10):
        if type(noise_inds) == list:
            # Noise inds of shape (batch, num_noise_inds)
            # Iterate through batch and noise num_noise_inds
            for i in range(act.shape[0]):
                act[i, noise_inds[i], :] = act[i, noise_inds[i], :] + torch.randn_like(act[i, noise_inds[i], :]) * 3 * embedding_std
        else:
            act[:, noise_inds, :] = act[:, noise_inds, :] + torch.randn_like(act[:, noise_inds, :]) * 3 * embedding_std
    elif hook.layer() == save_layer:
        # Save head/mlp
        if len(act.shape) == 4:
            act[:, :, save_head, :] = save_cache[hook.name][:, :, save_head, :]
        else:
            act = save_cache[hook.name]
    return act

### IOI
def ioi_ave_logit_diff(logits, io_toks, s_toks):
    batch = logits.shape[0]
    patch_logit_diff = (logits[list(range(batch)), -2, io_toks] - logits[list(range(batch)), -2, s_toks])
    return patch_logit_diff.mean()

def ioi_causal_tracing_logit_diff_metric(logits, io_toks, s_toks, CLEAN_LOGIT_DIFF, CORR_LOGIT_DIFF):
    patch_logit_diff = ioi_ave_logit_diff(logits, io_toks, s_toks)
    # 0 when patch_logit_diff = CORR_LOGIT_DIFF, 1 when patch_logit_diff = CLEAN_LOGIT_DIFF
    return (patch_logit_diff - CORR_LOGIT_DIFF) / (CLEAN_LOGIT_DIFF - CORR_LOGIT_DIFF)

def causal_tracing_ioi(model, ioi_task, verbose=True):
    # Create corrupt cache
    embedding_std = sample_embedding_std(model)
    io_indices = torch.where(
        ioi_task.clean_data.toks == torch.tensor(ioi_task.clean_data.io_tokenIDs)[:, None]
    )[1].reshape(len(ioi_task.clean_data.toks), -1)

    clean_logits, save_cache = model.run_with_cache(
        ioi_task.clean_data.toks,
        names_filter=lambda name: "hook_z" in name or "hook_q" in name or "post" in name
    )
    # Get corrupt logits by noising embeddings but not saving anything
    corrupt_logits = model.run_with_hooks(
        ioi_task.clean_data.toks,
        fwd_hooks=[
            (
                utils.get_act_name('embed'), 
                functools.partial(
                    causal_tracing_denoising_hook,
                    embedding_std=embedding_std,
                    noise_inds=io_indices,
                    save_cache=None,
                    save_layer=-1
                )
            )
        ]
    )
    if verbose:
        print( 'Clean: \n' + 
            model.tokenizer.decode(
                torch.argmax(
                    torch.nn.functional.softmax(
                        clean_logits[:, -2, :],
                        dim=-1
                    ),
                    dim=-1
                )
            )
        )
        print('Corrupt: \n' + 
            model.tokenizer.decode(
                torch.argmax(
                    torch.nn.functional.softmax(
                        corrupt_logits[:, -2, :],
                        dim=-1
                    ),
                    dim=-1
                )
            )
        )
    CLEAN_LOGIT_DIFF = ioi_ave_logit_diff(
        clean_logits, 
        ioi_task.clean_data.io_tokenIDs, 
        ioi_task.clean_data.s_tokenIDs
    ).item()
    CORR_LOGIT_DIFF = ioi_ave_logit_diff(
        corrupt_logits, 
        ioi_task.clean_data.io_tokenIDs, 
        ioi_task.clean_data.s_tokenIDs
    ).item()
    if verbose:
       print(f'{CLEAN_LOGIT_DIFF=}, {CORR_LOGIT_DIFF=}')
    logit_diff_metric = functools.partial(
        ioi_causal_tracing_logit_diff_metric,
        io_toks = ioi_task.clean_data.io_tokenIDs,
        s_toks = ioi_task.clean_data.s_tokenIDs,
        CLEAN_LOGIT_DIFF = CLEAN_LOGIT_DIFF,
        # corr logit diff is noisy version of clean tokens
        CORR_LOGIT_DIFF = CORR_LOGIT_DIFF,
    )

    results = {}
    for layer in tqdm(list(range(model.cfg.n_layers))):
        for head in tqdm(list(range(model.cfg.n_heads))):
            # print('Patching layer', layer, 'head', head)
            hook_fn = functools.partial(
                causal_tracing_denoising_hook,
                embedding_std=embedding_std,
                noise_inds=io_indices,
                save_cache=save_cache,
                save_layer=layer,
                save_head=head
            )
            model.reset_hooks()
            patched_logits = model.run_with_hooks(
                ioi_task.clean_data.toks,
                fwd_hooks=[
                    (utils.get_act_name('embed'), hook_fn),
                    (utils.get_act_name('z', layer), hook_fn)
                ]
            )
            model.reset_hooks()
            results[f'a{layer}.{head}'] = logit_diff_metric(patched_logits).item()
        # Do MLP
        hook_fn = functools.partial(
            causal_tracing_denoising_hook,
            embedding_std=embedding_std,
            noise_inds=io_indices,
            save_cache=save_cache,
            save_layer=layer,
            save_head=None
        )
        model.reset_hooks()
        patched_logits = model.run_with_hooks(
            ioi_task.clean_data.toks,
            fwd_hooks=[
                (utils.get_act_name('embed'), hook_fn),
                (utils.get_act_name('post', layer), hook_fn)
            ]
        )
        model.reset_hooks()
        results[f'm{layer}'] = logit_diff_metric(patched_logits).item()

    return results

### INDUCTION
def ind_causal_tracing_logit_diff_metric(logits, ind_task, tokens, CLEAN_LOGIT_DIFF, CORR_LOGIT_DIFF):
    patch_logit_diff = ind_task.ave_logit_diff(logits, tokens)
    return (patch_logit_diff - CORR_LOGIT_DIFF) / (CLEAN_LOGIT_DIFF - CORR_LOGIT_DIFF)

def causal_tracing_induction(model, ind_task, verbose=True):
    embedding_std = sample_embedding_std(model)

    # We want to noise pos = -1, -2, ((seq_len - 1) / 2 - 1), and ((seq_len - 1) / 2) - 2)
    noise_inds = torch.tensor([
        # -2, 
        -1, 
        # ind_task.seq_len - 1, 
        ind_task.seq_len
    ])
    clean_logits, save_cache = model.run_with_cache(
        ind_task.clean_data,
        names_filter=lambda name: "hook_z" in name or "hook_q" in name or "post" in name
    )

    # Get corrupt logits by noising embeddings but not saving anything
    corrupt_logits = model.run_with_hooks(
        ind_task.clean_data,
        fwd_hooks=[
            (
                utils.get_act_name('embed'), 
                functools.partial(
                    causal_tracing_denoising_hook,
                    embedding_std=embedding_std,
                    noise_inds=noise_inds,
                    save_cache=None,
                    save_layer=-1
                )
            )
        ]
    )
    if verbose:
        print( 'Clean: \n' + 
            model.tokenizer.decode(
                torch.argmax(
                    torch.nn.functional.softmax(
                        clean_logits[:, -2, :],
                        dim=-1
                    ),
                    dim=-1
                )
            )
        )
        print('Corrupt: \n' + 
            model.tokenizer.decode(
                torch.argmax(
                    torch.nn.functional.softmax(
                        corrupt_logits[:, -2, :],
                        dim=-1
                    ),
                    dim=-1
                )
            )
        )
    CLEAN_LOGIT_DIFF = ind_task.ave_logit_diff(clean_logits, ind_task.clean_data).item()
    CORR_LOGIT_DIFF = ind_task.ave_logit_diff(corrupt_logits, ind_task.clean_data).item()
    if verbose:
       print(f'{CLEAN_LOGIT_DIFF=}, {CORR_LOGIT_DIFF=}')

    logit_diff_metric = functools.partial(
        ind_causal_tracing_logit_diff_metric,
        ind_task=ind_task,
        tokens=ind_task.clean_data,
        CLEAN_LOGIT_DIFF = CLEAN_LOGIT_DIFF,
        # corr logit diff is noisy version of clean tokens
        CORR_LOGIT_DIFF = CORR_LOGIT_DIFF,
    )
    results = {}
    for layer in tqdm(list(range(model.cfg.n_layers))):
        for head in tqdm(list(range(model.cfg.n_heads))):
            hook_fn = functools.partial(
                causal_tracing_denoising_hook,
                embedding_std=embedding_std,
                noise_inds=noise_inds,
                save_cache=save_cache,
                save_layer=layer,
                save_head=head
            )
            model.reset_hooks()
            patched_logits = model.run_with_hooks(
                ind_task.clean_data,
                fwd_hooks=[
                    (utils.get_act_name('embed'), hook_fn),
                    (utils.get_act_name('z', layer), hook_fn)
                ]
            )
            results[f'a{layer}.{head}'] = logit_diff_metric(patched_logits).item()
        # Do MLP
        hook_fn = functools.partial(
            causal_tracing_denoising_hook,
            embedding_std=embedding_std,
            noise_inds=noise_inds,
            save_cache=save_cache,
            save_layer=layer,
            save_head=None
        )
        model.reset_hooks()
        patched_logits = model.run_with_hooks(
            ind_task.clean_data,
            fwd_hooks=[
                (utils.get_act_name('embed'), hook_fn),
                (utils.get_act_name('post', layer), hook_fn)
            ]
        )
        results[f'm{layer}'] = logit_diff_metric(patched_logits).item()

    return results

### SPORTS FACTS
from collections import defaultdict

def sports_causal_tracing_logit_diff_metric(
    logits, 
    sports_task, 
    CLEAN_LOGIT_DIFF, 
    CORR_LOGIT_DIFF,
    correct_ans=None,
    wrong_ans=None
):
    patch_logit_diff = sports_task.ave_logit_diff(logits, correct_ans, wrong_ans)
    return (patch_logit_diff - CORR_LOGIT_DIFF) / (CLEAN_LOGIT_DIFF - CORR_LOGIT_DIFF)

def causal_tracing_sports(
    model, 
    sports_task, 
    batch_size=5, 
    verbose=True,
    gemma2=False
):
    embedding_std = sample_embedding_std(model)

    results = defaultdict(lambda: 0)
    toks = sports_task.clean_data.toks
    deltas = sports_task.clean_data.deltas
    correct_ans_toks = sports_task.clean_answer_toks
    wrong_ans_toks = sports_task.clean_wrong_toks

    model.reset_hooks()
    for i in tqdm(range(0, len(toks), batch_size), position=0, leave=True):
        print("Looping, ", i, " out of ", len(toks), " batch size ", batch_size)
        model.reset_hooks()
        toks_slice = toks[i:i+batch_size]
        deltas_slice = deltas[i:i+batch_size]
        correct_ans_slice = correct_ans_toks[i:i+batch_size]
        wrong_ans_slice = wrong_ans_toks[i:i+batch_size]

        noise_inds = [
            torch.tensor(
                list(range(s['{player}'].start, s['{player}'].stop))
            )
            for s in deltas_slice
        ]
        print(toks_slice.shape)
        print(noise_inds)
        print(f"Noising{model.tokenizer.decode(toks_slice[0, noise_inds[0]])}")

        wanted_hooks = ['hook_result', 'mlp_in', 'mlp_out', 'hook_q', 'hook_k', 'hook_v']
        names_filter = lambda name: any([hook in name for hook in wanted_hooks]) and not 'input' in name
        model.reset_hooks()
        clean_logits, save_cache = model.run_with_cache(
            toks_slice,
            names_filter=names_filter
        )
        model.reset_hooks()
        # Get corrupt logits by noising embeddings but not saving anything
        corrupt_logits = model.run_with_hooks(
            toks_slice,
            fwd_hooks=[
                (
                    utils.get_act_name('embed'), 
                    functools.partial(
                        causal_tracing_denoising_hook,
                        embedding_std=embedding_std,
                        noise_inds=noise_inds,
                        save_cache=None,
                        save_layer=-1
                    )
                )
            ]
        )
        model.reset_hooks()
        if verbose:
            print('Clean: \n' + model.tokenizer.decode(
                    torch.argmax(torch.nn.functional.softmax(clean_logits[:, -1, :], dim=-1), dim=-1)
                )
            )
            print('Corrupt: \n' + 
                model.tokenizer.decode( 
                    torch.argmax(torch.nn.functional.softmax(corrupt_logits[:, -1, :], dim=-1), dim=-1)
                )
            )

        CLEAN_LOGIT_DIFF = sports_task.ave_logit_diff(clean_logits, correct_ans_slice, wrong_ans_slice).item()
        CORR_LOGIT_DIFF = sports_task.ave_logit_diff(corrupt_logits, correct_ans_slice, wrong_ans_slice).item()
        if verbose:
            print(f'{CLEAN_LOGIT_DIFF=}, {CORR_LOGIT_DIFF=}')

        logit_diff_metric = functools.partial(
            sports_causal_tracing_logit_diff_metric,
            sports_task=sports_task,
            CLEAN_LOGIT_DIFF = CLEAN_LOGIT_DIFF,
            # corr logit diff is noisy version of clean tokens
            CORR_LOGIT_DIFF = CORR_LOGIT_DIFF,
            correct_ans=correct_ans_slice,
            wrong_ans=wrong_ans_slice
        )

        for layer in tqdm(list(range(model.cfg.n_layers)), position=0, leave=True):
            for head in tqdm(list(range(model.cfg.n_heads)), position=0, leave=True):
                hook_fn = functools.partial(
                    causal_tracing_denoising_hook,
                    embedding_std=embedding_std,
                    noise_inds=noise_inds,
                    save_cache=save_cache,
                    save_layer=layer,
                    save_head=head
                )
                for hook_name in ['result', 'q', 'k', 'v']:
                    model.reset_hooks()
                    patched_logits = model.run_with_hooks(
                        toks_slice,
                        fwd_hooks=[
                            (utils.get_act_name('embed'), hook_fn),
                            (utils.get_act_name(hook_name, layer), hook_fn)
                        ]
                    )
                    results[f'a{layer}.{head}_{hook_name}'] += logit_diff_metric(patched_logits).item() / len(toks_slice)
            # Do MLP
            hook_fn = functools.partial(
                causal_tracing_denoising_hook,
                embedding_std=embedding_std,
                noise_inds=noise_inds,
                save_cache=save_cache,
                save_layer=layer,
                save_head=None
            )
            model.reset_hooks()
            patched_logits = model.run_with_hooks(
                toks_slice,
                fwd_hooks=[
                    (utils.get_act_name('embed'), hook_fn),
                    (utils.get_act_name('mlp_out', layer), hook_fn)
                ]
            )
            results[f'm{layer}_out'] += logit_diff_metric(patched_logits).item() / len(toks_slice)

            model.reset_hooks()
            patched_logits = model.run_with_hooks(
                toks_slice,
                fwd_hooks=[
                    (utils.get_act_name('embed'), hook_fn),
                    (utils.get_act_name('mlp_in', layer), hook_fn)
                ]
            )
            results[f'm{layer}_in'] += logit_diff_metric(patched_logits).item() / len(toks_slice)

    return results
