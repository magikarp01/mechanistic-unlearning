# %%

import torch
import torch.nn as nn
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
import plotly.express as px
import numpy as np
from typing import Union

from jaxtyping import Float
from functools import partial

import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, FactoredMatrix


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array, substring):
    """
    Find the start and end token indices of a substring in a token array.
    """
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)


def noise_embedding_hook(
    activation: Float[torch.Tensor, "batch pos d_model"],
    hook: HookPoint,
    corruption_start: int,
    corruption_end: int,
    noise_coefficient: float,
    seed: int,
) -> Float[torch.Tensor, "batch pos d_model"]:
    torch.manual_seed(seed)

    activation[
        :, corruption_start:corruption_end, :
    ] += noise_coefficient * torch.randn_like(
        activation[:, corruption_start:corruption_end, :]
    )

    return activation


def patching_hook(
    activation: Float[torch.Tensor, "batch pos d_model"],
    hook: HookPoint,
    position: int,
    clean_cache: dict,  # TODO: only works for one input at a time, need to batchify
    attention_head: int = None,
) -> Float[torch.Tensor, "batch (attn_head) pos d_model"]:
    if attention_head is None:
        activation[:, position, :] = clean_cache[hook.name][position, :]
    else:
        activation[:, position, attention_head, :] = clean_cache[hook.name][
            position, attention_head, :
        ]

    return activation


def debug_hook(
    activation,
    hook,
    **kwargs,
):
    argument_dict = kwargs
    if "mlp" not in argument_dict["hook_name"]:
        breakpoint()
    return activation


def get_corrupted_probs(
    model: HookedTransformer,
    tokens: torch.tensor,
    corruption_start: int,
    corruption_end: int,
    noise_coefficient: Float,
    num_seeds: int = 5,
    hooks: list = None,
):
    fwd_hooks = []
    if hooks is not None:
        fwd_hooks.extend(hooks)
    corrupted_probs = []
    for seed in range(1, num_seeds + 1):
        noise_hook_fn = partial(
            noise_embedding_hook,
            corruption_start=corruption_start,
            corruption_end=corruption_end,
            noise_coefficient=noise_coefficient,
            seed=seed,
        )
        fwd_hooks.insert(
            0, (utils.get_act_name("resid_pre", 0), noise_hook_fn)
        )  # put it at beginning of list
        corrupted_logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
        with torch.no_grad():
            corrupted_prob, _ = torch.max(
                torch.softmax(corrupted_logits[0, -1, :], dim=-1), dim=-1
            )
        corrupted_probs.append(corrupted_prob)
    corrupted_prob = torch.mean(torch.stack(corrupted_probs))

    return corrupted_prob


def get_patching_results(
    model,
    prompt: str,
    subject: str,
    hook_names: list,
    num_seeds=5,
    noise_coefficient=1,
) -> torch.tensor:  # layer, hook_idx, position
    tokens = model.to_tokens(prompt)
    clean_logits, clean_cache = model.run_with_cache(tokens, remove_batch_dim=True)
    corruption_start, corruption_end = find_token_range(
        model.tokenizer,
        tokens[0],
        subject,
    )
    with torch.no_grad():
        base_prob, answer_t = torch.max(
            torch.softmax(clean_logits[0, -1, :], dim=-1), dim=-1
        )

    corrupted_prob = get_corrupted_probs(
        model=model,
        tokens=tokens,
        corruption_start=corruption_start,
        corruption_end=corruption_end,
        noise_coefficient=noise_coefficient,
        num_seeds=num_seeds,
    )
    prob_diff = base_prob - corrupted_prob
    patching_result = torch.zeros(
        model.cfg.n_layers,
        model.cfg.n_heads + 1,
        len(tokens[0]),
        device=model.cfg.device,
    ).cpu()  # layer, hook_idx, position

    # TODO: batchify this
    for layer in range(model.cfg.n_layers):
        for position in range(len(tokens[0])):
            for hook_idx, hook_name in enumerate(hook_names):
                if "mlp" in hook_name:
                    hooks = [
                        (
                            hook_name.format(layer=layer),
                            partial(
                                patching_hook,
                                position=position,
                                clean_cache=clean_cache,
                            ),
                        ),
                    ]
                    patched_prob = get_corrupted_probs(
                        model=model,
                        tokens=tokens,
                        corruption_start=corruption_start,
                        corruption_end=corruption_end,
                        noise_coefficient=noise_coefficient,
                        num_seeds=num_seeds,
                        hooks=hooks,
                    )
                    patched_score = patched_prob / prob_diff
                    layer_node = 0
                    patching_result[layer, layer_node, position] = patched_score.cpu()
                elif "attn" in hook_name:
                    for attention_head in range(model.cfg.n_heads):
                        if attention_head >= 0:
                            break
                        hooks = [
                            (
                                hook_name.format(layer=layer),
                                partial(
                                    patching_hook,
                                    position=position,
                                    clean_cache=clean_cache,
                                    attention_head=attention_head,
                                ),
                            ),
                        ]
                        patched_prob = get_corrupted_probs(
                            model=model,
                            tokens=tokens,
                            corruption_start=corruption_start,
                            corruption_end=corruption_end,
                            noise_coefficient=noise_coefficient,
                            num_seeds=num_seeds,
                            hooks=hooks,
                        )
                        patched_score = patched_prob / prob_diff
                        layer_node = attention_head + 1
                        patching_result[
                            layer, layer_node, position
                        ] = patched_score.cpu()
                else:
                    raise ValueError("Hook name not recognised")

    return patching_result


def get_top_k_nodes(stacked_patching_result: torch.tensor, k: int) -> list:
    """
    Extracts the top k nodes from a tensor representing neural network results.

    Parameters:
    stacked_patching_result (torch.tensor): A tensor of shape (batch, layer, node, position).
    k (int): The number of top nodes to retrieve.

    Returns:
    list: A list of tuples, each containing the (layer, node) indices and the corresponding score.
    """

    # Maximise across the position dimension
    stacked_patching_result = torch.max(stacked_patching_result, dim=-1)[0]

    # Average across the batch dimension, if applicable
    if len(stacked_patching_result.shape) > 2:
        stacked_patching_result = torch.mean(stacked_patching_result, dim=0)

    # Get the original number of layers and nodes per layer
    num_layers, num_nodes = stacked_patching_result.shape[:2]

    # Flatten the layer and node dimensions
    stacked_patching_result = stacked_patching_result.flatten()

    # Get the top k indices and scores
    top_k_scores, top_k_indices = torch.topk(stacked_patching_result, k=k)

    # Convert indices to CPU and layer/node format
    top_k_indices = top_k_indices.cpu().int()
    layer_indices = top_k_indices // num_nodes
    node_indices = top_k_indices % num_nodes

    top_k_scores = top_k_scores.cpu().float()

    # Pair each layer/node index with its corresponding score
    matrix_indices = list(zip(layer_indices.tolist(), node_indices.tolist()))
    return list(zip(matrix_indices, top_k_scores.tolist()))


def convert_indices_to_names(
    top_k: list,
    verbose: bool = False,
):
    node_names = [f"Attention Head {i}" for i in range(12)]
    node_names.insert(0, "MLP")
    top_k_results = []

    for matrix_index, score in top_k:
        layer, node = matrix_index
        node_name = node_names[node]
        if node_name == "MLP":
            top_k_results.append((f"m.{layer}", score))
        else:
            top_k_results.append((f"a.{layer}.{node}", score))
        if verbose:
            print(f"Layer {layer} {node_name}, Score {score}")
        # top_k_results.append(
        #     (
        #         f"Layer {layer} {node_name}",
        #         score,
        #     )
        # )

    return top_k_results

#%%
# %cd ../../
#%%
# from tasks import IOITask

#%%
# device = torch.device("cuda:0" if torch.cuda.is_available() else "mps:0")
# model_name = "gpt2-small"
# model = HookedTransformer.from_pretrained(model_name, device=device)

# ioi_task = IOITask(batch_size=100, tokenizer=model.tokenizer)
# ioi_data = ioi_task.ioi_data

def get_causal_tracing_components(model, ioi_data, n):
    hook_names = ["blocks.{layer}.hook_mlp_out", "blocks.{layer}.attn.hook_z"]
    NUM_SEEDS = 1
    NOISE_COEFFICIENT = 10
    patching_result = torch.zeros(
        model.cfg.n_layers,
        model.cfg.n_heads + 1,
        len(ioi_data.toks[0]),
    )

    for data in tqdm.tqdm(ioi_data.ioi_prompts[:n]):
        patching_result += get_patching_results(
            model=model,
            prompt=data['text'],  # TODO: many prompts
            hook_names=hook_names,  # TODO: make this neater
            subject=data['IO'],
            num_seeds=NUM_SEEDS,
            noise_coefficient=NOISE_COEFFICIENT,
        )

    patching_result /= len(ioi_data)
    top_k = get_top_k_nodes(patching_result, k=145)
    top_k_results = convert_indices_to_names(top_k, verbose=True)

    return top_k_results

# get_important_components(3)
# %%