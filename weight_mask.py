# We want to mask the weights of a transformer
# First, we run a forward pass on all weights
# Our loss function then a regularization term, a forget term, and a maintain term
# Then, we run a backward pass on all weights
# Then, we zero the gradients of the components we don't want to update
# Then, we step the optimizer

# #%%
# %cd ~/mechanistic-unlearning
# %load_ext autoreload
# %autoreload 2
from transformer_lens import HookedTransformer, ActivationCache
import random
import os
import torch
from torch import nn
import numpy as np
import pandas as pd
import datasets
import transformers
import pickle
import gc
import wandb
import json
from collections import defaultdict

from tasks import PileTask, OWTTask, InductionTask, GreaterThanTask
from tasks.ioi.IOITask import IOITask, IOITask_NPO, IOITask_Uniform
from tasks.induction.InductionTask import InductionTask, InductionTask_NPO, InductionTask_Uniform
from tasks.facts.SportsTask import SportsTask, SportsTask_NPO, SportsTask_Uniform

from tqdm.auto import tqdm

from transformers import GPT2Tokenizer, GPTNeoXTokenizerFast, AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset

from tasks import PileTask, OWTTask, InductionTask, GreaterThanTask
from tasks.ioi.IOITask import IOITask, IOITask_NPO, IOITask_Uniform
from tasks.induction.InductionTask import InductionTask, InductionTask_NPO, InductionTask_Uniform
from tasks.facts.SportsTask import SportsTask, SportsTask_NPO, SportsTask_Uniform
from tasks.facts.SportsTaskAdversarial import adversarial_sports_eval
from tasks.facts.SportsTaskSideEffects import run_side_effects_evals
from weight_masked_transformer import WeightMaskedTransformer
### Hyperparams
def load_global_hyperparams(config_file):
    global train_batch_size, eval_batch_size, device, train_loss_type, forget_sport, maintain_sport, \
           model_name, model_type, learning_rate, n_epochs, grad_accum_steps, alpha, beta, clip_grad, \
           evaluate_every, n_eval_iters, do_adversarial_evals, do_side_effects_evals, \
           localization_type, localization_top_p, n_devices
    
    with open(config_file, "r") as f:
        config = json.load(f)
    
    train_batch_size = int(config['train_batch_size'])
    eval_batch_size = int(config['eval_batch_size'])
    device = config['device']
    train_loss_type = config['train_loss_type']
    forget_sport = config['forget_sport']

    maintain_sport = config['maintain_sport']
    model_name = config['model_name']
    model_type = config['model_type']
    learning_rate = float(config['learning_rate'])
    n_epochs = int(config['n_epochs'])
    grad_accum_steps = int(config['grad_accum_steps'])
    alpha = float(config['alpha'])
    beta = float(config['beta'])
    clip_grad = None if config['clip_grad'] == "null" else int(config['clip_grad'])
    evaluate_every = int(config['evaluate_every'])
    n_eval_iters = int(config['n_eval_iters'])
    do_adversarial_evals = config['do_adversarial_evals']
    do_side_effects_evals = config['do_side_effects_evals']
    localization_type = config['localization_type']
    localization_top_p = float(config['localization_top_p'])
    n_devices = int(config['n_devices'])
    
### Localization helper funcs
def create_random_weight_mask_dicts(model, top_p):
    # Creates random weight masks for testing
    weight_mask_attn_dict = {}
    weight_mask_mlp_dict = {}

    for layer in range(model.cfg.n_layers):
        weight_mask_attn_dict[layer] = {}
        weight_mask_mlp_dict[layer] = {}
        # Want bool of length n_head, randomly set to True
        weight_mask_attn_dict[layer]['W_Q'] = torch.rand(model.cfg.n_heads) > top_p
        weight_mask_attn_dict[layer]['W_K'] = torch.rand(model.cfg.n_heads) > top_p
        weight_mask_attn_dict[layer]['W_V'] = torch.rand(model.cfg.n_heads) > top_p
        weight_mask_attn_dict[layer]['W_O'] = torch.rand(model.cfg.n_heads) > top_p

        # Randomly set to true or false
        weight_mask_mlp_dict[layer]['W_in'] = random.random() > top_p
        weight_mask_mlp_dict[layer]['W_out'] = random.random() > top_p

    return weight_mask_attn_dict, weight_mask_mlp_dict

def create_mlp_only_mask_dicts(model):
    weight_mask_attn_dict = {}
    weight_mask_mlp_dict = {}

    for layer in range(model.cfg.n_layers):
        weight_mask_attn_dict[layer] = {}
        weight_mask_mlp_dict[layer] = {}

        # Set to false: we train a mask over these
        weight_mask_mlp_dict[layer]['W_in'] = not (layer >= 1 and layer <= 7)
        weight_mask_mlp_dict[layer]['W_out'] = not (layer >= 1 and layer <= 7)

    return weight_mask_attn_dict, weight_mask_mlp_dict

def get_mask_from_ap_graph(model, ap_graph, top_p):
    # Attention masks are of form:
    # {layer: {"W_Q": frozen_heads, "W_K": frozen_heads, "W_V": frozen_heads, "W_O": frozen_heads}}
    # TRUE for the heads we want to FREEZE, FALSE for heads we want to MASK over
    # MLP masks are of form:
    # {layer: bool}

    # Localizations are of form:
    # {alayer.head_{q,k,v,result}:int, mlayer_{in,out}: int}

    # Get threshold such that top_p% of weights are ABOVE threshold: We are masking over the top_p% of weights
    top_p *= 100
    all_weights = []
    for key, value in ap_graph.items():
        all_weights.append(value)

    all_weights = np.array(all_weights)
    threshold = np.percentile(all_weights, 100 - top_p)

    weight_mask_attn_dict = {}
    weight_mask_mlp_dict = {}

    for layer in range(model.cfg.n_layers):
        weight_mask_attn_dict[layer] = {}
        weight_mask_mlp_dict[layer] = {}

        if 'a0.0_q' in ap_graph:
            weight_mask_attn_dict[layer]['W_Q'] = torch.tensor(
                [
                    abs(ap_graph[f"a{layer}.{head}_q"]) < threshold 
                    for head in range(model.cfg.n_heads)
                ]
            )
        else:
            weight_mask_attn_dict[layer]['W_Q'] = None

        if 'a0.0_k' in ap_graph:
            weight_mask_attn_dict[layer]['W_K'] = torch.tensor(
                [
                    abs(ap_graph[f"a{layer}.{head}_k"]) < threshold 
                    for head in range(model.cfg.n_heads)
                ]
            )
        else:
            weight_mask_attn_dict[layer]['W_K'] = None
        
        if 'a0.0_v' in ap_graph:
            weight_mask_attn_dict[layer]['W_V'] = torch.tensor(
                [
                    abs(ap_graph[f"a{layer}.{head}_v"]) < threshold 
                    for head in range(model.cfg.n_heads)
                ]
            )
        else:
            weight_mask_attn_dict[layer]['W_V'] = None
        
        if 'a0.0_result' in ap_graph:
            weight_mask_attn_dict[layer]['W_O'] = torch.tensor(
                [
                    abs(ap_graph[f"a{layer}.{head}_result"]) < threshold 
                    for head in range(model.cfg.n_heads)
                ]
            )
        else:
            weight_mask_attn_dict[layer]['W_O'] = None
            
        if 'm0_in' in ap_graph:
            weight_mask_mlp_dict[layer]['W_in'] = abs(ap_graph[f"m{layer}_in"]) < threshold
        else:
            weight_mask_mlp_dict[layer]['W_in'] = None
        
        if 'm0_out' in ap_graph:
            weight_mask_mlp_dict[layer]['W_out'] = abs(ap_graph[f"m{layer}_out"]) < threshold
        else:
            weight_mask_mlp_dict[layer]['W_out'] = None

    return weight_mask_attn_dict, weight_mask_mlp_dict

def get_mask_from_ct_graph(model, ct_graph, top_p):
    # Attention masks are of form:
    # {layer: {"W_Q": frozen_heads, "W_K": frozen_heads, "W_V": frozen_heads, "W_O": frozen_heads}}
    # TRUE for the heads we want to FREEZE, FALSE for heads we want to MASK over
    # MLP masks are of form:
    # {layer: bool}

    # Localizations are of form:
    # {alayer.head:int, mlayer: int}

    top_p *= 100
    all_weights = []
    for key, value in ct_graph.items():
        all_weights.append(value)

    all_weights = np.array(all_weights)
    threshold = np.percentile(all_weights, 100 - top_p)

    weight_mask_attn_dict = {}
    weight_mask_mlp_dict = {}

    for layer in range(model.cfg.n_layers):
        weight_mask_attn_dict[layer] = {}
        weight_mask_mlp_dict[layer] = {}

        weight_mask_attn_dict[layer]['W_O'] = torch.tensor(
            [
                abs(ct_graph[f"a{layer}.{head}"]) < threshold 
                for head in range(model.cfg.n_heads)
            ]
        )

        weight_mask_mlp_dict[layer]['W_out'] = torch.tensor(
            [
                abs(ct_graph[f"m{layer}"]) < threshold
            ]
        )
    
    return weight_mask_attn_dict, weight_mask_mlp_dict

### Mask helper funcs
def get_unfrozen_weights(model, weight_mask_attn_dict, weight_mask_mlp_dict):
    # Get a copy of the weights that are not frozen
    unfrozen_weights = {}
    for layer, block in enumerate(model.blocks):
        unfrozen_weights[layer] = {}
        for component in ["W_Q", "_W_K", "_W_V", "W_O"]:
            if component in weight_mask_attn_dict[layer]:
                frozen_heads = weight_mask_attn_dict[layer][component]
                if all(frozen_heads):
                    getattr(block.attn, component).requires_grad = False
                else:
                    unfrozen_weights[layer][component] = getattr(block.attn, component)[~frozen_heads].clone()
        
        for component in ["W_in", "W_out"]:
            if component in weight_mask_mlp_dict[layer]:
                if not weight_mask_mlp_dict[layer][component]:
                    unfrozen_weights[layer][component] = getattr(block.mlp, component).clone()
                else:
                    getattr(block.mlp, component).requires_grad = False
    return unfrozen_weights

def zero_grad(model, weight_mask_attn_dict, weight_mask_mlp_dict):
    for layer, block in enumerate(model.blocks):
        # If head is frozen or doesnt exist, set grad to 0
        for component in ["W_Q", "_W_K", "_W_V", "W_O"]:
            if component in weight_mask_attn_dict[layer]:
                frozen_heads = weight_mask_attn_dict[layer][component]
                param = getattr(block.attn, component)
                if param.grad is not None:
                    param.grad[frozen_heads] = torch.zeros_like(param.grad[frozen_heads])
                else:
                    pass
                    # print(f"None grad for {component} in layer {layer}")
            else:
                # Set grad of all heads to 0
                param = getattr(block.attn, component)
                if param.grad is not None:
                    param.grad = torch.zeros_like(param.grad).to("cuda")
                else:
                    pass
                    # print(f"None grad for {component} in layer {layer}")

        # If mlp is frozen or doesnt exist, set grad to 0
        for component in ["W_in", "W_out"]:
            if component in weight_mask_mlp_dict[layer]:
                if weight_mask_mlp_dict[layer][component]:
                    param = getattr(block.mlp, component)
                    if param.grad is not None:
                        param.grad = torch.zeros_like(param.grad).to("cuda")
                    else:
                        pass
                        # print(f"None grad for {component} in layer {layer}")
            else:
                param = getattr(block.mlp, component)
                if param.grad is not None:
                    param.grad = torch.zeros_like(param.grad).to("cuda")
                else:
                    pass
                    # print(f"None grad for {component} in layer {layer}")

def regularization_loss(model, weight_mask_attn_dict, weight_mask_mlp_dict):
    # L1 sparsity, but only for components that are not frozen
    loss = 0
    for layer, block in enumerate(model.blocks):
        for component in ["W_Q", "_W_K", "_W_V", "W_O"]:
            # If the component exists, we take the non frozen heads
            if component in weight_mask_attn_dict[layer]:
                frozen_heads = weight_mask_attn_dict[layer][component]
                loss += torch.sum(torch.abs(getattr(block.attn, component)[~frozen_heads]))
        
        for component in ["W_in", "W_out"]:
            if component in weight_mask_mlp_dict[layer]:
                # If the mlp is not frozen, then we add the L1 loss
                if not weight_mask_mlp_dict[layer][component]:
                    loss += torch.sum(torch.abs(getattr(block.mlp, component)))
    return loss

def clamp_unfrozen_weights(model, original_weights, weight_mask_attn_dict, weight_mask_mlp_dict):
    # Clamp all the non-frozen weights to be between 0 and original_weights
    for layer, block in enumerate(model.blocks):
        for component in ["W_Q", "_W_K", "_W_V", "W_O"]:
            if component in weight_mask_attn_dict[layer]:
                frozen_heads = weight_mask_attn_dict[layer][component]
                if all(frozen_heads):
                    continue
                param = getattr(block.attn, component)
                orig = original_weights[layer][component]
                param.data[~frozen_heads] = torch.where(
                    orig > 0,
                    torch.clamp(
                        param[~frozen_heads],
                        torch.zeros_like(param[~frozen_heads]),
                        orig
                    ),
                    torch.clamp(
                        param[~frozen_heads],
                        orig,
                        torch.zeros_like(param[~frozen_heads])
                    )
                )
        
        for component in ["W_in", "W_out"]:
            if component in weight_mask_mlp_dict[layer]:
                if not weight_mask_mlp_dict[layer][component]:
                    param = getattr(block.mlp, component)
                    orig = original_weights[layer][component]
                    param.data = torch.where(
                        orig > 0,
                        torch.clamp(
                            param, 
                            torch.zeros_like(param),
                            orig
                        ),
                        torch.clamp(
                            param, 
                            orig, 
                            torch.zeros_like(param)
                        )
                    )
        
def get_mask(model, original_weights, weight_mask_attn_dict, weight_mask_mlp_dict):
    # Get masks, where a mask is model weight / original weight
    mask = {}
    for layer, block in enumerate(model.blocks):
        mask[layer] = {}
        for component in ["W_Q", "_W_K", "_W_V", "W_O"]:
            if component in weight_mask_attn_dict[layer]:
                frozen_heads = weight_mask_attn_dict[layer][component]
                if all(frozen_heads):
                    continue
                param = getattr(block.attn, component)

                total_original = torch.ones_like(param).to(device)
                total_original[~frozen_heads] = original_weights[layer][component]
                mask[layer][component] = param / total_original
                mask[layer][component][frozen_heads] = 1
        
        for component in ["W_in", "W_out"]:
            if component in weight_mask_mlp_dict[layer]:
                if not weight_mask_mlp_dict[layer][component]:
                    mask[layer][component] = getattr(block.mlp, component) / original_weights[layer][component]
    return mask

def run():
    os.environ['HF_TOKEN'] = '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'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = HookedTransformer.from_pretrained(
        model_name,
        tokenizer=tokenizer,
        device='cuda',
        default_padding_side="right",
        fold_ln=False,
        fold_value_biases=False,
        center_writing_weights=False,
        dtype=torch.bfloat16,
        # n_devices=2
    )
    model.W_E.requires_grad = False
    model.W_U.requires_grad = False
    for block in model.blocks:
        block.ln1.w.requires_grad = False
        block.ln2.w.requires_grad = False
    model.ln_final.w.requires_grad = False

    ### DATASETS
    # train_dataset = load_dataset('monology/pile-uncopyrighted', split='train', streaming=True)
    if forget_sport == "athlete":
        sports_1mp = SportsTask(
            batch_size=train_batch_size, 
            tokenizer=tokenizer, 
            device=device, 
            prep_acdcpp=False, 
            criterion="log_1_minus_p", 
            forget_player_subset={
                "Lance Stephenson",
                "Nnamdi Asomugha",
                "Philip Rivers",
                "Andre Iguodala",
                "Tyus Jones",
                "Jonathan Papelbon",
                "Brandon Roy",
                "Nick Markakis",
                "Deion Jones",
                "Thon Maker",
                "Tony Gonzalez",
                "David Robertson",
                "Tom Izzo",
                "Billy Hamilton",
                "Brendan Haywood",
                "Brett Hundley"
            },
            is_forget_dataset=True,
            train_test_split=False
        )
    else:
        sports_1mp = SportsTask(batch_size=train_batch_size, tokenizer=tokenizer, device=device, prep_acdcpp=False, criterion="log_1_minus_p", forget_sport_subset={forget_sport}, is_forget_dataset=True)

    if maintain_sport is None or maintain_sport == "null":
        if forget_sport == "athlete":
            maintain_sports = SportsTask(
                batch_size=train_batch_size, 
                tokenizer=tokenizer, 
                device=device, 
                prep_acdcpp=False, 
                criterion="cross_entropy", 
                forget_player_subset={
                    "Lance Stephenson",
                    "Nnamdi Asomugha",
                    "Philip Rivers",
                    "Andre Iguodala",
                    "Tyus Jones",
                    "Jonathan Papelbon",
                    "Brandon Roy",
                    "Nick Markakis",
                    "Deion Jones",
                    "Thon Maker",
                    "Tony Gonzalez",
                    "David Robertson",
                    "Tom Izzo",
                    "Billy Hamilton",
                    "Brendan Haywood",
                    "Brett Hundley"
                },
                is_forget_dataset=False,
                train_test_split=True
            )
        else:
            maintain_sports = SportsTask(batch_size=train_batch_size, tokenizer=tokenizer, device=device, prep_acdcpp=False, criterion="cross_entropy", forget_sport_subset={forget_sport}, is_forget_dataset=False)
    else:
        maintain_sports = SportsTask(batch_size=train_batch_size, tokenizer=tokenizer, device=device, prep_acdcpp=False, criterion="cross_entropy", forget_sport_subset={maintain_sport}, is_forget_dataset=True)

    train_pile = PileTask(batch_size=train_batch_size, tokenizer=tokenizer, device=device, ctx_length=100, shuffle=True, buffer_size=1000)
    train_tasks = {"sports_1mp": (sports_1mp, .3), "maintain_sports": (maintain_sports, 1), "pile": (train_pile, 1)}

    # want to eval on other sports

    if forget_sport == "athlete":
        forget_sport_eval = SportsTask(
            batch_size=train_batch_size, 
            tokenizer=tokenizer, 
            device=device, 
            prep_acdcpp=False, 
            criterion="cross_entropy", 
            forget_player_subset={
                "Lance Stephenson",
                "Nnamdi Asomugha",
                "Philip Rivers",
                "Andre Iguodala",
                "Tyus Jones",
                "Jonathan Papelbon",
                "Brandon Roy",
                "Nick Markakis",
                "Deion Jones",
                "Thon Maker",
                "Tony Gonzalez",
                "David Robertson",
                "Tom Izzo",
                "Billy Hamilton",
                "Brendan Haywood",
                "Brett Hundley"
            },
            is_forget_dataset=True,
            train_test_split=False
        )
    else:
        forget_sport_eval = SportsTask(batch_size=train_batch_size, tokenizer=tokenizer, device=device, prep_acdcpp=False, criterion="cross_entropy", forget_sport_subset={forget_sport}, is_forget_dataset=True)
    test_pile = PileTask(batch_size=eval_batch_size, tokenizer=tokenizer, device=device, ctx_length=100, shuffle=True, buffer_size=1000)

    induction_eval = InductionTask(batch_size=eval_batch_size, tokenizer=tokenizer, prep_acdcpp=False, seq_len=15, device=device)
    if maintain_sport is None or maintain_sport == "null":
        if forget_sport == "athlete":
            maintain_sports_eval = SportsTask(
                batch_size=train_batch_size, 
                tokenizer=tokenizer, 
                device=device, 
                prep_acdcpp=False, 
                criterion="cross_entropy", 
                forget_player_subset={
                    "Lance Stephenson",
                    "Nnamdi Asomugha",
                    "Philip Rivers",
                    "Andre Iguodala",
                    "Tyus Jones",
                    "Jonathan Papelbon",
                    "Brandon Roy",
                    "Nick Markakis",
                    "Deion Jones",
                    "Thon Maker",
                    "Tony Gonzalez",
                    "David Robertson",
                    "Tom Izzo",
                    "Billy Hamilton",
                    "Brendan Haywood",
                    "Brett Hundley"
                },
                is_forget_dataset=False,
                train_test_split=True
            )
        else:
            maintain_sports_eval = SportsTask(batch_size=train_batch_size, tokenizer=tokenizer, device=device, prep_acdcpp=False, criterion="cross_entropy", forget_sport_subset={forget_sport}, is_forget_dataset=False)
        eval_tasks = {"induction": induction_eval, "pile": test_pile, "forget_sport": forget_sport_eval, "maintain_sport": maintain_sports_eval}
    else:
        raise NotImplemented

    ### LOCALIZATIONS

    if localization_type == "ap":
        with open(f"models/{model_name.replace('/', '_')}_sports_{forget_sport}_{localization_type}_graph.pkl", "rb") as f:
            localization_graph = pickle.load(f)
        weight_mask_attn_dict, weight_mask_mlp_dict = get_mask_from_ap_graph(model, localization_graph, localization_top_p)
    elif localization_type == "ct":
        with open(f"models/{model_name.replace('/', '_')}_sports_{forget_sport}_{localization_type}_graph.pkl", "rb") as f:
            localization_graph = pickle.load(f)
        weight_mask_attn_dict, weight_mask_mlp_dict = get_mask_from_ap_graph(model, localization_graph, localization_top_p)
    elif localization_type == "random":
        weight_mask_attn_dict, weight_mask_mlp_dict = create_random_weight_mask_dicts(model, localization_top_p)
    elif localization_type == "none":
        weight_mask_attn_dict, weight_mask_mlp_dict = create_random_weight_mask_dicts(model, 1)
    elif localization_type == "manual":
        # Manual interp means only training the MLP weights from layer 1 to 7
        weight_mask_attn_dict, weight_mask_mlp_dict = create_mlp_only_mask_dicts(model)

    for layer in weight_mask_attn_dict.keys():
        if 'W_K' in weight_mask_attn_dict[layer]:
            weight_mask_attn_dict[layer]['_W_K'] = weight_mask_attn_dict[layer]['W_K']
        if 'W_V' in weight_mask_attn_dict[layer]:
            weight_mask_attn_dict[layer]['_W_V'] = weight_mask_attn_dict[layer]['W_V']
    gc.collect()
    torch.cuda.empty_cache()

    ### TRAIN 

    all_train_losses = defaultdict(list)
    all_test_losses = defaultdict(list)
    adversarial_evals = []
    side_effect_evals = []

    # Initialize optimizer, want to optimize over all weights
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_epochs)
    original_weights = get_unfrozen_weights(model, weight_mask_attn_dict, weight_mask_mlp_dict)


    wandb.login(key="6f39dedff978870c25e55aed36e504403271d404")
    ### LOGGING
    wandb.init(
        # set the wandb project where this run will be logged
        project="mech-unlearning",
        name=f"{model_name.split('/')[-1]}-{forget_sport}-{localization_type}",

        # track hyperparameters and run metadata
        config={
            "model_type": model_type,
            "model_name": model_name,
            "forget_sport": forget_sport,
            "learning_rate": learning_rate,
            "n_epochs": n_epochs,
            "grad_accum_steps": grad_accum_steps,
            "alpha": alpha,
            "beta": beta,
            "clip_grad": clip_grad,
            "evaluate_every": evaluate_every,
            "n_eval_iters": n_eval_iters,
            "do_adversarial_evals": do_adversarial_evals,
            "do_side_effects_evals": do_side_effects_evals,
            "train_task_weights": {k:v[1] for k, v in train_tasks.items()},
            "localization_type": localization_type,
            "localization_top_p": localization_top_p
        }
    )
    
    # Train a sparse mask
    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        # Reset grad
        optimizer.zero_grad()
        gc.collect()
        torch.cuda.empty_cache()

        with torch.autocast(device_type="cuda"):
            # Compute normal loss over retain
            for task_name, (task, task_weight) in train_tasks.items():
                print(f"Running {task_name}")
                task_loss = 0
                for i in range(grad_accum_steps):
                    loss = task.get_train_loss(model) / grad_accum_steps
                    task_loss += loss.item()
                    loss *= task_weight
                    # print(task_name, i, loss)
                    loss.backward()
                    del loss
                    gc.collect()
                    torch.cuda.empty_cache()
                all_train_losses[task_name].append(task_loss)

                gc.collect()
                torch.cuda.empty_cache()
                
            gc.collect()
            torch.cuda.empty_cache()
            # Add sparsity loss and backprop
            # Linearly increase from negative to positive, with 0 at 10
            loss = beta * regularization_loss(model, weight_mask_attn_dict, weight_mask_mlp_dict)
            loss.backward()
            print(f"reg loss, {loss.item()}")
            all_train_losses["reg"].append(loss.item())
            del loss
            gc.collect()
            torch.cuda.empty_cache()
            # Step and log
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            # print(model.blocks[0].attn.W_Q.grad[2])
            # print(model.blocks[0].attn.W_Q.grad[1])
            # Remove gradients from frozen components
            zero_grad(model, weight_mask_attn_dict, weight_mask_mlp_dict)
            # print(model.blocks[0].attn.W_Q[2])
            optimizer.step()
            scheduler.step()
            clamp_unfrozen_weights(model, original_weights, weight_mask_attn_dict, weight_mask_mlp_dict)
            gc.collect()
            torch.cuda.empty_cache()

            if epoch % evaluate_every == 0 or epoch == n_epochs - 1:
                for task_name, task in eval_tasks.items():
                    task_loss = 0
                    for i in range(n_eval_iters):
                        task_loss += task.get_test_loss(model).item()
                    all_test_losses[task_name].append(task_loss / n_eval_iters)
                if do_adversarial_evals:
                    print("Running adversarial evals")
                    adversarial_evals.append(adversarial_sports_eval(model, model_type=model_type, batch_size=eval_batch_size, use_system_prompt=True, include_evals=["MC"]))
                if do_side_effects_evals:
                    print("Running side effects evals")
                    side_effect_evals.append(run_side_effects_evals(model, model_type=model_type, batch_size=eval_batch_size, evals_to_run=["Sports Answers"]))
            gc.collect()
            torch.cuda.empty_cache()
            
            log_dict = {}
            for k, v in all_train_losses.items():
                log_dict[f"train_loss_{k}"] = v[-1]
            for k, v in all_test_losses.items():
                log_dict[f"test_loss_{k}"] = v[-1]
            if adversarial_evals:
                for k, v in adversarial_evals[-1].items():
                    log_dict[f"adversarial_{k}"] = v
            if side_effect_evals:
                for k, v in side_effect_evals[-1].items():
                    log_dict[f"side_effects_{k}"] = v
            wandb.log(log_dict)

    # Get masks
    wandb.finish()

    ### SAVE
    mask = get_mask(model, original_weights, weight_mask_attn_dict, weight_mask_mlp_dict) 
    torch.save(mask, f"results/{model_name.replace('/', '_')}-{forget_sport}-{localization_type}.pt")
    # torch.save(mask, f'results/test.pt')

if __name__ == "__main__":
    # Get config file as argument
    import argparse

    parser = argparse.ArgumentParser("weight_mask_script")
    parser.add_argument("--config_dir", help="Config file directory", type=str)
    args = parser.parse_args()

    load_global_hyperparams(args.config_dir)

    run()

#%%

# load_global_hyperparams("weight_masking_config.json")
# os.environ['HF_TOKEN'] = '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'
# # tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = HookedTransformer.from_pretrained(
#     model_name,
#     # tokenizer=tokenizer,
#     # device='cuda',
#     default_padding_side="right",
#     fold_ln=False,
#     fold_value_biases=False,
#     center_writing_weights=False,
#     dtype=torch.bfloat16
# )
# tokenizer=model.tokenizer
# ### DATASETS
# # train_dataset = load_dataset('monology/pile-uncopyrighted', split='train', streaming=True)
# sports_1mp = SportsTask(
#     batch_size=train_batch_size, 
#     tokenizer=tokenizer, 
#     device=device, 
#     prep_acdcpp=False, 
#     criterion="log_1_minus_p", 
#     forget_sport_subset={forget_sport}, 
#     is_forget_dataset=True,
# )

# if maintain_sport is None or maintain_sport == "null":
#     maintain_sports = SportsTask(
#         batch_size=train_batch_size, 
#         tokenizer=tokenizer, 
#         device=device, 
#         prep_acdcpp=False, 
#         criterion="cross_entropy", 
#         forget_sport_subset={forget_sport}, 
#         is_forget_dataset=False,
#     )
# else:
#     maintain_sports = SportsTask(batch_size=train_batch_size, tokenizer=tokenizer, device=device, prep_acdcpp=False, criterion="cross_entropy", forget_sport_subset={maintain_sport}, is_forget_dataset=True)

# train_pile = PileTask(
#     batch_size=train_batch_size, 
#     tokenizer=tokenizer, 
#     device=device, 
#     ctx_length=100, 
#     shuffle=True, 
#     buffer_size=1000
# )
# train_tasks = {"sports_1mp": (sports_1mp, .3), "maintain_sports": (maintain_sports, 1), "pile": (train_pile, 1)}

# # want to eval on other sports
# forget_sport_eval = SportsTask(batch_size=eval_batch_size, tokenizer=tokenizer, device=device, prep_acdcpp=False, criterion="cross_entropy", forget_sport_subset={forget_sport}, is_forget_dataset=True)
# test_pile = PileTask(batch_size=eval_batch_size, tokenizer=tokenizer, device=device, ctx_length=100, shuffle=True, buffer_size=1000)

# induction_eval = InductionTask(batch_size=eval_batch_size, tokenizer=tokenizer, prep_acdcpp=False, seq_len=15, device=device)
# if maintain_sport is None or maintain_sport == "null":
#     maintain_sports_eval = SportsTask(batch_size=eval_batch_size, tokenizer=tokenizer, device=device, prep_acdcpp=False, criterion="cross_entropy", forget_sport_subset={forget_sport}, is_forget_dataset=False)
#     eval_tasks = {"induction": induction_eval, "pile": test_pile, "forget_sport": forget_sport_eval, "maintain_sport": maintain_sports_eval}
# else:
#     raise NotImplemented

# ### LOCALIZATIONS

# if localization_type == "ap":
#     with open(f"models/{model_name.replace('/', '_')}_sports_{forget_sport}_{localization_type}_graph.pkl", "rb") as f:
#         localization_graph = pickle.load(f)
#     weight_mask_attn_dict, weight_mask_mlp_dict = get_mask_from_ap_graph(model, localization_graph, localization_top_p)
# elif localization_type == "ct":
#     with open(f"models/{model_name.replace('/', '_')}_sports_{forget_sport}_{localization_type}_graph.pkl", "rb") as f:
#         localization_graph = pickle.load(f)
#     weight_mask_attn_dict, weight_mask_mlp_dict = get_mask_from_ct_graph(model, localization_graph, localization_top_p)
# elif localization_type == "random":
#     weight_mask_attn_dict, weight_mask_mlp_dict = create_random_weight_mask_dicts(model, localization_top_p)
# elif localization_type == "none":
#     weight_mask_attn_dict, weight_mask_mlp_dict = create_random_weight_mask_dicts(model, 1)
# elif localization_type == "manual":
#     # Manual interp means only training the MLP weights from layer 1 to 7
#     weight_mask_attn_dict, weight_mask_mlp_dict = create_mlp_only_mask_dicts(model)

# for layer in weight_mask_attn_dict.keys():
#     if 'W_K' in weight_mask_attn_dict[layer]:
#         weight_mask_attn_dict[layer]['_W_K'] = weight_mask_attn_dict[layer]['W_K']
#     if 'W_V' in weight_mask_attn_dict[layer]:
#         weight_mask_attn_dict[layer]['_W_V'] = weight_mask_attn_dict[layer]['W_V']
# gc.collect()
# torch.cuda.empty_cache()

# ### TRAIN 

# all_train_losses = defaultdict(list)
# all_test_losses = defaultdict(list)
# adversarial_evals = []
# side_effect_evals = []

# # Initialize optimizer, want to optimize over all weights
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.01)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_epochs)
# original_weights = get_unfrozen_weights(model, weight_mask_attn_dict, weight_mask_mlp_dict)


# # Train a sparse mask
# pbar = tqdm(range(n_epochs))
# for epoch in pbar:
#     # Reset grad
#     optimizer.zero_grad()
#     gc.collect()
#     torch.cuda.empty_cache()

#     with torch.autocast(device_type="cuda"):
#         # Compute normal loss over retain
#         for task_name, (task, task_weight) in train_tasks.items():
#             print(f"Running {task_name}")
#             task_loss = 0
#             for i in range(grad_accum_steps):
#                 loss = task.get_train_loss(model) / grad_accum_steps
#                 task_loss += loss.item()
#                 loss *= task_weight
#                 # print(task_name, i, loss)
#                 loss.backward()
#                 del loss
#                 gc.collect()
#                 torch.cuda.empty_cache()
#             all_train_losses[task_name].append(task_loss)

#             gc.collect()
#             torch.cuda.empty_cache()
            
#         gc.collect()
#         torch.cuda.empty_cache()
#         # Add sparsity loss and backprop
#         # Linearly increase from negative to positive, with 0 at 10
#         loss = beta * regularization_loss(model, weight_mask_attn_dict, weight_mask_mlp_dict)
#         loss.backward()
#         print(f"reg loss, {loss.item()}")
#         all_train_losses["reg"].append(loss.item())
#         del loss
#         gc.collect()
#         torch.cuda.empty_cache()
#         # Step and log
#         if clip_grad is not None:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
#         # print(model.blocks[0].attn.W_Q.grad[2])
#         # print(model.blocks[0].attn.W_Q.grad[1])
#         # Remove gradients from frozen components
#         zero_grad(model, weight_mask_attn_dict, weight_mask_mlp_dict)
#         # print(model.blocks[0].attn.W_Q[2])
#         optimizer.step()
#         scheduler.step()
#         clamp_unfrozen_weights(model, original_weights, weight_mask_attn_dict, weight_mask_mlp_dict)
#         gc.collect()
#         torch.cuda.empty_cache()

#         if epoch % evaluate_every == 0 or epoch == n_epochs - 1:
#             for task_name, task in eval_tasks.items():
#                 task_loss = 0
#                 for i in range(n_eval_iters):
#                     task_loss += task.get_test_loss(model).item()
#                 all_test_losses[task_name].append(task_loss / n_eval_iters)
#             if do_adversarial_evals:
#                 print("Running adversarial evals")
#                 adversarial_evals.append(adversarial_sports_eval(model, model_type=model_type, batch_size=eval_batch_size, use_system_prompt=True, include_evals=["MC"]))
#             if do_side_effects_evals:
#                 print("Running side effects evals")
#                 side_effect_evals.append(run_side_effects_evals(model, model_type=model_type, batch_size=eval_batch_size, evals_to_run=["Sports Answers"]))
#         gc.collect()
#         torch.cuda.empty_cache()

# # %%
