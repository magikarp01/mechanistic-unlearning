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
    global train_batch_size, eval_batch_size, device, unlearning_task, maintain_sport, \
           model_name, model_type, learning_rate, n_epochs, grad_accum_steps, alpha, clip_grad, \
           evaluate_every, save_every, n_eval_iters, do_adversarial_evals, do_side_effects_evals, \
           localization_type, localization_location, localization_top_p, n_devices
    
    with open(config_file, "r") as f:
        config = json.load(f)
    
    train_batch_size = int(config['train_batch_size'])
    eval_batch_size = int(config['eval_batch_size'])
    device = config['device']
    model_name = config['model_name']
    model_type = config['model_type']
    unlearning_task = config['unlearning_task']
    learning_rate = float(config['learning_rate'])
    n_epochs = int(config['n_epochs'])
    grad_accum_steps = int(config['grad_accum_steps'])
    alpha = float(config['alpha'])
    clip_grad = None if config['clip_grad'] == "null" else int(config['clip_grad'])
    evaluate_every = int(config['evaluate_every'])
    save_every = int(config['save_every'])
    n_eval_iters = int(config['n_eval_iters'])
    do_adversarial_evals = config['do_adversarial_evals']
    do_side_effects_evals = config['do_side_effects_evals']
    localization_type = config['localization_type']
    localization_location = config['localization_location']
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

        if 'n_key_value_heads' in dir(model.cfg):
            n_heads = model.cfg.n_key_value_heads
        else:
            n_heads = model.cfg.n_heads
        weight_mask_attn_dict[layer]['W_Q'] = torch.rand(model.cfg.n_heads) > top_p
        weight_mask_attn_dict[layer]['W_K'] = torch.rand(n_heads) > top_p
        weight_mask_attn_dict[layer]['W_V'] = torch.rand(n_heads) > top_p
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
        weight_mask_mlp_dict[layer]['W_in'] = False
        weight_mask_mlp_dict[layer]['W_out'] = False
        print(f"Setting layer {layer} to {weight_mask_mlp_dict[layer]}")

    return weight_mask_attn_dict, weight_mask_mlp_dict

def create_mlp_only_mask_dicts(model):
    weight_mask_attn_dict = {}
    weight_mask_mlp_dict = {}
    layers = [3, 4, 5, 7, 8, 9, 10, 14, 15, 16, 17]

    for layer in range(model.cfg.n_layers):
        weight_mask_attn_dict[layer] = {}
        weight_mask_mlp_dict[layer] = {}

        # Set to false: we train a mask over these
        weight_mask_mlp_dict[layer]['W_in'] = not (layer in layers)
        weight_mask_mlp_dict[layer]['W_out'] = not (layer in layers)
        print(f"Setting layer {layer} to {weight_mask_mlp_dict[layer]}")

    return weight_mask_attn_dict, weight_mask_mlp_dict

def get_mask_from_localization(model, localization, top_p):
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
    for key, value in localization.items():
        all_weights.append(value)

    all_weights = np.array(all_weights)
    threshold = np.percentile(all_weights, 100 - top_p)

    weight_mask_attn_dict = {}
    weight_mask_mlp_dict = {}

    for layer in range(model.cfg.n_layers):
        weight_mask_attn_dict[layer] = {}
        weight_mask_mlp_dict[layer] = {}

        if 'a0.0_q' in localization:
            weight_mask_attn_dict[layer]['W_Q'] = torch.tensor(
                [
                    abs(localization[f"a{layer}.{head}_q"]) < threshold 
                    for head in range(model.cfg.n_heads)
                ]
            )
        else:
            weight_mask_attn_dict[layer]['W_Q'] = None

        if 'a0.0_k' in localization:
            if 'n_key_value_heads' in dir(model.cfg):
                n_heads = model.cfg.n_key_value_heads
            else:
                n_heads = model.cfg.n_heads
            weight_mask_attn_dict[layer]['W_K'] = torch.tensor(
                [
                    abs(localization[f"a{layer}.{head}_k"]) < threshold 
                    for head in range(n_heads)
                ]
            )
        else:
            weight_mask_attn_dict[layer]['W_K'] = None
        
        if 'a0.0_v' in localization:
            if 'n_key_value_heads' in dir(model.cfg):
                n_heads = model.cfg.n_key_value_heads
            else:
                n_heads = model.cfg.n_heads
            weight_mask_attn_dict[layer]['W_V'] = torch.tensor(
                [
                    abs(localization[f"a{layer}.{head}_v"]) < threshold 
                    for head in range(n_heads)
                ]
            )
        else:
            weight_mask_attn_dict[layer]['W_V'] = None
        
        if 'a0.0_result' in localization:
            weight_mask_attn_dict[layer]['W_O'] = torch.tensor(
                [
                    abs(localization[f"a{layer}.{head}_result"]) < threshold 
                    for head in range(model.cfg.n_heads)
                ]
            )
        else:
            weight_mask_attn_dict[layer]['W_O'] = None
            
        if 'm0_in' in localization:
            weight_mask_mlp_dict[layer]['W_in'] = abs(localization[f"m{layer}_in"]) < threshold
        else:
            weight_mask_mlp_dict[layer]['W_in'] = None
        
        if 'm0_out' in localization:
            weight_mask_mlp_dict[layer]['W_out'] = abs(localization[f"m{layer}_out"]) < threshold
        else:
            weight_mask_mlp_dict[layer]['W_out'] = None

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
                    print(f"Setting {layer}: {component} to not require grad")
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
                        print(f"None grad for {component} in layer {layer}")
            else:
                param = getattr(block.mlp, component)
                if param.grad is not None:
                    param.grad = torch.zeros_like(param.grad).to("cuda")
                else:
                    print(f"None grad for {component} in layer {layer}")

def regularization_loss(model, weight_mask_attn_dict, weight_mask_mlp_dict):
    # L1 sparsity, but only for components that are not frozen

    last_device = f'cuda:{n_devices - 1}' if torch.cuda.is_available() else 'cpu'
    loss = 0
    for layer, block in enumerate(model.blocks):
        for component in ["W_Q", "_W_K", "_W_V", "W_O"]:
            # If the component exists, we take the non frozen heads
            if component in weight_mask_attn_dict[layer]:
                frozen_heads = weight_mask_attn_dict[layer][component]
                loss += torch.sum(torch.abs(getattr(block.attn, component)[~frozen_heads])).to(last_device)
        
        for component in ["W_in", "W_out"]:
            if component in weight_mask_mlp_dict[layer]:
                # If the mlp is not frozen, then we add the L1 loss
                if not weight_mask_mlp_dict[layer][component]:
                    loss += torch.sum(torch.abs(getattr(block.mlp, component))).to(last_device)
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

                total_original = torch.ones_like(param)
                total_original[~frozen_heads] = original_weights[layer][component]
                mask[layer][component] = (param / total_original)
                mask[layer][component][frozen_heads] = 1
        
        for component in ["W_in", "W_out"]:
            if component in weight_mask_mlp_dict[layer]:
                if not weight_mask_mlp_dict[layer][component]:
                    mask[layer][component] = (getattr(block.mlp, component) / original_weights[layer][component])
    return mask

def run():
    from transformers import GPT2Tokenizer, GPTNeoXTokenizerFast, AutoModelForCausalLM, AutoTokenizer
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

    from zipfile import ZipFile
    from glob import glob

    os.chdir('/root/mechanistic-unlearning')

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
        n_devices=n_devices
    )
    model.W_E.requires_grad = False
    model.W_U.requires_grad = False
    for block in model.blocks:
        block.ln1.w.requires_grad = False
        block.ln2.w.requires_grad = False
    model.ln_final.w.requires_grad = False

    ### DATASETS
    
    from tasks.facts.CounterFactTask import CounterFactTask, CounterFactTask_Injection, adversarial_counterfact_eval
    from transformers import AutoTokenizer

    right_tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
    forget_facts = 16
    last_device = f'cuda:{n_devices - 1}' if torch.cuda.is_available() else 'cpu'
    forget_kwargs = {"forget_fact_subset": forget_facts, "is_forget_dataset": True, "train_test_split": False}
    maintain_kwargs = {"forget_fact_subset": forget_facts, "is_forget_dataset": False, "train_test_split": True}

    # Train
    inject_fact_train = CounterFactTask_Injection(batch_size=train_batch_size, tokenizer=tokenizer, device=last_device, **forget_kwargs)
    maintain_facts = CounterFactTask(batch_size=train_batch_size, tokenizer=tokenizer, device=last_device, criterion="cross_entropy", **maintain_kwargs)
    train_pile = PileTask(batch_size=train_batch_size, tokenizer=tokenizer, device=last_device, ctx_length=100, shuffle=True, buffer_size=1000)
    train_tasks = {"facts_injection": (inject_fact_train, .5), "maintain_facts": (maintain_facts, 1), "pile": (train_pile, 1)}

    # Test
    test_pile = PileTask(batch_size=eval_batch_size, tokenizer=tokenizer, device=last_device, ctx_length=100, shuffle=True, buffer_size=1000)
    induction_eval = InductionTask(batch_size=eval_batch_size, tokenizer=tokenizer, prep_acdcpp=False, seq_len=15, device=last_device)
    inject_fact_eval = CounterFactTask_Injection(batch_size=eval_batch_size, tokenizer=tokenizer, device=last_device, criterion="cross_entropy", **forget_kwargs)
    forget_fact_eval = CounterFactTask(batch_size=eval_batch_size, tokenizer=right_tokenizer, device=last_device, criterion="log_1_minus_p", **forget_kwargs)
    maintain_facts_eval = CounterFactTask(batch_size=eval_batch_size, tokenizer=right_tokenizer, device=last_device, criterion="cross_entropy", **maintain_kwargs)
    eval_tasks = {"induction": induction_eval, "pile": test_pile, "maintain_facts": maintain_facts_eval, "inject_facts": inject_fact_eval, "forget_fact": forget_fact_eval}

    ### LOCALIZATIONS

    if localization_type == "ap":
        with open(f"{localization_location}", "rb") as f:
            localization_graph = pickle.load(f)
        weight_mask_attn_dict, weight_mask_mlp_dict = get_mask_from_localization(model, localization_graph, localization_top_p)
    elif localization_type == "ct":
        with open(f"{localization_location}", "rb") as f:
            localization_graph = pickle.load(f)
        weight_mask_attn_dict, weight_mask_mlp_dict = get_mask_from_localization(model, localization_graph, localization_top_p)
    elif localization_type == "random":
        weight_mask_attn_dict, weight_mask_mlp_dict = create_random_weight_mask_dicts(model, localization_top_p)
    elif localization_type == "none":
        weight_mask_attn_dict, weight_mask_mlp_dict = create_random_weight_mask_dicts(model, 1)
    elif localization_type == "mlp":
        weight_mask_attn_dict, weight_mask_mlp_dict = create_mlp_only_mask_dicts(model)
    elif localization_type == "manual":
        weight_mask_attn_dict, weight_mask_mlp_dict = create_manual_mask_dicts(model)

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

    # Set beta such that the regularization loss is 1.5 at the start
    beta = 1.5 / regularization_loss(model, weight_mask_attn_dict, weight_mask_mlp_dict).item()
    zero_grad(model, weight_mask_attn_dict, weight_mask_mlp_dict)



    wandb.login(key="6f39dedff978870c25e55aed36e504403271d404")
    ### LOGGING
    wandb.init(
        # set the wandb project where this run will be logged
        project="mech-unlearning",
        name=f"{model_name.split('/')[-1]}-{unlearning_task}-{localization_type}",

        # track hyperparameters and run metadata
        config={
            "model_type": model_type,
            "model_name": model_name,
            "learning_rate": learning_rate,
            "unlearning_task": unlearning_task,
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
        torch.cuda.empty_cache()
        gc.collect()

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
                    torch.cuda.empty_cache()
                    gc.collect()
                all_train_losses[task_name].append(task_loss)

                torch.cuda.empty_cache()
                gc.collect()
                
            torch.cuda.empty_cache()
            gc.collect()
            # Add sparsity loss and backprop
            loss = beta * regularization_loss(model, weight_mask_attn_dict, weight_mask_mlp_dict)
            loss.backward()
            print(f"reg loss, {loss.item()}")
            all_train_losses["reg"].append(loss.item())
            del loss
            torch.cuda.empty_cache()
            gc.collect()
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
            torch.cuda.empty_cache()
            gc.collect()

            if epoch % evaluate_every == 0 or epoch == n_epochs - 1:
                for task_name, task in eval_tasks.items():
                    task_loss = 0
                    for i in range(n_eval_iters):
                        task_loss += task.get_test_loss(model).item()
                    all_test_losses[task_name].append(task_loss / n_eval_iters)
                if do_adversarial_evals:
                    print("Running adversarial evals")
                    adversarial_evals.append(
                        adversarial_counterfact_eval(
                            model, 
                            model_type=model_type, 
                            batch_size=eval_batch_size, 
                            forget_task_init_kwargs=forget_kwargs, 
                            maintain_task_init_kwargs=maintain_kwargs, 
                            continuous=True, include_evals=["Normal", "MC", "Paraphrase", "Neighborhood"], n_mc_shots=1, device=last_device
                        )
                    )
                if do_side_effects_evals:
                    print("Running side effects evals")
                    side_effect_evals.append(run_side_effects_evals(model, model_type=model_type, batch_size=eval_batch_size, evals_to_run=["General"], device=last_device))
            if epoch % save_every == 0 and epoch > 0:
                torch.cuda.empty_cache()
                gc.collect()

                mask = get_mask(model, original_weights, weight_mask_attn_dict, weight_mask_mlp_dict) 
                torch.save(mask, f"results/{model_name.replace('/', '_')}-{unlearning_task}-{localization_type}-{epoch}.pt")

                del mask
                torch.cuda.empty_cache()
                gc.collect()

                # Save to wandb, delete after
                wandb.save(f"results/{model_name.replace('/', '_')}-{unlearning_task}-{localization_type}-{epoch}.pt")
                os.remove(f"results/{model_name.replace('/', '_')}-{unlearning_task}-{localization_type}-{epoch}.pt")

            torch.cuda.empty_cache()
            gc.collect()
            
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
    torch.save(mask, f"results/{model_name.replace('/', '_')}-{unlearning_task}-{localization_type}.pt")
    # torch.save(mask, f'results/test.pt')

if __name__ == "__main__":
    # Get config file as argument
    import argparse

    parser = argparse.ArgumentParser("weight_mask_script")
    parser.add_argument("--config_dir", help="Config file directory", type=str)
    args = parser.parse_args()

    load_global_hyperparams(args.config_dir)
    # load_global_hyperparams("weight_masking_config.json")

    run()

#%%
