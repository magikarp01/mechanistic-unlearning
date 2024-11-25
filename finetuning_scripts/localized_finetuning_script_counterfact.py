# from circuit_breaking.src import *
import torch
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import os
print(os.getcwd())
import sys
sys.path.append(os.getcwd())
# from circuit_breaking.src.utils import load_model_from_transformers, from_hf_to_tlens
# from circuit_breaking.src.masks import MLPHiddenMask
from tqdm.auto import tqdm
import pickle

import argparse
parser = argparse.ArgumentParser()


parser.add_argument("--config_path", type=str, default=None, help="Path to a json config file containing all arguments. Will override all other arguments where specified.")
parser.add_argument("--save_dir", type=str, default=None, help="Path to a directory to save the results. If not specified, will be saved in same folder as config path")
parser.add_argument("--model_type", type=str, choices=["gemma-7b", "llama-2", "pythia-2.8b", "gemma-2-9b", "llama-3-8b"], default="gemma-7b")
# parser.add_argument("--forget_facts", type=int, default=None)
# parser.add_argument("--inject_fact", type=bool, default=False)
parser.add_argument("--forget_split", type=str, default=None)
parser.add_argument("--inject_fact", action="store_true")

parser.add_argument("--localization_type", type=str, choices=["localized_ct", "localized_ap", "localized_ct_mlps", "localized_ap_mlps", "manual_interp", "random", "random_mlps", "all_mlps", "nonlocalized"])
parser.add_argument("--run_id", type=str, default=None)

parser.add_argument("--combine_heads", type=bool, default=True)
parser.add_argument("--train_batch_size", type=int, default=4)
parser.add_argument("--eval_batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--grad_accum_steps", type=int, default=None)
parser.add_argument("--mixed_precision", type=bool, default=False)
parser.add_argument("--n_epochs", type=int, default=50)
parser.add_argument("--beta", type=int, default=3)
parser.add_argument("--clip_grad", type=float, default=1)
parser.add_argument("--evaluate_every", type=int, default=5)
parser.add_argument("--n_eval_iters", type=int, default=5)
parser.add_argument("--deep_evaluate_every", type=int, default=10)
parser.add_argument("--do_adversarial_evals", type=bool, default=True)
parser.add_argument("--n_mc_shots", type=int, default=8)
parser.add_argument("--do_side_effects_evals", type=bool, default=True)
parser.add_argument("--check_all_logits", type=bool, default=False)
parser.add_argument("--use_wandb", type=bool, default=True)
parser.add_argument("--save_model", type=bool, default=False)
parser.add_argument("--push_to_hub", type=bool, default=False)

parser.add_argument("--do_full_mmlu_evals", type=bool, default=False)

parser.add_argument("--do_relearning_evals", type=bool, default=False)
parser.add_argument("--n_relearn_iters", type=int, default=20)
parser.add_argument("--n_relearn_facts", type=int, default=32)
parser.add_argument("--lora_rank", type=int, default=512)
parser.add_argument("--target_modules", type=str, default="all-linear")
parser.add_argument("--relearning_lr", type=float, default=2e-4)
parser.add_argument("--forget_loss_coef", type=float, default=1)


parser.add_argument("--do_softprompt_evals", type=bool, default=False)
parser.add_argument("--softprompt_attack_batch_size", type=int, default=16)
parser.add_argument("--num_softprompts", type=int, default=4, help="Number of softprompts to train")

args = parser.parse_args()

import json
if args.config_path:
    print(f"Loading args from config file: {args.config_path}")
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    # Update args with config file values
    args.__dict__.update(config)
else:
    print("No config file provided, using command line args")

import os
if args.save_dir is None:
    args.save_dir = os.path.dirname(args.config_path)
    # args.save_dir = os.path.join(args.save_dir, "saved")
    os.makedirs(args.save_dir, exist_ok=True)

train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size

learning_rate = args.learning_rate
n_epochs = args.n_epochs
if args.grad_accum_steps is None:
    args.grad_accum_steps = 64 // train_batch_size
grad_accum_steps = args.grad_accum_steps
beta = args.beta
clip_grad = args.clip_grad
evaluate_every = args.evaluate_every
n_eval_iters = args.n_eval_iters
deep_evaluate_every = args.deep_evaluate_every
do_adversarial_evals = args.do_adversarial_evals
do_side_effects_evals = args.do_side_effects_evals
check_all_logits = args.check_all_logits
use_wandb = args.use_wandb
save_model = args.save_model
forget_loss_coef = args.forget_loss_coef

print("==========ARGS==========")
print(args)
print("==========END ARGS==========")


forget_split_name_dict = {"first_16_unsplit": "0_16", "first_64_unsplit": "0_64"}

from transformers import AutoTokenizer, AutoModelForCausalLM
if args.model_type == "gemma-7b":
    model_name_or_path = "google/gemma-7b"
    model_type = "gemma"

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)

    n_layers = 28
    n_heads = 16
    n_kv_heads = None
    param_count_dict = {"attn.hook_q": 3072*4096, "attn.hook_k": 3072*4096, "attn.hook_v": 3072*4096, "attn.hook_result": 4096*3072, "mlp.hook_pre": 3072 * 24576, "mlp.hook_post": 24576 * 3072, "mlp.hook_gate": 3072 * 24576}

    # get manual localization param count
    forget_split_file_name = forget_split_name_dict[args.forget_split]
    with open(f"experiments/counterfact_manual/results/google_gemma-7b_manual_layers_{forget_split_file_name}.json", "r") as f:
        manual_layers = json.load(f)["chosen_layers"]

    manual_param_count = len(manual_layers)*(param_count_dict["mlp.hook_pre"] + param_count_dict["mlp.hook_post"] + param_count_dict["mlp.hook_gate"])

    mmlu_batch_size = 2

    if args.do_softprompt_evals:
        cast_to_model_dtype = False

elif args.model_type == "gemma-2-9b":
    model_name_or_path = "google/gemma-2-9b"
    model_type = "gemma-2"
    other_model_type = "gemma2_9b"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    forget_split_file_name = forget_split_name_dict[args.forget_split]
    with open(f"experiments/counterfact_manual/results/google_gemma-2-9b_manual_layers_{forget_split_file_name}.json", "r") as f:
        manual_layers = json.load(f)["chosen_layers"]

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
    n_layers = 42
    n_heads = 16
    n_kv_heads = 8

    param_count_dict = {"attn.hook_q": 3584*4096, "attn.hook_k": 3584*2048, "attn.hook_v": 3584*2048, "attn.hook_result": 4096*3584, "mlp.hook_pre": 3584 * 14336, "mlp.hook_post": 14336 * 3584, "mlp.hook_gate": 3584 * 14336}
    manual_param_count = len(manual_layers)*(param_count_dict["mlp.hook_pre"] + param_count_dict["mlp.hook_post"] + param_count_dict["mlp.hook_gate"])

    mmlu_batch_size = 2

    if args.do_softprompt_evals:
        cast_to_model_dtype = True



elif args.model_type == "llama-3-8b":
    model_name_or_path = "meta-llama/Meta-Llama-3-8B"
    model_type = "llama-3"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    if os.path.exists("/data/public_models/Meta-Llama-3-8B/"):
        model = AutoModelForCausalLM.from_pretrained("/data/public_models/Meta-Llama-3-8B/", torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
    n_layers = 32
    n_heads = 32
    n_kv_heads = None
    param_count_dict = {"attn.hook_q": 4096*4096, "attn.hook_k": 4096*1024, "attn.hook_v": 4096*1024, "attn.hook_result": 4096*4096, "mlp.hook_pre": 4096 * 14336, "mlp.hook_post": 14336 * 4096, "mlp.hook_gate": 4096 * 14336}
    manual_layers = range(1, 5)
    manual_param_count = len(manual_layers)*(param_count_dict["mlp.hook_pre"] + param_count_dict["mlp.hook_post"] + param_count_dict["mlp.hook_gate"])

    mmlu_batch_size = 5

    if args.do_softprompt_evals:
        cast_to_model_dtype = False

else:
    raise NotImplementedError(f"Model type {args.model_type} not implemented")

print("Manual param count: ", manual_param_count)

### Unlearning and evaluation tasks

from tasks import PileTask
# from tasks.ioi.IOITask import IOITask, IOITask_NPO, IOITask_Uniform
# from tasks.induction.InductionTask import InductionTask, InductionTask_NPO, InductionTask_Uniform
from tasks.facts.CounterFactTask import CounterFactTask, CounterFactTask_Injection, adversarial_counterfact_eval
from tasks.facts.SportsTaskSideEffects import run_side_effects_evals

device = "cuda"

forget_split = args.forget_split
inject_fact = args.inject_fact

# if inject_fact:
#     save_dir = f"results/{model_name_or_path}_localized_finetuning_injection_counterfact/{args.localization_type}"
# else:
#     save_dir = f"results/{model_name_or_path}_localized_finetuning_counterfact/{args.localization_type}"
# forget_kwargs = {"forget_fact_subset": forget_facts, "is_forget_dataset": True, "train_test_split": False}
# maintain_kwargs = {"forget_fact_subset": forget_facts, "is_forget_dataset": False, "train_test_split": True}
# # forget_loss_coef = 0.5
# forget_loss_coef = 1

# if args.run_id is not None:
#     save_dir = f"{save_dir}_{args.run_id}"

# os.makedirs(save_dir, exist_ok=True)
save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)

forget_kwargs = {"forget_split": forget_split, "maintain_split": None, "model_type": model_type}
maintain_kwargs = {"forget_split": forget_split, "maintain_split": "split", "model_type": model_type}
inject_fact = args.inject_fact

maintain_facts = CounterFactTask(batch_size=train_batch_size, tokenizer=tokenizer, device=device, criterion="cross_entropy", **maintain_kwargs)

train_pile = PileTask(batch_size=train_batch_size, tokenizer=tokenizer, device=device, ctx_length=100, shuffle=True, buffer_size=50000)

if inject_fact:
    facts_injection = CounterFactTask_Injection(batch_size=train_batch_size, tokenizer=tokenizer, device=device, inject_fact=inject_fact, **forget_kwargs)
    train_tasks = {"facts_injection": (facts_injection, forget_loss_coef), "maintain_facts": (maintain_facts, 1), "pile": (train_pile, 1)}
    print(facts_injection.train_df)
else:
    facts_1mp = CounterFactTask(batch_size=train_batch_size, tokenizer=tokenizer, device=device, criterion="log_1_minus_p", **forget_kwargs)
    train_tasks = {"facts_1mp": (facts_1mp, forget_loss_coef), "maintain_facts": (maintain_facts, 1), "pile": (train_pile, 1)}
    print(facts_1mp.train_df)

# train_tasks = {"maintain_facts": (maintain_facts, 1)}

# want to eval on other facts
forget_fact_eval = CounterFactTask_Injection(batch_size=eval_batch_size, tokenizer=tokenizer, device=device, criterion="cross_entropy", **forget_kwargs)
test_pile = PileTask(batch_size=eval_batch_size, tokenizer=tokenizer, device=device, ctx_length=100, shuffle=True, buffer_size=50000)

maintain_facts_eval = CounterFactTask(batch_size=eval_batch_size, tokenizer=tokenizer, device=device, criterion="cross_entropy", **maintain_kwargs)
# if inject_fact:
#     inject_fact_eval = CounterFactTask_Injection(batch_size=eval_batch_size, tokenizer=tokenizer, device=device, criterion="cross_entropy", **forget_kwargs)
#     eval_tasks = {"pile": test_pile, "forget_fact": forget_fact_eval, "maintain_fact": maintain_facts_eval, "inject_fact": inject_fact_eval}
# else:
eval_tasks = {"pile": test_pile, "forget_fact": forget_fact_eval, "maintain_fact": maintain_facts_eval}
print(forget_fact_eval.train_dataset[0])


### localize model

from cb_utils.mask_utils import convert_attrs_to_components, get_top_components, get_top_components_no_subcomponents, get_random_components, load_mask_from_state_dict, get_parameter, apply_localized_gradients, find_component_params, get_top_components_no_subcomponents_gqa

if args.model_type == "gemma-7b":
    localization_model_name = "google_gemma-7b"
elif args.model_type == "gemma-2-9b":
    localization_model_name = "google_gemma-2-9b"
elif args.model_type == "llama-3-8b":
    localization_model_name = "meta-llama_Meta-Llama-3-8B"
else:
    raise NotImplementedError(f"Model type {model_type} not implemented")

# if forget_split == "first_16_unsplit":
#     localization_forget_split_name = "0_16"
# elif forget_split == "first_64_split":
#     localization_forget_split_name = "0_64"
# else:
#     raise NotImplementedError(f"Forget split {forget_split} not implemented")

with open(f"models/{localization_model_name}_counterfact_ap_graph_{forget_split_name_dict[forget_split]}.pkl", "rb") as f:
    ap_graph = pickle.load(f)
with open(f"models/{localization_model_name}_counterfact_ct_graph_{forget_split_name_dict[forget_split]}.pkl", "rb") as f:
    ct_graph = pickle.load(f)

# import pickle
# with open("models/google_gemma-2-9b_facts_all_ap_graph.pkl", "rb") as f:
#     ap_graph = pickle.load(f)
# print(ap_graph.keys())

# # ct components
# with open("models/google_gemma-2-9b_facts_all_ct_graph.pkl", "rb") as f:
#     ct_graph = pickle.load(f){args.run_id}
# print(ct_graph)

# top_p = 5
combine_heads = True

# localization_types = ["localized_ap", "random", "manual_interp", "all_mlps", "nonlocalized"]
# localization_types = ["all_mlps"]
# localization_types = ["manual_interp", "nonlocalized"]
# localization_types = ["localized_ap", "localized_ct"]
# localization_types = ["nonlocalized", "all_mlps", "sports_manual_interp", "forget_ct", "manual_interp", "random"]
# localization_types = ["manual_interp"]

localization_type = args.localization_type

if localization_type == 'localized_ct' or localization_type == 'localized_ct_mlps':
    mlp_only = localization_type == 'localized_ct_mlps'
    final_components, final_attn_heads = get_top_components_no_subcomponents_gqa(ct_graph, n_heads=n_heads, n_layers=n_layers, combine_heads=combine_heads, param_count=manual_param_count, param_count_dict=param_count_dict, n_kv_heads=n_kv_heads, mlp_in_is_pre=True, mlp_only=mlp_only)

elif localization_type == 'localized_ap' or localization_type == 'localized_ap_mlps':
    mlp_only = localization_type == 'localized_ap_mlps'
    final_components, final_attn_heads = get_top_components_no_subcomponents_gqa(ap_graph, n_heads=n_heads, n_layers=n_layers, combine_heads=True, param_count=manual_param_count, param_count_dict=param_count_dict, n_kv_heads=n_kv_heads, mlp_in_is_pre=False, mlp_only=mlp_only)
    
elif localization_type == 'manual_interp':
    final_components = []
    for mlp_layer in manual_layers:
        final_components.append(f"blocks.{mlp_layer}.mlp.hook_pre")
        final_components.append(f"blocks.{mlp_layer}.mlp.hook_gate")
        final_components.append(f"blocks.{mlp_layer}.mlp.hook_post")
    final_attn_heads = {}

elif localization_type == 'random':
    final_components, final_attn_heads = get_random_components(n_layers=n_layers, n_heads=n_heads, combine_subcomponents=False, param_count=manual_param_count, param_count_dict=param_count_dict)

elif localization_type == "all_mlps":
    final_components = []
    for mlp_layer in range(n_layers):
        final_components.append(f"blocks.{mlp_layer}.mlp.hook_pre")
        final_components.append(f"blocks.{mlp_layer}.mlp.hook_post")
        final_components.append(f"blocks.{mlp_layer}.mlp.hook_gate")
    final_attn_heads = {}

elif localization_type == 'random_mlps':
    final_components = []
    num_mlps = len(manual_layers)
    randomly_chosen_layers = torch.randperm(n_layers)[:num_mlps].sort().values
    for mlp_layer in randomly_chosen_layers:
        final_components.append(f"blocks.{mlp_layer}.mlp.hook_pre")
        final_components.append(f"blocks.{mlp_layer}.mlp.hook_gate")
        final_components.append(f"blocks.{mlp_layer}.mlp.hook_post")
    final_attn_heads = {}

elif localization_type == 'nonlocalized':
    final_components = []
    for layer in range(n_layers):
        final_components.append(f"blocks.{layer}.mlp.hook_pre")
        final_components.append(f"blocks.{layer}.mlp.hook_post")
        final_components.append(f"blocks.{layer}.mlp.hook_gate")
        final_components.append(f"blocks.{layer}.attn.hook_q")
        final_components.append(f"blocks.{layer}.attn.hook_k")
        final_components.append(f"blocks.{layer}.attn.hook_v")
        final_components.append(f"blocks.{layer}.attn.hook_result")
        
    final_attn_heads = None # don't actually think we need this
    # assert (torch.tensor([len(x) for x in final_attn_heads.values()]) == n_heads).all()

# get number of params
num_params = 0
for component in final_components:
    num_params += find_component_params(component, param_count_dict)
print(f"Number of parameters in {localization_type} localization: {num_params}")
print(f"{final_components=}")


apply_localized_gradients(model, final_components, model_type=model_type)


## train model
from collections import defaultdict

import wandb

# for localization_type in ["nonlocalized"]:
print(f"Memory at start for {localization_type}: {torch.cuda.memory_allocated() / 1024**3}")
if use_wandb:
    wandb.init(project="circuit_breaking", name=f"finetuning_counterfact_{localization_type}_{forget_split=}_{inject_fact=}_run_id={args.run_id}")
    wandb.config.update(args.__dict__)

model.cuda()

all_train_losses = defaultdict(list)
all_test_losses = defaultdict(list)
adversarial_evals = {}
side_effect_evals = {}

# Initialize optimizer

if args.mixed_precision:
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate, weight_decay=0)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_epochs)
# Cycle dataloaders
# Train a sparse mask
# print(f"Memory afterloading {localization_type} model: {torch.cuda.memory_allocated() / 1024**3}")


pbar = tqdm(range(n_epochs))
for epoch in pbar:
    # Sample batches
    # Reset grad
    optimizer.zero_grad()
    # Compute normal loss over retain
    for task_name, (task, task_weight) in train_tasks.items():
        task_loss = 0
        for i in range(grad_accum_steps):
            torch.cuda.empty_cache()
            # with torch.cuda.amp.autocast():
            loss = task.get_train_loss(model) / grad_accum_steps
            task_loss += loss.item()
            # if args.gradient_checkpointing:
                # Backward pass with scaling
                # scaler.scale(loss * task_weight).backward()
            # else:
            (loss * task_weight).backward()
        all_train_losses[task_name].append(task_loss)

        if use_wandb:
            wandb.log({f"{task_name}_train_loss": task_loss}, step=epoch)
        
    # print(f"Before backpropgating loss on epoch {epoch}: {torch.cuda.memory_allocated() / 1024**3}, max mem: {torch.cuda.max_memory_allocated() / 1024**3}")
    # Step and log
    # zero_nan_grads(mask)
    if clip_grad is not None:
        # if args.gradient_checkpointing:
        #     scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

    # if args.gradient_checkpointing:
    #     scaler.step(optimizer)
    #     scaler.update()
    # else:
    optimizer.step()
    scheduler.step()
    print("After epoch, mem is ", torch.cuda.memory_allocated() / 1024**3)
    optimizer.zero_grad()

    torch.cuda.empty_cache()

    # print(f"After backpropgating loss on epoch {epoch}: {torch.cuda.memory_allocated() / 1024**3}, max mem: {torch.cuda.max_memory_allocated() / 1024**3}")


    if epoch % evaluate_every == 0 or epoch == n_epochs - 1:
        for task_name, task in eval_tasks.items():
            task_loss = 0
            task_accuracy = 0
            for i in range(n_eval_iters):
                task_loss += task.get_test_loss(model).item()
                task_accuracy += task.get_test_accuracy(model)
            all_test_losses[task_name].append(task_loss / n_eval_iters)
            all_test_losses[f"{task_name}_accuracy"].append(task_accuracy / n_eval_iters)
            if use_wandb:
                wandb.log({f"{task_name}_test_loss": task_loss / n_eval_iters}, step=epoch)
                wandb.log({f"{task_name}_test_accuracy": task_accuracy / n_eval_iters}, step=epoch)

        if inject_fact:
            # check inject_fact_accuracy
            inject_fact_accuracy = 0
            for i in range(n_eval_iters):
                inject_fact_accuracy += eval_tasks["forget_fact"].get_test_accuracy(model, injected_accuracy=True)
            all_test_losses["inject_fact_accuracy"].append(inject_fact_accuracy / n_eval_iters)
            if use_wandb:
                wandb.log({f"inject_fact_accuracy": inject_fact_accuracy / n_eval_iters}, step=epoch)

    # print(f"After evaluating test loss on epoch {epoch}: {torch.cuda.memory_allocated() / 1024**3}, max mem: {torch.cuda.max_memory_allocated() / 1024**3}")

    if (deep_evaluate_every is not None and epoch % deep_evaluate_every == 0) or epoch == n_epochs - 1:
        if do_adversarial_evals:
            print("Running adversarial evals")
            adv_evals = adversarial_counterfact_eval(model, model_type=model_type, batch_size=eval_batch_size, 
                forget_task_init_kwargs=forget_kwargs, 
                maintain_task_init_kwargs=maintain_kwargs, 
                inject_fact=inject_fact,
                n_mc_shots=args.n_mc_shots,
                continuous=True, check_all_logits=args.check_all_logits,
                include_evals=["Normal", "MC", "Paraphrase", "Neighborhood"])
            adversarial_evals[epoch] = adv_evals
            if use_wandb:
                for eval_domain in adv_evals.keys():
                    for eval_type in adv_evals[eval_domain].keys():
                        wandb.log({f"adversarial_{eval_domain}_{eval_type}": adv_evals[eval_domain][eval_type]}, step=epoch)

        # print(f"After evaluating adversarial evals on epoch {epoch}: {torch.cuda.memory_allocated() / 1024**3}, max mem: {torch.cuda.max_memory_allocated() / 1024**3}")
        if do_side_effects_evals:
            print("Before side effect eval, mem is ", torch.cuda.memory_allocated() / 1024**3)
            print("Running side effects evals")
            side_effect_evals[epoch] = run_side_effects_evals(model, model_type=model_type, batch_size=eval_batch_size, evals_to_run=["General"], general_batch_size=mmlu_batch_size)
            if use_wandb:
                wandb.log(side_effect_evals[epoch]["General"], step=epoch)
        # print(f"After evaluating side effects evals on epoch {epoch}: {torch.cuda.memory_allocated() / 1024**3}, max mem: {torch.cuda.max_memory_allocated() / 1024**3}")
del optimizer
del scheduler
torch.cuda.empty_cache()
print("After empty cache and del optimizer and scheduler: ", torch.cuda.memory_allocated() / 1024**3)

if save_model:
    os.makedirs(f"{save_dir}/models", exist_ok=True)
    model.save_pretrained(f"{save_dir}/models/model.pt")
else:
    print(f"Not saving model for {localization_type}")
os.makedirs(f"{save_dir}/models", exist_ok=True)
with open(f"{save_dir}/models/model_metrics.pkl", "wb") as f:
    pickle.dump({"train_losses": all_train_losses, "test_losses": all_test_losses, "adversarial_evals": adversarial_evals, "side_effect_evals": side_effect_evals}, f)

## SAVE TO HF
if args.push_to_hub:
    print("Pushing to HF, path is ", f"PhillipGuo/{model_type}-{localization_type}-forget_{forget_split=}-inject_{inject_fact=}-{args.run_id}")
    hf_save_path = f"PhillipGuo/{model_type}-{localization_type}-forget_{forget_split}-inject_{inject_fact}-{args.run_id}"
    model.push_to_hub(hf_save_path)

## MMLU Evals
print(args.do_full_mmlu_evals)
if args.do_full_mmlu_evals:
    # model.cpu()
    print("Running full MMLU evals")
    
    import lm_eval
    from lm_eval import evaluate
    from lm_eval.models.huggingface import HFLM

    print(f"{save_dir=}")
    try:
        with open(f"{save_dir}/full_capability_dict.pkl", "rb") as f:
            capability_dict = pickle.load(f)
    except:
        print(f"Couldn't load capability dict from {save_dir}/full_capability_dict.pkl")
        capability_dict = {}

    # for name in model_init_and_load_funcs.keys():
    model.cuda()

    lm_model = HFLM(pretrained=model, dtype=torch.bfloat16, device="cuda")

    results = lm_eval.simple_evaluate(
        model=lm_model,
        tasks=["mmlu", "sciq"]
    )
    with open(f"{save_dir}/full_capability_dict.pkl", "wb") as f:
        pickle.dump(results, f)
    del lm_model
    model.cuda()


if args.do_softprompt_evals:
    print("Running softprompt evals")
    tokenizer.padding_side = "right"
    if "unsplit" in args.forget_split:
        relearn_forget_split = args.forget_split.replace("unsplit", "split")
        print(f"Original forget split is {args.forget_split}, relearning with {relearn_forget_split}")
    else:
        print("Why is the forget_split originally train-test-splitted? This probably shouldn't be happening")
        relearn_forget_split = args.forget_split

    relearn_forget_kwargs = {"forget_split": relearn_forget_split, "maintain_split": None, "model_type": model_type}
    relearn_maintain_kwargs = {"forget_split": relearn_forget_split, "maintain_split": "split", "model_type": model_type}
    forget_eval = CounterFactTask_Injection(batch_size=32, tokenizer=tokenizer, inject_fact=inject_fact, **relearn_forget_kwargs)


    from torch.nn.utils.rnn import pad_sequence

    def prepare_counterfact_classification_data(dataframe, tokenizer):
        """
        Prepares tokenized data for counterfact classification task by combining prompts with counterfact labels
        and creating appropriate masks for training.
        
        Args:
            dataframe: DataFrame containing 'prompt' and 'sport' columns
            tokenizer: Tokenizer instance for text processing
            
        Returns:
            tokenized_sequences: Padded tensor of tokenized input sequences
            label_positions: Boolean tensor marking positions of sport labels
            original_prompts: List of original prompt texts
        """
        tokenized_sequences = []
        original_prompts = []
        label_positions = []
        
        for _, row in dataframe.iterrows():
            prompt_text = row["prompt"]
            original_prompts.append(tokenizer.bos_token + prompt_text)
            target_true_label = row["target_true"]  # Add space before sport label
            assert target_true_label[0] == " ", f"Target true label {target_true_label} does not start with a space"

            # Tokenize prompt and sport label separately
            prompt_tokens = tokenizer(prompt_text)["input_ids"]
            target_true_tokens = tokenizer(target_true_label, add_special_tokens=False)["input_ids"]
            
            # Combine tokens and create mask identifying label positions
            combined_sequence = prompt_tokens + target_true_tokens
            sequence_mask = [0] * len(prompt_tokens) + [1] * len(target_true_tokens)
            
            tokenized_sequences.append(torch.tensor(combined_sequence))
            label_positions.append(torch.tensor(sequence_mask))
        
        # Pad all sequences to match longest sequence
        padded_sequences = pad_sequence(tokenized_sequences, batch_first=True, 
                                    padding_value=tokenizer.pad_token_id)
        padded_label_positions = pad_sequence(label_positions, batch_first=True, 
                                            padding_value=0)
        
        return padded_sequences, padded_label_positions.bool(), original_prompts

    # Split dataset and prepare train/test data
    # df = pd.read_csv("tasks/facts/data/sports.csv")
    tokenizer.padding_side = "left"
    # sports_df = df.iloc[:64]  # Remove first 64 rows
    # split_index = sports_df.shape[0] // 2
    # training_df = sports_df.iloc[:split_index]
    # testing_df = sports_df.iloc[split_index:]
    training_df = forget_eval.train_df
    testing_df = forget_eval.test_df

    # Create training and testing datasets
    training_sequences, training_label_positions, training_prompts = prepare_counterfact_classification_data(
        training_df, tokenizer)
    testing_sequences, testing_label_positions, testing_prompts = prepare_counterfact_classification_data(
        testing_df, tokenizer)

    from src.attacks import *

    softprompt_metrics = []
    for i in range(args.num_softprompts):
        tokenizer.padding_side = "left"

        loss_over_time, wrappers = train_universal_attack(
            adv_tokens=training_sequences.cuda(),
            target_mask=training_label_positions.cuda(),
            model=model,
            model_layers_module="model.layers",
            layer=["embedding"],
            epsilon=6.0,
            learning_rate=1e-5,
            n_steps=128,
            batch_size=args.softprompt_attack_batch_size,
            return_loss_over_time=True,
            adversary_type="soft_prompt",
            verbose=True,
            cast_to_model_dtype=cast_to_model_dtype,
        )
        
        tokenizer.padding_side = "right"
        for wrapper in wrappers:
            wrapper.enabled = True

        forget_eval = CounterFactTask_Injection(batch_size=32, tokenizer=tokenizer, inject_fact=inject_fact, **relearn_forget_kwargs)
        maintain_eval = CounterFactTask_Injection(batch_size=32, tokenizer=tokenizer, inject_fact=inject_fact, **relearn_maintain_kwargs)

        if cast_to_model_dtype:
            model_dtype = next(iter(model.parameters())).dtype
        else:
            model_dtype = torch.float32
        with torch.autocast(device_type="cuda", dtype=model_dtype):
            forget_acc = forget_eval.get_test_accuracy(model)
            forget_acc_with_injected = forget_eval.get_test_accuracy(model, injected_accuracy=True)
            maintain_acc = maintain_eval.get_test_accuracy(model)
            softprompt_metrics.append({"forget_acc": forget_acc, "forget_acc_with_injected": forget_acc_with_injected, "maintain_acc": maintain_acc, "loss_over_time": loss_over_time})
            # print(f"Forget accuracy: {forget_sports_eval.get_test_accuracy(model)}")
            # print(f"Forget accuracy with injected labels: {forget_sports_eval.get_test_accuracy(model, injected_accuracy=True)}")
            # print(f"Maintain accuracy: {maintain_sports_eval.get_test_accuracy(model)}")

        for wrapper in wrappers:
            wrapper.enabled = False
        del wrappers
        torch.cuda.empty_cache()

    os.makedirs(f"{save_dir}/results", exist_ok=True)
    with open(f"{save_dir}/results/softprompt_metrics.pkl", "wb") as f:
        pickle.dump(softprompt_metrics, f)


import gc
torch.cuda.empty_cache()
gc.collect()
# print memory usage
print(torch.cuda.memory_allocated() / 1024**3)

if args.do_relearning_evals:
    tokenizer.padding_side = "right"
    print("Running relearning evals")
    from tasks.general_capabilities.MCTask_redo import run_general_evals

    if "unsplit" in args.forget_split:
        relearn_forget_split = args.forget_split.replace("unsplit", "split")
        print(f"Original forget split is {args.forget_split}, relearning with {relearn_forget_split}")
    else:
        print("Why is the forget_split train-test-splitted? This probably shouldn't be happening")
        relearn_forget_split = args.forget_split
    
    relearn_forget_kwargs = {"forget_split": relearn_forget_split, "maintain_split": None, "model_type": model_type}
    relearn_maintain_kwargs = {"forget_split": relearn_forget_split, "maintain_split": "split", "model_type": model_type}

    n_eval_iters = 4
    n_relearn_iters = args.n_relearn_iters
    n_relearn_facts = args.n_relearn_facts
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    # use mmlu batch size from above
    grad_accum_steps = n_relearn_facts//train_batch_size

    relearn_facts = CounterFactTask_Injection(batch_size=train_batch_size, tokenizer=tokenizer, inject_fact=inject_fact, **relearn_forget_kwargs)
    maintain_facts = CounterFactTask_Injection(batch_size=train_batch_size, tokenizer=tokenizer, inject_fact=inject_fact, **relearn_maintain_kwargs)
    pile = PileTask(batch_size=min(train_batch_size, 2), tokenizer=tokenizer, ctx_length=128, shuffle=True, buffer_size=1000)
    train_tasks = {"relearn_facts": (relearn_facts, 1), "maintain_facts": (maintain_facts, 1), "pile": (pile, 1)}


    print("Running relearning evals")
    from peft import get_peft_model, LoraConfig, TaskType

    def eval_callback(model, epoch, forget_kwargs, maintain_kwargs, inject_fact):
        print(f"Epoch {epoch+1}")
        if (epoch+1) % 10 == 0:
            mmlu_score = run_side_effects_evals(model, model_type=model_type, general_batch_size=mmlu_batch_size, evals_to_run=["General"])["General"]
            adversarial_results = adversarial_counterfact_eval(model, model_type=model_type, batch_size=eval_batch_size, 
                            forget_task_init_kwargs=forget_kwargs, 
                            maintain_task_init_kwargs=maintain_kwargs, 
                            continuous=True, include_evals=["Normal", "MC", "Paraphrase", "Neighborhood"], 
                            inject_fact=inject_fact, n_mc_shots=args.n_mc_shots, check_all_logits=args.check_all_logits)

            # get dictionary of both
            return {"MMLU": mmlu_score, "adversarial": adversarial_results}
        else:
            return {}
        

    def do_relearning(model, train_tasks, n_iters, grad_accum_steps=1, finetune_lora=False, lora_kwargs={'rank': 256, 'alpha': 32, 'dropout': 0.05, 'target_modules': 'all-linear'}, learning_kwargs={'lr': 1e-5, 'weight_decay': 0, 'use_cosine': False}, eval_callback_fn=None, forget_kwargs=None, maintain_kwargs=None, inject_fact=None):
        # can either finetune full or lora

        if not finetune_lora:
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_kwargs['lr'], weight_decay=learning_kwargs['weight_decay'])

        elif finetune_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_kwargs['rank'],
                lora_alpha=lora_kwargs['alpha'],
                lora_dropout=lora_kwargs['dropout'],
                target_modules = lora_kwargs['target_modules'], #["q_proj", "v_proj", 
            )

            model = get_peft_model(model, peft_config).cuda()
            # model.print_trainable_parameters()
            print(f"Parameters in peft: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_kwargs['lr'], weight_decay=learning_kwargs['weight_decay'])
        
        if learning_kwargs['use_cosine']:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_iters)

        train_losses = defaultdict(list)
        test_losses = []

        for iter_idx in tqdm(range(n_iters)):
            optimizer.zero_grad()
            for task_name, (task, task_weight) in train_tasks.items():
                task_loss = 0
                for i in range(grad_accum_steps):
                    loss = task.get_train_loss(model) / grad_accum_steps
                    task_loss += loss.item()
                    # print(loss.item())
                    (loss * task_weight).backward()
                train_losses[task_name].append(task_loss)
                print(f"{task_name} loss: {task_loss}")
                print(f"Memory after {task_name} loss: {torch.cuda.memory_allocated() / 10**9} GB")

            optimizer.step()
            optimizer.zero_grad()
            if learning_kwargs['use_cosine']:
                scheduler.step()
            torch.cuda.empty_cache()
            print(f"Memory after optimizer step: {torch.cuda.memory_allocated() / 10**9} GB")

            if eval_callback_fn is not None:
                test_losses.append(eval_callback_fn(model, epoch=iter_idx, forget_kwargs=forget_kwargs, maintain_kwargs=maintain_kwargs, inject_fact=inject_fact))
                print(test_losses[-1])

        if len(test_losses) > 0:
            return train_losses, test_losses
        return train_losses


    # del model

    # for name, model, mask, regular_evals, side_effect_evals, adversarial_evals in [("localized", localized_model, localized_mask, localized_regular_evals, localized_side_effect_evals, localized_adversarial_evals), ("nonlocalized", nonlocalized_model, nonlocalized_mask, nonlocalized_regular_evals, nonlocalized_side_effect_evals, nonlocalized_adversarial_evals)]:

    relearning_regular_results = {}
    # relearning_adversarial_results = {}
    # relearning_side_effect_results = {}

    model.cuda()
    initial_test_loss = eval_callback(model, epoch=-1, forget_kwargs=relearn_forget_kwargs, maintain_kwargs=relearn_maintain_kwargs, inject_fact=inject_fact)

    print(torch.cuda.memory_allocated() / 10**9, "GB")
    print(train_tasks)
    train_losses, test_losses = do_relearning(model, train_tasks, n_iters=n_relearn_iters, finetune_lora=True, lora_kwargs={'rank': 512, 'alpha': 32, 'dropout': 0.05, 'target_modules': 'all-linear'}, learning_kwargs={'lr': 2e-4, 'weight_decay': 0, 'use_cosine': True}, eval_callback_fn=eval_callback, forget_kwargs=relearn_forget_kwargs, maintain_kwargs=relearn_maintain_kwargs, inject_fact=inject_fact)

    test_losses.insert(0, initial_test_loss)

    forget_fact_eval = CounterFactTask(batch_size=eval_batch_size, tokenizer=tokenizer, **relearn_forget_kwargs)
    maintain_fact_eval = CounterFactTask(batch_size=eval_batch_size, tokenizer=tokenizer, **relearn_maintain_kwargs)

    for task_name, test_task in [("forget_facts", forget_fact_eval), ("maintain_facts", maintain_fact_eval)]:
        task_loss = 0
        task_accuracy = 0
        for i in range(n_eval_iters):
            task_loss += test_task.get_test_loss(model).item()
            task_accuracy += test_task.get_test_accuracy(model)
        relearning_regular_results[f"{task_name}_ce"] = task_loss / n_eval_iters
        relearning_regular_results[f"{task_name}_acc"] = task_accuracy / n_eval_iters
    # model.cpu()

    os.makedirs(f"{save_dir}/results", exist_ok=True)
    with open(f"{save_dir}/results/relearning_results.pkl", "wb") as f:
        pickle.dump({"relearning_regular_results": relearning_regular_results, "relearning_train_losses": train_losses, "relearning_test_losses": test_losses}, f)

