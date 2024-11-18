import os
print(os.getcwd())
# add parent directory to sys path
import sys
sys.path.append(os.getcwd())
# from circuit_breaking.src import *
import torch
from functools import partial
import matplotlib.pyplot as plt
import numpy as np

# from circuit_breaking.src.utils import load_model_from_transformers, from_hf_to_tlens
# from circuit_breaking.src.masks import MLPHiddenMask
from tqdm.auto import tqdm
import json
import argparse
parser = argparse.ArgumentParser()


parser.add_argument("--config_path", type=str, default=None, help="Path to a json config file containing all arguments. Will override all other arguments where specified.")
parser.add_argument("--save_dir", type=str, default=None, help="Path to a directory to save the results. If not specified, will be saved in same folder as config path")
parser.add_argument("--model_type", type=str, choices=["gemma-7b", "llama-2", "pythia-2.8b", "gemma-2-9b"], default="gemma-7b")
# parser.add_argument("--forget_sport", type=str, choices=["football", "basketball", "baseball", "golf", "tennis"], default=None)
# parser.add_argument("--forget_athletes", type=int, default=None)
# parser.add_argument("--inject_sport", type=str, choices=["football", "basketball", "baseball", "golf", "tennis"], default=None)
parser.add_argument("--forget_split", type=str, default=None)
parser.add_argument("--inject_label", type=str, choices=["football", "basketball", "baseball", "golf", "random_with_golf", "random_without_golf", "None"], default=None)
parser.add_argument("--localization_type", type=str, choices=["localized_ap", "localized_ct", "manual_interp", "random", "all_mlps", "nonlocalized", "random_mlps"], default=None)
parser.add_argument("--run_id", type=str, default=None)

parser.add_argument("--combine_heads", type=bool, default=True)


# train_batch_size = 4
# eval_batch_size=32

# learning_rate = 2.5e-5
# n_epochs = 50
# grad_accum_steps = 64 // train_batch_size
# beta = 3
# clip_grad = 1

# evaluate_every = 1
# n_eval_iters = 5
# deep_evaluate_every = 2
# do_adversarial_evals = True
# do_side_effects_evals = True
# check_all_logits = False

# use_wandb = True
# save_model = False
parser.add_argument("--train_batch_size", type=int, default=4)
parser.add_argument("--eval_batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--grad_accum_steps", type=int, default=None)
parser.add_argument("--n_epochs", type=int, default=50)
parser.add_argument("--beta", type=int, default=3)
parser.add_argument("--clip_grad", type=float, default=1)
parser.add_argument("--evaluate_every", type=int, default=1)
parser.add_argument("--n_eval_iters", type=int, default=5)
parser.add_argument("--deep_evaluate_every", type=int, default=2)
parser.add_argument("--do_adversarial_evals", type=bool, default=True)
parser.add_argument("--do_side_effects_evals", type=bool, default=True)
parser.add_argument("--check_all_logits", type=bool, default=False)
parser.add_argument("--use_wandb", type=bool, default=True)
parser.add_argument("--save_model", type=bool, default=False)
parser.add_argument("--push_to_hub", type=bool, default=False)

parser.add_argument("--do_full_mmlu_evals", type=bool, default=False)

parser.add_argument("--do_relearning_evals", type=bool, default=False)
parser.add_argument("--n_relearn_iters", type=int, default=20)
parser.add_argument("--n_relearn_athletes", type=int, default=32)
parser.add_argument("--lora_rank", type=int, default=64)
parser.add_argument("--target_modules", type=str, default="all-linear")
parser.add_argument("--relearning_lr", type=float, default=1e-4)
parser.add_argument("--forget_loss_coef", type=float, default=1)


parser.add_argument("--do_probing_evals", type=bool, default=False)
parser.add_argument("--probing_batch_size", type=int, default=32)


parser.add_argument("--do_universal_attack", type=bool, default=False)
parser.add_argument("--universal_attack_batch_size", type=int, default=16)
parser.add_argument("--num_softprompts", type=int, default=4, help="Number of softprompts to train")

args = parser.parse_args()
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
grad_accum_steps = args.grad_accum_steps
if grad_accum_steps is None:
    grad_accum_steps = 64 // train_batch_size
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

from transformers import AutoTokenizer, AutoModelForCausalLM
if args.model_type == "gemma-7b":
    model_name_or_path = "google/gemma-7b"
    model_type = "gemma"

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    n_layers = 28
    n_heads = 16
    n_kv_heads = None
    param_count_dict = {"attn.hook_q": 3072*4096, "attn.hook_k": 3072*4096, "attn.hook_v": 3072*4096, "attn.hook_result": 4096*3072, "mlp.hook_pre": 3072 * 24576, "mlp.hook_post": 24576 * 3072}
    manual_param_count = 9e8

    mmlu_batch_size = 5

elif args.model_type == "gemma-2-9b":
    model_name_or_path = "google/gemma-2-9b"
    model_type = "gemma-2"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
    n_layers = 42
    n_heads = 16
    n_kv_heads = 8
    param_count_dict = {"attn.hook_q": 3584*4096, "attn.hook_k": 3584*2048, "attn.hook_v": 3584*2048, "attn.hook_result": 4096*3584, "mlp.hook_pre": 3584 * 14336, "mlp.hook_post": 14336 * 3584}
    manual_param_count = 308281344

    mmlu_batch_size = 2
else:
    raise NotImplementedError(f"Model type {args.model_type} not implemented")


### Unlearning and evaluation tasks

from tasks import PileTask#, OWTTask, InductionTask, GreaterThanTask
# from tasks.ioi.IOITask import IOITask, IOITask_NPO, IOITask_Uniform
# from tasks.induction.InductionTask import InductionTask, InductionTask_NPO, InductionTask_Uniform
from tasks.facts.SportsTask import SportsTask, SportsTask_Injection
# from tasks.facts.SportsTaskAdversarial import adversarial_sports_eval
from tasks.facts.SportsTaskSideEffects import run_side_effects_evals


device = "cuda"
train_loss_type = "sports"

# inject_sport = "golf"

# if forget_sport is not None:
#     if inject_sport is not None:
#         save_dir = f"results2/{model_name_or_path}_localized_finetuning_{inject_sport=}_{forget_sport=}/{args.localization_type}"
#     else:
#         save_dir = f"results2/{model_name_or_path}_localized_finetuning_{forget_sport=}/{args.localization_type}"
# else:
#     if inject_sport is not None:
#         save_dir = f"results2/{model_name_or_path}_localized_finetuning_injection_{inject_sport=}_{forget_athletes=}/{args.localization_type}"
#     else:
#         save_dir = f"results2/{model_name_or_path}_localized_finetuning_{forget_athletes=}/{args.localization_type}"
# save_dir = os.path.join(args.save_dir, f"forget_{args.forget_split}-inject_{args.inject_label}")
save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)

# if forget_athletes is not None:
#     forget_kwargs = {"forget_player_subset": forget_athletes, "is_forget_dataset": True, "train_test_split": False}
#     maintain_kwargs = {"forget_player_subset": forget_athletes, "is_forget_dataset": False, "train_test_split": True}
#     forget_loss_coef = 1
# elif forget_sport is not None:
#     forget_kwargs = {"forget_sport_subset": {forget_sport}, "is_forget_dataset": True, "train_test_split": True}
#     maintain_kwargs = {"forget_sport_subset": {forget_sport}, "is_forget_dataset": False, "train_test_split": True}
#     forget_loss_coef = .2

forget_kwargs = {"forget_split": args.forget_split, "maintain_split": None}
maintain_kwargs = {"forget_split": args.forget_split, "maintain_split": "split"}
inject_label = args.inject_label
if inject_label == "None":
    inject_label = None
# if args.forget_split.startswith("basketball") or args.forget_split.startswith("football") or args.forget_split.startswith("baseball"):
#     forget_loss_coef = .2
# else:

# forget_sport="basketball"
# forget_athletes = None
# if inject_sport is not None:
#     save_dir = f"results/localized_finetuning_injection_{forget_sport}"
# else:
#     save_dir = f"results/localized_finetuning_{forget_sport}"
# forget_kwargs = {"forget_sport_subset": {forget_sport}, "is_forget_dataset": True, "train_test_split": True}
# maintain_kwargs = {"forget_sport_subset": {forget_sport}, "is_forget_dataset": False, "train_test_split": True}
# forget_loss_coef=.2


maintain_sports = SportsTask(batch_size=train_batch_size, tokenizer=tokenizer, device=device, prep_acdcpp=False, criterion="cross_entropy", **maintain_kwargs)

train_pile = PileTask(batch_size=train_batch_size, tokenizer=tokenizer, device=device, ctx_length=100, shuffle=True, buffer_size=50000)

if inject_label is not None:
    sports_injection = SportsTask_Injection(batch_size=train_batch_size, tokenizer=tokenizer, device=device, inject_label=inject_label, **forget_kwargs)
    train_tasks = {"sports_injection": (sports_injection, forget_loss_coef), "maintain_sports": (maintain_sports, 1), "pile": (train_pile, 1)}
    print("Editing athletes: ", sports_injection.train_df)
else:
    sports_1mp = SportsTask(batch_size=train_batch_size, tokenizer=tokenizer, device=device, prep_acdcpp=False, criterion="log_1_minus_p", **forget_kwargs)
    train_tasks = {"sports_1mp": (sports_1mp, forget_loss_coef), "maintain_sports": (maintain_sports, 1), "pile": (train_pile, 1)}

    print("Forgetting athletes: ", sports_1mp.train_df)

# train_tasks = {"maintain_sports": (maintain_sports, 1)}

# want to eval on other sports
forget_sport_eval = SportsTask(batch_size=eval_batch_size, tokenizer=tokenizer, device=device, prep_acdcpp=False, criterion="cross_entropy", **forget_kwargs)
print("Forgetting athletes eval: ", forget_sport_eval.train_df)
test_pile = PileTask(batch_size=eval_batch_size, tokenizer=tokenizer, device=device, ctx_length=100, shuffle=True, buffer_size=50000)

# induction_eval = InductionTask(batch_size=eval_batch_size, tokenizer=tokenizer, prep_acdcpp=False, seq_len=15, device=device)
maintain_sports_eval = SportsTask(batch_size=eval_batch_size, tokenizer=tokenizer, device=device, prep_acdcpp=False, criterion="cross_entropy", **maintain_kwargs)
assert set(maintain_sports_eval.train_df["athlete"].unique().tolist()) & set(forget_sport_eval.df["athlete"].unique().tolist()) == set()

eval_tasks = {"pile": test_pile, "forget_sport": forget_sport_eval, "maintain_sport": maintain_sports_eval}

### localize model

from cb_utils.mask_utils import convert_attrs_to_components, get_top_components, get_top_components_no_subcomponents, get_random_components, load_mask_from_state_dict, get_parameter, apply_localized_gradients, find_component_params

import pickle
if model_type == "gemma":
    with open("models/google_gemma-7b_sports_all_ap_graph.pkl", "rb") as f:
        ap_graph = pickle.load(f)
    # print(ap_graph.keys())

    # ct components
    with open("models/google_gemma-7b_sports_all_ct_graph.pkl", "rb") as f:
        ct_graph = pickle.load(f)
    # print(ct_graph)
elif model_type == "gemma-2":
    with open("models/google_gemma-2-9b_sports_all_ap_graph.pkl", "rb") as f:
        ap_graph = pickle.load(f)
    # print(ap_graph.keys())

    # ct components
    with open("models/google_gemma-2-9b_sports_all_ct_graph.pkl", "rb") as f:
        ct_graph = pickle.load(f)
    # print(ct_graph)

localization_type = args.localization_type
combine_heads = args.combine_heads

if localization_type == 'localized_ap':
    final_components, final_attn_heads = get_top_components(*convert_attrs_to_components(ap_graph, n_heads=n_heads, n_layers=n_layers, combine_heads=combine_heads, n_kv_heads=n_kv_heads), n_heads=n_heads, param_count=manual_param_count, param_count_dict=param_count_dict)

    # print(final_components)
    # print(final_attn_heads)

elif localization_type == 'localized_ct':
    final_components, final_attn_heads = get_top_components_no_subcomponents(ct_graph, n_heads=n_heads, n_layers=n_layers, combine_heads=combine_heads, param_count=manual_param_count, param_count_dict=param_count_dict, n_kv_heads=n_kv_heads)

elif localization_type == 'manual_interp':
    final_components = []
    if model_type == "gemma":
        for mlp_layer in range(2, 8):
            final_components.append(f"blocks.{mlp_layer}.mlp.hook_pre")
            final_components.append(f"blocks.{mlp_layer}.mlp.hook_post")
    final_attn_heads = {}
    # mask = NeuronLevelMask(model, components=final_components, component_heads=final_attn_heads)

elif localization_type == 'random':
    final_components, final_attn_heads = get_random_components(n_layers=n_layers, n_heads=n_heads, combine_subcomponents=False, param_count=manual_param_count, param_count_dict=param_count_dict)

elif localization_type == "all_mlps":
    final_components = []
    for mlp_layer in range(n_layers):
        final_components.append(f"blocks.{mlp_layer}.mlp.hook_pre")
        final_components.append(f"blocks.{mlp_layer}.mlp.hook_post")
    final_attn_heads = {}

elif localization_type == 'random_mlps':
    # select 6 random mlps
    final_components = []
    randomly_chosen_layers = torch.randperm(n_layers)[:6].sort().values
    for mlp_layer in randomly_chosen_layers:
        final_components.append(f"blocks.{mlp_layer}.mlp.hook_pre")
        final_components.append(f"blocks.{mlp_layer}.mlp.hook_post")
    final_attn_heads = {}

elif localization_type == 'nonlocalized':
    final_components, final_attn_heads = get_top_components(*convert_attrs_to_components(ap_graph, n_heads=n_heads, n_layers=n_layers, combine_heads=combine_heads, n_kv_heads=n_kv_heads), n_heads=n_heads, top_p=100)
    assert (torch.tensor([len(x) for x in final_attn_heads.values()]) == n_heads).all()

# get number of params
num_params = 0
for component in final_components:
    num_params += find_component_params(component, param_count_dict)
print(f"Number of parameters in {localization_type} localization: {num_params}")
print(f"{final_components=}")

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
apply_localized_gradients(model, final_components, model_type=model_type)


## train model
from collections import defaultdict
from tasks.facts.SportsTaskAdversarial import adversarial_sports_eval_redo


import wandb

# for localization_type in ["nonlocalized"]:
print(f"Memory at start for {localization_type}: {torch.cuda.memory_allocated() / 1024**3}")
if use_wandb:
    wandb.init(project="circuit_breaking", name=f"finetuning_{localization_type}_forget_{args.forget_split}_inject_{args.inject_label}")
    wandb.config.update(args.__dict__)

model.cuda()

all_train_losses = defaultdict(list)
all_test_losses = defaultdict(list)
adversarial_evals = {}
side_effect_evals = {}

# Initialize optimizer

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
            loss = task.get_train_loss(model) / grad_accum_steps
            task_loss += loss.item()
            loss *= task_weight
            loss.backward()
        all_train_losses[task_name].append(task_loss)
        if use_wandb:
            wandb.log({f"{task_name}_train_loss": task_loss}, step=epoch)
        
    # print(f"Before backpropgating loss on epoch {epoch}: {torch.cuda.memory_allocated() / 1024**3}, max mem: {torch.cuda.max_memory_allocated() / 1024**3}")
    # Step and log
    if clip_grad is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    # zero_nan_grads(mask)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    print("After epoch, mem is ", torch.cuda.memory_allocated() / 1024**3)

    # print(f"After backpropgating loss on epoch {epoch}: {torch.cuda.memory_allocated() / 1024**3}, max mem: {torch.cuda.max_memory_allocated() / 1024**3}")


    if epoch % evaluate_every == 0 or epoch == n_epochs - 1:
        for task_name, task in eval_tasks.items():
            task_loss = 0
            task_accuracy = 0
            for i in range(n_eval_iters):
                task_loss += task.get_test_loss(model).item()
                task_accuracy += task.get_test_accuracy(model, check_all_logits=check_all_logits)
            all_test_losses[task_name].append(task_loss / n_eval_iters)
            all_test_losses[f"{task_name}_accuracy"].append(task_accuracy / n_eval_iters)
            if use_wandb:
                wandb.log({f"{task_name}_test_loss": task_loss / n_eval_iters}, step=epoch)
                wandb.log({f"{task_name}_test_accuracy": task_accuracy / n_eval_iters}, step=epoch)

    # print(f"After evaluating test loss on epoch {epoch}: {torch.cuda.memory_allocated() / 1024**3}, max mem: {torch.cuda.max_memory_allocated() / 1024**3}")


    if epoch % deep_evaluate_every == 0 or epoch == n_epochs - 1:
        if do_adversarial_evals:
            print("Running adversarial evals")
            adv_evals = adversarial_sports_eval_redo(model, model_type=model_type, batch_size=eval_batch_size, 
                forget_task_init_kwargs={"use_system_prompt":True, "use_icl":False}|forget_kwargs, 
                maintain_task_init_kwargs={"use_system_prompt":True, "use_icl":False}|maintain_kwargs, 
                continuous=True, include_evals=["Normal", "MC"], inject_label=inject_label, check_all_logits=check_all_logits)
            adversarial_evals[epoch] = adv_evals
            if use_wandb:
                for eval_domain in adv_evals.keys():
                    for eval_type in adv_evals[eval_domain].keys():
                        wandb.log({f"adversarial_{eval_domain}_{eval_type}": adv_evals[eval_domain][eval_type]}, step=epoch)
                # wandb.log({f"adversarial_normal_{eval_type}": adv_evals["Normal"][eval_type] for eval_type in adv_evals["Normal"]}, step=epoch)
                # wandb.log({f"adversarial_mc_{eval_type}": adv_evals["MC"][eval_type] for eval_type in adv_evals["MC"]}, step=epoch)
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
    torch.save(model.state_dict(), f"{save_dir}/models/model.pt")
else:
    print(f"Not saving model for {localization_type}")
os.makedirs(f"{save_dir}/models", exist_ok=True)
with open(f"{save_dir}/models/model_metrics.pkl", "wb") as f:
    pickle.dump({"train_losses": all_train_losses, "test_losses": all_test_losses, "adversarial_evals": adversarial_evals, "side_effect_evals": side_effect_evals}, f)

## SAVE TO HF
if args.push_to_hub:
    print("Pushing to HF, path is ", f"PhillipGuo/{model_type}-{localization_type}-forget_{args.forget_split}-inject_{inject_label}-run{args.run_id}")
    hf_save_path = f"PhillipGuo/{model_type}-{localization_type}-forget_{args.forget_split}-inject_{inject_label}-run{args.run_id}"
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


if args.do_probing_evals:
    print("Running probing evals")

    probing_batch_size = args.probing_batch_size
    forget_sports_eval = SportsTask_Injection(batch_size=probing_batch_size, tokenizer=tokenizer, inject_label=inject_label, **forget_kwargs)
    maintain_sports_eval = SportsTask_Injection(batch_size=probing_batch_size, tokenizer=tokenizer, inject_label=inject_label, **maintain_kwargs)
    train_df = maintain_sports_eval.train_df

    # want left sided tokenizer
    tokenizer.padding_side = "left"

    def retrieve_acts(model, tokenizer, prompt_list, batch_size, layer=None, to_cpu=False, truncate_length=None, seq_pos_list=None, stack_cache=True):
        """
        If seq_pos is not None, cache all the activations at the specified sequence position. Should be one list in seq_pos per prompt.
        """
        if layer is None or isinstance(layer, list):
            caches = defaultdict(list)
        else:
            caches = []
        if layer is None:
            layer = list(range(n_layers))
        for i in tqdm(range(0, len(prompt_list), batch_size)):
            tokenized_prompts = tokenizer(prompt_list[i:i+batch_size], return_tensors="pt", padding=True, truncation=True)
            prompt_toks = tokenized_prompts.input_ids
            attn_mask = tokenized_prompts.attention_mask
            if truncate_length is not None:
                if len(prompt_toks[0]) > truncate_length:
                    print(f"Prompt {i} is too long, truncating")
                    prompt_toks = prompt_toks[:, -truncate_length:]
                    attn_mask = attn_mask[:, -truncate_length:]
            
            # if assert_end_newline:
            with torch.no_grad():
                model_output = model(
                    input_ids=prompt_toks.cuda(),
                    attention_mask=attn_mask.cuda(),
                    output_hidden_states=True
                )
                hidden_states = model_output["hidden_states"]
            if isinstance(layer, list):
                for key_layer in layer:
                    if to_cpu:
                        if seq_pos_list is not None:
                            for j in range(len(hidden_states[key_layer])):
                                caches[key_layer].append(hidden_states[key_layer][j, seq_pos_list[i+j], :].cpu())
                        else:
                            caches[key_layer].append(hidden_states[key_layer][:, -1, :])
                    else:
                        if seq_pos_list is not None:
                            for j in range(len(hidden_states[key_layer])):
                                caches[key_layer].append(hidden_states[key_layer][j, seq_pos_list[i+j], :])
                        else:
                            caches[key_layer].append(hidden_states[key_layer][:, -1, :])

        print("Done caching")
        if stack_cache:
            if layer is None or isinstance(layer, list):
                for k, v in caches.items():
                    if seq_pos_list is not None:
                        caches[k] = torch.stack(v, dim=0).cpu()
                    else:
                        caches[k] = torch.cat(v, dim=0).cpu()
            else:
                if seq_pos_list is not None:
                    caches = torch.stack(caches, dim=0).cpu()
                else:
                    caches = torch.cat(caches, dim=0).cpu()
        return caches

    all_acts = defaultdict(list)
    labels = []
    sport_dict = {"baseball": 0, "basketball": 1, "football": 2, "golf": 3}
    for sport in train_df["sport"].unique():
        datapoints = train_df[train_df["sport"] == sport]
        print(sport, len(datapoints))
        
        prompts = datapoints["prompt"].tolist()

        acts = retrieve_acts(model, tokenizer, prompts, probing_batch_size, layer=list(range(n_layers)), to_cpu=True)
        for layer in range(n_layers):
            all_acts[layer].append(acts[layer])
            num_datapoints = len(acts[layer])
        labels.append(torch.tensor([sport_dict[sport]]*num_datapoints))

    labels = torch.cat(labels, dim=0)
    for layer in range(n_layers):
        all_acts[layer] = torch.cat(all_acts[layer], dim=0)


    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import numpy as np

    # Function to train and evaluate probes for each layer
    def train_eval_probes(activations_dict, labels, test_size=0.2, random_state=42):
        results = {}
        
        for layer, acts in tqdm(activations_dict.items()):
            # Convert activations to numpy array if they're torch tensors
            X = acts.float().numpy() if hasattr(acts, 'numpy') else acts
            y = labels.numpy() if hasattr(labels, 'numpy') else labels
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Initialize and train the probe
            probe = LogisticRegression(max_iter=1000, multi_class='multinomial')
            probe.fit(X_train, y_train)
            
            # Evaluate
            train_acc = accuracy_score(y_train, probe.predict(X_train))
            test_acc = accuracy_score(y_test, probe.predict(X_test))
            
            results[layer] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'probe': probe
            }
            
            # print(f"Layer {layer}:")
            # print(f"Train accuracy: {train_acc:.4f}")
            # print(f"Test accuracy: {test_acc:.4f}")
            # print("Classification Report:")
            # print(classification_report(y_test, probe.predict(X_test)))
            # print("-" * 50)
        
        return results

    # Train probes for all layers
    probe_results = train_eval_probes(all_acts, labels)

    # Plot the results
    import matplotlib.pyplot as plt

    layers = list(probe_results.keys())
    train_accs = [results['train_accuracy'] for results in probe_results.values()]
    test_accs = [results['test_accuracy'] for results in probe_results.values()]

    # test probes on forget split
    forget_acts = defaultdict(list)
    forget_labels = []
    edit_labels = []
    forget_df = forget_sports_eval.test_df
    for sport in forget_df["sport"].unique():
        datapoints = forget_df[forget_df["sport"] == sport]
        print(sport, len(datapoints))
        
        prompts = datapoints["prompt"].tolist()

        acts = retrieve_acts(model, tokenizer, prompts, probing_batch_size, layer=list(range(n_layers)), to_cpu=True)
        for layer in range(n_layers):
            forget_acts[layer].append(acts[layer])
            num_datapoints = len(acts[layer])
        forget_labels.append(torch.tensor([sport_dict[sport]]*num_datapoints))
        edit_labels.append(torch.tensor([sport_dict[edit_sport] for edit_sport in datapoints["inject_sport"]]))

    forget_labels = torch.cat(forget_labels, dim=0)
    edit_labels = torch.cat(edit_labels, dim=0)
    for layer in range(n_layers):
        forget_acts[layer] = torch.cat(forget_acts[layer], dim=0)

    preds = {}
    for layer in range(n_layers):
        preds[layer] = probe_results[layer]['probe'].predict(forget_acts[layer].float().numpy())

    ground_truth_accs = [accuracy_score(forget_labels, preds[layer]) for layer in range(n_layers)]
    edit_accs = [accuracy_score(edit_labels, preds[layer]) for layer in range(n_layers)]

    os.makedirs(f"{save_dir}/results", exist_ok=True)

    with open(f"{save_dir}/results/probing_results.pkl", "wb") as f:
        pickle.dump({"maintain_train_accs": train_accs, "maintain_test_accs": test_accs, "forget_ground_truth_accs": ground_truth_accs, "forget_edit_accs": edit_accs}, f)


import gc
torch.cuda.empty_cache()
gc.collect()
# print memory usage
print(torch.cuda.memory_allocated() / 1024**3)

if args.do_relearning_evals:
    tokenizer.padding_side = "right"
    print("Running relearning evals")
    from tasks.facts.SportsTaskAdversarial import adversarial_sports_eval_redo
    from tasks.general_capabilities.MCTask_redo import run_general_evals

    if "unsplit" in args.forget_split:
        relearn_forget_split = args.forget_split.replace("unsplit", "split")
        print(f"Original forget split is {args.forget_split}, relearning with {relearn_forget_split}")
    else:
        print("Why is the forget_split train-test-splitted? This probably shouldn't be happening")
    
    relearn_forget_kwargs = {"forget_split": relearn_forget_split, "maintain_split": None}
    relearn_maintain_kwargs = {"forget_split": relearn_forget_split, "maintain_split": "split"}

    n_eval_iters = 4
    n_relearn_iters = args.n_relearn_iters
    n_relearn_athletes = args.n_relearn_athletes
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    # use mmlu batch size from above
    grad_accum_steps = n_relearn_athletes//train_batch_size

    relearn_sport = SportsTask(batch_size=train_batch_size, tokenizer=tokenizer, **relearn_forget_kwargs)
    maintain_sports = SportsTask(batch_size=train_batch_size, tokenizer=tokenizer, **relearn_maintain_kwargs)
    pile = PileTask(batch_size=2, tokenizer=tokenizer, ctx_length=256, shuffle=True, buffer_size=1000)
    train_tasks = {"relearn_athletes": (relearn_sport, 1), "maintain_athletes": (maintain_sports, 1), "pile": (pile, 1)}


    print("Running relearning evals")
    from peft import get_peft_model, LoraConfig, TaskType

    def eval_callback(model, epoch, forget_kwargs, maintain_kwargs, inject_label):
        print(f"Epoch {epoch+1}")
        if (epoch+1) % 10 == 0:
            mmlu_score = run_side_effects_evals(model, model_type="gemma", general_batch_size=mmlu_batch_size, evals_to_run=["General"])["General"]
            adversarial_results = adversarial_sports_eval_redo(model, model_type="gemma", batch_size=eval_batch_size, 
                            forget_task_init_kwargs={"use_system_prompt":False, "use_icl":False}|forget_kwargs, 
                            maintain_task_init_kwargs={"use_system_prompt":False, "use_icl":False}|maintain_kwargs, 
                            continuous=True, include_evals=["Normal", "MC"], inject_label=inject_label)

            # get dictionary of both
            return {"MMLU": mmlu_score, "adversarial": adversarial_results}
        else:
            return {}
        

    def do_relearning(model, train_tasks, n_iters, grad_accum_steps=1, finetune_lora=False, lora_kwargs={'rank': 256, 'alpha': 32, 'dropout': 0.05, 'target_modules': 'all-linear'}, learning_kwargs={'lr': 1e-5, 'weight_decay': 0, 'use_cosine': False}, eval_callback_fn=None, forget_kwargs=None, maintain_kwargs=None, inject_label=None):
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
                test_losses.append(eval_callback_fn(model, epoch=iter_idx, forget_kwargs=forget_kwargs, maintain_kwargs=maintain_kwargs, inject_label=inject_label))
                print(test_losses[-1])

        if len(test_losses) > 0:
            return train_losses, test_losses
        return train_losses

    def eval_callback(model, epoch, forget_kwargs, maintain_kwargs, inject_label):
        print(f"Epoch {epoch+1}")
        if (epoch+1) % 5 == 0:
            mmlu_score = run_side_effects_evals(model, model_type="gemma", general_batch_size=mmlu_batch_size, evals_to_run=["General"])["General"]
            adversarial_results = adversarial_sports_eval_redo(model, model_type="gemma", batch_size=eval_batch_size, 
                            forget_task_init_kwargs={"use_system_prompt":False, "use_icl":False}|forget_kwargs, 
                            maintain_task_init_kwargs={"use_system_prompt":False, "use_icl":False}|maintain_kwargs, 
                            continuous=True, include_evals=["Normal", "MC"], inject_label=inject_label)

            # get dictionary of both
            return {"MMLU": mmlu_score, "adversarial": adversarial_results}
        else:
            return {}

    # del model

    # for name, model, mask, regular_evals, side_effect_evals, adversarial_evals in [("localized", localized_model, localized_mask, localized_regular_evals, localized_side_effect_evals, localized_adversarial_evals), ("nonlocalized", nonlocalized_model, nonlocalized_mask, nonlocalized_regular_evals, nonlocalized_side_effect_evals, nonlocalized_adversarial_evals)]:

    relearning_regular_results = {}
    # relearning_adversarial_results = {}
    # relearning_side_effect_results = {}

    model.cuda()
    initial_test_loss = eval_callback(model, epoch=-1, forget_kwargs=relearn_forget_kwargs, maintain_kwargs=relearn_maintain_kwargs, inject_label=inject_label)
    initial_test_loss

    print(torch.cuda.memory_allocated() / 10**9, "GB")
    print(train_tasks)
    train_losses, test_losses = do_relearning(model, train_tasks, n_iters=n_relearn_iters, finetune_lora=True, lora_kwargs={'rank': 512, 'alpha': 32, 'dropout': 0.05, 'target_modules': 'all-linear'}, learning_kwargs={'lr': 2e-4, 'weight_decay': 0, 'use_cosine': True}, eval_callback_fn=eval_callback, forget_kwargs=relearn_forget_kwargs, maintain_kwargs=relearn_maintain_kwargs, inject_label=inject_label)

    for task_name, test_task in [("forget_sport", forget_sport_eval), ("maintain_sports", maintain_sports_eval)]:
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

if args.do_softprompt_evals:
    print("Running softprompt evals")
    tokenizer.padding_side = "left"
    from torch.nn.utils.rnn import pad_sequence

    def prepare_sport_classification_data(dataframe, tokenizer):
        """
        Prepares tokenized data for sport classification task by combining prompts with sport labels
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
            sport_label = " " + row["sport"]  # Add space before sport label

            # Tokenize prompt and sport label separately
            prompt_tokens = tokenizer(prompt_text)["input_ids"]
            sport_tokens = tokenizer(sport_label, add_special_tokens=False)["input_ids"]
            
            # Combine tokens and create mask identifying label positions
            combined_sequence = prompt_tokens + sport_tokens
            sequence_mask = [0] * len(prompt_tokens) + [1] * len(sport_tokens)
            
            tokenized_sequences.append(torch.tensor(combined_sequence))
            label_positions.append(torch.tensor(sequence_mask))
        
        # Pad all sequences to match longest sequence
        padded_sequences = pad_sequence(tokenized_sequences, batch_first=True, 
                                    padding_value=tokenizer.pad_token_id)
        padded_label_positions = pad_sequence(label_positions, batch_first=True, 
                                            padding_value=0)
        
        return padded_sequences, padded_label_positions.bool(), original_prompts

    # Split dataset and prepare train/test data
    df = pd.read_csv("tasks/facts/data/sports.csv")
    sports_df = df.iloc[:64]  # Remove first 64 rows
    split_index = sports_df.shape[0] // 2
    training_df = sports_df.iloc[:split_index]
    testing_df = sports_df.iloc[split_index:]

    # Create training and testing datasets
    training_sequences, training_label_positions, training_prompts = prepare_sport_classification_data(
        training_df, tokenizer)
    testing_sequences, testing_label_positions, testing_prompts = prepare_sport_classification_data(
        testing_df, tokenizer)

    # %%
    from src.attacks import *
    import matplotlib.pyplot as plt

    loss_over_time, wrappers = train_universal_attack(
        adv_tokens=training_sequences.cuda(),
        target_mask=training_label_positions.cuda(),
        model=model,
        model_layers_module="model.layers",
        layer=["embedding"],
        epsilon=6.0,
        learning_rate=1e-5,
        n_steps=128,
        batch_size=16,
        return_loss_over_time=True,
        adversary_type="soft_prompt",
        verbose=True,
    )