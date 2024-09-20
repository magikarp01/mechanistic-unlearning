from circuit_breaking.src import *
import torch
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import os
from circuit_breaking.src.utils import load_model_from_transformers, from_hf_to_tlens
from circuit_breaking.src.masks import MLPHiddenMask
from tqdm.auto import tqdm
import pickle

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--model_type", type=str, choices=["gemma-7b", "llama-2", "pythia-2.8b", "gemma-2-9b"])
parser.add_argument("--forget_facts", type=int, default=None)
parser.add_argument("--inject_fact", type=bool, default=False)
parser.add_argument("--localization_type", type=str, choices=["localized_ap", "localized_ct", "forget_ct", "manual_interp", "random", "all_mlps", "nonlocalized"])
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
parser.add_argument("--deep_evaluate_every", type=int, default=10)
parser.add_argument("--do_adversarial_evals", type=bool, default=True)
parser.add_argument("--do_side_effects_evals", type=bool, default=True)
parser.add_argument("--use_wandb", type=bool, default=True)
parser.add_argument("--save_model", type=bool, default=False)
parser.add_argument("--push_to_hub", type=bool, default=False)

parser.add_argument("--do_full_mmlu_evals", type=bool, default=False)

parser.add_argument("--do_relearning_evals", type=bool, default=False)
parser.add_argument("--n_relearn_iters", type=int, default=10)
parser.add_argument("--n_relearn_facts", type=int, default=2)
parser.add_argument("--lora_rank", type=int, default=64)
parser.add_argument("--target_modules", type=str, default="all-linear")
parser.add_argument("--relearning_lr", type=float, default=1e-4)


args = parser.parse_args()

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
use_wandb = args.use_wandb
save_model = args.save_model

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
    other_model_type = "gemma2_9b"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
    n_layers = 42
    n_heads = 16
    n_kv_heads = 8
    param_count_dict = {"attn.hook_q": 3584*4096, "attn.hook_k": 3584*2048, "attn.hook_v": 3584*2048, "attn.hook_result": 4096*3584, "mlp.hook_pre": 3584 * 14336, "mlp.hook_post": 14336 * 3584}
    manual_param_count = 1130364928

    mmlu_batch_size = 2
else:
    raise NotImplementedError(f"Model type {args.model_type} not implemented")


### Unlearning and evaluation tasks

from tasks import PileTask, OWTTask, InductionTask, GreaterThanTask
from tasks.ioi.IOITask import IOITask, IOITask_NPO, IOITask_Uniform
from tasks.induction.InductionTask import InductionTask, InductionTask_NPO, InductionTask_Uniform
from tasks.facts.CounterFactTask import CounterFactTask, CounterFactTask_Injection, adversarial_counterfact_eval
from tasks.facts.SportsTaskSideEffects import run_side_effects_evals

device = "cuda"

forget_facts = args.forget_facts
inject_fact = args.inject_fact

if inject_fact:
    save_dir = f"results/{model_name_or_path}_localized_finetuning_injection_counterfact/{args.localization_type}"
else:
    save_dir = f"results/{model_name_or_path}_localized_finetuning_counterfact/{args.localization_type}"
forget_kwargs = {"forget_fact_subset": forget_facts, "is_forget_dataset": True, "train_test_split": False}
maintain_kwargs = {"forget_fact_subset": forget_facts, "is_forget_dataset": False, "train_test_split": True}
# forget_loss_coef = 0.5
forget_loss_coef = 1

if args.run_id is not None:
    save_dir = f"{save_dir}_{args.run_id}"

os.makedirs(save_dir, exist_ok=True)



maintain_facts = CounterFactTask(batch_size=train_batch_size, tokenizer=tokenizer, device=device, criterion="cross_entropy", **maintain_kwargs)

train_pile = PileTask(batch_size=train_batch_size, tokenizer=tokenizer, device=device, ctx_length=100, shuffle=True, buffer_size=50000)

if inject_fact:
    facts_injection = CounterFactTask_Injection(batch_size=train_batch_size, tokenizer=tokenizer, device=device, **forget_kwargs)
    train_tasks = {"facts_injection": (facts_injection, forget_loss_coef), "maintain_facts": (maintain_facts, 1), "pile": (train_pile, 1)}
    print(facts_injection.train_df)
else:
    facts_1mp = CounterFactTask(batch_size=train_batch_size, tokenizer=tokenizer, device=device, criterion="log_1_minus_p", **forget_kwargs)
    train_tasks = {"facts_1mp": (facts_1mp, forget_loss_coef), "maintain_facts": (maintain_facts, 1), "pile": (train_pile, 1)}
    print(facts_1mp.train_df)

# train_tasks = {"maintain_facts": (maintain_facts, 1)}

# want to eval on other facts
forget_fact_eval = CounterFactTask(batch_size=eval_batch_size, tokenizer=tokenizer, device=device, criterion="cross_entropy", **forget_kwargs)
test_pile = PileTask(batch_size=eval_batch_size, tokenizer=tokenizer, device=device, ctx_length=100, shuffle=True, buffer_size=50000)

maintain_facts_eval = CounterFactTask(batch_size=eval_batch_size, tokenizer=tokenizer, device=device, criterion="cross_entropy", **maintain_kwargs)
if inject_fact:
    inject_fact_eval = CounterFactTask_Injection(batch_size=eval_batch_size, tokenizer=tokenizer, device=device, criterion="cross_entropy", **forget_kwargs)
    eval_tasks = {"pile": test_pile, "forget_fact": forget_fact_eval, "maintain_fact": maintain_facts_eval, "inject_fact": inject_fact_eval}
else:
    eval_tasks = {"pile": test_pile, "forget_fact": forget_fact_eval, "maintain_fact": maintain_facts_eval}
print(forget_fact_eval.train_dataset[0])


### localize model

from cb_utils.mask_utils import convert_attrs_to_components, get_top_components, get_top_components_no_subcomponents, get_random_components, load_mask_from_state_dict, get_parameter, apply_localized_gradients, find_component_params, get_top_components_no_subcomponents_gqa

# import pickle
# with open("models/google_gemma-2-9b_facts_all_ap_graph.pkl", "rb") as f:
#     ap_graph = pickle.load(f)
# print(ap_graph.keys())

# # ct components
# with open("models/google_gemma-2-9b_facts_all_ct_graph.pkl", "rb") as f:
#     ct_graph = pickle.load(f)
# print(ct_graph)

# top_p = 5
combine_heads = True

# localization_types = ["localized_ap", "random", "manual_interp", "all_mlps", "nonlocalized"]
# localization_types = ["all_mlps"]
# localization_types = ["manual_interp", "nonlocalized"]
# localization_types = ["localized_ap", "localized_ct"]
# localization_types = ["nonlocalized", "all_mlps", "sports_manual_interp", "forget_ct", "manual_interp", "random"]
# localization_types = ["manual_interp"]

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
color_map = {"localized_ap": colors[0], "localized_ct": colors[1], "random": colors[2], "manual_interp": colors[3], "nonlocalized": colors[4], "all_mlps": colors[5]}
formal_name_dict = {"localized_ap": "Localized AP", "localized_ct": "Localized CT", "random": "Random", "manual_interp": "Manual Interp", "nonlocalized": "Nonlocalized", "all_mlps": "All MLPs"}


all_components = {}

localization_type = args.localization_type
if localization_type == 'forget_ct':
    with open(f"models/google_{other_model_type}_counterfact_forget_ct_graph.pkl", "rb") as f:
        ct_graph = pickle.load(f)
    final_components, final_attn_heads = get_top_components_no_subcomponents(ct_graph, n_heads=n_heads, n_layers=n_layers, combine_heads=combine_heads, param_count=manual_param_count, param_count_dict=param_count_dict, n_kv_heads=8, input_heads=False)

elif localization_type == 'general_ct':
    with open(f"models/google_{other_model_type}_counterfact_maintain_ct_graph.pkl", "rb") as f:
        ct_graph = pickle.load(f)
    final_components, final_attn_heads = get_top_components_no_subcomponents(ct_graph, n_heads=n_heads, n_layers=n_layers, combine_heads=combine_heads, param_count=manual_param_count, param_count_dict=param_count_dict, n_kv_heads=8, input_heads=False)

elif localization_type == 'sports_manual_interp':
    final_components = []
    for mlp_layer in range(2, 5):
        final_components.append(f"blocks.{mlp_layer}.mlp.hook_pre")
        final_components.append(f"blocks.{mlp_layer}.mlp.hook_post")
    final_attn_heads = {}
    # mask = NeuronLevelMask(model, components=final_components, component_heads=final_attn_heads)

elif localization_type == 'manual_interp':
    final_components = []
    for mlp_layer in [3, 4, 5, 7, 8, 9, 10, 14, 15, 16, 17]:
        final_components.append(f"blocks.{mlp_layer}.mlp.hook_pre")
        final_components.append(f"blocks.{mlp_layer}.mlp.hook_post")
    final_attn_heads = {}

elif localization_type == 'random':
    final_components, final_attn_heads = get_random_components(n_layers=n_layers, n_heads=n_heads, combine_subcomponents=False, param_count=manual_param_count, param_count_dict=param_count_dict)

elif localization_type == "all_mlps":
    final_components = []
    for mlp_layer in range(n_layers):
        final_components.append(f"blocks.{mlp_layer}.mlp.hook_pre")
        final_components.append(f"blocks.{mlp_layer}.mlp.hook_post")
    final_attn_heads = {}

elif localization_type == 'nonlocalized':
    final_components = []
    for layer in range(n_layers):
        final_components.append(f"blocks.{layer}.mlp.hook_pre")
        final_components.append(f"blocks.{layer}.mlp.hook_post")
        final_components.append(f"blocks.{layer}.attn.hook_q")
        final_components.append(f"blocks.{layer}.attn.hook_k")
        final_components.append(f"blocks.{layer}.attn.hook_v")
        final_components.append(f"blocks.{layer}.attn.hook_result")
        
    final_attn_heads = None # don't actually think we need this
    # assert (torch.tensor([len(x) for x in final_attn_heads.values()]) == n_heads).all()

all_components[localization_type] = (final_components, final_attn_heads)

# get number of params
num_params = 0
for component in final_components:
    num_params += find_component_params(component, param_count_dict)
print(f"Number of parameters in {localization_type} localization: {num_params}")


model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
apply_localized_gradients(model, final_components, model_type=model_type)


## train model
from collections import defaultdict
from tasks.facts.SportsTaskAdversarial import adversarial_sports_eval_redo


import wandb

# for localization_type in ["nonlocalized"]:
print(f"Memory at start for {localization_type}: {torch.cuda.memory_allocated() / 1024**3}")
if use_wandb:
    wandb.init(project="circuit_breaking", name=f"finetuning_counterfact_{localization_type}_{forget_facts=}_{inject_fact=}")
    wandb.config.update({"model_type": model_type, "localization_type": localization_type, "combine_heads": combine_heads, "forget_facts": forget_facts, "lr": learning_rate, "n_epochs": n_epochs, "grad_accum_steps": grad_accum_steps, "forget_loss_coef": forget_loss_coef, "clip_grad": clip_grad, "manual_param_count": manual_param_count})

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
                task_accuracy += task.get_test_accuracy(model)
            all_test_losses[task_name].append(task_loss / n_eval_iters)
            all_test_losses[f"{task_name}_accuracy"].append(task_accuracy / n_eval_iters)
            if use_wandb:
                wandb.log({f"{task_name}_test_loss": task_loss / n_eval_iters}, step=epoch)
                wandb.log({f"{task_name}_test_accuracy": task_accuracy / n_eval_iters}, step=epoch)

    # print(f"After evaluating test loss on epoch {epoch}: {torch.cuda.memory_allocated() / 1024**3}, max mem: {torch.cuda.max_memory_allocated() / 1024**3}")


    if epoch % deep_evaluate_every == 0 or epoch == n_epochs - 1:
        if do_adversarial_evals:
            print("Running adversarial evals")
            adv_evals = adversarial_counterfact_eval(model, model_type=model_type, batch_size=eval_batch_size, 
                forget_task_init_kwargs=forget_kwargs, 
                maintain_task_init_kwargs=maintain_kwargs, 
                continuous=True, include_evals=["Normal", "MC"])
            adversarial_evals[epoch] = adv_evals
            if use_wandb:
                wandb.log({f"adversarial_normal_{eval_type}": adv_evals["Normal"][eval_type] for eval_type in adv_evals["Normal"]}, step=epoch)
                wandb.log({f"adversarial_mc_{eval_type}": adv_evals["MC"][eval_type] for eval_type in adv_evals["MC"]}, step=epoch)
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
    model.save_pretrained(f"{save_dir}/models/{model_type}_{localization_type}_{combine_heads=}_unlearn_{forget_facts=}_{inject_fact=}")
else:
    print(f"Not saving model for {localization_type}")
os.makedirs(f"{save_dir}/models", exist_ok=True)
with open(f"{save_dir}/models/{model_type}_{localization_type}_{combine_heads=}_unlearn_{forget_facts=}_{inject_fact=}_metrics.pkl", "wb") as f:
    pickle.dump({"train_losses": all_train_losses, "test_losses": all_test_losses, "adversarial_evals": adversarial_evals, "side_effect_evals": side_effect_evals}, f)

## SAVE TO HF
if args.push_to_hub:
    print("Pushing to HF, path is ", f"PhillipGuo/{model_type}-{localization_type}-unlearn_{forget_facts=}_{inject_fact=}-{args.run_id}")
    hf_save_path = f"PhillipGuo/{model_type}-{localization_type}-unlearn_{forget_facts=}_{inject_fact=}-{args.run_id}"
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


import gc
torch.cuda.empty_cache()
gc.collect()
# print memory usage
print(torch.cuda.memory_allocated() / 1024**3)

if args.do_relearning_evals:
    print("Running relearning evals")
    from peft import get_peft_model, LoraConfig, TaskType
    def do_relearning(model, train_tasks, n_iters, finetune_lora=False, lora_kwargs={'rank': args.lora_rank, 'alpha': 32, 'dropout': 0.05, 'target_modules': args.target_modules}, learning_kwargs={'lr': args.relearning_lr, 'weight_decay': 0, 'use_cosine': False}, eval_callback_fn=None):
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

            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_kwargs['lr'], weight_decay=learning_kwargs['weight_decay'])
        
        if learning_kwargs['use_cosine']:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_iters)

        train_losses = defaultdict(list)
        test_losses = []

        for i in tqdm(range(n_iters)):
            optimizer.zero_grad()
            for task_name, (task, task_weight) in train_tasks.items():
                loss = task.get_train_loss(model)
                train_losses[task_name].append(loss.item())
                # print(loss.item())
                (loss * task_weight).backward()
            
            optimizer.step()
            if learning_kwargs['use_cosine']:
                scheduler.step()

            if eval_callback_fn is not None:
                test_losses.append(eval_callback_fn(model))

        if len(test_losses) > 0:
            return train_losses, test_losses
        return train_losses
    
    n_eval_iters = args.n_eval_iters
    n_relearn_iters = args.n_relearn_iters
    n_relearn_facts = args.n_relearn_facts


    relearn_facts = CounterFactTask(batch_size=train_batch_size, tokenizer=tokenizer, forget_fact_subset=n_relearn_facts, train_test_split=False, is_forget_dataset=True)

    pile = PileTask(batch_size=8, tokenizer=tokenizer, ctx_length=256, shuffle=True, buffer_size=1000)
    train_tasks = {"relearn_athletes": (relearn_facts, .2), "maintain_athletes": (maintain_facts, 1), "pile": (train_pile, 1)}

    from tasks.facts.CounterFactTask import adversarial_counterfact_eval
    from tasks.general_capabilities.MCTask_redo import run_general_evals

    def eval_callback(model):
        mmlu_score = run_general_evals(model, model_type=model_type, batch_size=mmlu_batch_size)["MMLU"]
        adversarial_results = adversarial_counterfact_eval(model, model_type=model_type, batch_size=eval_batch_size, 
                    forget_task_init_kwargs=forget_kwargs, 
                    maintain_task_init_kwargs=maintain_kwargs, 
                    continuous=True, include_evals=["Normal", "MC", "Paraphrase", "Neighborhood"], n_mc_shots=1)

        # get dictionary of both
        return {"MMLU": mmlu_score, "adversarial": adversarial_results}

    # del model

    # for name, model, mask, regular_evals, side_effect_evals, adversarial_evals in [("localized", localized_model, localized_mask, localized_regular_evals, localized_side_effect_evals, localized_adversarial_evals), ("nonlocalized", nonlocalized_model, nonlocalized_mask, nonlocalized_regular_evals, nonlocalized_side_effect_evals, nonlocalized_adversarial_evals)]:

    relearning_train_results = {}
    relearning_test_results = {}
    relearning_regular_results = {}
    relearning_adversarial_results = {}
    relearning_side_effect_results = {}

    model.cuda()

    train_losses, test_losses = do_relearning(model, train_tasks, n_iters=n_relearn_iters, finetune_lora=True, learning_kwargs={'lr': args.relearning_lr, 'weight_decay': 0, 'use_cosine': True}, eval_callback_fn=eval_callback)

    relearning_train_results[localization_type] = train_losses
    relearning_test_results[localization_type] = test_losses

    relearning_regular_results[localization_type] = {}
    for task_name, test_task in [("forget_facts", forget_fact_eval), ("maintain_facts", maintain_facts)]:
        task_loss = 0
        task_accuracy = 0
        for i in range(n_eval_iters):
            task_loss += test_task.get_test_loss(model).item()
            task_accuracy += test_task.get_test_accuracy(model)
        relearning_regular_results[localization_type][f"{task_name}_ce"] = task_loss / n_eval_iters
        relearning_regular_results[localization_type][f"{task_name}_acc"] = task_accuracy / n_eval_iters

    adversarial_eval_results = adversarial_counterfact_eval(model, model_type=model_type, batch_size=eval_batch_size, 
                    forget_task_init_kwargs=forget_kwargs, 
                    maintain_task_init_kwargs=maintain_kwargs, 
                    continuous=True, include_evals=["Normal", "MC", "Paraphrase", "Neighborhood"], n_mc_shots=1)
    relearning_adversarial_results[localization_type] = adversarial_eval_results

    side_effect_eval_results = run_side_effects_evals(model, model_type=model_type, batch_size=eval_batch_size, evals_to_run=["General"], general_batch_size=mmlu_batch_size)
    relearning_side_effect_results[localization_type] = side_effect_eval_results

    # model.cpu()

    os.makedirs(f"{save_dir}/results", exist_ok=True)
    with open(f"{save_dir}/results/relearning_{n_relearn_facts=}_{n_relearn_iters=}_{model_type}_{combine_heads=}_{beta=}_unlearn_{forget_facts=}_results.pkl", "wb") as f:
        pickle.dump({"relearning_regular_results": relearning_regular_results, "relearning_adversarial_results": relearning_adversarial_results, "relearning_side_effect_results": relearning_side_effect_results, "relearning_train_results": relearning_train_results, "relearning_test_results": relearning_test_results}, f)

