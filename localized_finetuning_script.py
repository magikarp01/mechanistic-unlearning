from circuit_breaking.src import *
import torch
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import os
from circuit_breaking.src.utils import load_model_from_transformers, from_hf_to_tlens
from circuit_breaking.src.masks import MLPHiddenMask
from tqdm.auto import tqdm

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--model_type", type=str, choices=["gemma-7b", "llama-2", "pythia-2.8b"])
parser.add_argument("--forget_sport", type=str, choices=["football", "basketball", "baseball", "golf", "tennis"], default=None)
parser.add_argument("--forget_athletes", type=int, default=None)
parser.add_argument("--inject_sport", type=str, choices=["football", "basketball", "baseball", "golf", "tennis"], default=None)
parser.add_argument("--localization_type", type=str, choices=["localized_ap", "localized_ct", "manual_interp", "random", "all_mlps", "nonlocalized"])
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
parser.add_argument("--learning_rate", type=float, default=2.5e-5)
parser.add_argument("--n_epochs", type=int, default=50)
parser.add_argument("--grad_accum_steps", type=int, default=None)
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
parser.add_argument("--n_relearn_iters", type=int, default=10)
parser.add_argument("--n_relearn_athletes", type=int, default=2)
parser.add_argument("--lora_rank", type=int, default=64)
parser.add_argument("--target_modules", type=str, default="all-linear")
parser.add_argument("--relearning_lr", type=float, default=1e-4)


parser.add_argument("--do_probing_evals", type=bool, default=False)
parser.add_argument("--probing_batch_size", type=int, default=16)

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
check_all_logits = args.check_all_logits
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
    param_count_dict = {"attn.hook_q": 3072*4096, "attn.hook_k": 3072*4096, "attn.hook_v": 3072*4096, "attn.hook_result": 4096*3072, "mlp.hook_pre": 3072 * 24576, "mlp.hook_post": 24576 * 3072}
    manual_param_count = 9e8
else:
    raise NotImplementedError(f"Model type {args.model_type} not implemented")


### Unlearning and evaluation tasks

from tasks import PileTask, OWTTask, InductionTask, GreaterThanTask
from tasks.ioi.IOITask import IOITask, IOITask_NPO, IOITask_Uniform
from tasks.induction.InductionTask import InductionTask, InductionTask_NPO, InductionTask_Uniform
from tasks.facts.SportsTask import SportsTask, SportsTask_NPO, SportsTask_Uniform, SportsTask_Injection
from tasks.facts.SportsTaskAdversarial import adversarial_sports_eval
from tasks.facts.SportsTaskSideEffects import run_side_effects_evals


device = "cuda"
train_loss_type = "sports"

maintain_sport = None

# inject_sport = "golf"
inject_sport = args.inject_sport

forget_sport = args.forget_sport
forget_athletes = args.forget_athletes

if inject_sport is not None:
    save_dir = f"results2/localized_finetuning_injection_{forget_athletes}_athletes/{args.localization_type}"
else:
    save_dir = f"results2/localized_finetuning_{forget_athletes}_athletes/{args.localization_type}"

if args.run_id is not None:
    save_dir = f"{save_dir}_{args.run_id}"

os.makedirs(save_dir, exist_ok=True)

if forget_athletes is not None:
    forget_kwargs = {"forget_player_subset": forget_athletes, "is_forget_dataset": True, "train_test_split": False}
    maintain_kwargs = {"forget_player_subset": forget_athletes, "is_forget_dataset": False, "train_test_split": True}
    forget_loss_coef = 1
elif forget_sport is not None:
    forget_kwargs = {"forget_sport_subset": {forget_sport}, "is_forget_dataset": True, "train_test_split": True}
    maintain_kwargs = {"forget_sport_subset": {forget_sport}, "is_forget_dataset": False, "train_test_split": True}
    forget_loss_coef = .2

# forget_sport="basketball"
# forget_athletes = None
# if inject_sport is not None:
#     save_dir = f"results/localized_finetuning_injection_{forget_sport}"
# else:
#     save_dir = f"results/localized_finetuning_{forget_sport}"
# forget_kwargs = {"forget_sport_subset": {forget_sport}, "is_forget_dataset": True, "train_test_split": True}
# maintain_kwargs = {"forget_sport_subset": {forget_sport}, "is_forget_dataset": False, "train_test_split": True}
# forget_loss_coef=.2

os.makedirs(save_dir, exist_ok=True)



if maintain_sport is None:
    maintain_sports = SportsTask(batch_size=train_batch_size, tokenizer=tokenizer, device=device, prep_acdcpp=False, criterion="cross_entropy", **maintain_kwargs)
else:
    maintain_sports = SportsTask(batch_size=train_batch_size, tokenizer=tokenizer, device=device, prep_acdcpp=False, criterion="cross_entropy", **maintain_kwargs)

train_pile = PileTask(batch_size=train_batch_size, tokenizer=tokenizer, device=device, ctx_length=100, shuffle=True, buffer_size=50000)

if inject_sport is not None:
    sports_injection = SportsTask_Injection(batch_size=train_batch_size, tokenizer=tokenizer, device=device, inject_sport=inject_sport, **forget_kwargs)
    train_tasks = {"sports_injection": (sports_injection, forget_loss_coef), "maintain_sports": (maintain_sports, 1), "pile": (train_pile, 1)}
else:
    sports_1mp = SportsTask(batch_size=train_batch_size, tokenizer=tokenizer, device=device, prep_acdcpp=False, criterion="log_1_minus_p", **forget_kwargs)
    train_tasks = {"sports_1mp": (sports_1mp, forget_loss_coef), "maintain_sports": (maintain_sports, 1), "pile": (train_pile, 1)}

# train_tasks = {"maintain_sports": (maintain_sports, 1)}

# want to eval on other sports
forget_sport_eval = SportsTask(batch_size=eval_batch_size, tokenizer=tokenizer, device=device, prep_acdcpp=False, criterion="cross_entropy", **forget_kwargs)
test_pile = PileTask(batch_size=eval_batch_size, tokenizer=tokenizer, device=device, ctx_length=100, shuffle=True, buffer_size=50000)

induction_eval = InductionTask(batch_size=eval_batch_size, tokenizer=tokenizer, prep_acdcpp=False, seq_len=15, device=device)
if maintain_sport is None:
    maintain_sports_eval = SportsTask(batch_size=eval_batch_size, tokenizer=tokenizer, device=device, prep_acdcpp=False, criterion="cross_entropy", **maintain_kwargs)
    eval_tasks = {"induction": induction_eval, "pile": test_pile, "forget_sport": forget_sport_eval, "maintain_sport": maintain_sports_eval}
else:
    maintain_sport_eval = SportsTask(batch_size=eval_batch_size, tokenizer=tokenizer, device=device, prep_acdcpp=False, criterion="cross_entropy", forget_sport_subset={maintain_sport}, is_forget_dataset=True)
    val_sport_eval = SportsTask(batch_size=eval_batch_size, tokenizer=tokenizer, device=device, prep_acdcpp=False, criterion="cross_entropy", forget_sport_subset={val_sport}, is_forget_dataset=True)
    eval_tasks = {"induction": induction_eval, "pile": test_pile, "forget_sport": forget_sport_eval, "maintain_sport": maintain_sport_eval, "val_sport": val_sport_eval}

### localize model

from cb_utils.mask_utils import convert_attrs_to_components, get_top_components, get_top_components_no_subcomponents, get_random_components, load_mask_from_state_dict, get_parameter, apply_localized_gradients, find_component_params

import pickle
with open("models/google_gemma-7b_sports_all_ap_graph.pkl", "rb") as f:
    ap_graph = pickle.load(f)
print(ap_graph.keys())

# ct components
with open("models/google_gemma-7b_sports_all_ct_graph.pkl", "rb") as f:
    ct_graph = pickle.load(f)
print(ct_graph)

localization_type = args.localization_type
combine_heads = args.combine_heads

if localization_type == 'localized_ap':
    final_components, final_attn_heads = get_top_components(*convert_attrs_to_components(ap_graph, n_heads=n_heads, n_layers=n_layers, combine_heads=combine_heads), n_heads=n_heads, param_count=manual_param_count, param_count_dict=param_count_dict)

    # print(final_components)
    # print(final_attn_heads)

elif localization_type == 'localized_ct':
    final_components, final_attn_heads = get_top_components_no_subcomponents(ct_graph, n_heads=n_heads, n_layers=n_layers, combine_heads=combine_heads, param_count=manual_param_count, param_count_dict=param_count_dict)

elif localization_type == 'manual_interp':
    final_components = []
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


elif localization_type == 'nonlocalized':
    final_components, final_attn_heads = get_top_components(*convert_attrs_to_components(ap_graph, n_heads=n_heads, n_layers=n_layers, combine_heads=combine_heads), n_heads=n_heads, top_p=100)
    assert (torch.tensor([len(x) for x in final_attn_heads.values()]) == n_heads).all()

# get number of params
num_params = 0
for component in final_components:
    num_params += find_component_params(component, param_count_dict)
print(f"Number of parameters in {localization_type} localization: {num_params}")


model = AutoModelForCausalLM.from_pretrained("google/gemma-7b", torch_dtype=torch.bfloat16)
apply_localized_gradients(model, final_components, model_type=model_type)


## train model
from collections import defaultdict
from tasks.facts.SportsTaskAdversarial import adversarial_sports_eval_redo


import wandb

# for localization_type in ["nonlocalized"]:
print(f"Memory at start for {localization_type}: {torch.cuda.memory_allocated() / 1024**3}")
if use_wandb:
    wandb.init(project="circuit_breaking", name=f"finetuning_{localization_type}_{forget_sport=}_{forget_athletes=}")
    wandb.config.update({"model_type": model_type, "localization_type": localization_type, "combine_heads": combine_heads, "beta": beta, "forget_sport": forget_sport, "forget_athletes": forget_athletes, "lr": learning_rate, "n_epochs": n_epochs, "grad_accum_steps": grad_accum_steps, "forget_loss_coef": forget_loss_coef, "clip_grad": clip_grad, "manual_param_count": manual_param_count, "run_id": args.run_id})

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
                continuous=True, include_evals=["Normal", "MC"], check_all_logits=check_all_logits)
            adversarial_evals[epoch] = adv_evals
            if use_wandb:
                wandb.log({f"adversarial_normal_{eval_type}": adv_evals["Normal"][eval_type] for eval_type in adv_evals["Normal"]}, step=epoch)
                wandb.log({f"adversarial_mc_{eval_type}": adv_evals["MC"][eval_type] for eval_type in adv_evals["MC"]}, step=epoch)
        # print(f"After evaluating adversarial evals on epoch {epoch}: {torch.cuda.memory_allocated() / 1024**3}, max mem: {torch.cuda.max_memory_allocated() / 1024**3}")
        if do_side_effects_evals:
            print("Running side effects evals")
            side_effect_evals[epoch] = run_side_effects_evals(model, model_type=model_type, batch_size=eval_batch_size, evals_to_run=["General"], general_batch_size=5)
            if use_wandb:
                wandb.log(side_effect_evals[epoch]["General"], step=epoch)
        # print(f"After evaluating side effects evals on epoch {epoch}: {torch.cuda.memory_allocated() / 1024**3}, max mem: {torch.cuda.max_memory_allocated() / 1024**3}")


if save_model:
    os.makedirs(f"{save_dir}/models", exist_ok=True)
    torch.save(model.state_dict(), f"{save_dir}/models/{model_type}_{localization_type}_{combine_heads=}_{beta=}_unlearn_{forget_sport=}_{forget_athletes=}.pt")
else:
    print(f"Not saving model for {localization_type}")
os.makedirs(f"{save_dir}/models", exist_ok=True)
with open(f"{save_dir}/models/{model_type}_{localization_type}_{combine_heads=}_{beta=}_unlearn_{forget_sport=}_{forget_athletes=}_metrics.pkl", "wb") as f:
    pickle.dump({"train_losses": all_train_losses, "test_losses": all_test_losses, "adversarial_evals": adversarial_evals, "side_effect_evals": side_effect_evals}, f)

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

    left_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    left_tokenizer.pad_token_id = left_tokenizer.eos_token_id
    left_tokenizer.padding_side = "left"

    from collections import defaultdict
    def layer_hook_function(layer, outputs, last_token_only=True, store_cpu=False):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                save_output = output[0].clone().detach()
            else:
                save_output = output.clone().detach()
            if last_token_only:
                save_output = save_output[:, -1]
            if store_cpu:
                save_output = save_output.cpu()
            outputs[layer].append(save_output)
            # return output
        return hook_fn

    def get_hf_residuals(texts, model, batch_size, last_token_only=True, layers_module=None, store_cpu=True, text_col="prompt"):
        # needs left_
        outputs = defaultdict(list)
        hooks = []
        if layers_module is None:
            layers_module = model.model.layers
        for layer, block in enumerate(layers_module):
            hook_fn = layer_hook_function(layer, outputs=outputs, last_token_only=last_token_only, store_cpu=store_cpu)
            hook_applied = block.register_forward_hook(hook_fn)
            hooks.append(hook_applied)

        for idx in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[idx:idx+batch_size]
            tokenized = left_tokenizer(batch_texts, return_tensors="pt", padding=True)
            tokenized = {k: v.cuda() for k, v in tokenized.items()}
            with torch.no_grad():
                model(**tokenized)
        
        for layer in outputs:
            outputs[layer] = torch.cat(outputs[layer], dim=0)
            if store_cpu:
                outputs[layer] = outputs[layer].cpu()

        for hook in hooks:
            hook.remove()
        
        return outputs

    batch_size = args.probing_batch_size
    def get_resids(sports_task, model):
        train_outputs = get_hf_residuals(sports_task.train_df["prompt"].tolist(), model, batch_size, last_token_only=True) # needs to not be last token only because of layernorm
        test_outputs = get_hf_residuals(sports_task.test_df["prompt"].tolist(), model, batch_size, last_token_only=True)

        train_labels = sports_task.train_df['sport'].tolist()
        test_labels = sports_task.test_df['sport'].tolist()
        return train_outputs, test_outputs, train_labels, test_labels

    forget_is_split = True if forget_sport is not None else False
    if forget_is_split:
        forget_train_outputs_dict = {}
        forget_test_outputs_dict = {}
        forget_train_labels_dict = {}
        forget_test_labels_dict = {}
    else:
        forget_outputs_dict = {}
        forget_labels_dict = {}

    maintain_train_outputs_dict = {}
    maintain_test_outputs_dict = {}
    maintain_train_labels_dict = {}
    maintain_test_labels_dict = {}

    model.cuda()
    if forget_is_split:
        forget_train_outputs_dict[localization_type], forget_test_outputs_dict[localization_type], forget_train_labels_dict[localization_type], forget_test_labels_dict[localization_type] = get_resids(forget_sport_eval, model)
    else:
        forget_outputs_dict[localization_type], _, forget_labels_dict[localization_type], _ = get_resids(forget_sport_eval, model)
    maintain_train_outputs_dict[localization_type], maintain_test_outputs_dict[localization_type], maintain_train_labels_dict[localization_type], maintain_test_labels_dict[localization_type] = get_resids(maintain_sports_eval, model)

    # model.cpu()

    # set train and test splits
    if not forget_is_split:
        print("Performing manual split of the unsplit training dataset")
        train_test_split = .5
        forget_train_outputs_dict = {}
        forget_test_outputs_dict = {}
        forget_train_labels_dict = {}
        forget_test_labels_dict = {}
        num_train = int(len(forget_labels_dict[localization_type]) * train_test_split)
        forget_train_labels_dict[localization_type] = forget_labels_dict[localization_type][:num_train]
        forget_test_labels_dict[localization_type] = forget_labels_dict[localization_type][num_train:]
        forget_train_outputs_dict[localization_type] = {}
        forget_test_outputs_dict[localization_type] = {}
        for layer in range(n_layers):
            forget_train_outputs_dict[localization_type][layer] = forget_outputs_dict[localization_type][layer][:num_train]
            forget_test_outputs_dict[localization_type][layer] = forget_outputs_dict[localization_type][layer][num_train:]


    from sklearn.linear_model import LogisticRegression

    def get_sport_labels(string_labels, return_np=True):
        # want three different lists of labels, one for each sport
        sports = ["baseball", "football", "basketball"]
        sport_labels = {sport: [] for sport in sports}
        for label in string_labels:
            for sport in sports:
                if sport in label:
                    sport_labels[sport].append(1)
                else:
                    sport_labels[sport].append(0)
        if return_np:
            for sport in sports:
                sport_labels[sport] = np.array(sport_labels[sport])
            
        assert sum(sport_labels["baseball"]) + sum(sport_labels["football"]) + sum(sport_labels["basketball"]) == len(string_labels)
        # assert each position always adds up to 1
        for i in range(len(string_labels)):
            assert sport_labels["baseball"][i] + sport_labels["football"][i] + sport_labels["basketball"][i] == 1
        return sport_labels

    # train probes
    all_probes = defaultdict(dict) # double-nested dictionary, first keys are model_name, second keys are layers, final values are dictionaries with keys "basketball", "football", "baseball" and values of probes

    all_train_accs = defaultdict(dict)
    all_test_accs = defaultdict(dict)
    all_forget_accs = defaultdict(dict)
    all_maintain_accs = defaultdict(dict)

    combine_accuracies = True

    shuffle_train = True
    
    forget_test_acts = forget_test_outputs_dict[localization_type]
    forget_test_labels = get_sport_labels(forget_test_labels_dict[localization_type])
    maintain_test_acts = maintain_test_outputs_dict[localization_type]
    maintain_test_labels = get_sport_labels(maintain_test_labels_dict[localization_type])

    forget_train_acts = forget_train_outputs_dict[localization_type]
    maintain_train_acts = maintain_train_outputs_dict[localization_type]
    # forget_test_labels_dict[model_name] + maintain_test_labels_dict[model_name]
    train_labels = forget_train_labels_dict[localization_type] + maintain_train_labels_dict[localization_type]
    train_labels = get_sport_labels(train_labels)

    test_labels = forget_test_labels_dict[localization_type] + maintain_test_labels_dict[localization_type]
    test_labels = get_sport_labels(test_labels)

    if shuffle_train:
        shuffle_idx = torch.randperm(len(list(train_labels.values())[0]))

    if shuffle_train:
        for sport in train_labels:
            train_labels[sport] = train_labels[sport][shuffle_idx]
        
    # print(f"Labels look like {train_labels}")

    for layer in range(n_layers):
        layer_train_acts = torch.cat([forget_train_acts[layer], maintain_train_acts[layer]], dim=0).float().cpu().numpy()
        layer_test_acts = torch.cat([forget_test_acts[layer], maintain_test_acts[layer]], dim=0).float().cpu().numpy()
        layer_forget_test_acts = forget_test_acts[layer].float().cpu().numpy()
        layer_maintain_test_acts = maintain_test_acts[layer].float().cpu().numpy()

        if shuffle_train:
            layer_train_acts = layer_train_acts[shuffle_idx]
        all_probes[localization_type][layer] = {}

        if not combine_accuracies:
            all_train_accs[localization_type][layer] = {}
            all_test_accs[localization_type][layer] = {}
            all_forget_accs[localization_type][layer] = {}
            all_maintain_accs[localization_type][layer] = {}

        sports_train_preds = {}
        sports_test_preds = {}
        sports_forget_preds = {}
        sports_maintain_preds = {}
        for sport in train_labels:
            if sum(train_labels[sport]) <= 0:
                print("No labels for sport", sport)
                continue
            probe = LogisticRegression(max_iter=10000)
            # print(f"Training probe for {sport} at layer {layer}, {layer_train_acts.shape=}, {train_labels[sport].shape=}, {train_labels[sport].mean()=}")
            probe.fit(layer_train_acts, train_labels[sport])
            all_probes[localization_type][layer][sport] = probe

            # test probes
            # print(f"{sport=}, {layer_train_acts.shape=}, {train_labels[sport].shape=}, {train_labels[sport].mean()=}")
            train_preds = probe.predict(layer_train_acts)
            if not combine_accuracies:
                train_acc = (train_preds == train_labels[sport]).sum() / len(train_labels[sport])
                all_train_accs[localization_type][layer][sport] = train_acc
            else:
                sports_train_preds[sport] = train_preds


            # print(f"Testing probe for {sport} at layer {layer}, {layer_test_acts.shape=}, {test_labels[sport].shape=}, {test_labels[sport].mean()=}")
            test_preds = probe.predict(layer_test_acts)
            if not combine_accuracies:
                test_acc = (test_preds == test_labels[sport]).sum() / len(test_labels[sport])
                all_forget_accs[localization_type][layer][sport] = test_acc
            else:
                sports_test_preds[sport] = test_preds

            # print(f"{sport=}, {layer_forget_test_acts.shape=}, {forget_test_labels[sport].shape=}, {forget_test_labels[sport].mean()=}")
            forget_test_preds = probe.predict(layer_forget_test_acts)
            if not combine_accuracies:
                forget_acc = (forget_test_preds == forget_test_labels[sport]).sum() / len(forget_test_labels[sport])
                all_test_accs[localization_type][layer][sport] = forget_acc
            else:
                sports_forget_preds[sport] = forget_test_preds

            # print(f"{sport=}, {layer_maintain_test_acts.shape=}, {maintain_test_labels[sport].shape=}, {maintain_test_labels[sport].mean()=}")
            maintain_test_preds = probe.predict(layer_maintain_test_acts)
            if not combine_accuracies:
                maintain_acc = (maintain_test_preds == maintain_test_labels[sport]).sum() / len(maintain_test_labels[sport])
                all_maintain_accs[localization_type][layer][sport] = maintain_acc 
            else:
                sports_maintain_preds[sport] = maintain_test_preds

        if combine_accuracies:
            # combine accuracies by saying probes correct if all sports are correct
            train_correct = np.ones(len(train_labels["baseball"]))
            test_correct = np.ones(len(test_labels["baseball"]))
            forget_correct = np.ones(len(forget_test_labels["baseball"]))
            maintain_correct = np.ones(len(maintain_test_labels["baseball"]))
            for sport in train_labels:
                if sum(train_labels[sport]) > 0:
                    train_correct *= (sports_train_preds[sport] == train_labels[sport])
                else:
                    print("No train labels for sport", sport)
                if sum(test_labels[sport]) > 0:
                    test_correct *= (sports_test_preds[sport] == test_labels[sport])
                else:
                    print("No test labels for sport", sport)
                if sum(forget_test_labels[sport]) > 0:
                    forget_correct *= (sports_forget_preds[sport] == forget_test_labels[sport])
                else:
                    print("No forget labels for sport", sport)
                if sum(maintain_test_labels[sport]) > 0:
                    maintain_correct *= (sports_maintain_preds[sport] == maintain_test_labels[sport])
                else:
                    print("No maintain labels for sport", sport)

            all_train_accs[localization_type][layer] = train_correct.mean()
            all_test_accs[localization_type][layer] = test_correct.mean()
            all_forget_accs[localization_type][layer] = forget_correct.mean()
            all_maintain_accs[localization_type][layer] = maintain_correct.mean()

    os.makedirs(f"{save_dir}/results", exist_ok=True)
    with open(f"{save_dir}/results/probes_{model_type}_{combine_heads=}_{beta=}_unlearn_{forget_sport=}_{forget_athletes=}.pkl", "wb") as f:
        pickle.dump({"all_probes": all_probes, "all_train_accs": all_train_accs, "all_test_accs": all_test_accs, "all_forget_accs": all_forget_accs, "all_maintain_accs": all_maintain_accs}, f)


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
    n_relearn_athletes = args.n_relearn_athletes


    if forget_sport is None:
        relearn_sport = SportsTask(batch_size=n_relearn_athletes, tokenizer=tokenizer, forget_player_subset=n_relearn_athletes, train_test_split=False, is_forget_dataset=True)
    else:
        relearn_sport = SportsTask(batch_size=n_relearn_athletes, tokenizer=tokenizer, forget_sport_subset={forget_sport}, forget_player_subset=n_relearn_athletes, train_test_split=False, is_forget_dataset=True)


    if forget_sport is None:
        relearn_sport = SportsTask(batch_size=n_relearn_athletes, tokenizer=tokenizer, forget_player_subset=n_relearn_athletes, train_test_split=False, is_forget_dataset=True)
    else:
        relearn_sport = SportsTask(batch_size=n_relearn_athletes, tokenizer=tokenizer, forget_sport_subset={forget_sport}, forget_player_subset=n_relearn_athletes, train_test_split=False, is_forget_dataset=True)

    pile = PileTask(batch_size=8, tokenizer=tokenizer, ctx_length=256, shuffle=True, buffer_size=1000)
    train_tasks = {"relearn_athletes": (relearn_sport, .2), "maintain_athletes": (maintain_sports, 1), "pile": (train_pile, 1)}

    from tasks.facts.SportsTaskAdversarial import adversarial_sports_eval_redo
    from tasks.general_capabilities.MCTask_redo import run_general_evals

    def eval_callback(model):
        mmlu_score = run_general_evals(model, model_type="gemma")["MMLU"]
        adversarial_results = adversarial_sports_eval_redo(model, model_type=model_type, batch_size=eval_batch_size, 
                        forget_task_init_kwargs={"use_system_prompt":True, "use_icl":False}|forget_kwargs, 
                        maintain_task_init_kwargs={"use_system_prompt":True, "use_icl":False}|maintain_kwargs, 
                        continuous=True, include_evals=["Normal", "MC"])

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
    for task_name, test_task in [("forget_sport", forget_sport_eval), ("maintain_sports", maintain_sports_eval)]:
        task_loss = 0
        task_accuracy = 0
        for i in range(n_eval_iters):
            task_loss += test_task.get_test_loss(model).item()
            task_accuracy += test_task.get_test_accuracy(model)
        relearning_regular_results[localization_type][f"{task_name}_ce"] = task_loss / n_eval_iters
        relearning_regular_results[localization_type][f"{task_name}_acc"] = task_accuracy / n_eval_iters

    adversarial_eval_results = adversarial_sports_eval_redo(model, model_type=model_type, batch_size=eval_batch_size, 
                    forget_task_init_kwargs={"use_system_prompt":True, "use_icl":False}|forget_kwargs, 
                    maintain_task_init_kwargs={"use_system_prompt":True, "use_icl":False}|maintain_kwargs, 
                    continuous=True, include_evals=["Normal", "MC"])
    relearning_adversarial_results[localization_type] = adversarial_eval_results

    side_effect_eval_results = run_side_effects_evals(model, model_type=model_type, batch_size=eval_batch_size, evals_to_run=["General"], general_batch_size=5)
    relearning_side_effect_results[localization_type] = side_effect_eval_results

    # model.cpu()

    os.makedirs(f"{save_dir}/results", exist_ok=True)
    with open(f"{save_dir}/results/relearning_{n_relearn_athletes=}_{n_relearn_iters=}_{model_type}_{combine_heads=}_{beta=}_unlearn_{forget_sport=}_{forget_athletes=}_results.pkl", "wb") as f:
        pickle.dump({"relearning_regular_results": relearning_regular_results, "relearning_adversarial_results": relearning_adversarial_results, "relearning_side_effect_results": relearning_side_effect_results, "relearning_train_results": relearning_train_results, "relearning_test_results": relearning_test_results}, f)

