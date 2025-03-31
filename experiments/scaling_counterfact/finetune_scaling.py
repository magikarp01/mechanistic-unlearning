#%%
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import DataCollatorForLanguageModeling
import datasets
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import random
import gc

#%%
DATASET_LENGTH = 100
MODE = "ATTN"
MODEL_NAME = "google/gemma-7b"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained(MODEL_NAME, default_padding_side="left", device=device)
# model = model.half()
model.tokenizer.padding_side = "left"

#%%
# def extract_dataset(model, dataset):
#     def extract(examples):
#         true_string = [' ' + r['target_true']['str'] for r in examples['requested_rewrite']]
#         edit_string = [' ' + r['target_new']['str'] for r in examples['requested_rewrite']]
#         question_string = [r['prompt'].format(r['subject']) for r in examples['requested_rewrite']]
#         subject_string = [r['subject'] for r in examples['requested_rewrite']]
#         return {"prompt": question_string, "true_string": true_string, "edit_string": edit_string, "subject": subject_string}

#     return dataset.map(
#         lambda examples: extract(examples),
#         batched=True,
#         batch_size=10_000
#     )
    
# def correctness_filter(model, dataset, verbose=False):
#     ''' Filter out any examples that the model gets wrong. '''
#     def get_correctness(model, examples):
#         ''' Populate dataset with first token of correct answer and model answer '''
#         true_string = [' ' + r['target_true']['str'] for r in examples['requested_rewrite']]
#         edit_string = [' ' + r['target_new']['str'] for r in examples['requested_rewrite']]
#         question_string = [r['prompt'].format(r['subject']) for r in examples['requested_rewrite']]

#         orig_padding_side = model.tokenizer.padding_side
#         model.tokenizer.padding_side = "right"
#         correct_tokens = list(model.tokenizer(true_string, return_tensors="pt", padding=True)["input_ids"][:, 0])
#         edit_tokens = model.tokenizer(edit_string, return_tensors="pt", padding=True)["input_ids"][:, 0]
#         model.tokenizer.padding_side = orig_padding_side

#         question_tokens = model.tokenizer(question_string, return_tensors="pt", padding=True)["input_ids"]
#         scaled_logits = torch.nn.functional.softmax(model(question_tokens)[:, -1, :], dim=-1)
#         is_model_correct = list(scaled_logits[range(len(scaled_logits)), correct_tokens] > scaled_logits[range(len(scaled_logits)), edit_tokens])
#         if verbose:
#             print(f"String: {true_string[0]} tokenized as {correct_tokens[0]}, model is correct? {is_model_correct[0]}")

#         return {"correct_token": correct_tokens, "is_model_correct": is_model_correct}
    
#     with torch.no_grad():
#         dataset = dataset.map(
#             lambda row: get_correctness(model, row),
#             batched=True,
#             batch_size=1_000
#         )
#         dataset = dataset.filter(lambda x: x["is_model_correct"])

#     assert all(dataset["is_model_correct"]), "Filter failed, model is not correct on all examples"

#     return dataset
#%%
cfact_retain = datasets.load_from_disk(f"~/mechanistic-unlearning/experiments/scaling_counterfact/{MODEL_NAME.replace('/', '_')}_cfact_forget.hf")["test"]
# cfact_retain = cfact_retain.map(
#     lambda row: {"prompt": "Fact: The Eiffel Tower is in Paris. Fact: " + row["prompt"]}
# )
# cfact_retain = extract_dataset(model, cfact_retain)
# cfact_retain = correctness_filter(model, cfact_retain, verbose=True)

# Important columns: prompt, correct_token, edit_str, localization_str
cfact_forget = pd.read_csv(f"~/mechanistic-unlearning/experiments/scaling_counterfact/cfact_forget_{MODEL_NAME.replace('/', '_')}.csv")[:DATASET_LENGTH]
mcq_path = "/root/mechanistic-unlearning/tasks/facts/data/counterfact_mc_questions.parquet"
mcq = pd.read_parquet(mcq_path)
mcq['case_id'] = mcq['prompt_id']
merged_q = pd.merge(mcq, cfact_forget, how="inner", on="case_id")

def generate_binary_string(length):
    result = ['0'] * length

    # Ensure at least one '1' in a random position
    guaranteed_one_pos = random.randint(0, length - 1)
    result[guaranteed_one_pos] = '1'

    # Fill in the rest with 10% chance of being '1'
    for i in range(length):
        if i != guaranteed_one_pos:
            result[i] = '1' if random.random() < 0.1 else '0'

    return ''.join(result)

# cfact_forget["prompt"] = "Fact: The Eiffel Tower is in Paris. Fact: " + cfact_forget["prompt"]
mlps_len = model.cfg.n_layers
if MODE == "MANUAL":
    cfact_forget["localization_str"] = '1111111111100000000000000000'
    # cfact_forget["localization_str"] = '1111111111111000000000000000'
    # pass
if MODE == "RANDOM":
    mlps_len = len(cfact_forget["localization_str"].iloc[0])
    cfact_forget["localization_str"] = [generate_binary_string(mlps_len) for _ in range(len(cfact_forget))]
elif MODE == "ALL":
    mlps_len = len(cfact_forget["localization_str"].iloc[0])
    cfact_forget["localization_str"] = ["1" * mlps_len for _ in range(len(cfact_forget))]
elif MODE == "ATTN":
    cfact_forget["localization_str"] = ["0" * mlps_len for _ in range(len(cfact_forget))]

pile_retain = load_dataset("EleutherAI/the_pile_deduplicated", split="train", streaming=True)

#%%
import wandb
from itertools import cycle
import random
from functools import partial
wandb.login()

def collate_fn(batch, tokenizer):
    def generate_mcqs(questions, correct_answers, incorrect_answers, edit_answers):
        formatted_questions = []
        orig_labels = []
        edit_labels = []
        icl_prompt = "Where is the Eiffel Tower?\nA. Seattle\nB. Rome\nC. Paris\nD. Madrid\nAnswer: C\n"

        for q, correct, incorrect, edit in zip(questions, correct_answers, incorrect_answers, edit_answers):
            # Combine correct and incorrect answers
            options = list(incorrect) + [correct]
            random.shuffle(options)

            # Map options to labels
            option_labels = ['A', 'B', 'C', 'D']

            # Find correct answer's label
            correct_idx = options.index(correct)
            correct_label = option_labels[correct_idx]
            if edit not in options:
                options[(correct_idx+1)%4] = edit

            edit_label = option_labels[options.index(edit)]

            labeled_options = [f"{label}. {opt}" for label, opt in zip(option_labels, options)]

            # Format question string
            question_text = icl_prompt + f"{q}\n" + "\n".join(labeled_options) + "\nAnswer:"
            # question_text = f"{q}\n" + "\n".join(labeled_options) + "\nAnswer:"
            formatted_questions.append(question_text)

            orig_labels.append(" " + correct_label)
            edit_labels.append(" " + edit_label)


        return formatted_questions, orig_labels, edit_labels

    qs = [q['question'] for q in batch]
    corrects = [q['target_true'] for q in batch]
    incorrects = [q['targets_false'] for q in batch]
    edits = [q['edit_string'].strip() for q in batch]
    questions, orig_labels, edit_labels = generate_mcqs(qs, corrects, incorrects, edits)

    tokenizer.padding_side = "right"
    orig_labels_toks = tokenizer(orig_labels, return_tensors="pt", padding=True, add_special_tokens=False)['input_ids'][:, 0]
    edit_labels_toks = tokenizer(edit_labels, return_tensors="pt", padding=True, add_special_tokens=False)['input_ids'][:, 0]
    tokenizer.padding_side = "left"
    question_toks = tokenizer(questions, return_tensors="pt", padding=True)['input_ids']

    return (question_toks, orig_labels_toks, edit_labels_toks)

def unfreeze_mlps(model, mlp_str):
    # Unfreeze MLPs according to mlp_str
    for block, do_train in zip(model.blocks, mlp_str):
        if do_train == "1":
            block.mlp.W_out.requires_grad = True


lambda_inject = 2e-4
lambda_retain = 1e-3
lambda_sft = 1e-3
learning_rate = 1e-5
BATCH_SIZE = 4
epochs=10

ce_loss = torch.nn.CrossEntropyLoss(ignore_index = model.tokenizer.pad_token_id)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
run = wandb.init(
    project="mech-unlearning",  # Specify your project
    name=f"cfact_scale_{MODEL_NAME.replace('/', '_')}_{DATASET_LENGTH}_{MODE}",
    config={                        # Track hyperparameters and metadata
        "dataset_length": DATASET_LENGTH,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "lambda_inject": lambda_inject,
        "lambda_retain": lambda_retain,
        "lambda_sft": lambda_sft,
        "batch_size": BATCH_SIZE,
        "model": MODEL_NAME,
        "mode": MODE
    },
)
# Dataloaders
cfact_retain_iter = iter(cycle(DataLoader(cfact_retain.with_format("torch"), batch_size=BATCH_SIZE)))
pile_retain_iter = iter(DataLoader(pile_retain.with_format("torch"), batch_size=BATCH_SIZE))
#%%
for epoch in range(epochs):
    print(f"EPOCH: {epoch}")
    torch.cuda.empty_cache()
    gc.collect()
    
    for localization_str in cfact_forget['localization_str'].unique():
        torch.cuda.empty_cache()
        gc.collect()
        if MODE == "ATTN":
            for name, param in model.named_parameters():
                if "attn" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False 
        else:
            for param in model.parameters():
                param.requires_grad = False
            unfreeze_mlps(model, localization_str)
        # if MODE == "ATTN":
        #     for attn in 

        forget_prompts = cfact_forget[cfact_forget['localization_str'] == localization_str]['prompt'].tolist()
        forget_labels = cfact_forget[cfact_forget['localization_str'] == localization_str]['edit_string'].tolist()
        forget_orig_labels = torch.tensor(cfact_forget[cfact_forget['localization_str'] == localization_str]['correct_token'].tolist(), device=device)

        forget_toks = model.tokenizer(forget_prompts, return_tensors="pt", padding=True)["input_ids"].to(device)

        model.tokenizer.padding_side = "right"
        forget_labels = model.tokenizer(forget_labels, return_tensors="pt", padding=True, add_special_tokens=False)["input_ids"][:, 0].to(device)
        model.tokenizer.padding_side = "left"

        optimizer.zero_grad()
        # Process in batches of BATCH_SIZE
        
        pbar = tqdm(range(0, len(forget_toks), BATCH_SIZE))
        for i in pbar:
            torch.cuda.empty_cache()
            gc.collect()
            # Injection
            forget_logits = model(forget_toks[i:i+BATCH_SIZE])[:, -1, :]
            forget_preds = torch.argmax(forget_logits, dim=-1).detach()
            # for pred, orig, edit in zip(forget_preds, forget_orig_labels[i:i+BATCH_SIZE], forget_labels[i:i+BATCH_SIZE]):
            #     print(model.tokenizer.decode(pred), model.tokenizer.decode(orig), model.tokenizer.decode(edit))

            # Percent of time model predicts original label over edit label
            # forget_acc = torch.sum(forget_preds == forget_orig_labels[i:i+BATCH_SIZE]).item() / forget_preds.shape[0]
            forget_acc = torch.sum(forget_logits[range(forget_logits.size(0)), forget_labels[i:i+BATCH_SIZE]] < forget_logits[range(forget_logits.size(0)), forget_orig_labels[i:i+BATCH_SIZE]]).item() / forget_preds.shape[0]

            # Percent of time model predicts edited label over original label
            # edit_acc = torch.sum(forget_preds == forget_labels[i:i+BATCH_SIZE]).item() / forget_preds.shape[0]
            edit_acc = torch.sum(forget_logits[range(forget_logits.size(0)), forget_labels[i:i+BATCH_SIZE]] > forget_logits[range(forget_logits.size(0)), forget_orig_labels[i:i+BATCH_SIZE]]).item() / forget_preds.shape[0]

            loss = ce_loss(forget_logits, forget_labels[i:i+BATCH_SIZE])
            inject_loss = loss.item()
            loss.backward()

            torch.cuda.empty_cache()
            gc.collect()
            # Retain
            cfact_retain_batch = next(cfact_retain_iter)
            retain_toks = model.tokenizer(cfact_retain_batch['prompt'], return_tensors="pt", padding=True)["input_ids"].to(device)

            model.tokenizer.padding_side = "right"
            retain_labels = model.tokenizer(cfact_retain_batch['true_string'], return_tensors="pt", padding=True, add_special_tokens=False)["input_ids"][:, 0].to(device)
            model.tokenizer.padding_side = "left"
            loss = ce_loss(model(retain_toks)[:, -1, :], retain_labels)
            retain_loss = loss.item()
            loss.backward()

            torch.cuda.empty_cache()
            gc.collect()

            # SFT
            pile_retain_batch = next(pile_retain_iter)
            pile_toks = model.tokenizer(pile_retain_batch['text'], return_tensors="pt", padding=True, truncation=True, max_length=100)['input_ids'].to(device)
            pile_labels = pile_toks[:, 1:]
            logits = model(pile_toks)[:, :-1, :]
            loss = ce_loss(logits.reshape(-1, logits.size(-1)), pile_labels.reshape(-1))
            sft_loss = loss.item()
            loss.backward()

            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                # mcq eval
                mcq_iter = iter(DataLoader(
                    Dataset.from_pandas(merged_q), 
                    collate_fn=partial(collate_fn, tokenizer=model.tokenizer), 
                    batch_size=32
                ))
                mc_edit_acc = 0
                mc_orig_acc = 0
                mc_edit_over_orig_acc = 0
                total = 0
                for mcq_batch in mcq_iter:
                    question_toks, orig_labels, edit_labels = mcq_batch
                    question_toks = question_toks.to(device)
                    orig_labels = orig_labels.to(device)
                    edit_labels = edit_labels.to(device)

                    logits = model(question_toks)[:, -1, :]
                    preds = torch.argmax(logits, dim=-1)
                    mc_edit_acc += (preds == edit_labels).float().sum().item() 
                    mc_orig_acc += (preds == orig_labels).float().sum().item()
                    mc_edit_over_orig_acc += (logits[range(len(edit_labels)), edit_labels] > logits[range(len(orig_labels)), orig_labels]).float().sum().item()
                    total += len(edit_labels)
                mc_edit_acc /= total
                mc_orig_acc /= total
                mc_edit_over_orig_acc /= total
                # if mc_edit_acc > 0.9: break

            pbar.set_postfix(
                inject_loss=inject_loss, 
                retain_loss=retain_loss, 
                sft_loss=sft_loss, 
                forget_acc=forget_acc, 
                edit_acc=edit_acc,
                mc_edit_acc=mc_edit_acc,
                mc_orig_acc=mc_orig_acc,
                mc_edit_over_orig_acc=mc_edit_over_orig_acc
            )
            wandb.log({
                "inject_loss": inject_loss, 
                "retain_loss": retain_loss, 
                "sft_loss": sft_loss, 
                "forget_acc": forget_acc, 
                "edit_acc": edit_acc, 
                "mc_edit_over_orig_acc": mc_edit_over_orig_acc, 
                "mc_edit_acc": mc_edit_acc, 
                "mc_orig_acc": mc_orig_acc
            })
            torch.cuda.empty_cache()
            gc.collect()

with torch.no_grad():
    forget_acc = 0
    edit_acc = 0
    total = 0
    for localization_str in cfact_forget['localization_str'].unique():
        pbar = tqdm(range(0, len(forget_toks), BATCH_SIZE))
        for i in pbar:
            torch.cuda.empty_cache()
            gc.collect()
            forget_logits = model(forget_toks[i:i+BATCH_SIZE])[:, -1, :]
            forget_preds = torch.argmax(forget_logits, dim=-1).detach()
            # for pred, orig, edit in zip(forget_preds, forget_orig_labels[i:i+BATCH_SIZE], forget_labels[i:i+BATCH_SIZE]):
            #     print(model.tokenizer.decode(pred), model.tokenizer.decode(orig), model.tokenizer.decode(edit))

            # Percent of time model predicts original label over edit label
            # forget_acc = torch.sum(forget_preds == forget_orig_labels[i:i+BATCH_SIZE]).item() / forget_preds.shape[0]
            forget_acc += torch.sum(forget_logits[range(forget_logits.size(0)), forget_labels[i:i+BATCH_SIZE]] < forget_logits[range(forget_logits.size(0)), forget_orig_labels[i:i+BATCH_SIZE]]).item()

            # Percent of time model predicts edited label over original label
            # edit_acc = torch.sum(forget_preds == forget_labels[i:i+BATCH_SIZE]).item() / forget_preds.shape[0]
            edit_acc += torch.sum(forget_logits[range(forget_logits.size(0)), forget_labels[i:i+BATCH_SIZE]] > forget_logits[range(forget_logits.size(0)), forget_orig_labels[i:i+BATCH_SIZE]]).item()
            total += forget_preds.shape[0]
            pbar.set_postfix(forget_acc=forget_acc/total, edit_acc=edit_acc/total)
    forget_acc /= total
    edit_acc /= total
    wandb.log({"final_forget_acc": forget_acc, "final_edit_acc": edit_acc})
# %%
torch.save(model.state_dict(), f"/root/mechanistic-unlearning/experiments/scaling_counterfact/{MODEL_NAME.replace('/', '_')}_forget_{DATASET_LENGTH}_{MODE}.pth")
print('saved')
wandb.finish()
# %%
# model.load_state_dict(torch.load(f"/root/mechanistic-unlearning/experiments/scaling_counterfact/{MODEL_NAME.replace('/', '_')}_forget_{DATASET_LENGTH}facts.pth"))
# %%
