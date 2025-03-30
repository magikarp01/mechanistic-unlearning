#%%
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import pandas as pd

#%%
DATASET_LENGTH = 1000
MODE = "MANUAL"

model = HookedTransformer.from_pretrained("Qwen/Qwen2-1.5B", default_padding_side="left")
model.tokenizer.padding_side = "left"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%%
def extract_dataset(model, dataset):
    def extract(examples):
        true_string = [' ' + r['target_true']['str'] for r in examples['requested_rewrite']]
        edit_string = [' ' + r['target_new']['str'] for r in examples['requested_rewrite']]
        question_string = [r['prompt'].format(r['subject']) for r in examples['requested_rewrite']]
        subject_string = [r['subject'] for r in examples['requested_rewrite']]
        return {"prompt": question_string, "true_string": true_string, "edit_string": edit_string, "subject": subject_string}

    return dataset.map(
        lambda examples: extract(examples),
        batched=True,
        batch_size=10_000
    )
    
def correctness_filter(model, dataset, verbose=False):
    ''' Filter out any examples that the model gets wrong. '''
    def get_correctness(model, examples):
        ''' Populate dataset with first token of correct answer and model answer '''
        true_string = [' ' + r['target_true']['str'] for r in examples['requested_rewrite']]
        edit_string = [' ' + r['target_new']['str'] for r in examples['requested_rewrite']]
        question_string = [r['prompt'].format(r['subject']) for r in examples['requested_rewrite']]

        orig_padding_side = model.tokenizer.padding_side
        model.tokenizer.padding_side = "right"
        correct_tokens = list(model.tokenizer(true_string, return_tensors="pt", padding=True)["input_ids"][:, 0])
        edit_tokens = model.tokenizer(edit_string, return_tensors="pt", padding=True)["input_ids"][:, 0]
        model.tokenizer.padding_side = orig_padding_side

        question_tokens = model.tokenizer(question_string, return_tensors="pt", padding=True)["input_ids"]
        scaled_logits = torch.nn.functional.softmax(model(question_tokens)[:, -1, :], dim=-1)
        is_model_correct = list(scaled_logits[range(len(scaled_logits)), correct_tokens] > scaled_logits[range(len(scaled_logits)), edit_tokens])
        if verbose:
            print(f"String: {true_string[0]} tokenized as {correct_tokens[0]}, model is correct? {is_model_correct[0]}")

        return {"correct_token": correct_tokens, "is_model_correct": is_model_correct}
    
    with torch.no_grad():
        dataset = dataset.map(
            lambda row: get_correctness(model, row),
            batched=True,
            batch_size=1_000
        )
        dataset = dataset.filter(lambda x: x["is_model_correct"])

    assert all(dataset["is_model_correct"]), "Filter failed, model is not correct on all examples"

    return dataset

cfact_retain = load_dataset("azhx/counterfact", split="test")
cfact_retain = extract_dataset(model, cfact_retain)
cfact_retain = correctness_filter(model, cfact_retain, verbose=True)

# Important columns: prompt, correct_token, edit_str, localization_str
cfact_forget = pd.read_csv("~/mechanistic-unlearning/experiments/scaling_counterfact/cfact_forget.csv")
if MODE == "RANDOM":
    mlps_len = len(cfact_forget["localization_str"].iloc[0])
    cfact_forget["localization_str"] = ["".join([str(int(torch.rand(1) > 0.2)) for _ in range(mlps_len)]) for _ in range(len(cfact_forget))]
elif MODE == "ALL":
    mlps_len = len(cfact_forget["localization_str"].iloc[0])
    cfact_forget["localization_str"] = ["1" * mlps_len for _ in range(len(cfact_forget))]

pile_retain = load_dataset("EleutherAI/the_pile_deduplicated", split="train", streaming=True)

#%%
def unfreeze_mlps(model, mlp_str):
    # Unfreeze MLPs according to mlp_str
    for block, do_train in zip(model.blocks, mlp_str):
        if do_train == "1":
            block.mlp.W_out.requires_grad = True


lambda_inject = 1e-3
lambda_retain = 1e-3
lambda_sft = 1e-3
BATCH_SIZE = 2

ce_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Dataloaders
cfact_retain_iter = iter(DataLoader(cfact_retain.with_format("torch"), batch_size=BATCH_SIZE))
pile_retain_iter = iter(DataLoader(pile_retain.with_format("torch"), batch_size=BATCH_SIZE))

for localization_str in cfact_forget['localization_str'].unique():
    for param in model.parameters():
        param.requires_grad = False
    unfreeze_mlps(model, localization_str)

    forget_prompts = cfact_forget[cfact_forget['localization_str'] == localization_str]['prompt'].tolist()
    forget_labels = cfact_forget[cfact_forget['localization_str'] == localization_str]['edit_string'].tolist()

    forget_toks = model.tokenizer(forget_prompts, return_tensors="pt", padding=True)["input_ids"].to(device)

    model.tokenizer.padding_side = "right"
    forget_labels = model.tokenizer(forget_labels, return_tensors="pt", padding=True)["input_ids"][:, 0].to(device)
    model.tokenizer.padding_side = "left"

    optimizer.zero_grad()
    # Process in batches of BATCH_SIZE
    pbar = tqdm(range(0, len(forget_toks), BATCH_SIZE))
    for i in pbar:
        # Injection
        loss = ce_loss(model(forget_toks[i:i+BATCH_SIZE])[:, -1, :], forget_labels[i:i+BATCH_SIZE])
        inject_loss = loss.item()
        loss.backward()

        # Retain
        cfact_retain_batch = next(cfact_retain_iter)
        retain_toks = model.tokenizer(cfact_retain_batch['prompt'], return_tensors="pt", padding=True)["input_ids"].to(device)

        model.tokenizer.padding_side = "right"
        retain_labels = model.tokenizer(cfact_retain_batch['edit_string'], return_tensors="pt", padding=True)["input_ids"][:, 0].to(device)
        model.tokenizer.padding_side = "left"
        loss = ce_loss(model(retain_toks)[:, -1, :], retain_labels)
        retain_loss = loss.item()
        loss.backward()

        # SFT
        pile_retain_batch = next(pile_retain_iter)
        pile_toks = model.tokenizer(pile_retain_batch['text'], return_tensors="pt", padding=True, truncation=True, max_length=100)["input_ids"].to(device)
        pile_labels = pile_toks[:, 1:]
        logits = model(pile_toks)[:, :-1, :]
        loss = ce_loss(logits.reshape(-1, logits.size(-1)), pile_labels.reshape(-1))
        sft_loss = loss.item()
        loss.backward()
        pbar.set_postfix(inject_loss=inject_loss, retain_loss=retain_loss, sft_loss=sft_loss)

    optimizer.step()
# %%
