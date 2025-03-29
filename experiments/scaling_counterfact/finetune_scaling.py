#%%
import torch
from transformer_lens import HookedTransformer
from datasets import load_dataset
import pandas as pd

#%%
DATASET_LENGTH = 1000

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

cfact_retain = load_dataset("azhx/counterfact", "test")
cfact_retain = extract_dataset(model, cfact_retain)
cfact_retain = correctness_filter(model, cfact_retain, verbose=True)

# Important columns: prompt, correct_token, edit_str, localization_str
cfact_forget = pd.read_csv("experiments/scaling_counterfact/cfact_forget.csv")

pile_retain = load_dataset("EleutherAI/the_pile_deduplicated", streaming=True)

#%%
def freeze_mlps(model, mlp_str):
    # Freeze all mlps
    for block, do_train in zip(model.blocks, mlp_str):
        block.mlp.W_in.requires_grad = False
        block.mlp.W_mid.requires_grad = False
        if do_train == "0":
            block.mlp.W_out.requires_grad = False
        elif do_train == "1":
            block.mlp.W_out.requires_grad = True


lambda_inject = 1e-3
lambda_retain = 1e-3
lambda_sft = 1e-3

inject_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for localization_str in cfact_forget['localization_str'].unique():
    print(localization_str)
    freeze_mlps(model, localization_str)
    prompts = cfact_forget['prompt'][cfact_forget['localization_str'] == localization_str]
    labels = cfact_forget['edit_string'][cfact_forget['localization_str'] == localization_str]

    toks = model.tokenizer(prompts, return_tensors="pt", padding=True)["input_ids"]

    model.tokenizer.padding_side = "right"
    label_toks = model.tokenizer(labels, return_tensors="pt", padding=True)["input_ids"][:, 0]
    model.tokenizer.padding_side = "left"

    optimizer.zero_grad()
    # Process in batches of 50
    for i in range(0, len(toks), 50):
        logits = model(toks[i:i+50])
        loss = inject_loss(logits, labels[i:i+50])
        print(loss.item())
        loss.backward()
    # Add retain loss for remaining counterfact tasks
        
    # Add SFT loss for Pile dataset

    optimizer.step()