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
import gc
import random

#%%

MODEL_NAME = "google/gemma-7b"
DATASET_LENGTH = 30
MODE = "MANUAL"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained(MODEL_NAME, default_padding_side="left", device=device)
model.tokenizer.padding_side = "left"
torch.cuda.empty_cache()
model.load_state_dict(
    torch.load(f"/root/mechanistic-unlearning/experiments/scaling_counterfact/{MODEL_NAME.replace('/', '_')}_forget_{DATASET_LENGTH}_{MODE}.pth")
)
torch.cuda.empty_cache()


#%%
mcq_path = "/root/mechanistic-unlearning/tasks/facts/data/counterfact_mc_questions.parquet"
cfact_path = "/root/mechanistic-unlearning/experiments/scaling_counterfact/cfact_forget_google_gemma-7b.csv"
mcq = pd.read_parquet(mcq_path)
mcq['case_id'] = mcq['prompt_id']
cfact = pd.read_csv(cfact_path)
merged_q = pd.merge(mcq, cfact, how="inner", on="case_id")
# %%
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
            labeled_options = [f"{label}. {opt}" for label, opt in zip(option_labels, options)]

            # Format question string
            question_text = icl_prompt + f"{q}\n" + "\n".join(labeled_options) + "\nAnswer:"
            # question_text = f"{q}\n" + "\n".join(labeled_options) + "\nAnswer:"
            formatted_questions.append(question_text)

            # Find correct answer's label
            correct_label = option_labels[options.index(correct)]
            edit_label = option_labels[options.index(edit)]
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
# %% MCQ
from functools import partial
dl = DataLoader(
    Dataset.from_pandas(merged_q), 
    collate_fn=partial(collate_fn, tokenizer=model.tokenizer), 
    batch_size=4
)

with torch.no_grad():
    for (question_toks, orig_labels, edit_labels) in dl:
        question_toks = question_toks.to(device)
        orig_labels = orig_labels.to(device)
        edit_labels = edit_labels.to(device)

        logits = model(question_toks)[:, -1, :]
        preds = torch.argmax(logits, dim=-1)
        print(model.tokenizer.batch_decode(preds))
        print(model.tokenizer.decode(orig_labels))
        print(model.tokenizer.decode(edit_labels))
        edit_acc = (preds == edit_labels).float().mean().item()
        orig_acc = (preds == orig_labels).float().mean().item()

        edit_over_orig_acc = (logits[range(len(edit_labels)), edit_labels] > logits[range(len(orig_labels)), orig_labels]).float().mean().item()

        print(f"{edit_acc=}, {orig_acc=}, {edit_over_orig_acc=}")

#%% Latent Probes
from functools import partial
dl = DataLoader(
    Dataset.from_pandas(merged_q), 
    collate_fn=partial(collate_fn, tokenizer=model.tokenizer), 
    batch_size=4
)

with torch.no_grad():
    for (question_toks, orig_labels, edit_labels) in dl:
        question_toks = question_toks.to(device)
        orig_labels = orig_labels.to(device)
        edit_labels = edit_labels.to(device)
