"""
Utils for inference with toxic and OWT data.
"""

# %% 
import torch
from tqdm import tqdm
from einops import repeat
from models import DEVICE

criterion = torch.nn.CrossEntropyLoss()
itemized_criterion = torch.nn.CrossEntropyLoss(reduction='none')
BATCH_SIZE_INFERENCE = 100

# %%

def prepare_fixed_demo(tokenizer, batch_size, demo="", device="cuda"):
    # first, encode the demos
    demo = tokenizer(demo, return_tensors="pt").input_ids.to(device)
    # remove batch dimension 
    demo = demo[0]
    # remove end token
    demo = demo[:-1]

    demos = repeat(demo, "l -> b l", b=batch_size).long()
    return demos

def infer_batch(model, criterion, batch, batch_size, demos, device=DEVICE, itemized=False):
    # cast the entire batch tensor to torch.long
    demos = demos.long().to(device)
    batch = batch.long().to(device)

    # remove start token 
    batch = batch[:, 1:]
    
    # concatenate the demos and the batch
    # if batch size is < batch_size, remove some demos
    if batch.shape[0] < batch_size:
        demos = demos[:batch.shape[0]]
    input = torch.cat([demos, batch], dim=1)

    out = model(input)[0]  # 0 is the logits

    return evaluate_sequence_loss(out, input, criterion, demos.shape[1], itemized=itemized)

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id
def batch_text_to_tokens(x, tokenizer=tokenizer, ctx_length=50, pad_max=False):
    if ctx_length is None:
        return tokenizer(x['text'], padding='max_length' if pad_max else True, truncation=True, return_tensors='pt').input_ids.long()
    else:
        return tokenizer(x['text'], max_length=ctx_length, padding='max_length' if pad_max else True, truncation=True, return_tensors='pt').input_ids.long()

def infer_batch_with_owt(model, criterion, toxic_batch, owt_batch, batch_size, demos, device="cuda", access_toxic_pos=None):
    # encode the batch
    # toxic_batch = tokenizer(toxic_batch, return_tensors="pt", padding=True).input_ids.to(device)
    losses = [0, 0]
    for idx, batch in enumerate([toxic_batch, owt_batch]):

        # batch = torch.cat([toxic_batch.to(device), owt_batch.to(device)], dim=0)
        # cast the entire batch tensor to torch.long
        # batch = batch.long()
        if idx == 0:
            batch = batch_text_to_tokens(batch, pad_max=False, ctx_length=None)
        else:
            batch = batch_text_to_tokens(batch, pad_max=True, ctx_length=50)
        # remove start token 
        batch = batch[:, 1:].to(device)
        
        # concatenate the demos and the batch
        # if batch size is < batch_size, remove some demos

        if batch.shape[0] < batch_size:
            demos = demos[:batch.shape[0]]
        # input = torch.cat([demos, batch], dim=1)
        input = batch

        # print(input.shape, input.dtype)

        # generate the output
        out = model(input)[0]  # 0 is the logits

        # print(f"{out.shape=}, {demos.shape=}")

        losses[idx] = evaluate_sequence_loss(out, input, criterion, demos.shape[1], access_seq_pos=access_toxic_pos if idx == 0 else None)
    return (losses[0], losses[1])

def evaluate_sequence_loss(logits, batch, criterion, demo_len=0, itemized=False, access_seq_pos=None):
    """
    If access_seq_pos is not None, then we only evaluate the loss for the token at that position. Else, it should be position of correct indirect object
    """
    # get the logits for all tokens after the last demo
    if access_seq_pos is None:
        logits = logits[:, demo_len:-1]
    else:
        logits = logits[:, access_seq_pos-1:access_seq_pos]

    # get the target labels by shifting the input batch to the left by one
    if access_seq_pos is None:
        target_labels = batch[:, demo_len+1:].long()
    else:
        target_labels = batch[:, access_seq_pos].long()

    # print(f"{logits.shape=}, {target_labels.shape=}")
    # print(f"logit_shape: {logits.shape}, target labels: {tokenizer.batch_decode(target_labels)}")
    # print(f"Sentence: {tokenizer.batch_decode(batch)}")

    if itemized is False:
        logits = logits.flatten(0,1)
        target_labels = target_labels.flatten()
    else:
        logits = logits.permute(0,2,1)
    
    return criterion(logits, target_labels)

def generate_text(model, tokenizer, prompt, max_length=20, temperature=0):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)
    output = model.generate(input_ids, temperature=temperature, max_new_tokens=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_from_tokens(model, input_ids, max_length=50, temperature=0, attention_mask=None, return_new_only=True):
    input_ids = input_ids.long()
    orig_len = input_ids.shape[1]
    for _ in tqdm(range(max_length)):
        if attention_mask is None:
            out = model(input_ids)[0]
        else:
            out = model(input_ids, attention_mask=attention_mask)[0]
        logits = out[:, -1, :]

        if temperature == 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits /= temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(dim=0)
        # next_token = torch.multinomial(probs, num_samples=1).squeeze()

        input_ids = torch.cat([input_ids,next_token], dim=-1)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1)).to(DEVICE)], dim=-1)
    if return_new_only:
        return input_ids[:,orig_len:]
    return input_ids

# batched
def generate_no_hf(model, tokenizer, prompts, max_length=50, temperature=0, return_new_only=True):
    prompts_batch = tokenizer(prompts, return_tensors='pt', padding=True).to(DEVICE)
    input_ids = prompts_batch['input_ids']
    attention_mask = prompts_batch['attention_mask']
    completions = generate_from_tokens(model, tokenizer, input_ids, max_length, temperature, attention_mask, return_new_only)
    return tokenizer.batch_decode(completions, skip_special_tokens=True)

# "non"-batched (data is still batched, but it's not batched model evaluation)
def generate_no_hf_new(model, tokenizer, prompts, max_length=50, temperature=0, return_new_only=True):
    outputs = []
    for prompt in tqdm(prompts):
        prompt = tokenizer.encode(prompt, return_tensors='pt', padding=True).to(DEVICE)
        # input_ids = prompts['input_ids']
        # attention_mask = prompts['attention_mask']
        orig_len = prompt.shape[1]
        
        for _ in range(max_length):
            out = model(prompt)[0]
            logits = out[:, -1, :]

            if temperature == 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits /= temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            # next_token = torch.multinomial(probs, num_samples=1).squeeze()

            prompt = torch.cat([prompt,next_token], dim=-1)
            # input_ids = torch.cat([input_ids,next_token], dim=-1)
            # attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1)).to(DEVICE)], dim=-1)
        if return_new_only:
            outputs.append(tokenizer.decode(prompt[orig_len:], skip_special_tokens=True))
        else:
            outputs.append(tokenizer.decode(prompt, skip_special_tokens=True))
    return outputs