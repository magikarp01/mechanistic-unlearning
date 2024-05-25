#%%
%cd ~/mechanistic-unlearning
import torch 
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformer_lens import HookedTransformer, utils
import einops

from typing import Callable, List, Tuple
from jaxtyping import Float, Int
import random
from tqdm import tqdm
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# %%
def get_valid_toks(tokenizer) -> List[int]:
    '''
        Get all valid tokens from the tokenizer.
        A token is valid if it is ascii, printable and not a special token.

        Args:
            tokenizer (AutoTokenizer): HuggingFace tokenizer
        
        Returns:
            List[int]: List of valid token ids
    '''

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(0, tokenizer.vocab_size):
        if is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)
    
    special_toks = [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id]
    ascii_toks = [tok for tok in ascii_toks if tok not in special_toks]
    
    return ascii_toks

def random_suffix(
    instruction: str, 
    suffix_tok_len: int,
    inst_format_fn = lambda x, y: x.format(instruction=y),
    inst_before_suffix: str = "{instruction}",
    inst_after_suffix: str = "",
) -> str:
    '''
        Generate a random suffix that has suffix_tok_len tokens.
        The suffix must tokenize to the same length as the suffix_tok_len. 

        Args:
            instruction (str): Instruction to generate suffix for
            suffix_tok_len (int): Length of the suffix in tokens
            inst_format_fn (Callable[[str, str], str]): Function to format the instruction with the suffix
            inst_before_suffix (str): Instruction string before the suffix eg: <|im_start|>user
            inst_after_suffix (str): Instruction string after the suffix eg: <|im_end|>\n<|im_start|>assistant

        Returns:
            str: Random suffix that tokenizes to the same length as suffix_tok_len
    '''
    random_suffix = None
    
    max_iter = 100
    prompt_tok_len = len(tokenizer.encode(
        inst_format_fn(
            instruction,
            inst_before_suffix + inst_after_suffix 
        )
    )) + suffix_tok_len

    for _ in range(max_iter):
        # try to find a random suffix that tokenizes to the same length as the suffix
        random_suffix_cand_toks = random.sample(valid_toks, suffix_tok_len)
        random_suffix_cand = tokenizer.decode(random_suffix_cand_toks)

        rand_suffix_cand_toks = tokenizer.encode(random_suffix_cand)
        if rand_suffix_cand_toks[0] == 0 or rand_suffix_cand_toks[0] == 2:
            rand_suffix_cand_toks = rand_suffix_cand_toks[1:]
        rand_suffix_cand_len = len(rand_suffix_cand_toks)
        rand_prompt_cand_len = len(
            tokenizer.encode(
                inst_format_fn(
                    instruction + random_suffix_cand,
                    inst_before_suffix + inst_after_suffix 
                )
            )
        )

        if rand_suffix_cand_len == suffix_tok_len and rand_prompt_cand_len == prompt_tok_len:
            # found a nice suffix
            random_suffix = random_suffix_cand
            break

    if random_suffix is None:
        raise Exception("Could not find a random suffix that preserves token length")

    return random_suffix

def fwd_pass_with_embeds(
    model: HookedTransformer, 
    prompt_toks, 
    suffix_embeds, 
    target_toks,
    inst_after_suffix: str = "",
):
    # List to hold inst_before_suffix toks, prompt toks, suffix toks, inst_after_suffix toks, target_toks
    embeds_list = []
    embeds_list.append(model.W_E[prompt_toks]) # <im_start>user\n{instruction}
    embeds_list.append(suffix_embeds)

    if inst_after_suffix != "":
        end_embeds = model.W_E[
            model.tokenizer.encode(
                inst_after_suffix, 
                return_tensors='pt'
            ).squeeze()
        ] # <im_end>\n<im_start>assistant\n
        embeds_list.append(end_embeds)

    embeds_list.append(model.W_E[target_toks]) # {target}

    embeddings = torch.cat(
        embeds_list, 
        dim=0
    )
    embeddings = embeddings.unsqueeze(0) # add artificial batch TODO implement proper batching
    # Pass through transformer blocks
    for block in model.blocks:
        embeddings = block(embeddings)
    
    # Pass through ln_final
    ln_final = model.ln_final(embeddings)
    # Pass through unembed
    unembed = model.unembed(ln_final)
    return unembed.squeeze(0)

def greedy_coordinate_descent(
    model,
    prompt,
    target,
    inst_format_fn = lambda x, y: x.format(instruction=y),
    inst_before_suffix: str = "{instruction}",
    inst_after_suffix: str = "",
    suffix_tok_len: int = 20,
    num_iter: int = 50,
    top_k: int = 128,
    batch_size: int = 64,
    suffix_update_size: int = 20,
    suffix_buffer_size: int = 16,
    verbose: bool = True
):
    '''
        Greedy Coordinate Descent

        Args:
            model: HookedTransformer
            prompt_toks: The tokens of the prompt, including the start instruction tokens
            suffix_toks: The tokens of the suffix
            target_toks: The tokens of the target
            num_iter: Number of iterations to run
            top_k: Number of top candidates to consider for each token
            batch_size: Number of candidates to sample from top_k, increases over num_iter
            suffix_update_size: Number of tokens to update in each suffix at once, decreases over num_iter
            suffix_buffer_size: Number of suffixes to keep track of in history

        Returns:
            The suffix with the lowest loss
    '''
    
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # Get embeds for prompt, suffix, end, target
    prompt_toks = model.tokenizer.encode(
        inst_format_fn(
            prompt,
            inst_before_suffix
        ),
        return_tensors="pt"
    ).squeeze()

    target_toks = model.tokenizer.encode(target, return_tensors='pt').squeeze().to(device) # seq

    target_seq = target_toks.shape[-1] # length of target sequence

    # Sorted list storing (loss, previous suffixes) (lowest to highest loss)
    suffix_history = []
    # Initialize list with suffix_buffer_size random suffixes
    for _ in range(suffix_buffer_size):
        rand_suffix = model.tokenizer.encode(random_suffix(prompt, suffix_tok_len, inst_format_fn, inst_before_suffix, inst_after_suffix), return_tensors='pt').squeeze() 

        # Evaluate this suffix
        with torch.no_grad():
            rand_suffix_embeds = model.W_E[rand_suffix].detach()
            rand_outputs = fwd_pass_with_embeds(
                model, 
                prompt_toks, 
                rand_suffix_embeds, 
                target_toks,
                inst_after_suffix
            )
            rand_target_logits = rand_outputs[-target_seq-1:-1]
            rand_ce_loss = loss_fn(rand_target_logits, target_toks).item()

        suffix_history.append((rand_ce_loss.item(), rand_suffix.clone()))

    init_suffix_update_size = suffix_update_size
    init_batch_size = batch_size

    # Perform num_iter iterations
    for curr_iter in tqdm(range(num_iter)):
        new_suffix = suffix_history[0][1]
        suffix_embeds = model.W_E[new_suffix].detach() # {suffix}
        suffix_embeds.requires_grad_(True)

        with torch.set_grad_enabled(True):
            # Forward pass
            outputs = fwd_pass_with_embeds(
                model, 
                prompt_toks, 
                suffix_embeds, 
                target_toks
            )

            # Get loss w.r.t suffix one hot encoded vector
            target_logits = outputs[-target_seq-1:-1]
            generated_target = outputs.argmax(-1)[-target_seq-1:-1] 

            ce_loss = loss_fn(target_logits, target_toks)
            ce_loss.backward()

        if verbose:
            print(
                f'Generated tokens: {repr(model.tokenizer.decode(generated_target.tolist()))}; Loss: {ce_loss.item()}'
            )

        # Check if target logits equal the target tokens, if so we are done!
        if torch.all(generated_target == target_toks):
            break

        # Get top k replacement candidates
        topk_subs = torch.topk(suffix_embeds.grad, top_k, dim=1).indices

        for _ in range(batch_size):
            updated_suffix = new_suffix.detach().clone()

            # Pick suffix_update_size number of tokens in the suffix to update
            suffix_update_indices = random.sample(range(suffix_embeds.shape[0]), suffix_update_size)
            # Update these indices by randomly picking from the corresponding top k candidates
            for i in suffix_update_indices:
                updated_suffix[i] = topk_subs[i, random.randint(0, top_k-1)]

            # Evaluate this new suffix
            with torch.no_grad():
                updated_suffix_embeds = model.W_E[updated_suffix].detach()
                updated_outputs = fwd_pass_with_embeds(
                    model, 
                    prompt_toks, 
                    updated_suffix_embeds, 
                    target_toks,
                    inst_after_suffix
                )
                updated_target_logits = updated_outputs[-target_seq-1:-1]
                updated_ce_loss = loss_fn(updated_target_logits, target_toks).item()
            
            # Update history list and sort if necessary
            if updated_ce_loss < suffix_history[-1][0]:
                suffix_history.pop()
                suffix_history.append((updated_ce_loss, updated_suffix.clone()))
                suffix_history = sorted(suffix_history, key=lambda x: x[0])

        # suffix_update_size needs to go to 1 as we get closer to num_iter
        # batch_size needs to increase as we get closer to num_iter
        # TODO: Experiment
        suffix_update_size -= 2 if curr_iter % (num_iter // init_suffix_update_size) == 0 else 0
        suffix_update_size = max(1, suffix_update_size)
        batch_size += 10 if curr_iter % 10 == 0 else 0 

        if verbose:
            print(f'Suffix update size: {suffix_update_size}; Batch size: {batch_size}')
            print(f'Best suffix: {repr(model.tokenizer.decode(suffix_history[0][1].tolist()))}; Loss: {suffix_history[0][0]}')

    # Return suffix with lowest loss
    return suffix_history[0]

#%%
from transformers import AutoTokenizer
import os
import gc

def threshold_mask(mask, threshold):
    for layer in mask.keys():
        for name, param in mask[layer].items():
            mask[layer][name] = torch.where(param < threshold, torch.zeros_like(param), torch.ones_like(param))

def apply_mask(model, mask):
    for layer in mask.keys():
        for name, mask_weight in mask[layer].items():
            if getattr(model.blocks[layer].attn, name, None) is not None:
                param = getattr(model.blocks[layer].attn, name)
                param.data = param * mask_weight
            elif getattr(model.blocks[layer].mlp, name, None) is not None:
                param = getattr(model.blocks[layer].mlp, name)
                param.data = param * mask_weight
            else:
                raise ValueError(f"Invalid mask name: {name} {layer=}")

def sort_mask_weights(mask):
    sorted_nonzero = []
    for layer in mask.keys():
        for param in mask[layer].values():
            sorted_nonzero.append(param[param < 1].flatten())
    return torch.cat(sorted_nonzero).sort().values 

os.environ['HF_TOKEN'] = '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'
model_name = 'google/gemma-7b'
# model_name = 'Qwen/Qwen1.5-0.5B-Chat'
tokenizer = AutoTokenizer.from_pretrained(model_name)
def load_model(model_name=model_name):
    model = HookedTransformer.from_pretrained(
        model_name,
        tokenizer=tokenizer,
        device='cuda',
        default_padding_side="right",
        fold_ln=False,
        fold_value_biases=False,
        center_writing_weights=False,
        dtype=torch.bfloat16
    )
    return model

num_weights = 400_000
forget_sport = "basketball"
localization_type="ap"
valid_toks = get_valid_toks(tokenizer)

model = load_model()
mask = torch.load(f"results/{model_name.replace('/', '_')}-{forget_sport}-{localization_type}.pt")

sorted_nonzero = sort_mask_weights(mask)
threshold = sorted_nonzero[num_weights - 1]
threshold_mask(mask, threshold)
apply_mask(model, mask)

del mask
gc.collect()
torch.cuda.empty_cache()
#%%
QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}"""
END_CHAT_TEMPLATE = """<|im_end|>
<|im_start|>assistant
"""

inst_before_suffix = "{instruction}" #QWEN_CHAT_TEMPLATE
inst_after_suffix = "" #END_CHAT_TEMPLATE
inst_format_fn = lambda inst, template: template.format(instruction=inst)

loss, suffix = greedy_coordinate_descent(
    model,
    "Fact: Tiger Woods plays the sport of golf\nFact: LeBron James plays the sport of",
    " basketball",
    inst_format_fn=inst_format_fn,
    inst_before_suffix=inst_before_suffix,
    inst_after_suffix=inst_after_suffix,
)

# %%
suffix = "atsныboxkerchoolatski edameляbucode<unused87> af conf K down betteronaprivate\x17"
model.generate(
    inst_format_fn(
        "Fact: Tiger Woods plays the sport of golf\nFact: LeBron James plays the sport of" ,
        inst_before_suffix
    ) + suffix + inst_after_suffix,
    do_sample=False,
    temperature=0,
    max_new_tokens=100
)

# %%
