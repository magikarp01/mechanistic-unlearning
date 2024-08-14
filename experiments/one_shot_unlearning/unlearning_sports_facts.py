#%%
%cd ~/mechanistic-unlearning
from tasks import SportsFactsTask
from transformer_lens import HookedTransformer
import torch

#%%
QWEN_CHAT_TEMPLATE_WITH_SYSTEM = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

def format_instruction_qwen_chat(
    instruction: str,
    system: str=None,
    include_trailing_newline: bool=True
):
    if system is not None:
        formatted_instruction = QWEN_CHAT_TEMPLATE_WITH_SYSTEM.format(instruction=instruction, system=system)
    else:
        formatted_instruction = QWEN_CHAT_TEMPLATE.format(instruction=instruction)
    if not include_trailing_newline:
        formatted_instruction = formatted_instruction.strip()
    return formatted_instruction

def tokenize_instructions_qwen_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    system: str=None,
    include_trailing_newline=True
):
    prompts = [
        format_instruction_qwen_chat(instruction=instruction, system=system, include_trailing_newline=include_trailing_newline)
        for instruction in instructions
    ]

    toks = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt"
    ).input_ids

    return toks
#%%

model = HookedTransformer.from_pretrained(
    'Qwen/Qwen-1_8B-Chat',
    device='cuda',
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    default_padding_side="left",
    dtype=torch.bfloat16
)
tokenizer = model.tokenizer

# %%
from functools import partial
format_func = partial(
    format_instruction_qwen_chat,
    tokenizer=tokenizer
)
sports_task = SportsFactsTask(
    model,
    tokenizer,
    tokenize_instructions=format_func,
    N=100
)

# %%
