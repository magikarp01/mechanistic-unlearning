from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer
import gc
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
hf_model = AutoModelForCausalLM.from_pretrained("google/gemma-7b", device_map="auto", torch_dtype=torch.float16).to(device)
hf_model.eval()
hf_model.half()

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to(device)

hf_logits = hf_model(input_ids.input_ids)

del hf_model
torch.cuda.empty_cache()
gc.collect()

tl_model = HookedTransformer.from_pretrained(
    "google/gemma-7b",
    tokenizer=tokenizer,
    device='cuda',
    fold_ln=False,
    fold_value_biases=False,
    center_writing_weights=False,
    dtype=torch.float16
)

# tl_model.set_use_attn_result(True)
# tl_model.set_use_hook_mlp_in(True)

tl_model.eval()
tl_logits = tl_model(input_ids.input_ids)

print(
    torch.allclose(
        torch.nn.functional.softmax(hf_logits.logits, dim=-1).type(torch.float16), 
        torch.nn.functional.softmax(tl_logits, dim=-1), 
        atol=1e-2
    )
)
