# %%
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from cb_utils.transformer import DemoTransformer as GPT2DemoTransformer, Config as GPT2Config
from cb_utils.pythia_transformer import DemoTransformer as PythiaDemoTransformer, Config as PythiaConfig
from easy_transformer import EasyTransformer
import torch
import pickle

# %%

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
RATIO = 0.2

# %% 

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id

# %%
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
model.to(DEVICE)

# %%

def import_finetuned_model(mode="finetuned", ratio=RATIO):
    model_ft = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
    if mode == "finetuned":
        model_ft.load_state_dict(torch.load(f"models/joint_inf_ratio={ratio}_max_20.pt"))
    elif mode == "ascent":
        model_ft.load_state_dict(torch.load(f"models/ascent_non_toxic_model_best_50_epochs.pt"))
    elif mode == "algebra":
        model_ft.load_state_dict(torch.load(f"models/task_algebra_non_toxic_model.pt"))
    else:
        raise Exception("Model not found")
    model_ft.to(DEVICE)
    return model_ft

# %%

def import_ablated_model(version, means):
    with open(f"models/masked_gpt2_mean_ablation_v{version}.pkl", "rb") as f: #"masked_gpt2.pkl", "rb") as f:
        gpt2_weights = pickle.load(f)() #()
    demo_gpt2 = GPT2DemoTransformer(GPT2Config(debug=False), means)
    demo_gpt2.load_state_dict(gpt2_weights, strict=False)
    demo_gpt2.to(DEVICE)
    return demo_gpt2

# %%
def load_gpt2_weights():
    reference_gpt2 = EasyTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
    with open("models/gpt2_weights.pkl", "wb") as f:
        pickle.dump(reference_gpt2.state_dict(), f)


# %%
# def load_demo_gpt2(means, edge_masks=True, mask_dict_superset=None, weight_masks_attn=False, weight_masks_mlp=False, weight_mask_attn_dict=None, weight_mask_mlp_dict=None, n_layers=12, n_heads=12):
def load_demo_gpt2(means, n_layers=12, n_heads=12, **kwargs):
    with open("models/gpt2_weights.pkl", "rb") as f:
        gpt2_weights = pickle.load(f)
    # demo_gpt2 = DemoTransformer(Config(debug=False, n_layers=n_layers, n_heads=n_heads), means, edge_masks=edge_masks, mask_dict_superset=mask_dict_superset, weight_masks_attn=weight_masks_attn, weight_masks_mlp=weight_masks_mlp,
    #                             weight_mask_attn_dict=weight_mask_attn_dict, weight_mask_mlp_dict=weight_mask_mlp_dict)
    demo_gpt2 = GPT2DemoTransformer(GPT2Config(debug=False, n_layers=n_layers, n_heads=n_heads), means, **kwargs)
    demo_gpt2.load_state_dict(gpt2_weights, strict=False)
    demo_gpt2.cuda()
    return demo_gpt2


# 2.8b: d_model = 2560, 32 layers, 32 heads
def load_demo_pythia(means, n_layers=32, n_heads=32, d_model=2560, **kwargs):
    with open("models/pythia_weights.pkl", "rb") as f:
        pythia_weights = pickle.load(f)
    if d_head is None:
        d_head = d_model // n_heads
    if d_mlp is None:
        d_mlp = 4 * d_model
    
    demo_pythia = PythiaDemoTransformer(PythiaConfig(debug=False, n_layers=n_layers, n_heads=n_heads, d_model=d_model, d_head=d_head, d_mlp=d_mlp), means, **kwargs)
    demo_pythia.load_state_dict(pythia_weights, strict=False)
    demo_pythia.cuda()
    return demo_pythia
