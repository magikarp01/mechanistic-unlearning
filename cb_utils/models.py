# %%
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer
from cb_utils.transformer import DemoTransformer as GPT2DemoTransformer, Config as GPT2Config
from cb_utils.transformers.gpt2.edge_masked_transformer import DemoTransformer as GPT2EdgeDemoTransformer
from cb_utils.transformers.gpt2.weight_masked_transformer import DemoTransformer as GPT2WeightDemoTransformer

# from cb_utils.pythia_transformer import DemoTransformer as PythiaEdgeDemoTransformer, Config as PythiaConfig
# from cb_utils.pythia_weight_masked_transformer import DemoTransformer as PythiaWeightDemoTransformer
from cb_utils.transformers.pythia.edge_masked_transformer import DemoTransformer as PythiaEdgeDemoTransformer, Config as PythiaConfig
from cb_utils.transformers.pythia.weight_masked_transformer import DemoTransformer as PythiaWeightDemoTransformer


from easy_transformer import EasyTransformer
import torch
import pickle
from transformer_lens import HookedTransformer

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
def load_demo_gpt2(
    means, 
    n_layers=12, 
    n_heads=12, 
    edge_mask=False, 
    weight_mask=False, 
    return_tokenizer=False,
    **kwargs
):
    with open("models/gpt2_weights.pkl", "rb") as f:
        gpt2_weights = pickle.load(f)
    # demo_gpt2 = DemoTransformer(Config(debug=False, n_layers=n_layers, n_heads=n_heads), means, edge_masks=edge_masks, mask_dict_superset=mask_dict_superset, weight_masks_attn=weight_masks_attn, weight_masks_mlp=weight_masks_mlp,
    #                             weight_mask_attn_dict=weight_mask_attn_dict, weight_mask_mlp_dict=weight_mask_mlp_dict)
    # demo_gpt2 = GPT2DemoTransformer(GPT2Config(debug=False, n_layers=n_layers, n_heads=n_heads), means, **kwargs)
    if edge_mask:
        demo_gpt2 = GPT2EdgeDemoTransformer(GPT2Config(debug=False, n_layers=n_layers, n_heads=n_heads), means, edge_masks=True, **kwargs)
        print("Loaded edge-masked transformer")
    elif weight_mask:
        demo_gpt2 = GPT2WeightDemoTransformer(GPT2Config(debug=False, n_layers=n_layers, n_heads=n_heads), weight_masks_attn=True, weight_masks_mlp=True, **kwargs)
        print("Loaded weight-masked transformer")
    else:
        print("Unsure which transformer, loading default transformer")
        demo_gpt2 = GPT2DemoTransformer(GPT2Config(debug=False, n_layers=n_layers, n_heads=n_heads), means, **kwargs)
    demo_gpt2.load_state_dict(gpt2_weights, strict=False)
    demo_gpt2.cuda()
    if return_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", add_bos_token=True)
        return demo_gpt2, tokenizer
    return demo_gpt2

from cb_utils.pythia_transformer import DemoTransformer, Config
def tl_config_to_demo_config(tl_config, debug=False):
    return Config(
            d_model = tl_config.d_model,
            debug = debug,
            layer_norm_eps = tl_config.eps,
            d_vocab = tl_config.d_vocab,
            init_range = tl_config.initializer_range,
            n_ctx = tl_config.n_ctx,
            d_head = tl_config.d_head,
            d_mlp = tl_config.d_mlp,
            n_heads = tl_config.n_heads,
            n_layers = tl_config.n_layers,
            positional_embedding_type = tl_config.positional_embedding_type,
            rotary_dim=tl_config.rotary_dim,
            rotary_base=10000,
            dtype=tl_config.dtype,
        )

# 2.8b: d_model = 2560, 32 layers, 32 heads
def load_demo_pythia(
    means=False, model_name="pythia-2.8b", n_layers=32, n_heads=32, d_model=2560, d_head=None, d_mlp=None, edge_mask=True, weight_mask=False, dtype=torch.float32, **kwargs):

    reference_pythia = HookedTransformer.from_pretrained(
        model_name,
        fold_ln=False,
        device="cpu",
        dtype=dtype
    )
    reference_pythia.set_use_attn_result(True)
    
    
    # demo_pythia = PythiaDemoTransformer(PythiaConfig(debug=False, n_layers=n_layers, n_heads=n_heads, d_model=d_model, d_head=d_head, d_mlp=d_mlp), means, **kwargs)
    # demo_pythia.load_state_dict(pythia_weights, strict=False)
    # demo_pythia.cuda()
    if edge_mask:
        demo_pythia = PythiaEdgeDemoTransformer(tl_config_to_demo_config(reference_pythia.cfg), means=means, edge_masks=True, dtype=dtype, **kwargs)
    elif weight_mask:
        demo_pythia = PythiaWeightDemoTransformer(tl_config_to_demo_config(reference_pythia.cfg), weight_masks_attn=True, weight_masks_mlp=True, dtype=dtype, **kwargs)
    else:
        raise NotImplementedError("No mask type specified")

    demo_pythia.load_state_dict(reference_pythia.state_dict(), strict=False)
    demo_pythia.to(DEVICE)
    return demo_pythia

# %%
