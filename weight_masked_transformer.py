#%%
import torch
from torch import nn

def make_partly_differentiable_mask(W, frozen_heads, device="cuda"):
    """
    W is Parameter of shape (n_heads, ...). 
    Returns baseline and frozen (both only 1d arrays of (n_heads,)), 
    and forward pass should be W_frozen.float() + W_baseline.float() * W 
    """
    W_frozen = torch.nn.Parameter(torch.zeros(W.shape[0], dtype=torch.bool), requires_grad=False).to(device)

    # unsqueeze to broadcast efficiently, until W_baseline has same shape as W
    while len(W_frozen.shape) < len(W.shape):
        W_frozen = W_frozen.unsqueeze(-1)
    
    W_frozen[frozen_heads] = True

    W_baseline = (~W_frozen).float()
    W_baseline = torch.nn.Parameter(W_baseline, requires_grad=True)
    # convert into float
    return W_frozen.float(), W_baseline.float()

class WeightMaskedLayer(nn.Module):
    """
        Implements one layer of a weight masked transformer
        Masks the W_Q, W_K, W_V, W_O of the attention layer and W_in, W_out of the MLP
    """

    def __init__(self, tl_layer, attn_mask_dict, mlp_mask_dict, torch_dtype=torch.bfloat16):
        '''
            tl_layer is a module containing attn.W_Q, attn.W_K, attn.W_V, attn.W_O, mlp.W_in, mlp.W_out
            attn_mask_dict: {"W_Q": frozen_heads, "W_K": frozen_heads, "W_V": frozen_heads, "W_O": frozen_heads} 
            mlp_mask_dict: {"W_in": bool, "W_out": bool}
        '''
        super().__init__()
        self.torch_dtype = torch.dtype
        self.tl_layer = tl_layer
        self.attn_mask_dict = attn_mask_dict
        self.mlp_mask_dict = mlp_mask_dict

        self.reference_attn_weights = {}
        self.reference_mlp_weights = {}

        self.attention_masks = {}
        self.mlp_masks = {}

        # Populate the reference weights for attention head
        for component, parameter in [("W_Q", tl_layer.attn.W_Q), ("W_K", tl_layer.attn.W_K), ("W_V", tl_layer.attn.W_V), ("W_O", tl_layer.attn.W_O)]:
            # If the component exists and not all heads are frozen
            if component in attn_mask_dict and not all(attn_mask_dict[component]):
                frozen_heads = attn_mask_dict[component]
                W_frozen, W_baseline = make_partly_differentiable_mask(parameter, frozen_heads)
                weight_mask = nn.Parameter(torch.ones_like(parameter).type(torch_dtype), requires_grad=True)
                self.attention_masks[component] = (W_frozen, W_baseline, weight_mask)
                self.reference_attn_weights[component] = parameter.clone()
        
        # Populate the reference weights for MLP
        for component, parameter in [("W_in", tl_layer.mlp.W_in), ("W_out", tl_layer.mlp.W_out)]:
            if component in mlp_mask_dict:
                weight_mask = nn.Parameter(torch.ones_like(parameter).type(torch_dtype), requires_grad=True)
                self.mlp_masks[component] = weight_mask
                self.reference_mlp_weights[component] = parameter.clone()

    def forward(self, *args, **kwargs):
        # Mask the tl layer weights, and then do a forward pass
        for component in ["W_Q", "W_K", "W_V", "W_O"]:
            if component in self.attention_masks:
                W_frozen, W_baseline, weight_mask = self.attention_masks[component]
                reference_data = self.reference_attn_weights[component].cuda()
                mask = W_frozen + W_baseline * weight_mask
                self.tl_layer.attn._parameters[component] = reference_data * mask

        for component in ["W_in", "W_out"]:
            if component in self.mlp_masks:
                weight_mask = self.mlp_masks[component]
                reference_data = self.reference_mlp_weights[component].to("cuda")
                self.tl_layer.mlp._parameters[component] = reference_data * weight_mask

        return self.tl_layer(*args, **kwargs)
    
    def regularization_loss(self):
        # Compute the L1 sparsity penalty using the masks
        loss = 0
        for component in ["W_Q", "W_K", "W_V", "W_O"]:
            attn_norm = 0
            num_comps = 0
            if component in self.attention_masks:
                num_comps += 1
                W_frozen, W_baseline, weight_mask = self.attention_masks[component]
                mask = W_frozen + (W_baseline * weight_mask)

                # Add up the L1 norm of the mask - 1 for the masks that are not frozen
                attn_norm += torch.sum(torch.abs(mask - 1))
                # This is probably a large number, so normalize it by the number of elements that are not frozen
                attn_norm /= (mask.numel() * (W_baseline.sum() / W_baseline.numel()) + 1e-5)

                del mask, W_frozen, W_baseline, weight_mask
                torch.cuda.empty_cache()
            loss += attn_norm / (num_comps + 1e-5)
        
        for component in ["W_in", "W_out"]:
            mlp_norm = 0
            num_comps = 0
            if component in self.mlp_masks:
                num_comps += 1
                weight_mask = self.mlp_masks[component]
                mlp_norm += torch.sum(torch.abs(weight_mask - 1)) / weight_mask.numel()
                del weight_mask
                torch.cuda.empty_cache()
            loss += mlp_norm / (num_comps + 1e-5)
        
        return loss
                

    def on_step_end(self):
        # Clip all the masks

        for component in ["W_Q", "W_K", "W_V", "W_O"]:
            if component in self.attention_masks:
                W_frozen, W_baseline, weight_mask = self.attention_masks[component]
                weight_mask.data = torch.clamp(weight_mask.data, 0, 1)

        for component in ["W_in", "W_out"]:
            if component in self.mlp_masks:
                weight_mask = self.mlp_masks[component]
                weight_mask.data = torch.clamp(weight_mask.data, 0, 1)


class WeightMaskedTransformer(nn.Module):
    def __init__(self, tl_transformer, weight_mask_attn_dict=None, weight_mask_mlp_dict=None, torch_dtype=torch.bfloat16):
        """
            Consists of an embed layer, n_layers of WeightMaskedLayer, and then the final layernorm and unembed layer

            tl_transformer: HookedTransformer
            weight_mask_attn_dict: {layer: {"W_Q": frozen_heads, "W_K": frozen_heads, "W_V": frozen_heads, "W_O": frozen_heads}}
            weight_mask_mlp_dict: {layer: {"W_in": bool, "W_out": bool}}
        """
        super().__init__()
        self.torch_dtype = torch_dtype
        self.tl_transformer = tl_transformer

        # Turn off gradients for tl_transformer
        for param in self.tl_transformer.parameters():
            param.requires_grad = False

        self.weight_mask_attn_dict = weight_mask_attn_dict
        self.weight_mask_mlp_dict = weight_mask_mlp_dict

        # Each layer is named layer{i}, and is a WeightMaskedLayer
        for layer in range(self.tl_transformer.cfg.n_layers):
            setattr(
                self,
                f"layer{layer}",
                WeightMaskedLayer(
                    tl_layer=self.tl_transformer.blocks[layer],
                    attn_mask_dict=weight_mask_attn_dict[layer] if weight_mask_attn_dict is not None else {},
                    mlp_mask_dict=weight_mask_mlp_dict[layer] if weight_mask_mlp_dict is not None else {},
                    torch_dtype=torch_dtype
                )
            )
                
    def forward(self, inp):
        # Forward pass through all the layers

        # Embedding
        # Positional encoding

        if len(inp.shape) == 1:
            x = self.tl_transformer.embed(inp.unsqueeze(0))
            print("Embed:", x.shape)
            # Does not have batch dim
            pos_x = self.tl_transformer.pos_embed(inp.unsqueeze(0))
        else:
            x = self.tl_transformer.embed(inp)
            print("Embed:", x.shape)
            # Does have batch dim >= 1
            pos_x = self.tl_transformer.pos_embed(inp[0].unsqueeze(0))

        print("Pos Embed:", pos_x.shape)
        x += pos_x
        print(f"Final Embed shape: {x.shape}")

        # Forward pass through all the layers
        for layer in range(self.tl_transformer.cfg.n_layers):
            x = getattr(self, f"layer{layer}")(x)
            print(f"Layer {layer} shape: {x.shape}")
        
        # Final layernorm and unembed
        x = self.tl_transformer.ln_final(x)
        print(f"LN shape: {x.shape}")
        x = self.tl_transformer.unembed(x)
        print(f"Unembed shape: {x.shape}")

        return x

    def generate(self, *args, **kwargs):
        return self.tl_transformer.generate(*args, **kwargs)

    def regularization_loss(self):
        # Compute the average L1 sparsity penalty over layers using the masks
        loss = 0
        for layer in range(self.tl_transformer.cfg.n_layers):
            loss += getattr(self, f"layer{layer}").regularization_loss()
        
        return loss / self.tl_transformer.cfg.n_layers

    def on_step_end(self):
        # Clip all the masks

        for layer in range(self.tl_transformer.cfg.n_layers):
            for component, parameter in [("W_Q", self.tl_transformer.blocks[layer].attn.W_Q), ("W_K", self.tl_transformer.blocks[layer].attn.W_K), ("W_V", self.tl_transformer.blocks[layer].attn.W_V), ("W_O", self.tl_transformer.blocks[layer].attn.W_O)]:
                if component in self.attention_masks[layer]:
                    _, _, weight_mask = self.attention_masks[layer][component]
                    weight_mask.data = torch.clamp(weight_mask.data, 0, 1)

            for component, parameter in [("W_in", self.tl_transformer.blocks[layer].mlp.W_in), ("W_out", self.tl_transformer.blocks[layer].mlp.W_out)]:
                if component in self.mlp_masks[layer]:
                    weight_mask = self.mlp_masks[layer][component]
                    weight_mask.data = torch.clamp(weight_mask.data, 0, 1)

#%%
from transformer_lens import HookedTransformer

gpt2_small = HookedTransformer.from_pretrained(
    "gpt2-small",
)
# %%
import random
import einops

def create_random_weight_mask_dicts(model, top_p):
    # Creates random weight masks for testing
    weight_mask_attn_dict = {}
    weight_mask_mlp_dict = {}

    for layer in range(model.cfg.n_layers):
        weight_mask_attn_dict[layer] = {}
        weight_mask_mlp_dict[layer] = {}
        # Want bool of length n_head, randomly set to True
        weight_mask_attn_dict[layer]['W_Q'] = torch.rand(model.cfg.n_heads) < top_p
        weight_mask_attn_dict[layer]['W_K'] = torch.rand(model.cfg.n_heads) < top_p
        weight_mask_attn_dict[layer]['W_V'] = torch.rand(model.cfg.n_heads) < top_p
        weight_mask_attn_dict[layer]['W_O'] = torch.rand(model.cfg.n_heads) < top_p

        # Randomly set to true or false
        weight_mask_mlp_dict[layer]['W_in'] = random.random() < top_p
        weight_mask_mlp_dict[layer]['W_out'] = random.random() < top_p

    return weight_mask_attn_dict, weight_mask_mlp_dict

weight_mask_attn_dict, weight_mask_mlp_dict = create_random_weight_mask_dicts(gpt2_small, 0.05)

mask = WeightMaskedTransformer(
    gpt2_small,
    weight_mask_attn_dict=weight_mask_attn_dict,
    weight_mask_mlp_dict=weight_mask_mlp_dict
)

#%%
# Test both gpt2_small and mask, and make sure they have the same output

toks = torch.tensor(gpt2_small.tokenizer.encode("Hello, my name is")).unsqueeze(0)

with torch.set_grad_enabled(False):
    gpt2_small_output = gpt2_small(toks)
    gpt2_small_logits = torch.nn.functional.softmax(gpt2_small_output, dim=-1)
    mask_output = mask(toks)
    mask_logits = torch.nn.functional.softmax(mask_output, dim=-1)

# # print(gpt2_small_logits)
# # print(mask_logits)
print(torch.allclose(gpt2_small_logits, mask_logits, atol=1e-3))
# %%
from pippy import pipeline, annotate_split_points, Pipe, SplitPoint
from pippy import split_into_equal_size

device="cuda"
split_policy = split_into_equal_size(2)

mask.to(device)
mask.eval()

batch_size = 4
example_input = torch.stack(
    [
        torch.tensor(gpt2_small.tokenizer.encode("Hello My name is")),
        torch.tensor(gpt2_small.tokenizer.encode("Hello My name is")),
        torch.tensor(gpt2_small.tokenizer.encode("Hello My name is")),
        torch.tensor(gpt2_small.tokenizer.encode("Hello My name is"))
    ]
).to("cuda")
chunks = 1

pipe = pipeline(
    mask, 
    num_chunks=chunks, 
    example_args=(), 
    example_kwargs={"inp": example_input},
    split_policy=split_policy
)
# print(pipe)

# %%
# import pippy
# %%
