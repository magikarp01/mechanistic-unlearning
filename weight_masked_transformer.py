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

class EmbedWrapper(nn.Module):
    """
        Use to reduce embed and pos embed into a single layer 
    """

    def __init__(self, embed, pos_embed=None):
        super().__init__()
        self.embed = embed
        self.pos_embed = pos_embed
    
    def forward(self, inp):

        if len(inp.shape) == 1:
            x = self.embed(inp.unsqueeze(0))
            # print("Embed:", x.shape)
            # Does not have batch dim
            if self.pos_embed is not None:
                pos_x = self.pos_embed(inp.unsqueeze(0))
        else:
            x = self.embed(inp)
            # print("Embed:", x.shape)
            # Does have batch dim >= 1
            if self.pos_embed is not None:
                pos_x = self.pos_embed(inp[0].unsqueeze(0))

        # print("Pos Embed:", pos_x.shape)
        if self.pos_embed is not None:
            x += pos_x
        # print(f"Final Embed shape: {x.shape}")

        return x

class WeightMaskedLayer(nn.Module):
    """
        Implements one layer of a weight masked transformer
        Masks the W_Q, W_K, W_V, W_O of the attention layer and W_in, W_out of the MLP
    """

    def __init__(self, tl_layer, attn_mask_dict, mlp_mask_dict, torch_dtype=torch.bfloat16, device="cuda:0"):
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
                W_frozen, W_baseline = make_partly_differentiable_mask(parameter, frozen_heads, device)
                weight_mask = nn.Parameter(torch.ones_like(parameter).type(torch_dtype).detach(), requires_grad=True).to(device)
                self.attention_masks[component] = (W_frozen, W_baseline, weight_mask)
                self.reference_attn_weights[component] = parameter.clone()
        
        # Populate the reference weights for MLP
        for component, parameter in [("W_in", tl_layer.mlp.W_in), ("W_out", tl_layer.mlp.W_out)]:
            if component in mlp_mask_dict:
                weight_mask = nn.Parameter(torch.ones_like(parameter).type(torch_dtype).detach(), requires_grad=True).to(device)
                self.mlp_masks[component] = weight_mask
                self.reference_mlp_weights[component] = parameter.clone()

    def forward(self, *args, **kwargs):
        # Mask the tl layer weights, and then do a forward pass
        for component in ["W_Q", "W_K", "W_V", "W_O"]:
            if component in self.attention_masks:
                W_frozen, W_baseline, weight_mask = self.attention_masks[component]
                reference_data = self.reference_attn_weights[component]
                mask = W_frozen + W_baseline * weight_mask
                self.tl_layer.attn.__dict__['_parameters'][component] = reference_data * mask

        for component in ["W_in", "W_out"]:
            if component in self.mlp_masks:
                weight_mask = self.mlp_masks[component]
                reference_data = self.reference_mlp_weights[component]
                self.tl_layer.mlp.__dict__['_parameters'][component] = reference_data * weight_mask

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

        for component in ["W_Q", "W_K", "W_V", "W_O"] :
            if component in self.attention_masks:
                _, _, weight_mask = self.attention_masks[component]
                weight_mask.data = torch.clamp(weight_mask.data, 0, 1)

        for component in ["W_in", "W_out"]:
            if component in self.mlp_masks:
                weight_mask = self.mlp_masks[component]
                weight_mask.data = torch.clamp(weight_mask.data, 0, 1)


class WeightMaskedTransformer(nn.Module):
    def __init__(
        self, 
        tl_transformer, 
        weight_mask_attn_dict=None, 
        weight_mask_mlp_dict=None, 
        torch_dtype=torch.bfloat16
    ):
        """
            Consists of an embed layer, n_layers of WeightMaskedLayer, and then the final layernorm and unembed layer

            tl_transformer: HookedTransformer
            weight_mask_attn_dict: {layer: {"W_Q": frozen_heads, "W_K": frozen_heads, "W_V": frozen_heads, "W_O": frozen_heads}}
            weight_mask_mlp_dict: {layer: {"W_in": bool, "W_out": bool}}
        """
        super().__init__()
        self.torch_dtype = torch_dtype
        self.tl_transformer = tl_transformer

        n_devices = torch.cuda.device_count()

        # Turn off gradients for tl_transformer
        # for param in self.tl_transformer.parameters():
            # param.requires_grad = False

        try:
            # If pos_embed is present
            self.embed = EmbedWrapper(tl_transformer.embed, tl_transformer.pos_embed).to("cuda:0")
        except AttributeError:
            self.embed = EmbedWrapper(tl_transformer.embed).to("cuda:0")

        self.ln_final = tl_transformer.ln_final.to(n_devices-1)
        self.unembed = tl_transformer.unembed.to(n_devices-1)
        # Each layer is named layer{i}, and is a WeightMaskedLayer
        self.num_layers_per_device = tl_transformer.cfg.n_layers // n_devices
        self.blocks = []
        device_id = 0
        for layer in range(self.tl_transformer.cfg.n_layers):
            if layer != 0 and layer % self.num_layers_per_device == 0:
                device_id += 1
            # print(layer, device_id)
            setattr(
                self,
                f"layer{layer}",
                WeightMaskedLayer(
                    tl_layer=self.tl_transformer.blocks[layer],
                    attn_mask_dict=weight_mask_attn_dict[layer] if weight_mask_attn_dict is not None else {},
                    mlp_mask_dict=weight_mask_mlp_dict[layer] if weight_mask_mlp_dict is not None else {},
                    torch_dtype=torch_dtype,
                    device=f"cuda:{device_id}"
                )
            )
            self.blocks.append(getattr(self, f"layer{layer}"))
                
    def forward(self, inp):
        # Forward pass through all the layers
        x = self.embed(inp)
        prev_id = 0
        device_id = 0
        for layer in range(self.tl_transformer.cfg.n_layers):
            if layer != 0 and layer % self.num_layers_per_device == 0:
                device_id += 1
            
            # print(layer, device_id)
            if device_id != prev_id:
                x = x.to(device_id)
                prev_id = device_id

            func = getattr(self, f"layer{layer}").to(device_id)
            # print(f'On device {device_id}. Func on {func.tl_layer.attn.W_K.device}, Tensor on {x.device}')
            x = func(x)

        x = self.ln_final(x)
        x = self.unembed(x)
        return x.to("cuda:0")

    def generate(self, *args, **kwargs):
        return self.tl_transformer.generate(*args, **kwargs)

    def regularization_loss(self, collect_device="cuda:0"):
        # Compute the average L1 sparsity penalty over layers using the masks
        # Returns result on collect_device
        loss = 0
        for layer in range(self.tl_transformer.cfg.n_layers):
            loss += getattr(self, f"layer{layer}").regularization_loss().to(collect_device)
        
        return loss / self.tl_transformer.cfg.n_layers

    def on_step_end(self):
        # Clip all the masks

        for layer in self.blocks:
            layer.on_step_end() 

#%% gpt2-small
# from transformer_lens import HookedTransformer

# model = HookedTransformer.from_pretrained(
#     "google/gemma-7b",
#     default_padding_side="right",
#     fold_ln=False,
#     fold_value_biases=False,
#     center_writing_weights=False,
#     n_devices=2
# )
# # %% Create Mask
# import random
# import einops

# def create_random_weight_mask_dicts(model, top_p):
#     # Creates random weight masks for testing
#     weight_mask_attn_dict = {}
#     weight_mask_mlp_dict = {}

#     for layer in range(model.cfg.n_layers):
#         weight_mask_attn_dict[layer] = {}
#         weight_mask_mlp_dict[layer] = {}
#         # Want bool of length n_head, randomly set to True
#         weight_mask_attn_dict[layer]['W_Q'] = torch.rand(model.cfg.n_heads) > top_p
#         weight_mask_attn_dict[layer]['W_K'] = torch.rand(model.cfg.n_heads) > top_p
#         weight_mask_attn_dict[layer]['W_V'] = torch.rand(model.cfg.n_heads) > top_p
#         weight_mask_attn_dict[layer]['W_O'] = torch.rand(model.cfg.n_heads) > top_p

#         # Randomly set to true or false
#         weight_mask_mlp_dict[layer]['W_in'] = random.random() > top_p
#         weight_mask_mlp_dict[layer]['W_out'] = random.random() > top_p

#     return weight_mask_attn_dict, weight_mask_mlp_dict

# weight_mask_attn_dict, weight_mask_mlp_dict = create_random_weight_mask_dicts(model, 0.05)

# example_input = torch.stack(
#     [
#         torch.tensor(model.tokenizer.encode("Hello My name is")),
#         torch.tensor(model.tokenizer.encode("Hello My name is")),
#         torch.tensor(model.tokenizer.encode("Hello My name is")),
#         torch.tensor(model.tokenizer.encode("Hello My name is"))
#     ]
# ).to("cuda")

# mask = WeightMaskedTransformer(
#     model, 
#     weight_mask_attn_dict=weight_mask_attn_dict, 
#     weight_mask_mlp_dict=weight_mask_mlp_dict
# )
# from tasks.facts.SportsTask import SportsTask

# mask_params = [
#     v[-1]
#     for layer in mask.blocks
#     for k, v in layer.attention_masks.items()
# ] + \
# [
#     v
#     for layer in mask.blocks
#     for k, v in layer.mlp_masks.items()
# ]
# sports_train = SportsTask(batch_size=2, tokenizer=model.tokenizer)

# optimizer = torch.optim.SGD(mask_params, lr=0.1, momentum=0.9, weight_decay=0.01)
# with torch.autocast(device_type="cuda"):
#     loss = sports_train.get_train_loss(mask, 1)
#     print(loss)
#     loss.backward()
#     optimizer.step()
# print(mask.blocks[0].attention_masks['W_O'][-1].grad)
#%%
# # Test both gpt2_small and mask, and make sure they have the same output

# # toks = torch.tensor(gpt2_small.tokenizer.encode("Hello, my name is")).unsqueeze(0)

# # with torch.set_grad_enabled(False):
# #     gpt2_small_output = gpt2_small(toks)
# #     gpt2_small_logits = torch.nn.functional.softmax(gpt2_small_output, dim=-1)
# #     mask_output = mask(toks)
# #     mask_logits = torch.nn.functional.softmax(mask_output, dim=-1)

# # # print(gpt2_small_logits)
# # # print(mask_logits)
# # print(torch.allclose(gpt2_small_logits, mask_logits, atol=1e-3))
