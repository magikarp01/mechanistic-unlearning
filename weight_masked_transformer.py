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

class WeightMaskedTransformer(nn.Module):
    def __init__(self, tl_transformer, weight_mask_attn_dict=None, weight_mask_mlp_dict=None, torch_dtype=torch.bfloat16):
        """
        weight_mask_attn_dict: {layer: {"W_Q": frozen_heads, "W_K": frozen_heads, "W_V": frozen_heads, "W_O": frozen_heads}} (frozen_heads is shape (n_heads,) of bools). If none, train mask over all heads
        weight_mask_mlp_dict: {layer: bool}. If none, train mask over all mlps

        """
        super().__init__()
        self.torch_dtype = torch_dtype
        # tl_transformer should be a HookedTransformer
        self.tl_transformer = tl_transformer
        # turn off gradients for tl_transformer
        # for param in self.tl_transformer.parameters():
        #     param.requires_grad = False

        self.weight_mask_attn_dict = weight_mask_attn_dict
        self.weight_mask_mlp_dict = weight_mask_mlp_dict
        # store weight masks for every component that is frozen
        
        # need to store reference weights so that you can reset W_Q, etc after a forward pass
        self.reference_attn_weights = {}
        self.reference_mlp_weights = {}

        self.attention_masks = {}
        self.mlp_masks = {}
        for layer in range(tl_transformer.cfg.n_layers):
            self.attention_masks[layer] = {}
            self.reference_attn_weights[layer] = {}
            self.mlp_masks[layer] = {}
            self.reference_mlp_weights[layer] = {}
            # Attention heads
            for component, parameter in [("W_Q", tl_transformer.blocks[layer].attn.W_Q), ("W_K", tl_transformer.blocks[layer].attn.W_K), ("W_V", tl_transformer.blocks[layer].attn.W_V), ("W_O", tl_transformer.blocks[layer].attn.W_O)]:
                if self.weight_mask_attn_dict is None:
                    frozen_heads = list(range(tl_transformer.cfg.n_heads)) # all heads are frozen
                else:
                    frozen_heads = self.weight_mask_attn_dict[layer][component]
                # make frozen and baseline masks, and also a copy of the original weights

                if not all(frozen_heads):
                    W_frozen, W_baseline = make_partly_differentiable_mask(parameter, frozen_heads)
                    weight_mask = nn.Parameter(torch.ones_like(parameter).type(torch_dtype), requires_grad=True)
                    
                    self.attention_masks[layer][component] = (W_frozen, W_baseline, weight_mask)
                    self.reference_attn_weights[layer][component] = parameter.cpu().clone()

            # MLPs

            for component, parameter in [("W_in", tl_transformer.blocks[layer].mlp.W_in), ("W_out", tl_transformer.blocks[layer].mlp.W_out)]:
                # If not frozen
                if not self.weight_mask_mlp_dict[layer][component]:
                    weight_mask = nn.Parameter(torch.ones_like(parameter).type(torch_dtype), requires_grad=True)

                    self.mlp_masks[layer][component] = weight_mask
                    self.reference_mlp_weights[layer][component] = parameter.cpu().clone()

                
    def forward(self, *args, **kwargs):
        for layer in range(self.tl_transformer.cfg.n_layers):
            for component, parameter in [("W_Q", self.tl_transformer.blocks[layer].attn.W_Q), ("W_K", self.tl_transformer.blocks[layer].attn.W_K), ("W_V", self.tl_transformer.blocks[layer].attn.W_V), ("W_O", self.tl_transformer.blocks[layer].attn.W_O)]:
                if component in self.attention_masks[layer]:
                    W_frozen, W_baseline, weight_mask = self.attention_masks[layer][component]
                    reference_data = self.reference_attn_weights[layer][component].cuda()
                    mask = W_frozen + W_baseline * weight_mask
                    self.tl_transformer.blocks[layer].attn.__dict__['_parameters'][component] = reference_data * mask
                    del reference_data
                    torch.cuda.empty_cache()

            for component, parameter in [("W_in", self.tl_transformer.blocks[layer].mlp.W_in), ("W_out", self.tl_transformer.blocks[layer].mlp.W_out)]:
                if component in self.mlp_masks[layer]:
                    weight_mask = self.mlp_masks[layer][component]
                    reference_data = self.reference_mlp_weights[layer][component].to("cuda")
                    self.tl_transformer.blocks[layer].mlp.__dict__['_parameters'][component] = reference_data * weight_mask
                    del reference_data
                    torch.cuda.empty_cache()

        return self.tl_transformer(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.tl_transformer.generate(*args, **kwargs)

    def regularization_loss(self):
        # Compute the L1 sparsity penalty using the masks
        loss = 0
        for layer in range(self.tl_transformer.cfg.n_layers):
            num_comps = 0
            comp_loss = 0
            for component, parameter in [("W_Q", self.tl_transformer.blocks[layer].attn.W_Q), ("W_K", self.tl_transformer.blocks[layer].attn.W_K), ("W_V", self.tl_transformer.blocks[layer].attn.W_V), ("W_O", self.tl_transformer.blocks[layer].attn.W_O)]:
                num_comps += 1
                if component in self.attention_masks[layer]:
                    W_frozen, W_baseline, weight_mask = self.attention_masks[layer][component]
                    mask = W_frozen + (W_baseline * weight_mask) # 1s for frozen, heads
                    # Add (weights away from 1) / (total weights * percent_masks_active)
                    comp_loss += torch.sum(torch.abs(mask - 1)) / (mask.numel() * (W_baseline.sum() / W_baseline.numel()) + 1e-5)
                    del mask, W_frozen, W_baseline, weight_mask
                    torch.cuda.empty_cache()
                    
            loss += comp_loss / (num_comps + 1e-5)

            for component, parameter in [("W_in", self.tl_transformer.blocks[layer].mlp.W_in), ("W_out", self.tl_transformer.blocks[layer].mlp.W_out)]:
                num_comps = 0
                comp_loss = 0
                if component in self.mlp_masks[layer]:
                    weight_mask = self.mlp_masks[layer][component]
                    comp_loss += torch.sum(torch.abs(weight_mask - 1)) / weight_mask.numel()
                    del weight_mask
                    torch.cuda.empty_cache()

            loss += comp_loss / (num_comps + 1e-5)
        loss /= self.tl_transformer.cfg.n_layers
        return loss
    
    def on_step_end(self):
        # Clip all the masks

        for layer in range(self.tl_transformer.cfg.n_layers):
            for component, parameter in [("W_Q", self.tl_transformer.blocks[layer].attn.W_Q), ("W_K", self.tl_transformer.blocks[layer].attn.W_K), ("W_V", self.tl_transformer.blocks[layer].attn.W_V), ("W_O", self.tl_transformer.blocks[layer].attn.W_O)]:
                if component in self.attention_masks[layer]:
                    W_frozen, W_baseline, weight_mask = self.attention_masks[layer][component]
                    weight_mask.data = torch.clamp(weight_mask.data, 0, 1)

            for component, parameter in [("W_in", self.tl_transformer.blocks[layer].mlp.W_in), ("W_out", self.tl_transformer.blocks[layer].mlp.W_out)]:
                if component in self.mlp_masks[layer]:
                    weight_mask = self.mlp_masks[layer][component]
                    weight_mask.data = torch.clamp(weight_mask.data, 0, 1)
