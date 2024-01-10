"""
A modified version of the transformer architecture to enable casaul path
interventions. Modified from Nanda.
"""

# %% 
import pickle
import einops
from fancy_einsum import einsum
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn as nn
import numpy as np
import math
from easy_transformer.utils import gelu_new
import tqdm.auto as tqdm
from typing import List
from cb_utils.mask_utils import get_edge_mask_template

@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12

cfg = Config()

class PartialFrozenMask(nn.Module):
    def __init__(self, mask):
        # Mask is a tensor of 1s and 0s that is the same shape as the parameter
        # Freeze all 1s and leave all 0s active
        super().__init__()
        # Create the first frozen parameter
        self.frozen_param1 = nn.Parameter(mask, requires_grad=False)
        # Create the second frozen parameter
        self.frozen_param2 = nn.Parameter(1 - mask, requires_grad=False)
        # Create the active parameter
        self.active_param = nn.Parameter(torch.ones_like(mask), requires_grad=True)

        self.full_mask = self.frozen_param1 + self.frozen_param2 * self.active_param

    def forward(self, x):
        # return masked x
        return x * self.full_mask


class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        self.b = nn.Parameter(torch.zeros(cfg.d_model))
    
    def forward(self, residual, parallel=False):
        # residual: [batch, position, n_heads, d_model]
        if self.cfg.debug: print("Residual:", residual.shape)
        if parallel:
            pattern = "batch position n_heads d_model -> batch position n_heads 1"
        else:
            pattern = "batch position d_model -> batch position 1"
        residual = residual - einops.reduce(residual, pattern, "mean")
        # Calculate the variance, square root it. Add in an epsilon to prevent divide by zero.
        scale = (einops.reduce(residual.pow(2), pattern, "mean") + cfg.layer_norm_eps).sqrt()
        normalized = residual / scale
        normalized = normalized * self.w + self.b
        if self.cfg.debug: print("Normalized:", residual.shape)
        return normalized

"""## Embedding

Basically a lookup table from tokens to residual stream vectors.
"""

class Embed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)
    
    def forward(self, tokens):
        # tokens: [batch, position]
        if self.cfg.debug: print("Tokens:", tokens.shape)
        embed = self.W_E[tokens, :] # [batch, position, d_model]
        if self.cfg.debug: print("Embeddings:", embed.shape)
        return embed

"""## Positional Embedding"""

class PosEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)
    
    def forward(self, tokens):
        # tokens: [batch, position]
        if self.cfg.debug: print("Tokens:", tokens.shape)
        pos_embed = self.W_pos[:tokens.size(1), :] # [position, d_model]
        pos_embed = einops.repeat(pos_embed, "position d_model -> batch position d_model", batch=tokens.size(0))
        if self.cfg.debug: print("pos_embed:", pos_embed.shape)
        return pos_embed

"""## Attention

* **Step 1:** Produce an attention pattern - for each destination token, probability distribution over previous tokens (incl current token)
    * Linear map from input -> query, key shape [batch, position, head_index, d_head]
    * Dot product every *pair* of queries and keys to get attn_scores [batch, head_index, query_pos, key_pos] (query = dest, key = source)
    * Scale and mask attn_scores to make it lower triangular, ie causal
    * softmax row-wise, to get a probability distribution along each the key_pos dimension - this is our attention pattern!
* **Step 2:** Move information from source tokens to destination token using attention pattern (move = apply linear map)
    * Linear map from input -> value [batch, key_pos, head_index, d_head]
    * Mix along the key_pos with attn pattern to get z, a mixed value [batch, query_pos, head_index, d_head]
    * Map to output, [batch, position, d_model] (position = query_pos, we've summed over all heads)

First, it's useful to visualize and play around with attention patterns - what exactly are we looking at here? (Click on a head to lock onto just showing that head's pattern, it'll make it easier to interpret)
"""
def make_partly_differentiable_mask(W, unfrozen_heads: List[int]):
    """
    W is Parameter of shape (n_heads, ...). Returns baseline and frozen, and forward pass should be W_baseline.float() + W_frozen.float() * W 
    """
    # W_baseline = torch.nn.Parameter(torch.ones_like(W))
    # W_baseline[unfrozen_heads] = 0
    W_frozen = torch.nn.Parameter(torch.zeros_like(W, dtype=torch.bool), requires_grad=False)
    W_frozen[unfrozen_heads] = True
    # W_baseline = ~W_frozen
    W_baseline = torch.nn.Parameter(~W_frozen, requires_grad=False)
    return W_baseline, W_frozen

class Attention(nn.Module):
    def __init__(self, cfg, weight_mask=False, mask_heads=None):
        """
        weight_mask tells you if you want to apply a weight mask to this attention head. If mask_heads is not none, it should be a list of ints (heads to mask).
        """
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        self.b_Q = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.W_K = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        self.b_K = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.W_V = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        self.b_V = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        
        self.W_O = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.b_O = nn.Parameter(torch.zeros((cfg.d_model)))
        
        self.weight_mask = weight_mask
        self.mask_heads = mask_heads

        if weight_mask:
            self.weight_mask_W_Q = nn.Parameter(torch.ones_like(self.W_Q), requires_grad=True)
            self.weight_mask_W_K = nn.Parameter(torch.ones_like(self.W_K), requires_grad=True)
            self.weight_mask_W_V = nn.Parameter(torch.ones_like(self.W_V), requires_grad=True)
            self.weight_mask_W_O = nn.Parameter(torch.ones_like(self.W_O), requires_grad=True)
        if mask_heads is not None:
            self.weight_mask_W_Q_baseline, self.weight_mask_W_Q_frozen = make_partly_differentiable_mask(self.weight_mask_W_Q, self.mask_heads)            
            self.weight_mask_W_K_baseline, self.weight_mask_W_K_frozen = make_partly_differentiable_mask(self.weight_mask_W_K, self.mask_heads)
            self.weight_mask_W_V_baseline, self.weight_mask_W_V_frozen = make_partly_differentiable_mask(self.weight_mask_W_V, self.mask_heads)
            self.weight_mask_W_O_baseline, self.weight_mask_W_O_frozen = make_partly_differentiable_mask(self.weight_mask_W_O, self.mask_heads)

        self.register_buffer("IGNORE", torch.tensor(-1e5, dtype=torch.float32, device="cuda"))
    
    def forward(self, normalized_resid_pre):
        # normalized_resid_pre: [batch, position, d_model]
        if self.cfg.debug: print("Normalized_resid_pre:", normalized_resid_pre.shape)

        if self.weight_mask:
            if self.mask_heads is not None:
                weight_mask_W_Q = self.weight_mask_W_Q_baseline.float() + self.weight_mask_W_Q_frozen.float() * self.weight_mask_W_Q
                weight_mask_W_K = self.weight_mask_W_K_baseline.float() + self.weight_mask_W_K_frozen.float() * self.weight_mask_W_K
                weight_mask_W_V = self.weight_mask_W_V_baseline.float() + self.weight_mask_W_V_frozen.float() * self.weight_mask_W_V
                weight_mask_W_O = self.weight_mask_W_O_baseline.float() + self.weight_mask_W_O_frozen.float() * self.weight_mask_W_O
            else:
                weight_mask_W_Q = self.weight_mask_W_Q
                weight_mask_W_K = self.weight_mask_W_K
                weight_mask_W_V = self.weight_mask_W_V
                weight_mask_W_O = self.weight_mask_W_O

            W_Q = self.W_Q * weight_mask_W_Q
            W_K = self.W_K * weight_mask_W_K
            W_V = self.W_V * weight_mask_W_V
            W_O = self.W_O * weight_mask_W_O
        else:
            W_Q = self.W_Q
            W_K = self.W_K
            W_V = self.W_V
            W_O = self.W_O

        q = einsum("batch query_pos n_heads d_model, n_heads d_model d_head -> batch query_pos n_heads d_head", normalized_resid_pre, W_Q) + self.b_Q

        k = einsum("batch key_pos n_heads d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", normalized_resid_pre, W_K) + self.b_K
        
        attn_scores = einsum("batch query_pos n_heads d_head, batch key_pos n_heads d_head -> batch n_heads query_pos key_pos", q, k)
        attn_scores = attn_scores / math.sqrt(self.cfg.d_head)
        attn_scores = self.apply_causal_mask(attn_scores)

        pattern = attn_scores.softmax(dim=-1) # [batch, n_head, query_pos, key_pos]
        v = einsum("batch key_pos n_heads d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", normalized_resid_pre, W_V) + self.b_V

        z = einsum("batch n_heads query_pos key_pos, batch key_pos n_heads d_head -> batch query_pos n_heads d_head", pattern, v)

        attn_out = einsum("batch query_pos n_heads d_head, n_heads d_head d_model -> batch query_pos n_heads d_model", z, W_O) + (self.b_O / cfg.n_heads)
        return attn_out

    def discretize_weight_masks(self, threshold=0.5):
        """
        Call to discretize weight masks. Sets all values below threshold to 0 and all values above threshold to 1.
        """
        assert self.weight_mask
        for p in [self.weight_mask_W_Q, self.weight_mask_W_K, self.weight_mask_W_V, self.weight_mask_W_O]:
            p.data[p.data < threshold] = 0
            p.data[p.data >= threshold] = 1

    def apply_causal_mask(self, attn_scores):
        # attn_scores: [batch, n_heads, query_pos, key_pos]
        mask = torch.triu(torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores

"""## MLP"""
class MLP(nn.Module):
    def __init__(self, cfg, weight_mask=False):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        self.b_in = nn.Parameter(torch.zeros((cfg.d_mlp)))
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model)))
        nn.init.normal_(self.W_out, std=self.cfg.init_range)
        self.b_out = nn.Parameter(torch.zeros((cfg.d_model)))

        if weight_mask:
            self.weight_mask_W_in = nn.Parameter(torch.ones_like(self.W_in), requires_grad=True)
            self.weight_mask_W_out = nn.Parameter(torch.ones_like(self.W_out), requires_grad=True)
            self.weight_mask_b_in = nn.Parameter(torch.ones_like(self.b_in), requires_grad=True)
            self.weight_mask_b_out = nn.Parameter(torch.ones_like(self.b_out), requires_grad=True)
        self.weight_mask = weight_mask

    def discretize_weight_masks(self, threshold=0.5):
        """
        Call to discretize weight masks. Sets all values below threshold to 0 and all values above threshold to 1.
        """
        assert self.weight_mask
        for p in [self.weight_mask_W_in, self.weight_mask_W_out, self.weight_mask_b_in, self.weight_mask_b_out]:
            p.data[p.data < threshold] = 0
            p.data[p.data >= threshold] = 1
    
    def forward(self, normalized_resid_mid):
        if self.weight_mask:
            W_in = self.W_in * self.weight_mask_W_in
            W_out = self.W_out * self.weight_mask_W_out
            b_in = self.b_in * self.weight_mask_b_in
            b_out = self.b_out * self.weight_mask_b_out
        else:
            W_in = self.W_in
            W_out = self.W_out
            b_in = self.b_in
            b_out = self.b_out

        # normalized_resid_mid: [batch, position, d_model]
        if self.cfg.debug: print("Normalized_resid_mid:", normalized_resid_mid.shape)
        pre = einsum("batch position d_model, d_model d_mlp -> batch position d_mlp", normalized_resid_mid, W_in) + b_in
        post = gelu_new(pre)
        mlp_out = einsum("batch position d_mlp, d_mlp d_model -> batch position d_model", post, W_out) + b_out
        return mlp_out

"""## Transformer Block"""
class TransformerBlock(nn.Module):
    def __init__(self, cfg, prev_layers: int, frozen_mask_edges=None, freeze_ones=True, weight_mask_attn=False, weight_mask_mlp=False, weight_mask_attn_heads=None):
        """
        frozen_mask_edges is a None or dictionary of the form: {'a': torch.tensor, 'm': torch.tensor}. Tells you what parts of mask to freeze (tensors are same shape as typical mask, 1s are not trained over while 0s are trained over if freeze_ones is True else vice-versa).

        weight_mask_attn_heads is list of what heads to selectively unfreeze in weight mask
        """
        super().__init__()
        self.cfg = cfg

        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg, weight_mask=weight_mask_attn, mask_heads=weight_mask_attn_heads)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg, weight_mask=weight_mask_mlp)

        self.frozen_mask = True if frozen_mask_edges is not None else False
        self.weight_mask_attn = weight_mask_attn
        self.weight_mask_mlp = weight_mask_mlp

        for name, p in self.named_parameters():
            if "weight_mask" not in name:
                p.requires_grad = False

        prev_nodes = (cfg.n_heads + 1) * prev_layers + 1
        edge_mask_attentions_init = torch.ones((prev_nodes, cfg.n_heads))
        self.edge_mask_attentions = torch.nn.Parameter(edge_mask_attentions_init, requires_grad=True)
        
        if self.frozen_mask:
            if freeze_ones:
                self.edge_mask_attentions_baseline = torch.nn.Parameter(frozen_mask_edges['a'], requires_grad=False)
                self.edge_mask_attentions_frozen = torch.nn.Parameter(1 - frozen_mask_edges['a'], requires_grad=False)
            else:
                self.edge_mask_attentions_baseline = torch.nn.Parameter(torch.zeros_like(frozen_mask_edges['a']), requires_grad=False)
                self.edge_mask_attentions_frozen = torch.nn.Parameter(frozen_mask_edges['a'], requires_grad=False)
        # if mask_params is not None:
        #     self.frozen_edge_mask_attentions = PartialFrozenMask(mask_params)
        # else:
        #     self.frozen_edge_mask_attentions = None

        edge_mask_mlp_init = torch.ones((prev_nodes + cfg.n_heads, ))
        self.edge_mask_mlp = torch.nn.Parameter(edge_mask_mlp_init, requires_grad=True)
        
        if self.frozen_mask:
            if freeze_ones:
                self.edge_mask_mlp_baseline = torch.nn.Parameter(frozen_mask_edges['m'], requires_grad=False)
                self.edge_mask_mlp_frozen = torch.nn.Parameter(1 - frozen_mask_edges['m'], requires_grad=False)
            else:
                self.edge_mask_mlp_baseline = torch.nn.Parameter(torch.zeros_like(frozen_mask_edges['m']), requires_grad=False)
                self.edge_mask_mlp_frozen = torch.nn.Parameter(frozen_mask_edges['m'], requires_grad=False)
    
    def discretize_weight_masks(self, threshold=0.5):
        """
        Call to discretize weight masks. Sets all values below threshold to 0 and all values above threshold to 1.
        """
        if self.weight_mask_attn:
            self.attn.discretize_weight_masks(threshold)
        if self.weight_mask_mlp:
            self.mlp.discretize_weight_masks(threshold)

    def forward(self, resid_pre, means=False):

        if self.frozen_mask:
            attn_mask = self.edge_mask_attentions_baseline + self.edge_mask_attentions_frozen * self.edge_mask_attentions
        else:
            attn_mask = self.edge_mask_attentions
        
        # resid_pre [batch, position, d_model, prev_head_idx]
        masked_residuals = einsum("batch position prev_head_idx d_model, prev_head_idx n_heads -> batch position n_heads d_model", resid_pre, attn_mask)

        if isinstance(means, torch.Tensor):
            masked_means = einsum("seq_len prev_head_idx d_model, prev_head_idx n_heads -> seq_len n_heads d_model", means[:, :self.edge_mask_attentions.shape[0]], 1 - attn_mask)
            if masked_means.shape[0] > masked_residuals.shape[1]:
                masked_residuals = masked_residuals + masked_means[:masked_residuals.shape[1]].unsqueeze(0)
            else:
                # print("WARNING: masked_means.shape[0] < masked_residuals.shape[1]")
                # print("masked_means.shape[0]", masked_means.shape[0], "masked_residuals.shape[1]", masked_residuals.shape[1])
                assert masked_means.shape[0] == 1
                masked_residuals = masked_residuals + masked_means # should broadcast


        # print(self.edge_mask_attentions)
        # torch.sum(masked_residuals, dim=2, keepdim=True)

        normalized_resid_pre = self.ln1(masked_residuals, parallel=True)
        # print(normalized_resid_pre[:,:,0])
        # print(torch.allclose(normalized_resid_pre[:,:,torch.randperm(normalized_resid_pre.shape[2])],normalized_resid_pre))

        attn_out = self.attn(normalized_resid_pre)

        # self.saved_output = attn_out

        residual = torch.cat((resid_pre, attn_out), dim=2)
        if self.frozen_mask:
            mlp_mask = self.edge_mask_mlp_baseline + self.edge_mask_mlp_frozen * self.edge_mask_mlp
        else:
            mlp_mask = self.edge_mask_mlp

        masked_mlp_residual = einsum("batch position prev_head_idx d_model, prev_head_idx -> batch position d_model", residual, mlp_mask)
        
        normalized_resid_mid = self.ln2(masked_mlp_residual)
        mlp_out = self.mlp(normalized_resid_mid)
        mlp_out = einops.rearrange(mlp_out, "batch position d_model -> batch position 1 d_model")

        residual = torch.cat((residual, mlp_out), dim=2)

        return residual

"""## Unembedding"""

class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(torch.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(torch.zeros((cfg.d_vocab), requires_grad=False))
    
    def forward(self, normalized_resid_final):
        # normalized_resid_final [batch, position, d_model]
        if self.cfg.debug: print("Normalized_resid_final:", normalized_resid_final.shape)
        logits = einsum("batch position d_model, d_model d_vocab -> batch position d_vocab", normalized_resid_final, self.W_U) + self.b_U
        return logits

"""## Full Transformer"""


def get_mask_dict_reformatted(layer, n_heads, mask_dict_superset=None):
    attn_mask = torch.stack([mask_dict_superset[f'a{layer}.{h}'] for h in range(n_heads)], dim=1)
    mlp_mask = mask_dict_superset[f'm{layer}']
    return {'a': attn_mask, 'm': mlp_mask}

class DemoTransformer(nn.Module):
    def __init__(self, cfg, means, 
                 edge_masks=False, 
                 mask_dict_superset=None, 
                 weight_masks_attn=False, 
                 weight_masks_mlp=False, 
                 weight_mask_attn_dict=None, 
                 weight_mask_mlp_dict=None):
        """
        edge_masks: if True, then have trainable masks for edges. If False, then no trainable masks for edges.
        mask_dict_superset: if not None, should be dictionary od 
        weight_mask_attn_dict, if not None, should be dictionary of layer: list of heads to mask. If want to use this, weight_masks_attn should be True.
        weight_mask_mlp_dict, if not None, should be dictionary of layer: bool, whether or not to mask that layer's MLP. Takes precendence over weight_masks_mlp if not None.
        """
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)
        self.weight_masks_attn = weight_masks_attn
        self.weight_masks_mlp = weight_masks_mlp
        for p in self.parameters():
            p.requires_grad = False

        if mask_dict_superset is not None:
            assert edge_masks, "edge_masks should be True if mask_dict_superset is not None (mask_dict_superset values take precedence over global edge_masks value)"
        else:
            if not edge_masks: # don't want to train edge masks
                # make our own mask_dict template, and everything should be 1 so that they're all frozen
                # can be optimized in the future, but this is a shortcut for now
                mask_dict_superset = get_edge_mask_template(num_layers=cfg.n_layers, num_heads=cfg.n_heads)

        if weight_mask_attn_dict is not None:
            assert weight_masks_attn, "weight_masks_attn should be True if weight_mask_attn_dict is not None (weight_mask_attn_dict values take precedence over global weight_masks_attn value)"
        if weight_mask_mlp_dict is not None:
            assert (weight_masks_mlp) or (weight_mask_mlp_dict is None), "weight_masks_mlp should be True if weight_mask_mlp_dict is not None (weight_mask_mlp_dict values take precedence over global weight_masks_mlp value)"

        self.blocks = nn.ModuleList([TransformerBlock(cfg, i, 
                                                      frozen_mask_edges=get_mask_dict_reformatted(i, cfg.n_heads, mask_dict_superset) if mask_dict_superset is not None else None, weight_mask_attn=weight_masks_attn, 
                                                      weight_mask_mlp=weight_mask_mlp_dict[i] if weight_mask_mlp_dict is not None else weight_masks_mlp, 
                                                      weight_mask_attn_heads=weight_mask_attn_dict[i] if weight_mask_attn_dict is not None else None) 
                                                      for i in range(cfg.n_layers)])
        
        total_nodes = (cfg.n_heads + 1) * cfg.n_layers + 1
        self.output_mask = torch.nn.Parameter(torch.ones((total_nodes,)), requires_grad=True)

        self.frozen_mask = True if mask_dict_superset is not None else False
        if self.frozen_mask:
            self.output_mask_baseline = torch.nn.Parameter(mask_dict_superset['output'], requires_grad=False)
            self.output_mask_frozen = torch.nn.Parameter(1 - mask_dict_superset['output'], requires_grad=False)

        self.means = means
    
    def discretize_weight_masks(self, threshold=0.5):
        """
        Call to discretize weight masks. Sets all values below threshold to 0 and all values above threshold to 1.
        """
        for block in self.blocks:
            block.discretize_weight_masks(threshold)
    
    def forward(self, tokens, return_states=False):
        # tokens [batch, position]
        embed = self.embed(tokens)
        pos_embed = self.pos_embed(tokens)
        residual = embed + pos_embed
        residual = einops.rearrange(residual, "batch position d_model -> batch position 1 d_model")
        
        for i, block in enumerate(self.blocks):
            # print(i)
            residual = block(residual, self.means)
            # if hasattr(self,"saved_states"):
            #     self.saved_states = torch.cat((self.saved_states, block.saved_output.unsqueeze(0)), dim=0)
            # else:
            #     self.saved_states = block.saved_output.unsqueeze(0)
        
        if return_states:
            return residual
        
        if self.frozen_mask:
            output_mask = self.output_mask_baseline + self.output_mask_frozen * self.output_mask
        else:
            output_mask = self.output_mask
        residual = einsum("batch position prev_head_idx d_model, prev_head_idx -> batch position d_model", residual, output_mask)
        normalized_resid_final = self.ln_final(residual)
        logits = self.unembed(normalized_resid_final)
        # logits have shape [batch, position, logits]
        # with open("saved_states_new.pkl", "wb") as f:
        #     pickle.dump(self.saved_states, f)
        return [logits]

# %%

# """Take a test string - the intro paragraph of today's featured Wikipedia article. Let's calculate the loss!"""

# model = demo_gpt2

# test_string = """Mini scule is a species of microhylid frog endemic to Madagascar that was described in 2019. The scientific name of the species refers to its size, being a pun on the word minuscule. It is very small, measuring only 8.4 to 10.8 mm (0.33 to 0.43 in) in snout–vent length. It has bronze underparts with a brown groin and back of the thigh, cream upperparts with brown flecking, a dark brown side of the head, and a red iris. On the hind feet, the first toe is absent and the second and fifth toes are strongly reduced. The frog is known only from the Sainte Luce Reserve, where it inhabits areas with deep leaf litter near semi-permanent water bodies. Specimens of frogs from Mandena, the Vohimena mountains, the southern Anosy Mountains, and Tsitongambarika may also be of this species. Along with Mini mum and Mini ature, the other two species in its genus, it received media attention when first described due to the wordplay in its scientific name. (Full article...)"""

# test_tokens = reference_gpt2.to_tokens(test_string).cuda()
# demo_logits = demo_gpt2(test_tokens)

# def lm_cross_entropy_loss(logits, tokens):
#     # Measure next token loss
#     # Logits have shape [batch, position, d_vocab]
#     # Tokens have shape [batch, position]
#     log_probs = logits.log_softmax(dim=-1)
#     pred_log_probs = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
#     return -pred_log_probs.mean()
# loss = lm_cross_entropy_loss(demo_logits, test_tokens)
# print(loss)
# print("Loss as average prob", (-loss).exp())
# print("Loss as 'uniform over this many variables'", (loss).exp())
# print("Uniform loss over the vocab", math.log(demo_gpt2.cfg.d_vocab))

# # %% 
# """We can also greedily generate text:"""

# test_string = "Breaking News: President Trump has been impeached by the House of Representatives for abuse of power and obstruction of Congress. The vote was 230 to 197, with 10 Republicans joining all Democrats in voting to impeach. The president is now only the third in American history to be impeached, and the first to be impeached twice. The House will now send the articles of impeachment to the Senate, where a trial will be held to determine whether to remove the president from office. The Senate is expected to begin the trial on"
# for i in tqdm.tqdm(range(100)):
#     test_tokens = reference_gpt2.to_tokens(test_string).cuda()
#     demo_logits = demo_gpt2(test_tokens)
#     test_string += reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())
# print(test_string)

# %%
