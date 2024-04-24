# File for me to test out the Pythia model without the overhead of duplicate weights for masking
# This should be a drop in replacement into the regular transfomer.py file
"""
A modified version of the transformer architecture to enable casaul path
interventions. Modified from Nanda.
"""

# %% 
import pickle
import einops
from fancy_einsum import einsum
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import numpy as np
import math
from cb_utils.transformer_modules import gelu_new
import tqdm.auto as tqdm
from transformer_lens.utils import get_offset_position_ids
from cb_utils.mask_utils import get_edge_mask_template
from typing import Optional, Union, Tuple, List, Dict
from jaxtyping import Float, Int

device = "cuda" if torch.cuda.is_available() else "mps:0"

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
    # positional_embedding_type: str = "learned"
    # rotary_dim: int = None
    # rotary_base: int = None
    dtype: torch.dtype = torch.float32


cfg = Config()

class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        self.b = nn.Parameter(torch.zeros(cfg.d_model))
    
    def forward(self, residual):
        # residual: [batch, position, d_model]
        if self.cfg.debug: print("Residual:", residual.shape)
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
    W is Parameter of shape (n_heads, ...). Returns baseline and frozen (both only 1d arrays of (n_heads,)), and forward pass should be W_baseline.float() + W_frozen.float() * W 

    Confusingly, W_frozen is actually the mask that allows gradients to flow through unfrozen part. However, I've been calling things frozen for too long to change this without likely hitting some kind of bug.
    """
    W_frozen = torch.nn.Parameter(torch.zeros(W.shape[0], dtype=torch.bool), requires_grad=False).to(device)

    # unsqueeze to broadcast efficiently, until W_frozen has same shape as W
    while len(W_frozen.shape) < len(W.shape):
        W_frozen = W_frozen.unsqueeze(-1)
    
    W_frozen[unfrozen_heads] = True
    # W_baseline = ~W_frozen
    W_baseline = torch.nn.Parameter(~W_frozen, requires_grad=False)
    # convert into float
    return W_baseline.float(), W_frozen.float()

class Attention(nn.Module):
    def __init__(self, cfg, finetune=False, ft_heads=None):

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
        
        self.finetune = finetune
        self.ft_heads = ft_heads

        self.combine_frozen = False

        if finetune and ft_heads is not None:
            self.W_Q_trainable = nn.Parameter(self.W_Q.clone().detach(), requires_grad=True)
            self.W_K_trainable = nn.Parameter(self.W_K.clone().detach(), requires_grad=True)
            self.W_V_trainable = nn.Parameter(self.W_V.clone().detach(), requires_grad=True)
            self.W_O_trainable = nn.Parameter(self.W_O.clone().detach(), requires_grad=True)

            self.W_Q_baseline, self.W_Q_frozen = make_partly_differentiable_mask(self.W_Q, ft_heads)
            self.W_K_baseline, self.W_K_frozen = make_partly_differentiable_mask(self.W_K, ft_heads)
            self.W_V_baseline, self.W_V_frozen = make_partly_differentiable_mask(self.W_V, ft_heads)
            self.W_O_baseline, self.W_O_frozen = make_partly_differentiable_mask(self.W_O, ft_heads)
            self.combine_frozen = True

        self.register_buffer("IGNORE", torch.tensor(-torch.inf, dtype=torch.float32, device=device))


    def forward(self, normalized_resid_pre):
        # normalized_resid_pre: [batch, position, d_model]
        if self.cfg.debug: print("Normalized_resid_pre:", normalized_resid_pre.shape)
        if self.combine_frozen:
            W_Q = self.W_Q_baseline * self.W_Q + self.W_Q_frozen * self.W_Q_trainable
            W_K = self.W_K_baseline * self.W_K + self.W_K_frozen * self.W_K_trainable
            W_V = self.W_V_baseline * self.W_V + self.W_V_frozen * self.W_V_trainable
            W_O = self.W_O_baseline * self.W_O + self.W_O_frozen * self.W_O_trainable           
        
        else:
            W_Q = self.W_Q
            W_K = self.W_K
            W_V = self.W_V
            W_O = self.W_O
        q = einsum("batch query_pos d_model, n_heads d_model d_head -> batch query_pos n_heads d_head", normalized_resid_pre, W_Q) + self.b_Q

        k = einsum("batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", normalized_resid_pre, W_K) + self.b_K
        
        attn_scores = einsum("batch query_pos n_heads d_head, batch key_pos n_heads d_head -> batch n_heads query_pos key_pos", q, k)
        attn_scores = attn_scores / math.sqrt(self.cfg.d_head)
        attn_scores = self.apply_causal_mask(attn_scores)
        pattern = attn_scores.softmax(dim=-1) # [batch, n_head, query_pos, key_pos]
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
        v = einsum("batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", normalized_resid_pre, W_V) + self.b_V
        z = einsum("batch n_heads query_pos key_pos, batch key_pos n_heads d_head -> batch query_pos n_heads d_head", pattern, v)
        # print(z)
        # print('----')
        # print(self.W_O)
        attn_out = einops.einsum(
            z, 
            W_O,
            "batch query_pos n_heads d_head, n_heads d_head d_model -> batch query_pos n_heads d_model"
        ) + (self.b_O / self.cfg.n_heads)
        return attn_out.sum(dim=2)
    
    def apply_causal_mask(self, attn_scores):
        # attn_scores: [batch, n_heads, query_pos, key_pos]
        mask = torch.triu(torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores

"""## MLP"""
class MLP(nn.Module):
    def __init__(self, cfg, finetune=False):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        self.b_in = nn.Parameter(torch.zeros((cfg.d_mlp)))
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model)))
        nn.init.normal_(self.W_out, std=self.cfg.init_range)
        self.b_out = nn.Parameter(torch.zeros((cfg.d_model)))

        self.finetune = finetune
            
    def forward(self, normalized_resid_mid):
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
    def __init__(self, cfg, finetune_attn=False, ft_attn_heads=None, finetune_mlp=False):
        """
        If finetune_attn is True, then the attention weights will be trained.
            If ft_attn_heads is not None, it should be a list of heads to train over.
        If finetune_mlp is True, then the MLP weights will be trained.
        """
        assert not (not finetune_attn and ft_attn_heads is not None), "If ft_attn_heads is not None, finetune_attn must be True"

        super().__init__()
        self.cfg = cfg

        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg, finetune=finetune_attn, ft_heads=ft_attn_heads)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg, finetune=finetune_mlp)

        # for p in self.parameters():
        #     p.requires_grad = False
        for name, p in self.named_parameters():
            p.requires_grad=False
        # manually set the attn and mlp weights to be trainable
        if finetune_attn:
            if ft_attn_heads is not None:
                self.attn.W_Q_trainable.requires_grad = True
                self.attn.W_K_trainable.requires_grad = True
                self.attn.W_V_trainable.requires_grad = True
                self.attn.W_O_trainable.requires_grad = True
            else:
                self.attn.W_Q.requires_grad = True
                self.attn.W_K.requires_grad = True
                self.attn.W_V.requires_grad = True
                self.attn.W_O.requires_grad = True
        
        if finetune_mlp:
            self.mlp.W_in.requires_grad = True
            self.mlp.W_out.requires_grad = True
            self.mlp.b_in.requires_grad = True
            self.mlp.b_out.requires_grad = True

    def forward(self, resid_pre, means=False):
        assert len(resid_pre.shape) == 3, f"resid_pre shape: {resid_pre.shape}"
        normalized_resid_pre = self.ln1(resid_pre)
        attn_out = self.attn(normalized_resid_pre)

        residual = resid_pre + attn_out

        normalized_resid_mid = self.ln2(residual)
        mlp_out = self.mlp(normalized_resid_mid)

        residual = residual + mlp_out
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
    def __init__(self, cfg, 
                 finetune_attn=False, 
                 finetune_mlp=False, 
                 ft_attn_dict=None, 
                 ft_mlp_dict=None,
                 ):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        # Rotary embedding handled by attention at every layer
        self.pos_embed = PosEmbed(cfg)
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)
        for p in self.parameters():
            p.requires_grad = False

            
        self.blocks = nn.ModuleList([TransformerBlock(cfg, 
                finetune_attn = finetune_attn,
                finetune_mlp = ft_mlp_dict[i] if (finetune_mlp and ft_mlp_dict is not None) else finetune_mlp,
                ft_attn_heads = ft_attn_dict[i] if (finetune_attn and ft_attn_dict is not None) else None)
            for i in range(cfg.n_layers)])


    def load_state_dict(self, state_dict, strict=True):
        """
        Need custom loading to set the trainable weights for the attention heads at first equal to the original weights.
        """
        super().load_state_dict(state_dict, strict=strict)
        # After the original state dict has been loaded, manually update the trainable weights
        for block in self.blocks:
            attn = block.attn
            if attn.finetune and attn.ft_heads is not None:
                attn.W_Q_trainable.data = attn.W_Q.data.clone()
                attn.W_K_trainable.data = attn.W_K.data.clone()
                attn.W_V_trainable.data = attn.W_V.data.clone()
                attn.W_O_trainable.data = attn.W_O.data.clone()
    
    def forward(self, tokens, return_states=False):
        # tokens [batch, position]

        embed = self.embed(tokens)
        pos_embed = self.pos_embed(tokens)
        residual = embed + pos_embed

        # residual = einops.rearrange(residual, "batch position d_model -> batch position 1 d_model")
        
        for i, block in enumerate(self.blocks):
            # print(i)
            assert len(residual.shape) == 3, f"residual shape: {residual.shape}"
            residual = block(residual)
            # if hasattr(self,"saved_states"):
            #     self.saved_states = torch.cat((self.saved_states, block.saved_output.unsqueeze(0)), dim=0)
            # else:
            #     self.saved_states = block.saved_output.unsqueeze(0)
        
        if return_states:
            return residual
        
        # if self.frozen_mask:
        #     output_mask = self.output_mask_baseline + self.output_mask_frozen * self.output_mask
        # else:
        #     output_mask = self.output_mask
        # residual = einsum("batch position prev_head_idx d_model, prev_head_idx -> batch position d_model", residual, output_mask)

        normalized_resid_final = self.ln_final(residual)
        logits = self.unembed(normalized_resid_final)
        # logits have shape [batch, position, logits]
        # with open("saved_states_new.pkl", "wb") as f:
        #     pickle.dump(self.saved_states, f)
        return [logits]
