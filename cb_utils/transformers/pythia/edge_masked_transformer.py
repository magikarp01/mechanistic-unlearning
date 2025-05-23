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
    d_vocab: int = 50304
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12
    positional_embedding_type: str = "learned"
    rotary_dim: int = None
    rotary_base: int = None
    dtype: torch.dtype = torch.float32


cfg = Config()

class LayerNorm(nn.Module):
    
    def __init__(self, cfg, dtype=torch.float32):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_model, dtype=dtype))
        self.b = nn.Parameter(torch.zeros(cfg.d_model, dtype=dtype))

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

    def __init__(self, cfg, dtype=torch.float32):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty((cfg.d_vocab, cfg.d_model), dtype=dtype))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens):
        # tokens: [batch, position]
        if self.cfg.debug: print("Tokens:", tokens.shape)
        embed = self.W_E[tokens, :] # [batch, position, d_model]
        if self.cfg.debug: print("Embeddings:", embed.shape)
        return embed

"""## Positional Embedding"""

class PosEmbed(nn.Module):
    def __init__(self, cfg, dtype=torch.float32):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty((cfg.n_ctx, cfg.d_model), dtype=dtype))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens):
        # tokens: [batch, position]
        if self.cfg.debug: print("Tokens:", tokens.shape)
        pos_embed = self.W_pos[:tokens.size(1), :] # [position, d_model]
        pos_embed = einops.repeat(pos_embed, "position d_model -> batch position d_model", batch=tokens.size(0))
        if self.cfg.debug: print("pos_embed:", pos_embed.shape)
        return pos_embed

"""## Rotary Embedding"""

class RotaryEmbed():
    def __init__(self, cfg):
        self.cfg = cfg

    def calculate_sin_cos_rotary(self, rotary_dim, n_ctx, rotary_base, dtype):
        '''
            Calculate the sin and cos for the rotary embedding

            IMPORTANT: This does it the GPT-NeoX way, which is the case for models
            like Pythia. This is different from the GPT-Neo way, and will lead to problems.
        '''

        high_precision = torch.float32 if dtype != torch.float64 else torch.float64
        pos = torch.arange(n_ctx, dtype=high_precision)
        dim = torch.arange(rotary_dim // 2, dtype=high_precision)

        # A set of frequencies evenly spaced in log space
        freq = rotary_base ** (dim / (rotary_dim / 2))
        freq = einops.repeat(freq, "d -> (2 d)") # If working with GPT-Neo, replace (2 d) with (d 2)

        # Create a n_ctx x rotary_dim tensor, where each column is an arithmetic sequence of angles in that frequency
        angles = pos[:, None] / freq[None, :]
        return torch.sin(angles).to(dtype), torch.cos(angles).to(dtype)

    def rotate_every_two(
        self, x: Float[torch.Tensor, "... rotary_dim"]
    ) -> Float[torch.Tensor, "... rotary_dim"]:
        """
        Rotary helper function, splits x into blocks of size 2 along the final axis and maps [x0, x1] to [-x1, x0]

        The final axis of x must have even length.

        GPT-NeoX and GPT-J do rotary subtly differently, see calculate_sin_cos_rotary for details.
        """
        rot_x = x.clone()
        n = x.size(-1) // 2
        rot_x[..., :n] = -x[..., n:]
        rot_x[..., n:] = x[..., :n]

        return rot_x

    def apply_rotary(
        self,
        x: Float[torch.Tensor, "batch pos head_index d_head"],
        past_kv_pos_offset=0,
        attention_mask: Optional[Int[torch.Tensor, "batch offset_pos"]] = None,
        rotary_cos=None,
        rotary_sin=None,
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        # Only apply rotary to first rotary_dim dimensions (eg, if rotary_dim=64 and d_head=256, only apply to first 1/4 of dimensions)
        x_pos = x.size(1)
        x_rot = x[..., : self.cfg.rotary_dim]
        x_pass = x[..., self.cfg.rotary_dim :]
        x_flip = self.rotate_every_two(x_rot)

        if attention_mask is None:
            rotary_cos = rotary_cos[
                None, past_kv_pos_offset : past_kv_pos_offset + x_pos, None, :
            ]
            rotary_sin = rotary_sin[
                None, past_kv_pos_offset : past_kv_pos_offset + x_pos, None, :
            ]
            x_rotated = x_rot * rotary_cos + x_flip * rotary_sin
        else:
            offset_position_ids = get_offset_position_ids(
                past_kv_pos_offset, attention_mask
            )
            mask_rotary_cos = rotary_cos[offset_position_ids, None, :]
            mask_rotary_sin = rotary_sin[offset_position_ids, None, :]
            x_rotated = x_rot * mask_rotary_cos + x_flip * mask_rotary_sin

        return torch.cat([x_rotated, x_pass], dim=-1)

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
class Attention(nn.Module):
    def __init__(self, cfg, dtype=torch.float32):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head), dtype=dtype))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        self.b_Q = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head), dtype=dtype))
        self.W_K = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head), dtype=dtype))
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        self.b_K = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head), dtype=dtype))
        self.W_V = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head), dtype=dtype))
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        self.b_V = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head), dtype=dtype))
        
        self.W_O = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_head, cfg.d_model), dtype=dtype))
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.b_O = nn.Parameter(torch.zeros((cfg.d_model), dtype=dtype))
        
        self.register_buffer("IGNORE", torch.tensor(-torch.inf, dtype=dtype, device=device))
        if cfg.positional_embedding_type == "rotary":
            self.rotary_embed = RotaryEmbed(cfg)
            sin, cos = self.rotary_embed.calculate_sin_cos_rotary(
                self.cfg.rotary_dim,
                self.cfg.n_ctx,
                rotary_base=self.cfg.rotary_base,
                dtype=dtype
            )
            self.register_buffer("rotary_sin", sin)
            self.register_buffer("rotary_cos", cos)

    def forward(self, normalized_resid_pre):
        # normalized_resid_pre: [batch, position, d_model]
        if self.cfg.debug: print("Normalized_resid_pre:", normalized_resid_pre.shape)
        q = einsum("batch query_pos n_heads d_model, n_heads d_model d_head -> batch query_pos n_heads d_head", normalized_resid_pre, self.W_Q) + self.b_Q

        k = einsum("batch key_pos n_heads d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", normalized_resid_pre, self.W_K) + self.b_K
        
        if self.cfg.positional_embedding_type == "rotary":
            q = self.rotary_embed.apply_rotary(q, rotary_cos=self.rotary_cos, rotary_sin=self.rotary_sin)
            k = self.rotary_embed.apply_rotary(k, rotary_cos=self.rotary_cos, rotary_sin=self.rotary_sin)

        attn_scores = einsum("batch query_pos n_heads d_head, batch key_pos n_heads d_head -> batch n_heads query_pos key_pos", q, k)
        attn_scores = attn_scores / math.sqrt(self.cfg.d_head)
        attn_scores = self.apply_causal_mask(attn_scores)
        pattern = attn_scores.softmax(dim=-1) # [batch, n_head, query_pos, key_pos]
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
        v = einsum("batch key_pos n_heads d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", normalized_resid_pre, self.W_V) + self.b_V
        z = einsum("batch n_heads query_pos key_pos, batch key_pos n_heads d_head -> batch query_pos n_heads d_head", pattern, v)
        # print(z)
        # print('----')
        # print(self.W_O)
        attn_out = einops.einsum(
            z, 
            self.W_O,
            "batch query_pos n_heads d_head, n_heads d_head d_model -> batch query_pos n_heads d_model"
        ) + (self.b_O / self.cfg.n_heads)
        return attn_out

    def apply_causal_mask(self, attn_scores):
        # attn_scores: [batch, n_heads, query_pos, key_pos]
        mask = torch.triu(torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores

"""## MLP"""
class MLP(nn.Module):
    def __init__(self, cfg, dtype=torch.float32):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp), dtype=dtype))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        self.b_in = nn.Parameter(torch.zeros((cfg.d_mlp), dtype=dtype))
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model), dtype=dtype))
        nn.init.normal_(self.W_out, std=self.cfg.init_range)
        self.b_out = nn.Parameter(torch.zeros((cfg.d_model), dtype=dtype))
    
    def forward(self, normalized_resid_mid):
        # normalized_resid_mid: [batch, position, d_model]
        if self.cfg.debug: print("Normalized_resid_mid:", normalized_resid_mid.shape)
        pre = einsum("batch position d_model, d_model d_mlp -> batch position d_mlp", normalized_resid_mid, self.W_in) + self.b_in
        post = gelu_new(pre)
        mlp_out = einsum("batch position d_mlp, d_mlp d_model -> batch position d_model", post, self.W_out) + self.b_out
        return mlp_out

"""## Transformer Block"""
class TransformerBlock(nn.Module):
    def __init__(self, cfg, prev_layers: int, frozen_mask_edges=None, freeze_ones=True, dtype=torch.float32):
        super().__init__()
        self.cfg = cfg

        self.ln1 = LayerNorm(cfg, dtype=dtype)
        self.attn = Attention(cfg, dtype=dtype)
        self.ln2 = LayerNorm(cfg, dtype=dtype)
        self.mlp = MLP(cfg, dtype=dtype)

        self.frozen_mask = True if frozen_mask_edges is not None else False

        for p in self.parameters():
            p.requires_grad = False

        prev_nodes = (cfg.n_heads + 1) * prev_layers + 1
        edge_mask_attentions_init = torch.ones((prev_nodes, cfg.n_heads), dtype=dtype)
        self.edge_mask_attentions = torch.nn.Parameter(edge_mask_attentions_init, requires_grad=True)

        if self.frozen_mask:
            if freeze_ones:
                self.edge_mask_attentions_baseline = torch.nn.Parameter(frozen_mask_edges['a'], requires_grad=False)
                self.edge_mask_attentions_frozen = torch.nn.Parameter(1 - frozen_mask_edges['a'], requires_grad=False)
            else:
                self.edge_mask_attentions_baseline = torch.nn.Parameter(torch.zeros_like(frozen_mask_edges['a']), requires_grad=False)
                self.edge_mask_attentions_frozen = torch.nn.Parameter(frozen_mask_edges['a'], requires_grad=False)

        edge_mask_mlp_init = torch.ones((prev_nodes,), dtype=dtype)
        self.edge_mask_mlp = torch.nn.Parameter(edge_mask_mlp_init, requires_grad=True)
        
        if self.frozen_mask:
            if freeze_ones:
                self.edge_mask_mlp_baseline = torch.nn.Parameter(frozen_mask_edges['m'], requires_grad=False)
                self.edge_mask_mlp_frozen = torch.nn.Parameter(1 - frozen_mask_edges['m'], requires_grad=False)
            else:
                self.edge_mask_mlp_baseline = torch.nn.Parameter(torch.zeros_like(frozen_mask_edges['m']), requires_grad=False)
                self.edge_mask_mlp_frozen = torch.nn.Parameter(frozen_mask_edges['m'], requires_grad=False)

    def forward(self, resid_pre, means=False):
        # resid_pre [batch, position, d_model, prev_head_idx]

        if self.frozen_mask:
            attn_mask = self.edge_mask_attentions_baseline + self.edge_mask_attentions_frozen * self.edge_mask_attentions
        else:
            attn_mask = self.edge_mask_attentions

        masked_resid_pre = einsum(
            "batch position prev_head_idx d_model, prev_head_idx n_heads -> batch position n_heads d_model", 
            resid_pre, 
            attn_mask
        )
        if isinstance(means, torch.Tensor):
            masked_resid_pre_means = einsum("prev_head_idx d_model, prev_head_idx n_heads -> n_heads d_model", means[:self.edge_mask_attentions.shape[0]], 1 - attn_mask)
            masked_resid_pre = masked_resid_pre + masked_resid_pre_means 

        # print(f'resid_pre: {resid_pre.shape}; masked_residuals: {masked_resid_pre.shape}')
        
        normalized_resid_pre = self.ln1(masked_resid_pre, parallel=True)
        # print(f"{masked_resid_pre.shape}, {normalized_resid_pre.shape=}")

        attn_out = self.attn(normalized_resid_pre)
        residual = torch.cat((resid_pre, attn_out), dim=2)

        if self.frozen_mask:
            mlp_mask = self.edge_mask_mlp_baseline + self.edge_mask_mlp_frozen * self.edge_mask_mlp
        else:
            mlp_mask = self.edge_mask_mlp

        masked_mlp_residual = einsum(
            "batch position prev_head_idx d_model, prev_head_idx -> batch position d_model", 
            resid_pre, 
            mlp_mask
        )
        
        normalized_resid_mid = self.ln2(masked_mlp_residual)
        # print(f"{masked_mlp_residual.shape}, {normalized_resid_mid.shape=}")

        mlp_out = self.mlp(normalized_resid_mid)
        mlp_out = einops.rearrange(mlp_out, "batch position d_model -> batch position 1 d_model")

        residual = torch.cat((residual, mlp_out), dim=2)

        # residual should have resid_pre, attn_out, mlp_out cat 
        return residual


    def get_mask_reg(self, norm='l1'):
        # calculate the mask regularization term
        if norm == 'l1':
            if not self.frozen_mask:
                # add edge_mask_attentions and edge_mask_mlp
                edge_reg = self.edge_mask_attentions.abs().sum() + self.edge_mask_mlp.abs().sum()
                tot_params = self.edge_mask_attentions.numel() + self.edge_mask_mlp.numel()
            else:
                # only add edge_mask_attentions that aren't masked by the frozen mask
                # edges with 0s in frozen mask are not added to reg
                edge_reg = (self.edge_mask_attentions_frozen * self.edge_mask_attentions).abs().sum() + (self.edge_mask_mlp_frozen * self.edge_mask_mlp).abs().sum()

                # only count 1s in the frozen mask for total trainable parameters
                tot_params = self.edge_mask_attentions_frozen.sum() + self.edge_mask_mlp_frozen.sum()
            return edge_reg, tot_params

        else:
            raise ValueError(f"Unknown norm {norm}")

"""## Unembedding"""

class Unembed(nn.Module):
    def __init__(self, cfg, dtype=torch.float32):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(torch.empty((cfg.d_model, cfg.d_vocab), dtype=dtype))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(torch.zeros((cfg.d_vocab), dtype=dtype), requires_grad=False)
        
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
                 dtype=torch.float32):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg, dtype=dtype)
        
        if cfg.positional_embedding_type == "learned":
            self.pos_embed = PosEmbed(cfg, dtype=dtype)
        self.ln_final = LayerNorm(cfg, dtype=dtype)
        self.unembed = Unembed(cfg, dtype=dtype)
        for p in self.parameters():
            p.requires_grad = False

        if mask_dict_superset is not None:
            assert edge_masks, "edge_masks should be True if mask_dict_superset is not None (mask_dict_superset values take precedence over global edge_masks value)"
        else:
            if not edge_masks: # don't want to train edge masks
                # make our own mask_dict template, and everything should be 1 so that they're all frozen
                # can be optimized in the future, but this is a shortcut for now
                mask_dict_superset = get_edge_mask_template(num_layers=cfg.n_layers, num_heads=cfg.n_heads, neox=True)
            
        self.blocks = nn.ModuleList([TransformerBlock(cfg, i,
                                                    frozen_mask_edges=get_mask_dict_reformatted(i, cfg.n_heads, mask_dict_superset) if mask_dict_superset is not None else None,
                                                    dtype=dtype) for i in range(cfg.n_layers)])
        total_nodes = (cfg.n_heads + 1) * cfg.n_layers + 1
        self.output_mask = torch.nn.Parameter(torch.ones((total_nodes,), dtype=dtype), requires_grad=True)
        self.frozen_mask = True if mask_dict_superset is not None else False
        if self.frozen_mask:
            self.output_mask_baseline = torch.nn.Parameter(mask_dict_superset['output'], requires_grad=False)
            self.output_mask_frozen = torch.nn.Parameter(1 - mask_dict_superset['output'], requires_grad=False)


        self.means = means

    def forward(self, tokens, return_states=False):
        # tokens [batch, position]

        embed = self.embed(tokens)
        if self.cfg.positional_embedding_type == "learned":
            pos_embed = self.pos_embed(tokens)
            residual = embed + pos_embed
        else:
            residual = embed

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

    def get_edge_reg(self, norm='l1'):
        edge_reg = 0
        tot_params = 0
        for block in self.blocks:
            block_reg, block_params = block.get_mask_reg(norm)
            edge_reg += block_reg
            tot_params += block_params
        
        # add output mask 
        if self.frozen_mask and norm == 'l1':
            edge_reg += (self.output_mask_frozen * self.output_mask).abs().sum()
            tot_params += self.output_mask_frozen.sum()
        return edge_reg, tot_params

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
