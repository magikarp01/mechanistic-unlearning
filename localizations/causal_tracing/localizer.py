#%%
from transformer_lens import HookedTransformer
MODEL_NAME = "google/gemma-2-9b"
DEVICE = "cuda"
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    default_padding_side="left",
    device=DEVICE
)
model.set_use_attn_result(True)
model.set_use_split_qkv_input(True)
# model.set_use_hook_mlp_in(True)
#%%
from tasks.facts.CounterFactTask import CounterFactTask
from transformers import AutoTokenizer

right_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
forget_facts = 16
forget_kwargs = {"forget_fact_subset": forget_facts, "is_forget_dataset": True, "train_test_split": False}
maintain_kwargs = {"forget_fact_subset": forget_facts, "is_forget_dataset": False, "train_test_split": True}
forget_fact_eval = CounterFactTask(batch_size=32, tokenizer=right_tokenizer, device=DEVICE, criterion="cross_entropy", **forget_kwargs)
maintain_facts_eval = CounterFactTask(batch_size=32, tokenizer=right_tokenizer, device=DEVICE, criterion="cross_entropy", **maintain_kwargs)

#%%
import functools

import torch
from tqdm.auto import tqdm
from transformer_lens import utils
import enum

class RunType(enum.Enum):
    CLEAN = 0
    CORRUPT = 1
    PATCH = 2

class CausalTracingLocalizer():
    def __init__(self, model, toks, noise_indices, correct_toks, incorrect_toks=None, verbose=False):
        ''' 
        Creates a causal tracing localizer object

        Args:
            model: HookedTransformer model
            toks: Clean tokens 
            noise_indices: Indices to add noise to, shape (batch, noise_inds)
        '''

        self.model = model
        self.toks = toks 
        self.noise_indices = noise_indices
        self.correct_toks = correct_toks
        self.incorrect_toks = incorrect_toks

        # ct params: noise level, clean cache
        _, self.save_cache = self.model.run_with_cache(
            self.toks,
        )
        self.noise_level = torch.sqrt(3 * self.get_embedding_std())

        # Model has (n_layers, n_heads+1) components (including MLP)
        # Probability of outputting correct result, clean model
        self.p_clean = self.do_clean_run()
        # Probability of outputting correct result, corrupt model
        self.p_corr = self.do_corrupt_run()
        # Probability of outputting correct result, patched model
        self.p_patch = self.patch_model_components(verbose=verbose)

    def get_total_effect(self):
        # return self.p_clean - self.p_corr
        return self.p_clean - self.p_corr
    
    def get_indirect_effect(self):
        # return self.p_patch - self.p_corr
        return {key: self.p_patch[key] - self.p_corr for key in self.p_patch.keys()}
    
    def get_normalized_indirect_effect(self):
        # 0 if no effect, 1 if full effect
        return {key: (self.p_patch[key] - self.p_corr) / (self.p_clean - self.p_corr) for key in self.p_patch.keys()}

    def get_embedding_std(self):
        # Get embedding std of import tokens, i.e the tokens we will be noising
        
        noise_stds = []
        for idx in self.noise_indices:
            noise_stds.append(self.model.W_E[self.toks[idx]].std().item())
        return torch.tensor(noise_stds)
    
    def get_correct_prob(self, logits):
        # Return sum probability of correct token(s) at last token position, avg over batch
        return torch.nn.functional.softmax(logits[:, -1, :], dim=-1)[:, self.correct_toks].mean(0).sum()
    
    def ct_embedding_noise_hook(self, act, hook):
        '''
            Add 3*embedding_std std noise to embedding, 
        '''
        if 'embed' in hook.name:
            # Noise embedding
            if type(self.noise_indices) == list:
                # Noise inds of shape (batch, num_noise_inds)
                # Iterate through batch and noise num_noise_inds
                for i in range(act.shape[0]):
                    act[i, self.noise_indices[i], :] = act[i, self.noise_indices[i], :] + torch.randn_like(act[i, self.noise_indices[i], :]) * self.noise_level[i]
            else:
                act[:, self.noise_indices, :] = act[:, self.noise_indices, :] + torch.randn_like(act[:, self.noise_indices, :]) * self.noise_level
        return act

    def ct_hook(self, act, hook, save_layer, save_head=None):
        '''
            Patch clean into save_layer component
        '''

        if hook.layer() == save_layer:
            # print(hook.name, self.save_cache[hook.name].shape, act.shape)
            if len(act.shape) == 4:
                # Save head
                act[:, :, save_head, :] = self.save_cache[hook.name][:, :, save_head, :]
            else:
                # Save mlp
                act = self.save_cache[hook.name]
        return act

    def do_clean_run(self):
        self.model.reset_hooks()
        patched_logits = self.model(self.toks)
        self.model.reset_hooks()
        return self.get_correct_prob(patched_logits).item()

    def do_corrupt_run(self):
        self.model.reset_hooks()
        patched_logits = self.model.run_with_hooks(
            self.toks,
            fwd_hooks = [
                (utils.get_act_name('embed'), self.ct_embedding_noise_hook)
            ]
        )
        self.model.reset_hooks()
        return self.get_correct_prob(patched_logits).item()

    def patch_model_components(self, verbose=False):
        results_mat = {}
        for layer in tqdm(list(range(self.model.cfg.n_layers))):
            for head_type in ['q', 'k', 'v', 'result']:
                # Handle GroupedQueryAttention
                if head_type in ['k', 'v'] and 'n_key_value_heads' in dir(self.model.cfg):
                    n_heads = self.model.cfg.n_key_value_heads
                else:
                    n_heads = self.model.cfg.n_heads

                # Iterate through heads
                for head in tqdm(list(range(n_heads))):
                    patch_hook = functools.partial(
                        self.ct_hook,
                        save_layer=layer,
                        save_head=head
                    )
                    self.model.reset_hooks()

                    fwd_hooks = [
                        (utils.get_act_name('embed'), self.ct_embedding_noise_hook),
                        (utils.get_act_name(head_type, layer), patch_hook)
                    ]

                    patched_logits = self.model.run_with_hooks(
                        self.toks,
                        fwd_hooks=fwd_hooks
                    )
                    self.model.reset_hooks()
                    prob = self.get_correct_prob(patched_logits).item()
                    results_mat[f"a{layer}.{head}_{head_type}"] = prob
                    if verbose:
                        print(
                            'Layer', layer, 
                            'Head', head, 
                            'Head type', head_type,
                            'Correct prob', prob
                        )
            # Do MLP
            for mlp_type in ['pre', 'mlp_out']:
                patch_hook = functools.partial(
                    self.ct_hook,
                    save_layer=layer,
                    save_head=None
                )
                self.model.reset_hooks()

                fwd_hooks = [
                    (utils.get_act_name('embed'), self.ct_embedding_noise_hook),
                    (utils.get_act_name(mlp_type, layer), patch_hook)
                ]
                patched_logits = self.model.run_with_hooks(
                    self.toks,
                    fwd_hooks=fwd_hooks
                )
                self.model.reset_hooks()
                prob = self.get_correct_prob(patched_logits).item()
                results_mat[f'm{layer}_{mlp_type}'] = prob
            if verbose:
                print(
                    'Layer', layer, 
                    'MLP Type', mlp_type,
                    'MLP Correct prob', prob
                )

        return results_mat

#%%
import numpy as np

def find_sublist_indices(tensor, sublists):
    # Store indices of starting positions for each sublist
    indices = []
    
    # Iterate over each row in the tensor and corresponding sublist
    for row, sublist in zip(tensor, sublists):
        row_len = len(row)
        sublist_len = len(sublist)
        found_index = -1  # Default value if sublist is not found
        
        # Slide over the row to find the sublist
        for i in range(row_len - sublist_len + 1):
            if np.array_equal(row[i:i+sublist_len], sublist):
                found_index = list(range(i, i+sublist_len))
                break
        
        indices.append(found_index)
    
    return indices

#%%

# 'prompt' is list of string prompts
# 'subject' is the string main subject of the prompt
# 'first_token' is int correct answer first tokens
prompt_toks = model.tokenizer(
    forget_fact_eval.train_df['prompt'].tolist(),
    padding=True,
    return_tensors="pt"
)['input_ids']

subject_toks = model.tokenizer(
    forget_fact_eval.train_df['subject'].tolist(),
    add_special_tokens=False
)['input_ids']

subject_idxs = find_sublist_indices(prompt_toks, subject_toks)

correct_toks = torch.tensor(
    forget_fact_eval.train_df['first_token'].tolist()
)

#%%
localizer = CausalTracingLocalizer(
    model,
    toks=prompt_toks,
    noise_indices=subject_idxs,
    correct_toks=correct_toks,
    incorrect_toks=None,
    verbose=False
)
# %%
