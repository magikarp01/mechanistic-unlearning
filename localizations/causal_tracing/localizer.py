#%%
from transformer_lens import HookedTransformer
MODEL_NAME = "google/gemma-2-2b"
DEVICE = "cuda"
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    default_padding_side="left",
    device=DEVICE
)
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
        _, self.save_cache = model.run_with_cache(
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
        return self.p_clean - self.p_corr
    
    def get_indirect_effect(self):
        return self.p_patch - self.p_corr

    def get_formatted_importance(self):
        results_mat = self.get_indirect_effect() / self.get_total_effect()
        result = {}
        for layer in tqdm(list(range(model.cfg.n_layers))):
            for head in tqdm(list(range(model.cfg.n_heads))):
                result[f'a{layer}.{head}'] = results_mat[layer, head]
            result[f'm{layer}'] = results_mat[layer, model.cfg.n_heads]
    
        return result

    def get_embedding_std(self):
        # Get embedding std of import tokens, i.e the tokens we will be noising
        
        noise_stds = []
        for idx in self.noise_indices:
            noise_stds.append(model.W_E[self.toks[idx]].std().item())
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
            if len(act.shape) == 4:
                # Save head
                act[:, :, save_head, :] = self.save_cache[hook.name][:, :, save_head, :]
            else:
                # Save mlp
                act = self.save_cache[hook.name]
        return act

    def do_clean_run(self):
        results_mat = torch.ones((model.cfg.n_layers, model.cfg.n_heads+1))
        model.reset_hooks()
        patched_logits = model(self.toks)
        model.reset_hooks()
        results_mat *= self.get_correct_prob(patched_logits).item()

        return results_mat 

    def do_corrupt_run(self):
        results_mat = torch.ones((model.cfg.n_layers, model.cfg.n_heads+1))
        model.reset_hooks()
        patched_logits = model.run_with_hooks(
            self.toks,
            fwd_hooks = [
                (utils.get_act_name('embed'), self.ct_embedding_noise_hook)
            ]
        )
        model.reset_hooks()
        results_mat *= self.get_correct_prob(patched_logits).item()

        return results_mat 

    def patch_model_components(self, verbose=False):
        results_mat = torch.zeros((model.cfg.n_layers, model.cfg.n_heads+1))
        for layer in tqdm(list(range(model.cfg.n_layers))):
            for head in tqdm(list(range(model.cfg.n_heads))):
                # print('Patching layer', layer, 'head', head)
                patch_hook = functools.partial(
                    self.ct_hook,
                    save_layer=layer,
                    save_head=head
                )
                model.reset_hooks()

                fwd_hooks = [
                    (utils.get_act_name('embed'), self.ct_embedding_noise_hook),
                    (utils.get_act_name('z', layer), patch_hook)
                ]

                patched_logits = model.run_with_hooks(
                    self.toks,
                    fwd_hooks=fwd_hooks
                )
                model.reset_hooks()
                results_mat[layer, head] = self.get_correct_prob(patched_logits).item()
                if verbose:
                    print('Layer', layer, 'Head', head, 'Correct prob', results_mat[layer, head])
            # Do MLP
            patch_hook = functools.partial(
                self.ct_hook,
                save_layer=layer,
                save_head=None
            )
            model.reset_hooks()

            fwd_hooks = [
                (utils.get_act_name('embed'), self.ct_embedding_noise_hook),
                (utils.get_act_name('post', layer), patch_hook)
            ]
            patched_logits = model.run_with_hooks(
                self.toks,
                fwd_hooks=fwd_hooks
            )
            model.reset_hooks()
            results_mat[layer, model.cfg.n_heads] = self.get_correct_prob(patched_logits).item()
            if verbose:
                print('Layer', layer, 'MLP Correct prob', results_mat[layer, model.cfg.n_heads])

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
