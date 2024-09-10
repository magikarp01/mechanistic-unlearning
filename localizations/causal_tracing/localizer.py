#%%
from transformer_lens import HookedTransformer
MODEL_NAME = "google/gemma-2-2b"
DEVICE = "cuda"
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
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
    def __init__(self, model, toks, noise_indices, correct_toks, incorrect_toks=None):
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
        self.clean_cache = model.run_with_cache(
            self.toks,

        )
        self.noise_level = torch.sqrt(3 * self.get_embedding_std())

        # Model has (n_layers, n_heads+1) components (including MLP)
        # Probability of outputting correct result, clean model
        self.p_clean = self.patch_model_components(run_type=RunType.CLEAN)
        # Probability of outputting correct result, corrupt model
        self.p_corr = self.patch_model_components(run_type=RunType.CORRUPT)
        # Probability of outputting correct result, patched model
        self.p_patch = self.patch_model_components(run_type=RunType.PATCH)


    def get_embedding_std(self):
        # Get embedding std of import tokens, i.e the tokens we will be noising
        noise_tokens = self.toks[self.noise_indices]
        return model.W_E[noise_tokens].std()
    
    def get_correct_prob(self, logits):
        # Return probability of correct token(s) at last token position
        return torch.nn.functional.softmax(logits, dim=-1)[:, -1, self.correct_toks].mean(0)
    
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
                    act[i, self.noise_indices[i], :] = act[i, self.noise_indices[i], :] + torch.randn_like(act[i, self.noise_indices[i], :]) * self.noise_level
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

    def patch_model_components(self, run_type=RunType.PATCH, verbose=False):
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

                if run_type == RunType.CLEAN:
                    if verbose:
                        print(f'Running clean on {layer}, {head}')
                    fwd_hooks = []
                elif run_type == RunType.CORRUPT:
                    if verbose:
                        print(f'Running corrupt on {layer}, {head}')
                    fwd_hooks = [
                        (utils.get_act_name('embed'), self.ct_embedding_noise_hook)
                    ]
                elif run_type == RunType.PATCH:
                    if verbose:
                        print(f'Running patch on {layer}, {head}')
                    fwd_hooks = [
                        (utils.get_act_name('embed'), self.ct_embedding_noise_hook),
                        (utils.get_act_name('z', layer), patch_hook)
                    ]

                patched_logits = model.run_with_hooks(
                    self.toks,
                    fwd_hooks=fwd_hooks
                )
                model.reset_hooks()
                results_mat[layer, head] = self.get_correct_prob(patched_logits)
            # Do MLP
            patch_hook = functools.partial(
                self.ct_hook,
                save_layer=layer,
                save_head=None
            )
            model.reset_hooks()
            if run_type == RunType.CLEAN:
                if verbose:
                    print(f'Running clean on {layer}, mlp')
                fwd_hooks = []
            elif run_type == RunType.CORRUPT:
                if verbose:
                    print(f'Running corrupt on {layer}, mlp')
                fwd_hooks = [
                    (utils.get_act_name('embed'), self.ct_embedding_noise_hook)
                ]
            elif run_type == RunType.PATCH:
                if verbose:
                    print(f'Running patch on {layer}, mlp')
                fwd_hooks = [
                    (utils.get_act_name('embed'), self.ct_embedding_noise_hook),
                    (utils.get_act_name('post', layer), patch_hook)
                ]
            patched_logits = model.run_with_hooks(
                self.toks,
                fwd_hooks=fwd_hooks
            )
            model.reset_hooks()
            results_mat[layer, model.cfg.n_heads] = self.get_correct_prob(patched_logits)

        if run_type == RunType.CLEAN:
            self.p_clean = results_mat
        elif run_type == RunType.CORRUPT:
            self.p_corr = results_mat
        elif run_type == RunType.PATCH:
            self.p_patch = results_mat

