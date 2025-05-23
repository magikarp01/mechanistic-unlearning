import copy

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


# Custom hook for non-transformer-lens models
class CustomHook(nn.Module):
    def __init__(self, module, hook_fn):
        super().__init__()
        self.module = module
        self.hook_fn = hook_fn
        self.enabled = True

    def forward(self, *args, **kwargs):
        if self.enabled:
            module_output = self.module(*args, **kwargs)
            # If output is a tuple, apply intervention to the first element
            if isinstance(module_output, tuple):
                # Apply intervention to the first element (usually hidden states)
                modified_first = self.hook_fn(module_output[0])
                assert isinstance(modified_first, torch.Tensor)
                # Return a new tuple with the modified first element and the rest unchanged
                return (modified_first,) + module_output[1:]
            else:
                # If output is not a tuple, just modify and return it
                assert isinstance(module_output, torch.Tensor)
                return self.hook_fn(module_output)
        else:
            return self.module(*args, **kwargs)


def _remove_hook(parent, target):
    for name, module in parent.named_children():
        if name == target:
            setattr(parent, name, module.module)
            return


def insert_hook(parent, target, hook_fn):
    hook = None
    for name, module in parent.named_children():
        if name == target and hook is None:
            hook = CustomHook(module, hook_fn)
            setattr(parent, name, hook)
        elif name == target and hook is not None:
            _remove_hook(parent, target)
            raise ValueError(
                f"Multiple modules with name {target} found, removed hooks"
            )

    if hook is None:
        raise ValueError(f"No module with name {target} found")

    return hook


def remove_hook(parent, target):
    is_removed = False
    for name, module in parent.named_children():
        if name == target and isinstance(module, CustomHook):
            setattr(parent, name, module.module)
            is_removed = True
        elif name == target and not isinstance(module, CustomHook):
            raise ValueError(f"Module {target} is not a hook")
        elif name == target:
            raise ValueError(f"FATAL: Multiple modules with name {target} found")

    if not is_removed:
        raise ValueError(f"No module with name {target} found")


def clear_hooks(model):
    for name, module in model.named_children():
        if isinstance(module, CustomHook):
            setattr(model, name, module.module)
            clear_hooks(module.module)
        else:
            clear_hooks(module)


# Main function for adding adversaries
def add_hooks(
    model,
    create_adversary,
    adversary_locations,
):
    if len(adversary_locations) == 0:
        raise ValueError("No hook points provided")

    adversaries = []
    hooks = []

    for layer, subcomponent in adversary_locations:
        parent = model.get_submodule(layer)
        adversaries.append(create_adversary((layer, subcomponent)))
        hooks.append(insert_hook(parent, subcomponent, adversaries[-1]))

    return adversaries, hooks


# Deepspeed version of add_hooks
class AdversaryWrapper(nn.Module):
    def __init__(self, module: nn.Module, adversary):
        super().__init__()
        self.module = module
        self.adversary = adversary

    def forward(self, *inputs, **kwargs):
        outputs = self.module(*inputs, **kwargs)
        return self.adversary(outputs)


def deepspeed_add_hooks(
    model: nn.Module,
    create_adversary,
    adversary_locations,
):
    if len(adversary_locations) == 0:
        raise ValueError("No hook points provided")

    adversaries = []
    hooks = []

    for layer, subcomponent in adversary_locations:
        parent = model.get_submodule(layer)
        adversary = create_adversary((layer, subcomponent))
        adversaries.append(adversary)
        submodule = parent.get_submodule(subcomponent)
        wrapped_module = AdversaryWrapper(submodule, adversary)
        hooks.append(wrapped_module)
        setattr(parent, subcomponent, wrapped_module)

    return adversaries, hooks


# Adversary classes
class VectorAdversary(nn.Module):

    def __init__(self, dim, batch_size, epsilon, device):
        super().__init__()
        self.dim = dim
        self.device = device
        self.epsilon = epsilon
        self.vector = torch.nn.Parameter(torch.zeros(batch_size, dim, device=device))

    def forward(self, x):
        return x + self.vector.unsqueeze(1)

    def clip_attack(self):
        with torch.no_grad():
            norms = torch.norm(self.vector, dim=-1, keepdim=True)
            scale = torch.clamp(norms / self.epsilon, min=1)
            self.vector.div_(scale)


class LowRankAdversary(nn.Module):

    def __init__(self, dim, rank, device, bias=False, zero_init=True):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.device = device
        self.lora_A = torch.nn.Linear(dim, rank, bias=False).to(device)
        self.lora_B = torch.nn.Linear(rank, dim, bias=bias).to(device)
        if zero_init:
            self.lora_B.weight.data.zero_()

    def forward(self, x):
        return self.lora_B(self.lora_A(x)) + x

    def clip_attack(self):
        # Not defined for low rank adversaries
        pass


class SoftPromptAdversary(nn.Module):

    def __init__(self, dim, batch_size, length, epsilon, device):
        super().__init__()
        self.dim = dim
        self.device = device
        self.epsilon = epsilon
        self.length = length
        self.sp_keys = nn.Parameter(torch.zeros(batch_size, length, dim, device=device))
        self.sp_values = nn.Parameter(
            torch.zeros(batch_size, length, dim, device=device)
        )
        torch.nn.init.kaiming_uniform_(self.sp_keys)
        torch.nn.init.kaiming_uniform_(self.sp_values)
        self.clip_attack()

        # Add attention layers with single head and no bias
        self.query_proj = nn.Linear(dim, dim, bias=False).to(self.device)
        self.attention = nn.MultiheadAttention(
            dim, num_heads=1, batch_first=True, bias=False
        ).to(self.device)

    def forward(self, x):
        query = self.query_proj(x)
        attn_output, _ = self.attention(query, self.sp_keys, self.sp_values)
        return x + attn_output

    def clip_attack(self):
        with torch.no_grad():
            norms = torch.norm(self.sp_keys, dim=-1, keepdim=True)
            scale = torch.clamp(norms / self.epsilon, min=1)
            self.sp_keys.data.div_(scale)

            norms = torch.norm(self.sp_values, dim=-1, keepdim=True)
            scale = torch.clamp(norms / self.epsilon, min=1)
            self.sp_values.data.div_(scale)


class GDAdversary(nn.Module):
    def __init__(self, dim, epsilon, attack_mask, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.epsilon = epsilon

        if dtype:
            self.attack = torch.nn.Parameter(
                torch.zeros(
                    attack_mask.shape[0],
                    attack_mask.shape[1],
                    dim,
                    device=self.device,
                    dtype=dtype,
                )
            )
        else:
            self.attack = torch.nn.Parameter(
                torch.zeros(
                    attack_mask.shape[0], attack_mask.shape[1], dim, device=self.device
                )
            )
        torch.nn.init.kaiming_uniform_(self.attack)
        self.clip_attack()
        self.attack_mask = attack_mask

    def forward(self, x):
        if (
            x.shape[1] == 1 and self.attack.shape[1] != 1
        ):  # generation mode (perturbation already applied)
            return x
        else:
            if self.device is None or self.device != x.device:
                self.device = x.device
                self.attack.data = self.attack.data.to(self.device)
                self.attack_mask = self.attack_mask.to(self.device)

            # Throw an error when attack is shorter than x
            if self.attack.shape[1] < x.shape[1]:
                raise ValueError(
                    f"Attack shape {self.attack.shape} is shorter than input shape {x.shape}"
                )

            perturbed_acts = x[self.attack_mask[:, : x.shape[1]]] + self.attack[
                :, : x.shape[1]
            ][self.attack_mask[:, : x.shape[1]]].to(x.dtype)
            x[self.attack_mask[:, : x.shape[1]]] = perturbed_acts

            return x

    def clip_attack(self):
        with torch.no_grad():
            norms = torch.norm(self.attack, dim=-1, keepdim=True)
            scale = torch.clamp(norms / self.epsilon, min=1)
            self.attack.div_(scale)


class UniversalVectorAdversary(nn.Module):

    def __init__(self, dim, epsilon, device):
        super().__init__()
        self.dim = dim
        self.device = device
        self.epsilon = epsilon
        self.vector = torch.nn.Parameter(torch.zeros(1, dim, device=device))
        torch.nn.init.kaiming_uniform_(self.vector)

    def forward(self, x):
        return x + self.vector.unsqueeze(0)

    def clip_attack(self):
        with torch.no_grad():
            norms = torch.norm(self.vector, dim=-1, keepdim=True)
            scale = torch.clamp(norms / self.epsilon, min=1)
            self.vector.div_(scale)


class UniversalSoftPromptAdversary(nn.Module):

    def __init__(self, dim, length, epsilon, device):
        super().__init__()
        self.dim = dim
        self.device = device
        self.epsilon = epsilon
        self.length = length
        self.sp_keys = nn.Parameter(torch.zeros(length, dim, device=device))
        self.sp_values = nn.Parameter(torch.zeros(length, dim, device=device))
        torch.nn.init.kaiming_uniform_(self.sp_keys)
        torch.nn.init.kaiming_uniform_(self.sp_values)
        self.clip_attack()

        # Add attention layers with single head and no bias
        self.query_proj = nn.Linear(dim, dim, bias=False).to(self.device)
        self.attention = nn.MultiheadAttention(
            dim, num_heads=1, batch_first=True, bias=False
        ).to(self.device)

    def forward(self, x):
        query = self.query_proj(x)
        batched_sp_keys = einops.repeat(self.sp_keys, "l d -> b l d", b=x.shape[0])
        batched_sp_values = einops.repeat(self.sp_values, "l d -> b l d", b=x.shape[0])
        attn_output, _ = self.attention(query, batched_sp_keys, batched_sp_values)
        return x + attn_output

    def clip_attack(self):
        with torch.no_grad():
            norms = torch.norm(self.sp_keys, dim=-1, keepdim=True)
            scale = torch.clamp(norms / self.epsilon, min=1)
            self.sp_keys.data.div_(scale)

            norms = torch.norm(self.sp_values, dim=-1, keepdim=True)
            scale = torch.clamp(norms / self.epsilon, min=1)
            self.sp_values.data.div_(scale)


def zero_nan_grads(model):
    flag = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                flag = True
                p.grad[torch.isnan(p.grad)] = 0.0
    if flag:
        print(f"{type(name)} has nan gradient. Setting it to zero.")


def compute_adversarial_loss(
    model, towards_tokens, towards_labels_mask, coef, probe_loss_coef, losses, probes, cast_to_model_dtype=False
):
    if cast_to_model_dtype:
        model_dtype = next(iter(model.parameters())).dtype
    else:
        model_dtype = torch.float32
    with torch.autocast(device_type="cuda", dtype=model_dtype):
        model_output = model(
            input_ids=towards_tokens, output_hidden_states=probes is not None
        )
        logits = model_output.logits
        final_logits = logits[:, :-1][towards_labels_mask[:, 1:]]
        towards_labels = towards_tokens[:, 1:][towards_labels_mask[:, 1:]]
        toward_loss = F.cross_entropy(final_logits, towards_labels)
        losses["toward"] = toward_loss.item()
        total_loss = toward_loss * coef

        if probes is not None:
            total_probe_loss = 0
            for probe_layer, probe in probes.items():
                probe = probe.cuda()
                layer_acts = model_output.hidden_states[probe_layer + 1]
                probe_outs = probe.predict(layer_acts)[towards_labels_mask]
                # labels = torch.zeros_like(
                #    probe_outs
                # )  # Adjust this if your labels are different
                # probe_loss = F.binary_cross_entropy(probe_outs, labels)
                probe_loss = probe_outs.mean()
                total_probe_loss += probe_loss
            total_loss += total_probe_loss * probe_loss_coef
            losses["probe"] = total_probe_loss.item()

    total_loss.backward()
    losses["total"] = total_loss.item()


def train_attack(
    adv_tokens,
    prompt_mask,
    target_mask,
    model,
    model_layers_module,
    layer,
    epsilon,
    learning_rate,
    pgd_iterations,
    probes=None,
    probe_loss_coef=1.0,
    towards_loss_coef=1.0,
    l2_regularization=0,
    return_loss_over_time=False,
    device="cuda",
    clip_grad=1,
    add_completions_pgd=False,
    adversary_type="pgd",
    verbose=False,
):
    # Clear and initialize the adversary
    clear_hooks(model)
    if isinstance(layer, int):
        layer = [layer]

    if add_completions_pgd:
        attack_mask = torch.any(torch.stack([prompt_mask, target_mask]), dim=0)
    else:
        attack_mask = prompt_mask
    attack_mask = attack_mask.to(device)

    if adversary_type == "pgd":
        create_adversary = lambda x: GDAdversary(
            dim=model.config.hidden_size,
            device=device,
            epsilon=epsilon,
            attack_mask=attack_mask,
        )
    elif adversary_type == "low_rank":
        create_adversary = lambda x: LowRankAdversary(
            dim=model.config.hidden_size,
            rank=16,
            device=device,
            zero_init=True,
        )
    elif adversary_type == "vector":
        create_adversary = lambda x: VectorAdversary(
            dim=model.config.hidden_size,
            batch_size=adv_tokens.shape[0],
            epsilon=epsilon,
            device=device,
        )
    elif adversary_type == "soft_prompt":
        create_adversary = lambda x: SoftPromptAdversary(
            dim=model.config.hidden_size,
            batch_size=adv_tokens.shape[0],
            epsilon=epsilon,
            length=15,
            device=device,
        )
    else:
        raise ValueError(f"Adversary type {adversary_type} not recognized")

    adversary_locations = [
        (f"{model_layers_module}", f"{layer_i}")
        for layer_i in layer
        if isinstance(layer_i, int)
    ]
    if "embedding" in layer:
        adversary_locations.append(
            (model_layers_module.replace(".layers", ""), "embed_tokens")
        )

    adversaries, wrappers = add_hooks(
        model,
        create_adversary=create_adversary,
        adversary_locations=adversary_locations,
    )
    params = [p for adv in adversaries for p in adv.parameters()]

    # Define optimization utils
    adv_optim = torch.optim.AdamW(params, lr=learning_rate)
    loss_over_time = [] if return_loss_over_time else None
    losses = {}

    # Optimize adversary to elicit attack labels
    for _ in tqdm(range(pgd_iterations), disable=not verbose):
        adv_optim.zero_grad()

        # Compute the adversary loss
        compute_adversarial_loss(
            model=model,
            towards_tokens=adv_tokens,
            towards_labels_mask=target_mask,
            coef=towards_loss_coef,
            probe_loss_coef=probe_loss_coef,
            losses=losses,
            probes=probes,
        )

        # Add L2 penalty if specified
        if l2_regularization:
            reg_loss = sum(torch.norm(adv.attack) for adv in adversaries)
            num_el = sum(torch.numel(adv.attack) for adv in adversaries)
            (l2_regularization * reg_loss / np.sqrt(num_el)).backward()
            losses["l2_norm"] = reg_loss.item() / np.sqrt(num_el)

        # Optimizer step
        for adv in adversaries:
            zero_nan_grads(adv)

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(params, clip_grad)
        adv_optim.step()

        for adv in adversaries:
            adv.clip_attack()

        if return_loss_over_time:
            loss_over_time.append(copy.deepcopy(losses))

    return (loss_over_time, wrappers) if return_loss_over_time else (losses, wrappers)


def train_universal_attack(
    adv_tokens,
    target_mask,
    model,
    model_layers_module,
    layer,
    epsilon,
    learning_rate,
    n_steps,
    batch_size,
    gradient_accumulation_steps=1,
    probes=None,
    probe_loss_coef=1.0,
    towards_loss_coef=1.0,
    l2_regularization=0,
    return_loss_over_time=False,
    device="cuda",
    clip_grad=1,
    adversary_type="soft_prompt",
    verbose=False,
    adversaries=None,  # Pass in existing adversaries directly
    wrappers=None,  # Pass in existing wrappers directly
    return_adversaries=False,  # Option to return adversaries
    cast_to_model_dtype=False,
):
    # Clear hooks
    if adversaries is None:
        # We dont want to clear any hooks if adversaries are provided
        clear_hooks(model)

    # Only create new adversaries if none are provided
    if adversaries is None:
        if isinstance(layer, int):
            layer = [layer]

        if adversary_type == "low_rank":
            create_adversary = lambda x: LowRankAdversary(
                dim=model.config.hidden_size,
                rank=16,
                device=device,
                zero_init=True,
            )
        elif adversary_type == "vector":
            create_adversary = lambda x: UniversalVectorAdversary(
                dim=model.config.hidden_size,
                epsilon=epsilon,
                device=device,
            )
        elif adversary_type == "soft_prompt":
            create_adversary = lambda x: UniversalSoftPromptAdversary(
                dim=model.config.hidden_size,
                epsilon=epsilon,
                length=15,
                device=device,
            )
        else:
            raise ValueError(f"Adversary type {adversary_type} not recognized")

        adversary_locations = [
            (f"{model_layers_module}", f"{layer_i}")
            for layer_i in layer
            if isinstance(layer_i, int)
        ]
        if "embedding" in layer:
            adversary_locations.append(
                (model_layers_module.replace(".layers", ""), "embed_tokens")
            )

        adversaries, wrappers = add_hooks(
            model,
            create_adversary=create_adversary,
            adversary_locations=adversary_locations,
        )
    elif wrappers is None:
        raise ValueError("Wrappers must be provided if adversaries are provided")
    else:
        for wrapper in wrappers:
            wrapper.enabled = True

    # Get parameters of each part of the attack for optimization
    params = [p for adv in adversaries for p in adv.parameters()]

    # Define optimization utils
    adv_optim = torch.optim.AdamW(params, lr=learning_rate)
    adv_optim.zero_grad()  # Zero out gradients in case

    # Ensure n_steps is divisible by gradient_accumulation_steps
    assert (
        n_steps % gradient_accumulation_steps == 0
    ), "n_steps must be divisible by gradient_accumulation_steps"

    # Construct a dataloader
    dataset = TensorDataset(adv_tokens, target_mask)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    data_iterator = iter(dataloader)

    # Optimize adversary for n_steps forward passes
    losses = None
    loss_over_time = [] if return_loss_over_time else None

    # Create progress bar
    pbar = tqdm(range(n_steps), disable=not verbose)

    for step in pbar:
        try:
            batch_tokens, batch_target_mask = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            batch_tokens, batch_target_mask = next(data_iterator)

        batch_tokens = batch_tokens.to(device)
        batch_target_mask = batch_target_mask.to(device)

        # Compute the adversary loss
        losses = {}
        compute_adversarial_loss(
            model=model,
            towards_tokens=batch_tokens,
            towards_labels_mask=batch_target_mask,
            coef=towards_loss_coef,
            probe_loss_coef=probe_loss_coef,
            losses=losses,
            probes=probes,
            cast_to_model_dtype=cast_to_model_dtype,
        )

        # Add L2 penalty if specified
        if l2_regularization:
            reg_loss = sum(torch.norm(adv.attack) for adv in adversaries)
            num_el = sum(torch.numel(adv.attack) for adv in adversaries)
            l2_loss = (
                l2_regularization * reg_loss / np.sqrt(num_el)
            ) / gradient_accumulation_steps
            l2_loss.backward()
            losses["l2_norm"] = reg_loss.item() / np.sqrt(num_el)

        # Update progress bar with current losses
        pbar.set_postfix(losses)

        # Perform optimization step if gradient accumulation is complete
        if (step + 1) % gradient_accumulation_steps == 0:
            for adv in adversaries:
                zero_nan_grads(adv)

            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(params, clip_grad)

            adv_optim.step()
            adv_optim.zero_grad()

            for adv in adversaries:
                adv.clip_attack()

            # Record the average loss over the accumulation steps
            if return_loss_over_time:
                loss_over_time.append(copy.deepcopy(losses))

    if return_loss_over_time and return_adversaries:
        return loss_over_time, wrappers, adversaries
    elif return_loss_over_time:
        return loss_over_time, wrappers
    elif return_adversaries:
        return losses, wrappers, adversaries
    else:
        return losses, wrappers
