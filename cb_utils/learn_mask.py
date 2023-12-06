from cb_utils.models import DEVICE, load_demo_gpt2

def train_mask(tasks, task_weights, 
               num_epochs, 
               weight_masks_attn=False,
               weight_masks_mlp=False,
               max_steps_per_epoch=100,
               lr=0.5, 
               log_every=10, 
               clamp_every=50, 
               threshold=0.5, 
               edge_mask_reg_strength=1, 
               weight_mask_reg_strength=1, 
               means=None
               ):
    
    model = load_demo_gpt2(means=means, weight_masks_attn=weight_masks_attn, weight_masks_mlp=weight_masks_mlp)
    for epoch in 
