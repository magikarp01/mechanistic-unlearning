# Implement a variety of loss functions for mask learning
import torch
import torch.nn.functional as F

def suppression_loss(model, masked_model, x, vocab_size):
    """
    Want model to be close to uniform distribution over vocab for data to forget.
    From DISCOVERING KNOWLEDGE-CRITICAL SUBNETWORKS IN PRETRAINED LANGUAGE MODELS
    """
    pu = torch.full((vocab_size,), 1.0 / vocab_size).to(x.device)

    predictions = masked_model(x)
    log_probs = F.log_softmax(predictions, dim=-1)
    
    return F.kl_div(log_probs, pu, reduction='batchmean')


def maintenance_loss(model, masked_model, x):
    """
    Maintain the original model's predictions for other data points (every data point that is not related to unforgetting) by minimizing KL divergence with original model.
    From DISCOVERING KNOWLEDGE-CRITICAL SUBNETWORKS IN PRETRAINED LANGUAGE MODELS
    """
    with torch.no_grad():
        original_predictions = model(x)
        original_probs = F.softmax(original_predictions, dim=-1)
    
    masked_predictions = masked_model(x)
    masked_probs = F.softmax(masked_predictions, dim=-1)
    
    return F.kl_div(masked_probs.log(), original_probs, reduction='batchmean')


def sparsity_loss(masked_model):
    """
    Minimize mask density to encourage sparsity. 
    From DISCOVERING KNOWLEDGE-CRITICAL SUBNETWORKS IN PRETRAINED LANGUAGE MODELS
    """
    total_density = 0.0
    for mask in masked_model.masks:
        if mask is not None:
            total_density += torch.sigmoid(mask).mean()
    
    return total_density / len(masked_model.masks)
