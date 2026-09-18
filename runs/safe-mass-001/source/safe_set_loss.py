"""Negative log probability of choosing any certified safe action."""
import torch

def safe_set_loss(logits,legal,labels):
    if not torch.all(labels.any(1)) or torch.any(labels & ~legal):
        raise ValueError('Every row needs a nonempty legal safe set')
    all_z=logits.masked_fill(~legal,float('-inf')).logsumexp(1)
    safe_z=logits.masked_fill(~labels,float('-inf')).logsumexp(1)
    return (all_z-safe_z).mean()
