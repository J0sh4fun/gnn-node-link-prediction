"""Existing Week 4 toy primitive; not a complete GCN or GAT layer."""

import torch

def receiver_softmax(scores,targets,num_nodes):
    if scores.ndim != 2 or targets.ndim != 1 or scores.shape[0] != targets.shape[0]:
        raise ValueError('scores [E,K], targets [E]')
    if not scores.is_floating_point() or targets.dtype != torch.long or scores.device != targets.device:
        raise ValueError('Sai dtype/device')
    if not torch.isfinite(scores).all() or (targets.numel() and (targets.min()<0 or targets.max()>=num_nodes)):
        raise ValueError('Score/index không hợp lệ')
    result = torch.zeros_like(scores)
    for node in range(num_nodes):
        mask = targets == node
        if mask.any():
            block = scores[mask]
            exp = torch.exp(block-block.max(dim=0).values)
            result[mask] = exp/exp.sum(dim=0)
    return result

