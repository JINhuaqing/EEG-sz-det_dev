import torch.nn as nn
import torch

def ordinal_mse_loss(output_prob, target_cls, num_cls):
    """calculate the loss with MSE for ordinal classification 
        output_prob: tensor (*dim, num_cls), prob of each cls
        target_cls: tensor (*dim), cls label, 0, 1, ldots, num_cls-1
    """
    mse = nn.MSELoss()
    output_cum_prob = output_prob.cumsum(dim=-1);
    target_one_hot_cum = nn.functional.one_hot(target_cls, num_classes=num_cls).cumsum(dim=-1).double();
    return mse(output_cum_prob, target_one_hot_cum)

def my_nllloss(probs, target, num_cls=None):
    """The cross-entropy loss when inputing probs.
       The pytorch's CELoss has a softmax step
    """
    if num_cls is not None:
        target = nn.functional.one_hot(target, num_classes=num_cls)
    probs = probs + 1e-50
    assert target.shape == probs.shape
    return torch.mean(-torch.sum(torch.log(probs) * target, axis=-1))