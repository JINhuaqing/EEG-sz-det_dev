"""Some netout2pred functions.
   The output of the network can be a matrix or a vector.
"""
import torch
import torch.nn as nn
from torch.functional import F
import numpy as np


def netout2pred_VARK(netout, X_input, config):
    """ Model is X_t+k = \sum_i=t^{t+k-1} X_i + A_i X_i
    """
    other_dim = netout.shape[:-1];
    netout_mat = netout.reshape(*other_dim, config.nfeature, config.nfeature);
    AX = torch.matmul(netout_mat, X_input.unsqueeze(-1)).squeeze(-1) + X_input;
    pred_Y = F.conv1d(AX.transpose(1, 2), torch.ones(AX.shape[-1], 1, config.move_step)/config.move_step, groups=AX.shape[-1]);
    pred_Y = pred_Y.transpose(1, 2);
    return pred_Y


def netout2pred_dlt(netout, X_input):
    """ X_t+k = dlt + X_t
    """
    Y_pred = netout + X_input
    return Y_pred

def netout2loss_tvdn(netout, X_input, Y, loss_fn, config, num_seg=None):
    """X_t+k = exp(A_t k*dlt)X_t for k =1, ldots, K
        args:
            num_seg: num of segments in the loop. In fact, it does not affect the speed. (on May 8, 2023) 
    """
    if num_seg is None:
        num_seg = config.move_step
    assert config.move_step % num_seg == 0
    seg_len = int(config.move_step//num_seg)
    dims = netout.shape[:-1]
    
    netout = netout.reshape(*dims, config.nfeature, config.nfeature);
    # add identity matrix 
    netout = netout +  torch.eye(netout.shape[-1])[None, None, :, :];
    # time sequence
    ts = torch.arange(1, config.move_step+1) * (1/config.fs);
    
    def Y2curY(Y):
        cur_Y = torch.stack([Y[:, idx:(idx+config.block_size)] for idx in range(low_idx, up_idx)], axis=1)
        return cur_Y.transpose(0, 1)
    loss = 0
    for cur_seg_idx in range(num_seg):
        low_idx, up_idx = cur_seg_idx*seg_len, (cur_seg_idx+1)*seg_len
        ts_part = ts[cur_seg_idx*seg_len:(cur_seg_idx+1)*seg_len]
        log_mats = netout.unsqueeze(-1) * ts_part;
        mats = torch.linalg.matrix_exp(log_mats.permute(-1, *range(log_mats.ndim-1)).contiguous());
        preds = torch.matmul(mats,  X_input.unsqueeze(-1)).squeeze(-1);
        cur_Y = Y2curY(Y)
        loss = loss + loss_fn(preds, cur_Y)/num_seg
    return loss

def netout2loss_tvdn1(netout, X_input, Y, loss_fn, config):
    """X_t+k = exp(A_t k*dlt)X_t for k =1, ldots, K
       It is a simpler version of netout2loss_tvdn, for checking
    """
    dims = netout.shape[:-1]
    
    netout = netout.reshape(*dims, config.nfeature, config.nfeature);
    # add identity matrix 
    netout = netout +  torch.eye(netout.shape[-1])[None, None, :, :];
    # time sequence
    ts = torch.arange(1, config.move_step+1) * (1/config.fs);
    
    loss = 0
    for cur_idx in range(config.move_step):
        cur_Y = Y[:, cur_idx:(cur_idx+config.block_size)]
        ts_cur = ts[cur_idx].item()
        log_mat = netout * ts_cur;
        mat = torch.linalg.matrix_exp(log_mat.contiguous());
        pred = torch.matmul(mat,  X_input.unsqueeze(-1)).squeeze(-1);
        loss = loss + loss_fn(pred, cur_Y)/config.move_step
    return loss