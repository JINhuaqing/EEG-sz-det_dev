# the main model we use for gTVDN
# It is cumstmized from https://github.com/karpathy/nanoGPT
# I do not use SC-layer as it may not be causal (on May 8, 2023)
import numpy as np
import torch
import torch.nn as nn
from torch.functional import F


def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.ndim))
        self.bias = nn.Parameter(torch.zeros(config.ndim)) if config.is_bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    

class Block(nn.Module):
    """Two attention in one blocks
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config)
        self.tattn = timeAttention(config)
        self.ln_2 = LayerNorm(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.tattn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x    
    
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.ndim, 4 * config.ndim, bias=config.is_bias)
        self.c_proj  = nn.Linear(4 * config.ndim, config.ndim, bias=config.is_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class timeAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.ndim % config.n_head == 0
        self.is_mask = config.is_mask
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.ndim, 3 * config.ndim, bias=config.is_bias)
        # output projection
        self.c_proj = nn.Linear(config.ndim, config.ndim, bias=config.is_bias)
        
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.ndim = config.ndim
        self.dropout = config.dropout
        
        # causal mask to ensure that attention is only applied to the left in the input sequence
        if self.is_mask:
            self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, feature dimensionality (ndim)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.ndim, dim=2)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
        if self.is_mask:
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class spAttention(nn.Module):
    """Attention on the spatial axis.
       This layer is a spatial attention for each time point
    """

    def __init__(self, config):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.nfeature, 3 * config.nfeature, bias=config.is_bias)
        # output projection
        self.c_proj = nn.Linear(config.nfeature, config.nfeature, bias=config.is_bias)
        self.nfeature = config.nfeature
        
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        
    def forward(self, x):
        """args:
                x: batchsize x len_seq x nroi
        """
        q, k, v = self.c_attn(x).split(self.nfeature, dim=2)
        q = q.unsqueeze(-1)
        k = k.unsqueeze(-2)
        v = v.unsqueeze(-2)
        weight = (q * k); # no need to divide by sqrt()
        nws = F.softmax(weight, dim=-1);
        nws = self.attn_dropout(nws)
        
        sp_attn_v = torch.sum(nws * v, axis=-1);
        y = self.resid_dropout(self.c_proj(sp_attn_v))
        return y    

class SC(nn.Module):
    """This layer is to use a cos-sin basis to approximate a time series
       i.e., sx(t) = A0 + sum_i Ai cos(it) + sum_i Bi sin(it)
       It is partially causal.
       It is only causal when I predict X_t+k from X_t but do not use X_t to X_t+1 (on May 8, 2023). 
    """

    def __init__(self, config):
        super().__init__()
        assert config.block_size % config.move_step == 0
        
        num_seq = int(config.block_size / config.move_step)
        ts = torch.arange(config.block_size, dtype=float)
        ts = 2*np.pi*ts.unsqueeze(-1) * torch.arange(1, config.num_Bs+1)/config.fs
        sin_ts = torch.sin(ts)
        cos_ts = torch.cos(ts)
        
        # the mask tensor
        mask_mat = torch.ones_like(sin_ts);
        mask_tensor = []
        for ix in range(num_seq):
            mask_mat_c = mask_mat.clone()
            mask_mat_c[0:ix*config.move_step] = 0
            mask_tensor.append(mask_mat_c)
        mask_tensor = torch.stack(mask_tensor) # num_seq x block size x numBs
        
        
        sin_ts_mask = mask_tensor * sin_ts.repeat(num_seq, 1, 1);# num_seq x block size x numBs
        cos_ts_mask = mask_tensor * cos_ts.repeat(num_seq, 1, 1);# num_seq x block size x numBs
        all_ts_mask = torch.cat([sin_ts_mask, cos_ts_mask, mask_tensor[:, :, :1]], axis=2);
        all_ts_mask = all_ts_mask.transpose(0, 1)[None, :, None, :, :];

        
        self.register_buffer("all_ts_mask", all_ts_mask) # register cos_ts and sin_ts 
        self.num_seq = num_seq
        self.fc_sc = nn.Linear(config.move_step, config.num_Bs*2+1, bias=config.is_bias) 

    def forward(self, x):
        """ x: batch_size x len_seq x nfs
        """
        B, T, C = x.shape
        x_trans = x.transpose(1, 2).view(B, C, self.num_seq, -1)
        x_trans = self.fc_sc(x_trans).unsqueeze(1)
        sx = (x_trans * self.all_ts_mask).sum(axis=(-1, -2))
        return sx    
      
    
class myNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.block_size is not None
        self.config = config

        self.sp_attn = spAttention(config)
        self.transformer = nn.ModuleDict(dict(
            fc_init = nn.Linear(config.nfeature, config.ndim),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config),
        ))
        #self.sc = SC(config)
        self.fc_out = nn.Linear(config.ndim, config.target_dim, bias=False)
        
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/np.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    def get_num_params(self):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, inputs):
        b, t, nf = inputs.size() # batchsize x len_seq x num_features
        assert t == self.config.block_size, f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"

        # forward the GPT model itself
        x = self.sp_attn(inputs) + inputs
        x = self.transformer.fc_init(x) 
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        #x = self.sc(x) + x # Remove the SC layers
        x = self.transformer.ln_f(x)
        x = self.fc_out(x)
        return x
