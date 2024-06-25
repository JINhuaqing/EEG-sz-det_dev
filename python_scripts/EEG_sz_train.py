#!/usr/bin/env python
# coding: utf-8

# This file is to test my model on the TUH-EEG-seizure data
# 
# In this file, I train the model with two loss, 
# 
# - loss1: the loss predicting X_t from X_{t-1}
# - loss2: the loss predicting seizure label from X_t
# 
# Note that I always discretize X into 2^K classes, so the loss1 is also a classification loss.
# 

# # Pre-params

# In[1]:


MODEL_NAME = "two_loss_autoreg_small"
SAVED_MODEL = None
MODEL_CLASS = "my_main_model_dis_base.py"


# # Load pkgs 

# In[2]:


import sys
sys.path.append("../mypkg")
from constants import RES_ROOT, FIG_ROOT, DATA_ROOT, MODEL_ROOT


# In[3]:


import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from easydict import EasyDict as edict
import time
# copy file
import shutil
from pprint import pprint



import importlib


# In[6]:


from models.my_main_model_dis_base import myNet
from models.losses import my_nllloss, ordinal_mse_loss
from models.model_utils import generate_position_encode 
from data_utils.eeg_load_sz import EEGDataSZ
from data_utils import digitize_data, rec_data, MyDataLoader
from utils.misc import delta_time, load_pkl_folder2dict, save_pkl_dict2folder, truncated_mean_upper


# In[7]:


# pkgs for pytorch (on Apr 3, 2023)
import torch
import torch.nn as nn
from torch.functional import F
from torch.optim.lr_scheduler import ExponentialLR

torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
torch.set_default_device(device)
device


# In[ ]:





# # training

# ## Model and training params

# In[23]:


if SAVED_MODEL is None:
    config = edict()
    config.nfeature = 19 # the dim of features at each time point
    config.ndim = 256 # the output of the first FC layer
    config.dropout = 0.5 # the dropout rate
    config.n_layer = 3 # the number of self-attention layers
    config.n_head = 8 # numher of heads for multi-head attention
    config.is_mask = True # Use mask to make the attention causal
    config.is_bias = True # Bias  for layernorm
    config.block_size = 256 # the preset length of seq, 
    config.batch_size = 8 # the batch size
    config.move_step = 1 # k, movestep
    config.fs = 90
    config.target_dim = 19
    config.k = 8 # discretize to 2^k levels
    config.ncls = 2 # number of classes, 2 for my seizure data
    
    train_params = edict()
    train_params.nepoch= 2
    train_params.loss_out = 20
    train_params.test_loss_out = 50
    train_params.eval_size = 8
    train_params.clip = 1 # 
    # lr step decay, if lr_step is 0, then no decay
    # if '1epoch', then decay every epoch
    train_params.lr_step = 5000
    train_params.lr = 1e-4 
    train_params.lr_gamma = 0.1
    train_params.lr_weight_decay = 0
    # save the model 
    # if '1epoch', then save every epoch
    train_params.save_interval = 1000

    # data parameters
    data_params = edict()
    data_params.move_params=dict(winsize=config.block_size+config.move_step, 
                     stepsize=config.block_size+config.move_step, 
                     marginsize=None)
    data_params.pre_params=dict(is_detrend=True, 
                    is_drop=True,
                    target_fs=90, 
                    filter_limit=[1, 45], 
                    is_diff=False)
    data_params.rm_params=dict(rm_len=50,
                   keep_len=20)
    data_params.subset = "AR"

else:
    saved_model_path = RES_ROOT/SAVED_MODEL
    assert saved_model_path.exists(), "No such model"
    saved_model = load_pkl_folder2dict(saved_model_path)
    
    config = saved_model.config
    train_params = saved_model.train_params
    data_params = saved_model.data_params


# ## load data

# In[16]:


verbose = 1
move_params = data_params.move_params
pre_params = data_params.pre_params
rm_params = data_params.rm_params
subset = data_params.subset

train_data_bckg = EEGDataSZ(
    dataset="train", 
    subset=subset,
    label="bckg", 
    discrete_k=config.k, 
    verbose=verbose, 
    move_params=move_params,
    pre_params=pre_params,
    rm_params=rm_params
    )
# to generate the cutoff of the background data for discretization
train_data_bckg.get_dis_cutoffs();

train_data_sz = EEGDataSZ(
    dataset="train", 
    subset=subset,
    label="sz", 
    discrete_k=config.k, 
    verbose=verbose, 
    move_params=move_params,
    pre_params=pre_params,
    rm_params=rm_params
    )

test_data_bckg = EEGDataSZ(
    dataset="dev", 
    subset=subset,
    label="bckg", 
    discrete_k=config.k, 
    verbose=verbose, 
    move_params=move_params,
    pre_params=pre_params,
    rm_params=rm_params
    )

test_data_sz = EEGDataSZ(
    dataset="dev", 
    subset=subset,
    label="sz", 
    discrete_k=config.k, 
    verbose=verbose, 
    move_params=move_params,
    pre_params=pre_params,
    rm_params=rm_params
    )


train_data_bckg_loader = MyDataLoader(train_data_bckg, 
                                      batch_size=config.batch_size, 
                                      shuffle=True)
train_data_sz_loader = MyDataLoader(train_data_sz, 
                                    batch_size=config.batch_size, 
                                    shuffle=True)
test_data_bckg_loader = MyDataLoader(test_data_bckg, 
                                      batch_size=config.batch_size, 
                                      shuffle=False)
test_data_sz_loader = MyDataLoader(test_data_sz, 
                                    batch_size=config.batch_size, 
                                    shuffle=False)
                                
print(f"Num of data: train_bckg: {len(train_data_bckg)}, train_sz: {len(train_data_sz)}, test_bckg: {len(test_data_bckg)}, test_sz: {len(test_data_sz)}")


# ## Prepare training

# In[17]:


def trans_batch(batch_sz, batch_bckg, shuffle=False):
    """transform the batch to make it easy for training
    args: 
        - batch_sz: the seizure data batch from the dataloader
        - batch_bckg: the background data batch from the dataloader
        - shuffle: whether to shuffle the batch
    return: res (both seizure and background data)
        - res[0]: X_rec, the input of the model 
        - res[1]: Y_dis, the output of the model, the discrete value
        - res[2]: labels, the labels of the data, 1 for seizure, 0 for background
    """
    def _trans_batch_single(batch):
        batch_dis, batch_rec = batch

        X_rec, Y_rec = batch_rec[:, :-config.move_step], batch_rec[:, config.move_step:]
        Y_dis = batch_dis[:, config.move_step:]
        Y_move_dis = batch_dis[:, (config.move_step-1):-1] # use X_t as prediction of X_t+1
        Y_move_prob = nn.functional.one_hot(Y_move_dis, num_classes=2**config.k).double()
        return X_rec, Y_dis

    res_sz = _trans_batch_single(batch_sz)
    res_bckg = _trans_batch_single(batch_bckg)
    labels = torch.cat([torch.ones(res_sz[0].size(0)), torch.zeros(res_bckg[0].size(0))], dim=0).long()
    res = []
    if shuffle:
        n_totol = res_sz[0].size(0) + res_bckg[0].size(0)
        idx = torch.randperm(n_totol)
        labels = labels[idx]
    for re1, re2 in zip(res_sz, res_bckg):
        re = torch.cat([re1, re2], dim=0)
        if shuffle:
            re = re[idx]
        res.append(re)
    res.append(labels)
    return res


# In[18]:


pos_enc = generate_position_encode(config.block_size, config.nfeature).unsqueeze(0)
loss_fn1 = ordinal_mse_loss
# logSoftmax + NLLLoss = CrossEntropyLoss
loss_fn2 = nn.NLLLoss() 

if SAVED_MODEL is None:
    net = myNet(config)
else:
    net = saved_model.model
if torch.cuda.is_available():
    net = net.cuda()
optimizer = torch.optim.Adam(net.parameters(), 
                             lr=train_params.lr,
                             weight_decay=train_params.lr_weight_decay)
scheduler = ExponentialLR(optimizer, 
                          gamma=train_params.lr_gamma);


# In[19]:


# to evaluate the model on the test data
def evaluate(net, test_size=32):
    sub_idxs = torch.randint(low=0, high=len(test_data_sz_loader), size=(test_size, ))
    losses1 = []
    losses2 = []
    for sub_idx in sub_idxs:
        batch_sz_test = test_data_sz_loader(sub_idx.item())
        batch_bckg_test = test_data_bckg_loader(sub_idx.item())
        X_rec, Y_dis, szlabels = trans_batch(batch_sz=batch_sz_test, batch_bckg=batch_bckg_test, shuffle=False)
        X_rec_wpos = X_rec + pos_enc;
        net.eval()
        with torch.no_grad():
            probs1, log_probs2 = net(X_rec_wpos)
            loss1 = loss_fn1(probs1, Y_dis, num_cls=2**config.k)
            loss2 = loss_fn2(log_probs2, szlabels)
        losses1.append(loss1.item())
        losses2.append(loss2.item())
    net.train()
    return np.median(losses1), np.median(losses2)


# ## training the model

# In[20]:


print("Start training")
pprint(train_params)
pprint(data_params)


# In[26]:


# training
loss1_cur = []
loss2_cur = []
if SAVED_MODEL is None:
    loss_save = edict()
    loss_save.train_niter = []
    loss_save.test_niter = []
    loss_save.train1 = []
    loss_save.train2 = []
    loss_save.test1 = []
    loss_save.test2 = []
else:
    loss_save = edict(saved_model.loss_save)

if isinstance(train_params.lr_step, str):
    lr_step = int(len(train_data_sz_loader) * float(train_params.lr_step[:-5]))
else:
    lr_step = train_params.lr_step
if isinstance(train_params.save_interval, str):
    save_interval = int(len(train_data_sz_loader) * float(train_params.save_interval[:-5]))
else:
    save_interval = train_params.save_interval


t0 = time.time()
total_iter = 0
for iep in range(train_params.nepoch):
    net.cuda()
    net.train()
    print(f"The current lr is {scheduler.get_last_lr()}.")
    for ix in range(len(train_data_sz_loader)):
        batch_sz = train_data_sz_loader(ix)
        batch_bckg = train_data_bckg_loader(ix)
        X_rec, Y_dis, szlabels  = trans_batch(batch_sz=batch_sz, batch_bckg=batch_bckg, shuffle=True)
        X_rec_wpos = X_rec + pos_enc
        # Zero the gradients
        optimizer.zero_grad()
        
        probs1, log_probs2 = net(X_rec_wpos)
        loss1 = loss_fn1(probs1, Y_dis, num_cls=2**config.k)
        loss2 = loss_fn2(log_probs2, szlabels)
        # TODO: add weight to the loss
        loss = loss1 + loss2
        
        # Perform backward pass
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(net.parameters(), train_params.clip)
        # Perform optimization
        optimizer.step()
        
        loss1_cur.append(loss1.item())
        loss2_cur.append(loss2.item())
        if ix % train_params.loss_out == (train_params.loss_out-1):
            loss_save.train_niter.append(total_iter)
            loss_save.train1.append(np.median(loss1_cur))
            loss_save.train2.append(np.median(loss2_cur))
            print(f"At iter {ix+1}/{len(train_data_sz_loader)}, epoch {iep+1}, "
                  f"the losses (autoreg-train) are {loss_save.train1[-1]:.3f} . "
                  f"the losses (cls-train) are {loss_save.train2[-1]:.3f} . "
                  f"The time used is {delta_time(t0):.3f}s. "
                 )
            loss1_cur = []
            loss2_cur = []
            t0 = time.time()
            
        if ix % train_params.test_loss_out == (train_params.test_loss_out-1):
            loss_save.test_niter.append(total_iter)
            loss_test1, loss_test2 = evaluate(net, test_size=train_params.eval_size)
            loss_save.test1.append(loss_test1)
            loss_save.test2.append(loss_test2)
            print("="*50)
            print(f"At iter {ix+1}/{len(train_data_sz_loader)}, epoch {iep+1}, "
                  f"the losses (autoreg-test) are {loss_save.test1[-1]:.3f} . "
                  f"the losses (cls-test) are {loss_save.test2[-1]:.3f} . "
                 )
            print("="*50)
        
        if total_iter % lr_step == (lr_step-1):
            scheduler.step()

        if total_iter % save_interval == (save_interval-1):
    
            # save the model 
            model_res = edict()
            model_res.config = config
            model_res.loss_fns = [loss_fn1, loss_fn2]
            model_res.loss_save = loss_save
            model_res.train_params = train_params
            model_res.data_params = data_params
    
            if SAVED_MODEL is None:
                cur_model_name = f"{MODEL_NAME}_epoch{iep+1}_iter{ix+1}"
            else:
                cur_model_name = f"{MODEL_NAME}_epoch{iep+1}_iter{ix+1}_w_{SAVED_MODEL}"

            save_pkl_dict2folder(RES_ROOT/cur_model_name, model_res, is_force=True)
            # save model     
            torch.save(net.state_dict(), RES_ROOT/cur_model_name/"model.pth")
            torch.save(optimizer.state_dict(), RES_ROOT/cur_model_name/"optimizer.pth")
            torch.save(scheduler.state_dict(), RES_ROOT/cur_model_name/"scheduler.pth")
            # copy class file 
            shutil.copy(MODEL_ROOT/MODEL_CLASS, RES_ROOT/cur_model_name/"model_class.py")

        total_iter += 1


# In[ ]:




