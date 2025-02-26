#!/usr/bin/env python
# coding: utf-8

# This notebook is to train the model to reconstruct the input
# 

# In[1]:


MODEL_NAME_PREFIX = "EEG_sz_rec_init"
SAVED_MODEL = None
MODEL_CLASS = "my_net_selflearning.py"


# In[2]:


import sys
from jin_utils import get_mypkg_path
pkgpath = get_mypkg_path()
sys.path.append(pkgpath)
from constants import RES_ROOT, MODEL_ROOT


# In[3]:


import numpy as np
from easydict import EasyDict as edict
import time
# copy file
import shutil
from pprint import pprint
from collections import defaultdict as ddict


import logging
import datetime

current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'training_{current_time}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # 同时输出到控制台
    ]
)


# In[4]:




# In[5]:


from models.model_utils import generate_position_encode, eval_model_selflearning
from models.my_net_selflearning import myNet
from data_utils.eeg_load import EEGData
from data_utils.utils import MyDataLoader
from jin_utils import  load_pkl_folder2dict, save_pkl_dict2folder
from utils.misc import delta_time


# In[6]:


import torch
import torch.nn as nn
from torch.functional import F
from torch.optim.lr_scheduler import ExponentialLR


df_dtype = torch.float32

torch.set_default_dtype(df_dtype)
if torch.cuda.is_available():
    df_device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    df_device = torch.device("cpu")
torch.set_default_device(df_device)


# # training

# In[7]:


if SAVED_MODEL is None:
    config = edict()
    nroi = 19
    # the dim of features at each time point, not used in this net
    config.nfeature = nroi 
    config.ndim0 = 128 # the number of queries for the first FC layer
    config.ndim = 256 # the output of the first FC layer
    config.dropout = 0.5 # the dropout rate
    config.n_layer = 2 # the number of self-attention layers
    config.n_head = 8 # numher of heads for multi-head attention
    config.is_mask = True # Use mask to make the attention causal
    config.is_bias = True # Bias  for layernorm
    config.block_size = 256 # the preset length of seq, 
    config.fs = 90
    config.target_dim = nroi # the target dim 

    train_params = edict()
    train_params.nepoch= 2
    train_params.loss_out = 10
    train_params.val_loss_out = 100
    train_params.clip = 1 # 
    # lr step decay, if lr_step is 0, then no decay
    # if '1epoch', then decay every epoch
    train_params.lr_step = '1epoch'
    train_params.lr = 1e-4 
    train_params.lr_gamma = 0.1
    train_params.lr_weight_decay = 0
    # save the model 
    # if '1epoch', then save every epoch
    train_params.save_interval = 10000

    train_params.ntrain_batch = 0 # the number of batches for training
    train_params.train_batch_size = 64 # the batch size for training
    train_params.val_batch_size = 64 # the batch size for validation
    train_params.train_size = 4
    train_params.val_size = 4
    train_params.seed = 0 # random seed

    # data parameters
    data_params = edict()
    data_params.move_params=dict(winsize=config.block_size,
                     stepsize=config.block_size,
                     marginsize=None)
    data_params.pre_params=dict(is_detrend=True, 
                    is_drop=True,
                    target_fs=90, 
                    filter_limit=[1, 45], 
                    is_diff=False)
    data_params.rm_params=dict(rm_len=50, keep_len=20)
    data_params.subset = "AR"
else:
    saved_model_path = RES_ROOT/SAVED_MODEL
    assert saved_model_path.exists(), "No such model"
    saved_model = load_pkl_folder2dict(saved_model_path)
    
    config = saved_model.config
    train_params = saved_model.train_params
    data_params = saved_model.data_params


# In[8]:


move_params = data_params.move_params
pre_params = data_params.pre_params
rm_params = data_params.rm_params
subset = data_params.subset
train_data = EEGData(dataset="train", 
                     subset=subset,
                     move_params=move_params,
                     pre_params=pre_params,
                     )
val_data = EEGData(dataset="eval", 
                     subset=subset,
                     move_params=move_params,
                     pre_params=pre_params,
                     )
train_data_loader = MyDataLoader(train_data, batch_size=train_params.train_batch_size, shuffle=True)
val_data_loader = MyDataLoader(val_data, batch_size=train_params.val_batch_size, shuffle=False)
logging.info(f"The number of training samples is {len(train_data_loader)} and the number of validation samples is {len(val_data_loader)}.")


# In[9]:


pos_enc = generate_position_encode(config.block_size, config.nfeature).unsqueeze(0)
loss_fn = nn.MSELoss()

if SAVED_MODEL is None:
    net = myNet(config)
else:
    net = saved_model.model.to(df_device)
optimizer = torch.optim.Adam(net.parameters(), 
                             lr=train_params.lr,
                             weight_decay=train_params.lr_weight_decay)
scheduler = ExponentialLR(optimizer, 
                          gamma=train_params.lr_gamma);


# In[10]:


def _save_model():
    model_res = edict()
    model_res.config = config
    model_res.loss_fns = loss_fn
    model_res.loss_save = loss_save
    model_res.train_params = train_params
    model_res.data_params = data_params
    
    if SAVED_MODEL is None:
        cur_model_name = f"{MODEL_NAME_PREFIX}_epoch{iep+1}_iter{ix+1}"
    else:
        cur_model_name = f"{MODEL_NAME_PREFIX}_epoch{iep+1}_iter{ix+1}_w_{SAVED_MODEL}"

    save_pkl_dict2folder(RES_ROOT/cur_model_name, model_res, is_force=True)
    # save model     
    torch.save(net.state_dict(), RES_ROOT/cur_model_name/"model.pth")
    torch.save(optimizer.state_dict(), RES_ROOT/cur_model_name/"optimizer.pth")
    torch.save(scheduler.state_dict(), RES_ROOT/cur_model_name/"scheduler.pth")
    # copy class file 
    shutil.copy(MODEL_ROOT/MODEL_CLASS, RES_ROOT/cur_model_name/"model_class.py")


# In[11]:


logging.info("Start training")
logging.info(train_params)
logging.info(data_params)


# In[ ]:


ntrain_batch = len(train_data_loader)
# training
if SAVED_MODEL is None:
    loss_save = {}
    loss_save["train"] = ddict(list)
    loss_save["val"] = ddict(list)
else:
    loss_save = saved_model.loss_save

if isinstance(train_params.lr_step, str):
    lr_step = int(ntrain_batch * float(train_params.lr_step[:-5]))
else:
    lr_step = train_params.lr_step
if isinstance(train_params.save_interval, str):
    save_interval = int(ntrain_batch * float(train_params.save_interval[:-5]))
else:
    save_interval = train_params.save_interval


t0 = time.time()
total_iter = 0
for iep in range(train_params.nepoch):
    logging.info(f"The current lr is {scheduler.get_last_lr()}.")
    for ix in range(ntrain_batch):
        net.train()
        batch = train_data_loader(ix)
        batch = batch.to(df_dtype)

        batch_wpos = batch + pos_enc
        # Zero the gradients
        optimizer.zero_grad()
        
        rec_batch = net(batch_wpos)
        loss = loss_fn(rec_batch, batch_wpos)
        loss_save["train"]["loss"].append(loss.item())
        loss_save["train"]["niter"].append(total_iter)
        
        #print("training:", net.training)
        # Perform backward pass
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(net.parameters(), train_params.clip)
        # Perform optimization
        optimizer.step()
        
        if ix % train_params.loss_out == (train_params.loss_out-1):
            tr_loss = np.mean(loss_save["train"]["loss"][-20:])
            loss_msg = f"At iter {total_iter}, the train loss is {tr_loss:.3f}, the time used is {delta_time(t0):.3f}s."
            logging.info(loss_msg)
            t0 = time.time()
            net.train()
            
        if ix % train_params.val_loss_out == (train_params.val_loss_out-1):
            res = eval_model_selflearning(trained_model=net,
                                          data_loader=val_data_loader,
                                          loss_fn=loss_fn,
                                          df_dtype=df_dtype,
                                          n_batch=20,
                                          random=True, 
                                          verbose=True)
            val_loss = np.mean(res) 
            loss_save["val"]["loss"].append(val_loss)
            loss_save["val"]["niter"].append(total_iter)
            loss_msg = f"At iter {total_iter}, the val loss is {val_loss:.3f}, the time used is {delta_time(t0):.3f}s."
            logging.info("="*50)
            logging.info(loss_msg)
            logging.info("="*50)
            t0 = time.time()
            net.train()
        
        if total_iter % lr_step == (lr_step-1):
            scheduler.step()

        if total_iter % save_interval == (save_interval-1):
            _save_model()
            t0 = time.time()

    
            # save the model 
        total_iter += 1
        #print("training:", net.training)
    _save_model()




