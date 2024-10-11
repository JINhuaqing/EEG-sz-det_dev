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


MODEL_NAME = "RAWINPUT-MULTI-LONGEP"
SAVED_MODEL = None
MODEL_CLASS = "my_net_multi.py"


# # Load pkgs 

# In[2]:


import sys
sys.path.append("../mypkg")
from constants import RES_ROOT, MODEL_ROOT


# In[3]:


import numpy as np
from easydict import EasyDict as edict
from collections import defaultdict as ddict
import time
# copy file
import shutil
from pprint import pprint



# In[5]:


from models.my_net_multi import myNet
from models.losses import  ordinal_mse_loss
from models.model_utils import generate_position_encode, trans_batch_multi, eval_model_multi 
from data_utils.eeg_load_sz import EEGDataSZ
from data_utils import MyDataLoader
from utils.misc import delta_time, load_pkl_folder2dict, save_pkl_dict2folder


# In[6]:


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
print(device)


import argparse

argparser = argparse.ArgumentParser()

argparser.add_argument("--lr", type=float, default=1e-4)
argparser.add_argument("--lr_gamma", type=float, default=0.1)
argparser.add_argument("--aux_loss", action="store_true") # if not specified, then False
argparser.add_argument("--aux_loss_weight", type=float, default=1)
argparser.add_argument("--ntrain_batch", type=int, default=0)
argparser.add_argument("--move_steps", type=int, nargs="+", default=[1, 5, 10])
argparser.add_argument("--nepoch", type=int, default=0)

args = argparser.parse_args()
# In[ ]:


if args.aux_loss:
    move_steps_str = "-".join([str(i) for i in args.move_steps])
    name_update = f"_aux_loss{args.aux_loss}_lr{args.lr*1000000:.0f}_lr_gamma{args.lr_gamma*100:.0f}_ntrain_batch{args.ntrain_batch}_aux_loss_weight{args.aux_loss_weight*100:.0f}_move_steps{move_steps_str}"
else:
    name_update = f"_aux_loss{args.aux_loss}_lr{args.lr*1000000:.0f}_lr_gamma{args.lr_gamma*100:.0f}_ntrain_batch{args.ntrain_batch}"

MODEL_NAME = MODEL_NAME + name_update
if SAVED_MODEL is not None:
    SAVED_MODEL = SAVED_MODEL + name_update



# # training

# ## Model and training params

# In[7]:


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
    # if [1, 5], use X_t to predict X_{t+1}, X_{t+5}
    config.move_steps = args.move_steps # move steps
    config.fs = 90
    config.target_dim = 19 # TODO: the target dim can be deprecated in the future 
    config.k = 8 # discretize to 2^k levels
    config.ncls = 2 # number of classes, 2 for my seizure data
    # while include auxiliary loss or not 
    # the weight of the auxiliary loss is config.aux_loss_weight
    config.aux_loss = args.aux_loss
    config.aux_loss_weight = args.aux_loss_weight
    
    train_params = edict()
    train_params.nepoch= args.nepoch
    train_params.loss_out = 20
    train_params.val_loss_out = 50
    train_params.clip = 1 # 
    # lr step decay, if lr_step is 0, then no decay
    # if '1epoch', then decay every epoch
    train_params.lr_step = f'{int(args.nepoch/2):.0f}epoch'
    train_params.lr = args.lr
    train_params.lr_gamma = args.lr_gamma
    train_params.lr_weight_decay = 0
    # save the model 
    # if '1epoch', then save every epoch
    train_params.save_interval = '1epoch'
    # if 0, use all the training sz
    train_params.ntrain_batch = args.ntrain_batch # the number of batches for training
    train_params.train_batch_size = 8 # the batch size for training
    train_params.val_batch_size = 8 # the batch size for validation
    train_params.test_batch_size = 8 # the batch size for test
    train_params.train_01_ratio = 1 # the ratio of 0 and 1 in the training set
    train_params.val_01_ratio = 2 # the ratio of 0 and 1 in the validation set
    train_params.test_01_ratio = 2 # the ratio of 0 and 1 in the test set
    train_params.train_size = 24
    train_params.val_size = 24
    train_params.test_size = 0 # if 0, use all the test set
    train_params.seed = 0 # random seed


    # data parameters
    data_params = edict()
    data_params.move_params=dict(winsize=config.block_size+np.max(config.move_steps), 
                     stepsize=config.block_size+np.max(config.move_steps), 
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

# In[8]:


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

val_data_bckg = EEGDataSZ(
    dataset="dev", 
    subset=subset,
    label="bckg", 
    discrete_k=config.k, 
    verbose=verbose, 
    move_params=move_params,
    pre_params=pre_params,
    rm_params=rm_params
    )

val_data_sz = EEGDataSZ(
    dataset="dev", 
    subset=subset,
    label="sz", 
    discrete_k=config.k, 
    verbose=verbose, 
    move_params=move_params,
    pre_params=pre_params,
    rm_params=rm_params
    )

test_data_bckg = EEGDataSZ(
    dataset="eval", 
    subset=subset,
    label="bckg", 
    discrete_k=config.k, 
    verbose=verbose, 
    move_params=move_params,
    pre_params=pre_params,
    rm_params=rm_params
    )

test_data_sz = EEGDataSZ(
    dataset="eval", 
    subset=subset,
    label="sz", 
    discrete_k=config.k, 
    verbose=verbose, 
    move_params=move_params,
    pre_params=pre_params,
    rm_params=rm_params
    )


train_data_bckg_loader = MyDataLoader(train_data_bckg, 
                                      batch_size=train_params.train_batch_size*train_params.train_01_ratio, 
                                      shuffle=True,
                                      seed=train_params.seed)
train_data_sz_loader = MyDataLoader(train_data_sz, 
                                    batch_size=train_params.train_batch_size,
                                    shuffle=True, 
                                    seed=train_params.seed)
val_data_bckg_loader = MyDataLoader(val_data_bckg, 
                                      batch_size=train_params.val_batch_size*train_params.val_01_ratio,
                                      shuffle=False, 
                                      seed=train_params.seed)
val_data_sz_loader = MyDataLoader(val_data_sz, 
                                    batch_size=train_params.val_batch_size,
                                    shuffle=False, 
                                    seed=train_params.seed)

test_data_bckg_loader = MyDataLoader(test_data_bckg, 
                                      batch_size=train_params.test_batch_size*train_params.test_01_ratio,
                                      shuffle=False, 
                                      seed=train_params.seed)
test_data_sz_loader = MyDataLoader(test_data_sz, 
                                    batch_size=train_params.test_batch_size,
                                    shuffle=False, 
                                    seed=train_params.seed)
                                
print(f"Num of data: train_bckg: {len(train_data_bckg)}, train_sz: {len(train_data_sz)}", 
      f"val_bckg: {len(val_data_bckg)}, val_sz: {len(val_data_sz)}", 
      f"test_bckg: {len(test_data_bckg)}, test_sz: {len(test_data_sz)}")


# ## Prepare training

# In[9]:


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


# ## training the model

# In[10]:


print("Start training")
pprint(train_params)
pprint(data_params)
pprint(config)


# In[11]:


def _save_model():
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


# In[12]:


# training
if SAVED_MODEL is None:
    loss_save = {}
    loss_save["train"] = ddict(list)
    loss_save["val"] = ddict(list)
    loss_save["test"] = ddict(list)
else:
    loss_save = saved_model.loss_save

if train_params.ntrain_batch == 0:
    ntrain_batch = len(train_data_sz_loader)
else: 
    ntrain_batch = train_params.ntrain_batch
if ntrain_batch > len(train_data_sz_loader):
    ntrain_batch = len(train_data_sz_loader)
    print(f"The number of training batches is larger than the number of training data, use all the training data to train, i.e., ntrain_batch={ntrain_batch}.")
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
    net.cuda()
    print(f"The current lr is {scheduler.get_last_lr()}.")
    for ix in range(ntrain_batch):
        net.train()
        batch_sz = train_data_sz_loader(ix)
        batch_bckg = train_data_bckg_loader(ix)
        trans_res  = trans_batch_multi(batch_sz=batch_sz, batch_bckg=batch_bckg, 
                                              config=config,
                                              shuffle=True)
        X_org = trans_res[0]
        Y_diss = trans_res[1:-1]
        szlabels = trans_res[-1]
        X_org_wpos = X_org + pos_enc
        # Zero the gradients
        optimizer.zero_grad()
        
        if config.aux_loss:
            probss_aux, log_probs_cls = net(X_org_wpos)
            loss1 = 0
            for Y_dis, probs_aux in zip(Y_diss, probss_aux):
                loss1p = loss_fn1(probs_aux, Y_dis, num_cls=2**config.k)
                loss1 += loss1p
            loss1 = loss1/len(Y_diss)
            loss2 = loss_fn2(log_probs_cls, szlabels)
            loss = config.aux_loss_weight*loss1 + loss2
        
            # record the loss
            loss_save["train"]["aux_loss"].append(loss1.item())
            loss_save["train"]["cls_loss"].append(loss2.item())
            loss_save["train"]["loss"].append(loss.item())
        else:
            log_probs_cls = net(X_org_wpos)
            loss = loss_fn2(log_probs_cls, szlabels)
            loss_save["train"]["cls_loss"].append(loss.item())
            loss_save["train"]["loss"].append(loss.item())
        loss_save["train"]["niter"].append(total_iter)
        
        assert net.training, "The model is not in training mode."
        # Perform backward pass
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(net.parameters(), train_params.clip)
        # Perform optimization
        optimizer.step()
        
        if ix % train_params.loss_out == (train_params.loss_out-1):
            loss_save["train"]["niter_auc"].append(total_iter)
            curlosses = eval_model_multi(net, 
                                   data_loader_sz=train_data_sz_loader, 
                                   data_loader_bckg=train_data_bckg_loader,
                                   n_batch=train_params.train_size, random=True)
                            
            for ky in curlosses.keys():
                loss_save["train"][ky].append(curlosses[ky])
            loss_msg = f"At iter {ix+1}/{ntrain_batch}, epoch {iep+1}," 
            for k, v in loss_save["train"].items():
                if k in ["probs_sz", "labs_sz", "niter", "niter_auc"]:
                    continue
                elif k in ["auc"]:
                    loss_msg += f"{k}: {v[-1]:.3f}, "
                else:
                    loss_msg += f"{k}: {np.mean(v[-train_params.loss_out:]):.3f}, "
            loss_msg += f"the time used is {delta_time(t0):.3f}s."
            print(loss_msg)
            t0 = time.time()
            
        if ix % train_params.val_loss_out == (train_params.val_loss_out-1):
            loss_save["val"]["niter"].append(total_iter)
            curlosses = eval_model_multi(net, 
                                   data_loader_sz=val_data_sz_loader, 
                                   data_loader_bckg=val_data_bckg_loader, 
                                   cls_loss_fn=loss_fn2,
                                   aux_loss_fn=loss_fn1 if config.aux_loss else None,
                                   n_batch=train_params.val_size, random=True)
            for ky in curlosses.keys():
                loss_save["val"][ky].append(curlosses[ky])
            print("="*50)
            loss_msg = f"At iter {ix+1}/{ntrain_batch}, epoch {iep+1}," 
            for k, v in loss_save["val"].items():
                if k in ["probs_sz", "labs_sz", "niter"]:
                    continue
                loss_msg += f"{k}: {np.mean(v[-1]):.3f}. "
            print("Val results: " + loss_msg)
            print("="*50)
            t0 = time.time()
        
        if total_iter % lr_step == (lr_step-1):
            scheduler.step()

        if total_iter % save_interval == (save_interval-1):
            # when saving model, test the model on the test data
            loss_save["test"]["niter"].append(total_iter)

            curlosses = eval_model_multi(net, 
                                   data_loader_sz=test_data_sz_loader, 
                                   data_loader_bckg=test_data_bckg_loader, 
                                   cls_loss_fn=loss_fn2,
                                   aux_loss_fn=loss_fn1 if config.aux_loss else None,
                                   n_batch=train_params.test_size, 
                                   random=False)
            for ky in curlosses.keys():
                loss_save["test"][ky].append(curlosses[ky])
            print("*"*50)
            loss_msg = f"At iter {ix+1}/{ntrain_batch}, epoch {iep+1}," 
            for k, v in loss_save["test"].items():
                if k in ["probs_sz", "labs_sz", "niter"]:
                    continue
                loss_msg += f"{k}: {np.mean(v[-1]):.3f}. "
            print("Test results: " + loss_msg)
            print("*"*50)

            _save_model()
            t0 = time.time()

    
            # save the model 
        total_iter += 1
    _save_model()


# In[ ]:




