#!/usr/bin/env python
# coding: utf-8

# This file is to test my model

# In[1]:


run_python_script = True


# In[2]:


import sys
sys.path.append("../mypkg")
from constants import RES_ROOT, FIG_ROOT, DATA_ROOT


# In[3]:


import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from easydict import EasyDict as edict
import time

if not run_python_script:
    plt.style.use(FIG_ROOT/"base.mplstyle")
    get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


import importlib
import models.main_model
importlib.reload(models.main_model)


# In[5]:


from models.main_model import myNet
from utils.misc import delta_time, load_pkl_folder2dict, save_pkl_dict2folder


# In[6]:


# pkgs for pytorch (on Apr 3, 2023)
import torch
import torch.nn as nn
from torch.functional import F
from torch.optim.lr_scheduler import ExponentialLR

torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    torch.cuda.set_device(2)
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
else:
    torch.set_default_tensor_type(torch.DoubleTensor)
    device = torch.device("cpu")


# # Load data

# In[7]:

print("Prepare the data")

AD_dir = DATA_ROOT/"AD_data"
ctrl_p1 = list(AD_dir.glob("70*.mat"))[0]
ctrl_p2 = list(AD_dir.glob("time*.mat"))[0]


# In[8]:


from scipy.io import loadmat
import mat73

data_p1 = loadmat(ctrl_p1)["dk10"]
data_p2 = mat73.loadmat(ctrl_p2)["dk10"]
data = np.concatenate([data_p1,  data_p2], axis=0)


# In[9]:

#
from scipy.signal import detrend
## detrend the data
if not run_python_script:
    data = detrend(data)


# In[10]:


test_data = data[-22:].transpose(0, 2, 1)
train_data = data[:-22].transpose(0, 2, 1)


# In[ ]:





# # training

# In[11]:

print("Prepare the some fns")

def get_batch(dataset, batch_size, block_size):
    sub_ix = torch.randint(len(dataset), (1, ))
    ix = torch.randint(len(dataset[sub_ix]) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((dataset[sub_ix, i:i+block_size])) for i in ix])
    y = torch.stack([torch.from_numpy((dataset[sub_ix, i+1:i+1+block_size])) for i in ix])
    return x.double(), y.double()


# In[12]:


def generate_position_encode(block_size, nfeature):
    # create a matrix with shape (blocksize, nfeature)
    position = torch.arange(block_size, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, nfeature, 2).float() * (-np.log(10000.0) / nfeature))
    # apply sine to even indices in the array
    pos_enc = torch.zeros((block_size, nfeature))
    pos_enc[:, 0::2] = torch.sin(position * div_term)
    # apply cosine to odd indices in the array
    pos_enc[:, 1::2] = torch.cos(position * div_term)
    return pos_enc


# In[16]:


def out2pred(net_out, X):
    A_mats = net_out.reshape(config.batch_size, config.block_size, paras.nroi, -1)
    Y_pred = (A_mats @ X.unsqueeze(-1)).squeeze()
    return Y_pred

def evaluate(net):
    X_test, Y_test = get_batch(test_data, 
                     batch_size=config.batch_size,  
                     block_size=config.block_size)
    X_test = X_test + pos_enc
    net.eval()
    net_out = net(X_test)
    Y_pred = out2pred(net_out, X_test)
    loss = loss_fn(Y_test, Y_pred)
    net.train()
    return loss.item()


# In[17]:


print("Prepare the paras")
paras = edict()
paras.nroi = data.shape[1]

config = edict()
config.nfeature = paras.nroi # the dim of features at each time point
config.ndim = 256 # the output of the first FC layer
config.target_dim = paras.nroi * paras.nroi # the target dim 
config.dropout = 0.5 # the dropout rate
config.n_layer = 1 # the number of self-attention layers
config.n_head = 8 # numher of heads for multi-head attention
config.is_mask = True # Use mask to make the attention causal
config.is_bias = True # Bias  for layernorm
config.block_size = 256 # the preset length of seq, 
config.batch_size = 32 # the preset length of seq, 

paras_train = edict()
paras_train.batch_size = 64
paras_train.niter = 500
paras_train.loss_out = 50
paras_train.clip = 1 # 
paras_train.lr_step = 200


# In[18]:

print("The Net")
net = myNet(config)
print("The Net1")
pos_enc = generate_position_encode(config.block_size, config.nfeature).unsqueeze(0)

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)
scheduler = ExponentialLR(optimizer, gamma=0.3, verbose=True)


# In[19]:


# training
loss_cur = 0
losses = []
losses_test = []

print("Start training")
t0 = time.time()
for ix in range(paras_train.niter):
    X, Y = get_batch(train_data, 
                      batch_size=config.batch_size,  
                      block_size=config.block_size)
    X = X + pos_enc
    # Zero the gradients
    optimizer.zero_grad()
    
    net_out = net(X)
    Y_pred = out2pred(net_out, X)
    loss = loss_fn(Y, Y_pred)
    
    # Perform backward pass
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(net.parameters(), paras_train.clip)
    # Perform optimization
    optimizer.step()
    
    loss_cur = loss_cur + loss.item()
    if ix % paras_train.loss_out == (paras_train.loss_out-1):
        losses.append(loss_cur/paras_train.loss_out)
        losses_test.append(evaluate(net))
        print(f"At iter {ix+1}/{paras_train.niter}, "
              f"the losses are {loss_cur/paras_train.loss_out:.5E} (train). "
              f"the losses are {losses_test[-1]:.5E} (test). "
              f"The time used is {delta_time(t0):.3f}s. "
             )
        loss_cur = 0
        t0 = time.time()
    
    #if ix % paras_train.lr_step == (paras_train.lr_step-1):
    #    scheduler.step()


# In[ ]:




