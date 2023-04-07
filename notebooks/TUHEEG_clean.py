#!/usr/bin/env python
# coding: utf-8

# This file is to clearn the TUH EEG seizure data
# 
# 1. Create a csv file to contains all edf and the labels
# 
# 2. A class to load data
# 

# In[1]:


import sys
sys.path.append("../mypkg")
from constants import RES_ROOT, FIG_ROOT, DATA_ROOT


# In[7]:


import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from easydict import EasyDict as edict
import torch
import mne

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use(FIG_ROOT/"base.mplstyle")


# In[3]:


from data_utils.eeg_load import txt2labinfo, len2_numsps, montage_txt_parse, EEG_data
from utils.misc import load_txt


# In[ ]:





# In[4]:


root = list(DATA_ROOT.glob("EEG_seizure"))[0]
root


# In[ ]:





# # Pre-analysis

# In[23]:


# all session paths
# count number of folder in each session
nums = []
for p in list(root.rglob("s0*")):
    num = len(list(p.glob("*")))
    nums.append(num)
    if num > 1:
        print(p)
    
nums = np.array(nums)
print((nums==2).sum())
print(np.max(nums))


# # Output csv file
# 
# Csv file lists all files

# In[5]:


names =["dataset", "sub", "session", "montage", "file_stem", "relative_path", "is_seizure", "total_dur"]
working_set = "dev" # train, eval, dev 
working_path = root/("edf/"+working_set)


# In[7]:


def path2info(p):
    p_dir = "/".join(str(p).split("/")[-5:]).split('.')[0]
    txt = load_txt(p.with_suffix(".csv_bi"))
    infos = txt2labinfo(txt)
    return str(p.parent).split("/")[-4:] + [p.stem] + [p_dir] + [infos["is_seizure"]] + [infos["total_dur"]]


# In[ ]:


all_infos = []
for p in tqdm(working_path.rglob("*.edf")):
    all_infos.append(path2info(p))


# In[21]:


df = pd.DataFrame(all_infos, columns=names)
df.to_csv(root/f"all_data_{working_set}.csv")


# In[ ]:





# # Load the data

# In[133]:


import time
from utils.misc import delta_time
all_data = pd.read_csv(root/"all_data_train.csv")
class EEG_data(Dataset):
    """The main data class
    """
    def __init__(self, dataset, subset, 
                 move_dict=dict(winsize=256, stepsize=64, paddingsize=None),
                 preprocess_dict=dict(is_detrend=True, 
                                      is_drop=True,
                                      target_fs=100, 
                                      filter_limit=[2, 45], 
                                      is_diff=True), 
                 root=None):
        """
        Initializes a new instance of the EEG data class.
        The EEG data with sampling freq 250

        Args:
            dataset: The data set we want to load, train, dev or eval
            subset: the subset to choose: LE, AR, AR_A, ALL
            move_dict: the moving parameters
            root: The root of dataset,
        """
        if root is None:
            root = list(DATA_ROOT.glob("EEG_seizure"))[0]
        self.root = root
        self.move_dict = edict(move_dict)
        self.preprocess_dict = edict(preprocess_dict)
        if self.preprocess_dict.target_fs is None:
            self.preprocess_dict.target_fs = 250
        
        if subset.lower().startswith("all"):
            montage_p = list((root/"DOCS").glob(f"*ar_montage*"))[0]
        else:
            montage_p = list((root/"DOCS").glob(f"*{subset.lower()}_montage*"))[0]
        montage_txt = load_txt(montage_p)
        self.montage = montage_txt_parse(montage_txt)
            

        all_data = pd.read_csv(self.root/f"all_data_{dataset}.csv")
        if subset.lower().endswith("all"):
            self.all_data = all_data
        else:
            self.all_data = all_data[all_data["montage"].str.endswith(subset.lower())]
        self.num_sps_persub = np.array(len2_numsps(
                                          np.array(self.all_data["total_dur"])*self.preprocess_dict.target_fs, 
                                          self.move_dict.winsize, 
                                          self.move_dict.stepsize, 
                                          self.move_dict.paddingsize), dtype=int)
            
    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return int(self.num_sps_persub.sum())

    def __getitem__(self, idx):
        """
        Gets the item at the specified index.

        Args:
            idx (int or str): The index of the item to get.

        Returns:
            tuple: A tuple containing the input data and target data.
        """
        if isinstance(idx, int):
            if idx < 0:
                idx = self.__len__() + idx
            num_cumsum = np.cumsum(self.num_sps_persub)
            sub_idx = np.sum(num_cumsum < (idx+1))
            if sub_idx != 0:
                loc_idx = idx - num_cumsum[sub_idx-1]
            else:
                loc_idx = idx
                
            _, low_idxs = len2_numsps(np.array(self.all_data["total_dur"])[sub_idx] * self.preprocess_dict.target_fs, 
                                                      self.move_dict.winsize, 
                                                      self.move_dict.stepsize, 
                                                      self.move_dict.paddingsize, True)
            loc_idx_low, loc_idx_up = int(low_idxs[loc_idx]), int(low_idxs[loc_idx] + self.move_dict.stepsize)
            if sub_idx > (self.__len__()-1):
                raise IndexError
        elif isinstance(idx, str) and idx.lower().startswith("sub"):
            sub_idx = int(idx.split("sub")[-1])
            if sub_idx < 0:
                sub_idx = len(self.all_data) + sub_idx
            if sub_idx > (self.__len__()-1):
                raise IndexError
            loc_idx_low = loc_idx_up = None
        else:
            raise NotImplementedError
            
        #print(idx)
        t0 = time.time()
        relative_path = self.all_data["relative_path"].iloc[sub_idx]
        data_path = (self.root/"edf"/relative_path).with_suffix(".edf")
        data = mne.io.read_raw_edf(data_path, preload=True, verbose=False)
        print(delta_time(t0))
        
        
        if self.preprocess_dict.is_drop:
            # only keep some channels
            pick_names = np.unique(np.concatenate([i[1:] for i in self.montage]))
            data.pick_channels(pick_names)
        print(delta_time(t0))
        
        # to make sure the order is the same
        data.reorder_channels(np.sort(data.ch_names))
        print(delta_time(t0))
        
        if self.preprocess_dict.is_detrend:
            data.apply_function(mne.epochs.detrend, verbose=False)
        #print(delta_time(t0))
        if self.preprocess_dict.target_fs < 250:
            data.resample(self.preprocess_dict.target_fs, verbose=False)
        #print(delta_time(t0))
        if self.preprocess_dict.filter_limit is not None:
            data.filter(
                self.preprocess_dict.filter_limit[0], 
                self.preprocess_dict.filter_limit[1], 
                verbose=False);
        #print(delta_time(t0))
            
        if self.preprocess_dict.is_diff:
            # to be down
            # it is the very bad way to do it, below
            #final_data = []
            #for chs in self.montage:
            #    final_data.append(data[chs[1]][0] - data[chs[2]][0])
            #final_data = np.array(final_data).squeeze()
            pass
        final_data = data.get_data()
        
        
        if loc_idx_low is not None:
            final_data = final_data[:, loc_idx_low:loc_idx_up]
            
            if final_data.shape[1] < self.move_dict.winsize:
                padding = np.zeros((final_data.shape[0], self.move_dict.winsize-final_data.shape[1]))
                final_data = np.concatenate([final_data, padding], axis=1)
        #final_data = torch.tensor(final_data)
        return final_data, delta_time(t0)


# In[134]:


dataset = EEG_data("train", "AR")
sel_idxs = torch.randint(len(dataset), (10, ))


# In[122]:



xys = [dataset[int(sel_idx)] for sel_idx in sel_idxs]


# In[139]:


dataset[1284004]


# In[126]:


for sel_idx in sel_idxs:
    tt =  int(sel_idx)
    print(111, dataset[tt][1])
    #print(tt)


# In[38]:





# In[ ]:



train_data_loader = DataLoader(dataset, batch_size=10, shuffle=True)
for ix in train_data_loader:
    print(ix.shape)


# In[ ]:





# In[ ]:




