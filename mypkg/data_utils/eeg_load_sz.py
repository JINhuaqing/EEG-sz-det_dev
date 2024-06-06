from .eeg_load import EEGData
import numpy as np
from constants import DATA_ROOT
import pandas as pd
from easydict import EasyDict as edict
import torch
import mne
import time
import pdb

class EEGdataSZ(EEGData):
    """The main data class
    """
    def __init__(self, dataset, subset, 
                 label, 
                 move_param={},
                 preprocess_dict=dict(is_detrend=True, 
                                      is_drop=True,
                                      target_fs=125, 
                                      filter_limit=[1, 45], 
                                      is_diff=False), 
                 root=None, scale_fct=None):
        """
        Initializes a new instance of the EEG data class.
        The EEG data with sampling freq 250

        Args:
            dataset: The data set we want to load, 
                     train, dev or eval
            subset: the subset to choose: LE, AR, AR_A, ALL
            label: the label of the data, sz, bckg, bckg-str
            move_dict: the moving parameters
            root: The root of dataset,
            scale_fct: EEG = EEG/scale_fct
        """
        label = label.lower()
        
        # set default values of move_dict and preprocess_dict
        move_param_def = edict(dict(winsize=256, stepsize=64, marginsize=None))
        self.move_dict=edict(dict(winsize=256, stepsize=64, marginsize=None))
        self.preprocess_dict=edict(dict(is_detrend=True, 
                                        is_drop=True,
                                        target_fs=125, 
                                        filter_limit=[1, 45], 
                                        is_diff=False))
        
        self.rm_len = 10
        self.keep_len = 20
        self.outliers_rate = 0.05
        # hook is only for get the variable
        self.hook = edict()
        self.hook.sub_idxs = []
        
        if root is None:
            root = list(DATA_ROOT.glob("EEG_seizure"))[0]/"edf"
        self.root = root
        self.scale_fct = scale_fct
        self.move_dict.update(move_dict)
        self.preprocess_dict.update(preprocess_dict)
        self.edf_data = None
        if self.preprocess_dict.target_fs is None:
            self.preprocess_dict.target_fs = 250
        
        all_data = pd.read_csv(self.root/f"all_data_{dataset}.csv")
        if subset.lower().endswith("all"):
            all_data = all_data
        else:
            all_data = all_data[all_data["montage"].str.endswith(subset.lower())]
        
        # remove the data not including all channels in SEL_CHS
        all_data = all_data[all_data.apply(sel_fn, axis=1)]

        # remove rows based on label
        if label.startswith("sz"):
            all_data = all_data[all_data["is_seizure"]]
            
        # remove the first and last 10 seconds and remove the subject with too-short seq
        all_data = all_data.copy()
        all_data.loc[:, "total_dur"] = all_data.loc[:, "total_dur"] - 2* self.rm_len
        all_data = all_data[all_data["total_dur"] > self.keep_len]
        all_data = all_data.reset_index()


        self.all_data = all_data

        # now count data seg in total
        if label == "sz":
            self.sz_margin = (self.move_dict.winsize/self.preprocess_dict.target_fs) * 0.5
            def _tmp_fn(x):
                rv = [(vs[0]-self.rm_len-self.sz_margin, vs[1]-self.rm_len+self.sz_margin) for 
                     vs in ast.literal_eval(x) 
                     if (vs[2] == "seiz") and ((vs[1]-vs[0])>self.keep_len) and (vs[0]>self.rm_len)]
                return rv
            self.eff_segs = self.all_data["lab"].map(_tmp_fn)
            num_sps_persub = []
            for eff_seg in self.eff_segs:
                len2num(total_dur*self.preprocess_dict.target_fs, 
                                  self.move_dict.winsize, 
                                  self.move_dict.stepsize, 
                                  self.move_dict.marginsize)[0]
                
            
            


    

        
            
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
                
            if sub_idx > (self.__len__()-1):
                raise IndexError
                
        elif isinstance(idx, str) and idx.lower().startswith("sub"):
            sub_idx = int(idx.split("sub")[-1])
            if sub_idx < 0:
                sub_idx = len(self.all_data) + sub_idx
            if sub_idx > (self.__len__()-1):
                raise IndexError
            loc_idx = None
            
        else:
            raise NotImplementedError
            
        
        self.hook.sub_idxs.append(sub_idx)
        data = self.get_preprocess_data(sub_idx)
        # remove the first and last pts
        data = data[:, int(self.preprocess_dict.target_fs*self.rm_len):-int(self.preprocess_dict.target_fs*self.rm_len)]
        data = robust_EEG_rescale(data, data_max=self.scale_fct)
        
        if loc_idx is not None:
            _, low_idxs = len2num(np.array(self.all_data["total_dur"])[sub_idx] * self.preprocess_dict.target_fs, 
                                                      self.move_dict.winsize, 
                                                      self.move_dict.stepsize, 
                                                      self.move_dict.marginsize)
            loc_idx = self.rm_outlier_seg(data, low_idxs, loc_idx)
            loc_idx_low, loc_idx_up = int(low_idxs[loc_idx]), int(low_idxs[loc_idx] + self.move_dict.winsize)
            data = data[:, loc_idx_low:loc_idx_up]
            
            if data.shape[1] < self.move_dict.winsize:
                padding = np.zeros((data.shape[0], self.move_dict.winsize-data.shape[1]))
                data = np.concatenate([data, padding], axis=1)
        #data = torch.tensor(data)
        return data.transpose(1, 0)
    
