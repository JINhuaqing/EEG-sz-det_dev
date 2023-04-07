from utils.misc import load_txt
import numpy as np
from constants import DATA_ROOT
import pandas as pd
from torch.utils.data import Dataset
from easydict import EasyDict as edict
import torch
import mne
import time


def txt2labinfo(txt):
    """extract the useful info from csv_bi data, 
       including
           1. is_seizure or not
           2. start time
           3. end time
           4. total duration time
           5. type
        data like 
            ['# version = csv_v1.0.0\n',
             '# bname = aaaaatvr_s005_t008\n',
             '# duration = 301.00 secs\n',
             '# montage_file = $NEDC_NFC/lib/nedc_eas_default_montage.txt\n',
             '#\n',
             'channel,start_time,stop_time,label,confidence\n',
             'TERM,0.0000,301.0000,bckg,1.0000\n']
    """
    ress = dict(total_dur = float(txt[2].split("=")[-1].split("secs")[0]))
    labs = txt[6:]
    for ix, lab in enumerate(txt[6:]):
        res = dict(
        st = float(lab.split(",")[1]),
        et = float(lab.split(",")[2]),
        typ = lab.split(",")[3], 
        )
        ress[f"lab{ix+1}"] = res
   
    ress["is_seizure"] = bool(np.sum(['seiz' in i for i in txt]))
    return ress

def len2_numsps(len_seg, winsize, stepsize, paddingsize=None, is_lowidx=False):
    """return the num of samples we have given a seq
        args: len_seg: length of the seq
              winsize: winsize (to train model)
              stepsize: 
              paddingsize: how long we pad with 0
    """
    if paddingsize is None:
        paddingsize = winsize
    if is_lowidx:
        # it will cause error if len_seq is an array
        low_idxs = np.arange(0, len_seg-paddingsize, stepsize)
    numsps = np.floor((len_seg - paddingsize -1)/stepsize) + 1
    if not is_lowidx:
        return numsps
    else:
        return numsps, low_idxs
    
def montage_txt_parse(montage_txt):
    usefuls = [ix for ix in montage_txt if ix.startswith("mon")]
    usefuls = [(int(useful.split("=")[1].split(",")[0]),
        useful.split("=")[1].split(",")[1].split(":")[-1].split("--")[0].strip(),
        useful.split("=")[1].split(",")[1].split(":")[-1].split("--")[1].strip()) 
     for useful in usefuls]
    return usefuls

class EEG_data(Dataset):
    """The main data class
    """
    def __init__(self, dataset, subset, 
                 move_dict=dict(winsize=256, stepsize=64, paddingsize=None),
                 preprocess_dict=dict(is_detrend=True, 
                                      is_drop=True,
                                      target_fs=100, 
                                      filter_limit=[2, 45], 
                                      is_diff=False), 
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
            
        relative_path = self.all_data["relative_path"].iloc[sub_idx]
        data_path = (self.root/"edf"/relative_path).with_suffix(".edf")
        data = mne.io.read_raw_edf(data_path, preload=True, verbose=False)
        
        
        if self.preprocess_dict.is_drop:
            # only keep some channels
            pick_names = np.unique(np.concatenate([i[1:] for i in self.montage]))
            data.pick_channels(pick_names)
        
        # to make sure the order is the same
        data.reorder_channels(np.sort(data.ch_names))
        
        if self.preprocess_dict.is_detrend:
            data.apply_function(mne.epochs.detrend, verbose=False)
        if self.preprocess_dict.target_fs < 250:
            data.resample(self.preprocess_dict.target_fs, verbose=False)
        if self.preprocess_dict.filter_limit is not None:
            data.filter(
                self.preprocess_dict.filter_limit[0], 
                self.preprocess_dict.filter_limit[1], 
                verbose=False);
            
        if self.preprocess_dict.is_diff:
            # to be down
            # below is the very bad way to do it
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
        return final_data
