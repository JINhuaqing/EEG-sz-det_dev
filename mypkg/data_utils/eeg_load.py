from utils.misc import load_txt, save_pkl, load_pkl
import numpy as np
from constants import DATA_ROOT
import pandas as pd
from torch.utils.data import Dataset
from easydict import EasyDict as edict
import torch
import mne
import time

#sel_chs_raw = "Fp1, Fp2, F7, F3, F4, F8, A1, T3, T4, A2, T5, T6, C3, Cz, C4, P3, P4, O1, O2"
# The below is from Fei,  but some data has no Fz and Pz, so I replace them with A1 A2 
# following the montage file (on Apr 10, 2023)
sel_chs_raw = "Fp1, Fp2, F7, F3, Fz, F4, F8, T3, T4, T5, T6, C3, Cz, C4, P3, Pz, P4, O1, O2"
# the selected channels for analysis
SEL_CHS = [ch.strip().upper() for ch in sel_chs_raw.split(",")]
EEG_ROOT = list(DATA_ROOT.glob("EEG_seizure"))[0]

def robust_EEG_rescale(data, data_max=None):
    """rescale data robustly.
    """
    if data_max is None:
        data_max = np.quantile(np.abs(data), [0.95]) # to remove the effet of outliers
    
    data_rescaled = data/data_max
    #data_min = -data_max
    #data_minmax = 2*(data-data_min)/(data_max-data_min) - 1;
    return data_rescaled

def rec_data(data_dis, k, verbose=False, typ="_rescale_health"):
    """ Reconstruct the digitized data for 2^k levels.
        Approximate inverse operator of digitize_data
    """
    fil = EEG_ROOT/f"discrete_cuts/cuts_2power{k}{typ}.pkl";
    assert fil.exists(), "No cutoff values"
    cuts = load_pkl(fil, verbose=verbose);
    cuts_all = np.sort(np.concatenate([-cuts, [0], cuts]));
    filled_vs = (cuts_all[1:] + cuts_all[:-1])/2;
    filled_vs_full = np.concatenate([cuts_all[:1], filled_vs, cuts_all[-1:]]);
    return filled_vs_full[data_dis]

def digitize_data(data, k, verbose=False, typ="_rescale_health"):
    """ Discretize the data into 2^k levels.
    """
    fil = EEG_ROOT/f"discrete_cuts/cuts_2power{k}{typ}.pkl";
    assert fil.exists(), "No cutoff values"
    cuts = load_pkl(fil, verbose=verbose);
    cuts_all = np.sort(np.concatenate([-cuts, [0], cuts]));
    data_discrete = np.digitize(data, cuts_all);
    return data_discrete


def seiz_lab_fn(lab_info_cl):
    """Parse the seizure lab and return (start_time, end_time)
        if no seizure, [(0, 0)]
    """
    lab_info_out = []
    if len(lab_info_cl) == 0:
        lab_info_out.append((0, 0))
    else:
        for lab in lab_info_cl:
            dat = lab.split(",")[1:3]
            dat = [float(la) for la in dat]
            lab_info_out.append(tuple(dat))
    return lab_info_out

def seiz_lab_from_row_fn(obs_row, root=None):
    """A convinient wrapper of `seiz_lab_fn`
    """
    if root is None:
        root = DATA_ROOT/"EEG_seizure"
    fil_path = root/"edf"/obs_row["relative_path"]
    lab_info = load_txt(fil_path.with_suffix(".csv_bi"))
    lab_info_cl =  [lab for lab in lab_info if lab.startswith("TERM") and "seiz" in lab]
    return seiz_lab_fn(lab_info_cl)

def sel_fn(row):
    """ Select the row including all the chs in SEL_CHS
    """
    all_chs = row["all_chs"].split(";")
    ex_chs = np.setdiff1d(SEL_CHS, all_chs)
    return len(ex_chs) == 0


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

def len2num(len_seq, winsize, stepsize, marginsize=None):
    """
    Compute the number of segments and their starting indices for a given sequence length, window size, step size, and margin size.

    Args:
    - len_seq (int): length of the sequence
    - winsize (int): size of the window
    - stepsize (int): size of the step
    - marginsize (int): size of the margin, only keep segments with len > marginsize

    Returns:
    - num_segs (int): number of segments
    - low_idxs (numpy.ndarray): starting indices of the segments
    """
    if marginsize is None:
        marginsize = winsize
    assert winsize >= marginsize
    if len_seq <= winsize:
        num_segs = 1
        low_idxs = np.array([0])
    else:
        low_idxs = np.arange(0, len_seq, stepsize)
        len_segs = np.minimum(len_seq - low_idxs, winsize)
        kp_idxs = len_segs >= marginsize
        num_segs = len(len_segs[kp_idxs])
        low_idxs = low_idxs[kp_idxs]
    return num_segs, low_idxs

    
def montage_txt_parse(montage_txt):
    usefuls = [ix for ix in montage_txt if ix.startswith("mon")]
    infos = []
    for useful in usefuls:
        splits = useful.split("=")[1].split(",")
        iloc = int(splits[0])
        left_roi = splits[1].split(":")[-1].split("--")[0].strip()
        if len(splits[1].split(":")[-1].split("--")) > 1:
            right_roi = splits[1].split(":")[-1].split("--")[1].strip()
            infos.append((iloc, left_roi, right_roi))
        else:
            infos.append((iloc, left_roi))
    return infos
    


def find_closest_false_idx(bool_array, idx):
    n = len(bool_array)

    distance_array = np.arange(n)

    abs_distance_array = np.abs(distance_array - idx)

    abs_distance_array[bool_array] = n

    new_idx = np.argmin(abs_distance_array)
    
    return new_idx

class EEG_data(Dataset):
    """The main data class
    """
    def __init__(self, dataset, subset, 
                 move_dict=dict(winsize=256, stepsize=64, marginsize=None),
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
            dataset: The data set we want to load, train, dev or eval
            subset: the subset to choose: LE, AR, AR_A, ALL
            move_dict: the moving parameters
            root: The root of dataset,
            scale_fct: EEG = EEG/scale_fct
        """
        # set default values of move_dict and preprocess_dict
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
            root = list(DATA_ROOT.glob("EEG_seizure"))[0]
        self.root = root
        self.scale_fct = scale_fct
        self.move_dict.update(move_dict)
        self.preprocess_dict.update(preprocess_dict)
        self.edf_data = None
        if self.preprocess_dict.target_fs is None:
            self.preprocess_dict.target_fs = 250
        
        # Currently, do not need montage file (on Apr 9, 2023)
        #if subset.lower().startswith("all"):
        #    #montage_p = list((root/"DOCS").glob(f"*ar_montage*"))[0]
        #    pass
        #else:
        #    montage_p = list((root/"DOCS").glob(f"*{subset.lower()}_montage*"))[0]
        #montage_txt = load_txt(montage_p)
        #self.montage = montage_txt_parse(montage_txt)
            

        all_data = pd.read_csv(self.root/f"all_data_{dataset}.csv")
        if subset.lower().endswith("all"):
            self.all_data = all_data
        else:
            self.all_data = all_data[all_data["montage"].str.endswith(subset.lower())]
        
        # remove the data not including all channels in SEL_CHS
        self.all_data = self.all_data[self.all_data.apply(sel_fn, axis=1)]
            
        # remove the first and last 10 seconds and remove the subject with too-short seq
        self.all_data = self.all_data.copy()
        self.all_data.loc[:, "total_dur"] = self.all_data.loc[:, "total_dur"] - 2* self.rm_len
        self.all_data = self.all_data[self.all_data["total_dur"] > self.keep_len]
        self.all_data = self.all_data.reset_index()
        
        self.num_sps_persub = np.array([len2num(total_dur*self.preprocess_dict.target_fs, 
                                                self.move_dict.winsize, 
                                                self.move_dict.stepsize, 
                                                self.move_dict.marginsize)[0] 
                                        for total_dur in np.array(self.all_data["total_dur"])], dtype=int)
            
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
    
    def _preprocess_data(self, sub_idx):
        """ preprocess data
        """
        relative_path = self.all_data["relative_path"].iloc[sub_idx]
        data_path = (self.root/"edf"/relative_path).with_suffix(".edf")
        data = mne.io.read_raw_edf(data_path, preload=True, verbose=False)
        
        
        if self.preprocess_dict.is_drop:
            # only keep some channels
            #pick_names = np.unique(np.concatenate([i[1:] for i in self.montage]))
            if "ar" in self.all_data["montage"].iloc[sub_idx]:
                pick_names = [f"EEG {ch}-REF" for ch in SEL_CHS]
            elif "le" in self.all_data["montage"].iloc[sub_idx]:
                pick_names = [f"EEG {ch}-LE" for ch in SEL_CHS]
            data.pick_channels(pick_names)
            
            # to make sure the order is the same
            data.reorder_channels(pick_names)
        else:
            # to make sure the order is the same, alphebetical
            data.reorder_channels(np.sort(data.ch_names))
            
        # substract mean over all electrods at each time point
        # make AR, LE the same (on Apr 9, 2023)
        AR_fn = lambda x: x-x.mean(axis=0, keepdims=True)
        data.apply_function(AR_fn, channel_wise=False)
        
        if self.preprocess_dict.is_detrend:
            data.apply_function(mne.epochs.detrend, verbose=False)
        if self.preprocess_dict.filter_limit is not None:
            data.filter(
                self.preprocess_dict.filter_limit[0], 
                self.preprocess_dict.filter_limit[1], 
                verbose=False);
        if self.preprocess_dict.target_fs != data.info["sfreq"]:
            data.resample(self.preprocess_dict.target_fs, verbose=False)
            
        self.edf_data = data
        if self.preprocess_dict.is_diff:
            ## to be down
            pass
            #onlydata = []
            #for chs in self.montage:
            #    if len(chs) == 3:
            #        onlydata.append(data[chs[1]][0] - data[chs[2]][0])
            #    elif len(chs) == 2:
            #        onlydata.append(data[chs[1]][0])
            #data = np.array(onlydata).squeeze()
        else:
            data = data.get_data()
        return data
    
    def _dict2name(self, d):
        """a small fn to convert preprocess_dict as a name string
        """
        # sort via key
        dvs = sorted(list(d.items()), key=lambda x: x[0])
        filename_parts = []
        for key, value in dvs:
            if isinstance(value, list):
                value_str = "-".join(map(str, value))
            else:
                value_str = str(value)
            filename_parts.append(f"{key}_{value_str}")
        return "_".join(filename_parts)
    
    def get_preprocess_data(self, sub_idx, verbose=False, regen=False):
        """Get the preprocessed data give subject idx, if no, generate it and save it
        """
        relative_path = self.all_data["relative_path"].iloc[sub_idx]
        relative_path = relative_path + "_" + self._dict2name(self.preprocess_dict)
        data_path = (self.root/"edf"/relative_path).with_suffix(".pkl")
        if (not data_path.exists()) or regen:
            data = self._preprocess_data(sub_idx)
            save_pkl(data_path, data, verbose=verbose)
        else:
            if verbose:
                print(f"{relative_path} is alreadly generated")
            data = load_pkl(data_path, verbose=verbose)
        return data
    
    def rm_preprocess_data(self, sub_idx, verbose=False):
        """Remove the preprocessed data give subject idx.
        """
        relative_path = self.all_data["relative_path"].iloc[sub_idx]
        relative_path = relative_path + "_" + self._dict2name(self.preprocess_dict)
        data_path = (self.root/"edf"/relative_path).with_suffix(".pkl")
        if not data_path.exists():
            if verbose:
                print("No data need to be removed")
        else:
            if verbose:
                print(f"Delete {relative_path}.")
            data_path.unlink()
        return data
    
    def rm_outlier_seg(self, data, low_idxs, loc_idx):
        up_idxs = low_idxs + self.move_dict.winsize
        
        max_vs = []
        for low_idx, up_idx in zip(low_idxs, up_idxs):
            data_part = data[:, int(low_idx):int(up_idx)]
            max_vs.append(np.abs(data_part).max())
        max_vs = np.array(max_vs)
        outliers_rate_v = np.quantile(max_vs, 1-self.outliers_rate)
        false_idxs = max_vs > outliers_rate_v
            
        # remove the segment containing outliers
        loc_idx = find_closest_false_idx(false_idxs, loc_idx)
        return loc_idx
    
class MyDataLoader:
    def __init__(self, dataset, batch_size, shuffle=False):
        """
        DataLoader constructor
        
        Args:
        - dataset: the dataset to load
        - batch_size: the size of each batch
        - shuffle: whether to shuffle the dataset before loading
        """
        num_segs = len(dataset)
        all_idxs = np.arange(num_segs, dtype=int)
        if shuffle: 
            np.random.shuffle(all_idxs)
        self.all_idxs = all_idxs
        self.batch_size = batch_size
        self.dataset = dataset
    
    def __len__(self):
        """
        Returns the number of batches in the dataset
        """
        return int(len(self.dataset)/self.batch_size)
    
    def __call__(self, ix):
        """
        Loads a batch of data
        
        Args:
        - ix: the index of the batch to load
        
        Returns:
        - a tensor containing the batch of data
        """
        if ix >= self.__len__():
            ix = self.__len__() - 1
        low, up = ix*self.batch_size, (ix+1)*self.batch_size
        batch = []
        for idx in self.all_idxs[low:up]:
            batch.append(self.dataset[int(idx)])
        batch = np.array(batch)
        return torch.tensor(batch)