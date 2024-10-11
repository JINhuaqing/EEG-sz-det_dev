from constants import DATA_ROOT
from .utils import rec_data, digitize_data
from utils.misc import save_pkl, load_pkl, _set_verbose_level, _update_params
import pandas as pd

import torch
import mne
import pdb
import numpy as np 
from torch.utils.data import Dataset
from easydict import EasyDict as edict
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    ch = logging.StreamHandler() # for console. 
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch) 

    
# currently not used
#def montage_txt_parse(montage_txt):
#    usefuls = [ix for ix in montage_txt if ix.startswith("mon")]
#    infos = []
#    for useful in usefuls:
#        splits = useful.split("=")[1].split(",")
#        iloc = int(splits[0])
#        left_roi = splits[1].split(":")[-1].split("--")[0].strip()
#        if len(splits[1].split(":")[-1].split("--")) > 1:
#            right_roi = splits[1].split(":")[-1].split("--")[1].strip()
#            infos.append((iloc, left_roi, right_roi))
#        else:
#            infos.append((iloc, left_roi))
#    return infos
    




class EEGData(Dataset):
    """The main data class
    """
    def __init__(self, 
                 dataset, 
                 subset, 
                 discrete_k=None,
                 verbose=1,
                 root=None, 
                 move_params={},
                 pre_params={},
                 rm_params={}
                 ):
        """
        Initializes a new instance of the EEG data class.
        The EEG data with sampling freq 250

        Args:
            dataset: The data set we want to load, 
                     train, dev or eval
            subset: the subset to choose: LE, AR, AR_A, ALL
            discrete_k: the k to discretize the data, if None, do not discretize
                if not None, discretize the data and return both discrete and reconstructed data
            verbose: whether to print the information, 1-4, 1 least, 4 most
            root: The root of dataset,
            move_params: the moving parameters, determine how to get data segment from org EEG
                - winsize: the length of data segment
                - stepsize: the stepsize moveing along the time series
                - marginsize: The minimal length of data to keep, <= winsize
                    if None, marginsize = winsize
            pre_params: the preprocess parameters, determine how to preprocess the data
                - is_detrend: whether to detrend the data
                - is_drop: whether to drop the channels not in SEL_CHS
                - target_fs: the target sampling frequency 
                - filter_limit: the limit of the filter, [low, high]
                - is_diff: whether to get the difference of the channels, use False, not implemented
                - scale_fct: EEG = EEG/scale_fct to scale the EEG
                    - if None, scale_fct = np.quantile(np.abs(EEG), [0.95])
            rm_params: the remove parameters, determine how to remove the outlier segments
                - rm_len: the length of the first and last part to remove, in seconds
                - keep_len: the minimal length of the data to keep, in seconds
                - outliers_rate: the rate of the outliers, to remove the segments with the max value > quantile(1-outliers_rate)
        """
        _set_verbose_level(verbose, logger)

        # the selected channels for analysis
        sel_chs_raw = "Fp1, Fp2, F7, F3, Fz, F4, F8, T3, T4, T5, T6, C3, Cz, C4, P3, Pz, P4, O1, O2"
        self.SEL_CHS = [ch.strip().upper() for ch in sel_chs_raw.split(",")]

        # set default values of move_params and pre_params
        move_params_def = edict(dict(
            winsize=256, 
            stepsize=64, 
            marginsize=None
        ))
        pre_params_def = edict(dict(
            is_detrend=True, 
            is_drop=True,
            target_fs=125, 
            filter_limit=[1, 45], 
            is_diff=False, 
            scale_fct=None,
        ))
        rm_params_def = edict(dict(
            rm_len = 50, 
            keep_len = 20,
            outliers_rate = 0.05,
        ))
        move_params = _update_params(move_params, move_params_def, logger)
        pre_params = _update_params(pre_params, pre_params_def, logger)
        rm_params = _update_params(rm_params, rm_params_def, logger)
        
        
        if root is None:
            root = list(DATA_ROOT.glob("EEG_seizure"))[0]/"edf"
        
        # Currently, do not need montage file (on Apr 9, 2023)
        #if subset.lower().startswith("all"):
        #    #montage_p = list((root/"DOCS").glob(f"*ar_montage*"))[0]
        #    pass
        #else:
        #    montage_p = list((root/"DOCS").glob(f"*{subset.lower()}_montage*"))[0]
        #montage_txt = load_txt(montage_p)
        #self.montage = montage_txt_parse(montage_txt)
            

        all_data = pd.read_csv(root/f"all_data_{dataset}.csv")
        if subset.lower().endswith("all"):
            all_data = all_data
        else:
            all_data = all_data[all_data["montage"].str.endswith(subset.lower())]
        
        # remove the data not including all channels in SEL_CHS
        all_data = all_data[all_data.apply(self._sel_fn, axis=1)]
            
        # remove the first and last 10 seconds and remove the subject with too-short seq
        all_data["total_dur"] = all_data["total_dur"] - 2* rm_params.rm_len
        all_data = all_data[all_data["total_dur"] > rm_params.keep_len]
        all_data = all_data.reset_index(drop=True)

        # get the number of segments for each subject
        num_sps_persub = []
        for total_dur in np.array(all_data["total_dur"]):
            num_sps, _ = self._len2num(total_dur*pre_params.target_fs, 
                                 move_params.winsize, 
                                 move_params.stepsize, 
                                 move_params.marginsize)
            num_sps_persub.append(num_sps)
        num_sps_persub = np.array(num_sps_persub).astype(int)


        self.num_sps_persub = num_sps_persub
        self.rm_params = rm_params
        self.pre_params = pre_params
        self.move_params = move_params
        self.root = root
        self.subset = subset
        self.all_data = all_data
        self.verbose = verbose
        self.discrete_k = discrete_k
        # hook is only for get the variable
        self.hook = edict()
        self.hook.sub_idxs = []
            
    def _sel_fn(self, row):
        """ Select the row including all the chs in SEL_CHS
        """
        all_chs = row["all_chs"].split(";")
        ex_chs = np.setdiff1d(self.SEL_CHS, all_chs)
        return len(ex_chs) == 0
    

    def _robust_EEG_rescale(self, data, data_max=None):
        """rescale data robustly.
        """
        if data_max is None:
            data_max = np.quantile(np.abs(data), [0.95]) # to remove the effet of outliers
        data_rescaled = data/data_max
        #data_min = -data_max
        #data_minmax = 2*(data-data_min)/(data_max-data_min) - 1;
        return data_rescaled

    def _len2num(self, len_seq, winsize, stepsize, marginsize=None):
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

    # find the closest False idx to the given idx
    def _find_closest_false_idx(self, bool_array, idx):
        n = len(bool_array)
    
        distance_array = np.arange(n)
        abs_distance_array = np.abs(distance_array - idx)
        abs_distance_array[bool_array] = n
    
        new_idx = np.argmin(abs_distance_array)
        
        return new_idx

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
        if isinstance(idx, (int, np.integer)):
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
        data = self.get_pre_data(sub_idx)
        # remove the first and last pts
        data = data[:, int(self.pre_params.target_fs*self.rm_params.rm_len):-int(self.pre_params.target_fs*self.rm_params.rm_len)]
        data = self._robust_EEG_rescale(data, data_max=self.pre_params.scale_fct)
        
        if loc_idx is not None:
            _, low_idxs = self._len2num(np.array(self.all_data["total_dur"])[sub_idx] * self.pre_params.target_fs, 
                                                      self.move_params.winsize, 
                                                      self.move_params.stepsize, 
                                                      self.move_params.marginsize)
            loc_idx = self.rm_outlier_seg(data, low_idxs, loc_idx)
            loc_idx_low, loc_idx_up = int(low_idxs[loc_idx]), int(low_idxs[loc_idx] + self.move_params.winsize)
            data = data[:, loc_idx_low:loc_idx_up]
            
            if data.shape[1] < self.move_params.winsize:
                padding = np.zeros((data.shape[0], self.move_params.winsize-data.shape[1]))
                data = np.concatenate([data, padding], axis=1)
        #data = torch.tensor(data)
        data = data.transpose(1, 0)
        if self.discrete_k is not None:
            data_dis = self._get_dis_data(data, self.discrete_k)
            return data_dis, data
            #data_rec = self._rec_dis_data(data_dis, self.discrete_k)
            #return data_dis, data_rec
        else:
            return data
    
    def _preprocess_data(self, sub_idx):
        """ preprocess data
        """
        relative_path = self.all_data["relative_path"].iloc[sub_idx]
        data_path = (self.root/relative_path).with_suffix(".edf")
        data = mne.io.read_raw_edf(data_path, preload=True, verbose=False)
        
        
        if self.pre_params.is_drop:
            # only keep some channels
            #pick_names = np.unique(np.concatenate([i[1:] for i in self.montage]))
            if "ar" in self.all_data["montage"].iloc[sub_idx]:
                pick_names = [f"EEG {ch}-REF" for ch in self.SEL_CHS]
            elif "le" in self.all_data["montage"].iloc[sub_idx]:
                pick_names = [f"EEG {ch}-LE" for ch in self.SEL_CHS]
            data.pick(pick_names)
            
            # to make sure the order is the same
            data.reorder_channels(pick_names)
        else:
            # to make sure the order is the same, alphebetical
            data.reorder_channels(np.sort(data.ch_names))
            
        # substract mean over all electrods at each time point
        # make AR, LE the same (on Apr 9, 2023)
        AR_fn = lambda x: x-x.mean(axis=0, keepdims=True)
        data.apply_function(AR_fn, channel_wise=False)
        
        if self.pre_params.is_detrend:
            data.apply_function(mne.epochs.detrend, verbose=False)
        if self.pre_params.filter_limit is not None:
            data.filter(
                self.pre_params.filter_limit[0], 
                self.pre_params.filter_limit[1], 
                verbose=False);
        if self.pre_params.target_fs != data.info["sfreq"]:
            data.resample(self.pre_params.target_fs, verbose=False)
            
        if self.pre_params.is_diff:
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
        """a small fn to convert pre_params as a name string
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
    
    def get_pre_data(self, sub_idx, verbose=False, regen=False):
        """Get the preprocessed data give subject idx, if no, generate it and save it
        """
        relative_path = self.all_data["relative_path"].iloc[sub_idx]
        relative_path = relative_path + "_" + self._dict2name(self.pre_params)
        data_path = (self.root/relative_path).with_suffix(".pkl")
        if (not data_path.exists()) or regen:
            data = self._preprocess_data(sub_idx)
            save_pkl(data_path, data, verbose=verbose)
        else:
            if verbose:
                print(f"{relative_path} is alreadly generated")
            data = load_pkl(data_path, verbose=verbose)
        return data
    
    def rm_pre_data(self, sub_idx, verbose=False):
        """Remove the preprocessed data give subject idx.
        """
        relative_path = self.all_data["relative_path"].iloc[sub_idx]
        relative_path = relative_path + "_" + self._dict2name(self.pre_params)
        data_path = (self.root/relative_path).with_suffix(".pkl")
        if not data_path.exists():
            if verbose:
                print("No data need to be removed")
        else:
            if verbose:
                print(f"Delete {relative_path}.")
            data_path.unlink()
        return data_path
    
    def rm_outlier_seg(self, data, low_idxs, loc_idx):
        """This function is to remove the outlier segments and 
        return the new loc_idx which is close to the original loc_idx after removing the outlier segments.
        the outlier segments are defined as the segments with the max value > quantile(1-outliers_rate) of the max values of all segments.
        """

        up_idxs = low_idxs + self.move_params.winsize
        
        max_vs = []
        for low_idx, up_idx in zip(low_idxs, up_idxs):
            data_part = data[:, int(low_idx):int(up_idx)]
            max_vs.append(np.abs(data_part).max())
        max_vs = np.array(max_vs)
        outliers_rate_v = np.quantile(max_vs, 1-self.rm_params.outliers_rate)
        false_idxs = max_vs > outliers_rate_v
            
        # remove the segment containing outliers
        loc_idx = self._find_closest_false_idx(false_idxs, loc_idx)
        return loc_idx

    def _get_dis_cutoffs(self, k, seed=0, regen=False):
        """Get the cutoffs to discretize the data
        args: 
            k (int): 2^k is the number of the bins
            seed (int): the seed to generate the cutoffs
            regen (bool): whether to regenerate the cutoffs
        """
        np.random.seed(seed)
        num_seg = 100

        name_p1 = self._dict2name(self.pre_params)
        name_p2 = self._dict2name(self.rm_params)
        name = f"dis_cutoffs_pool_subset_{self.subset}_{name_p1}_{name_p2}_k{k}_seed{seed}.pkl"
        save_folder = self.root/"dis_cutoffs"
        if not save_folder.exists():
            save_folder.mkdir(exist_ok=True)

        if (not (save_folder/name).exists()) or regen:
            logger.info(f"Generate the cutoffs for discretization")
            seg_idxs = np.sort(np.random.choice(np.arange(len(self)), num_seg)).astype(int)
            # get the data, and pool them together
            if self.verbose >= 2:
                pbar = tqdm(seg_idxs, total=len(seg_idxs), desc="Get the data to calculate the cutoffs")
            else:
                pbar = seg_idxs
            vec_uni = [
                self[ix].flatten() for ix in pbar 
            ]
            vec_uni_abs = np.abs(np.concatenate(vec_uni))

            # get the cutoffs
            # not that the positive and negative are the same
            num_cls_half = 2**(k-1)
            qs = np.arange(1, num_cls_half)/num_cls_half
            cutoffs = np.quantile(vec_uni_abs, qs)

            # save it 
            save_pkl(save_folder/name, cutoffs, verbose=self.verbose>2, is_force=True)
        else: 
            cutoffs = load_pkl(save_folder/name, verbose=self.verbose>2)
        return cutoffs

    def get_dis_cutoffs(self):
        """Get the cutoffs to discretize the data
        """
        return self._get_dis_cutoffs(self.discrete_k)
    
    def _get_dis_data(self, data, k):
        """Get the discretized data
        """
        cutoffs = self._get_dis_cutoffs(k)
        data_dis = digitize_data(data, cutoffs)
        return data_dis
    
    def _rec_dis_data(self, data_dis, k):
        """Get the reconstructed data
        """
        cutoffs = self._get_dis_cutoffs(k)
        data_rec = rec_data(data_dis, cutoffs)
        return data_rec


