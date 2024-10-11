from constants import DATA_ROOT
from .eeg_load import EEGData
from utils.misc import save_pkl, load_pkl, _set_verbose_level, _update_params

import pdb
import ast
import numpy as np
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

class EEGDataSZ(EEGData):
    """The main data class
    """
    def __init__(self, 
                 dataset, 
                 subset, 
                 label, 
                 discrete_k=None,
                 verbose=1, 
                 root=None, 
                 move_params={},
                 pre_params={},
                 rm_params={}):
        """
        Initializes a new instance of the EEG data class.
        The EEG data with sampling freq 250
        This class will load data with seizure and background label

        Args:
            dataset: The data set we want to load, 
                     train, dev or eval
            subset: the subset to choose: LE, AR, AR_A, ALL
            label: the label of the data, sz, bckg, bckg-str
            discrete_k: the number of discrete levels, if None, use the original data
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
                - outliers_rate: the rate of the outliers, to remove the segments with the max value > quantile(1-outliers_rate); it is not used in this class
        """
        _set_verbose_level(verbose, logger)
        label = label.lower()

        super().__init__(
            dataset=dataset,
            subset=subset,
            verbose=verbose, 
            root=root, 
            move_params=move_params, 
            pre_params=pre_params, 
            rm_params=rm_params, 
            )
        
        all_data = self.all_data
        move_params = self.move_params
        pre_params = self.pre_params
        rm_params = self.rm_params
        
        # get the winlen, winsize in seconds
        winlen = move_params.winsize/pre_params.target_fs
        
        # remove rows based on label
        if label.startswith("sz"):
            all_data = all_data[self.all_data["is_seizure"]]
            all_data = all_data.reset_index(drop=True)
            


        # now count data seg in total
        if label == "sz":
            # sz_margin is the margin of segment to still consider as seizure segment, in seconds
            sz_margin_len = (move_params.winsize/pre_params.target_fs) * 0.5
            def _get_effseg_fn(row):
                lab = row["lab"] 
                total_dur = row["total_dur"]
                if isinstance(lab, str):
                    lab = ast.literal_eval(lab)
                eff_seg = []
                for vs in lab:
                    if vs[2] != "seiz":
                        continue
                    # the start and end of the effective segment under removal
                    vl = np.maximum(vs[0] - rm_params.rm_len, 0)
                    vu = np.minimum(vs[1] - rm_params.rm_len, total_dur)
                    if (vu-vl)<winlen:
                        continue

                    vl = np.maximum(vl - sz_margin_len, 0)
                    vu = np.minimum(vu + sz_margin_len, total_dur)
                    eff_seg.append((vl, vu))
                eff_seg = sorted(eff_seg, key=lambda x: x[0])
                return eff_seg
        elif label == "bckg-str":
            # the label bckg-str is not recommended to use, because bckg label in the dataset is very limited.
            # In any case, we have better bckg label, so we can use it.
            raise NotImplementedError("The label bckg-str is not recommended to use, use bckg instead.")
            def _get_effseg_fn(row):
                lab = row["lab"] 
                total_dur = row["total_dur"]
                if isinstance(lab, str):
                    lab = ast.literal_eval(lab)
                eff_seg = []
                for vs in lab:
                    if vs[2] != "bckg":
                        continue
                    # the start and end of the effective segment under removal
                    vl = np.maximum(vs[0] - rm_params.rm_len, 0)
                    vu = np.minimum(vs[1] - rm_params.rm_len, total_dur)
                    if (vu-vl)<winlen:
                        continue

                    eff_seg.append((vl, vu))
                eff_seg = sorted(eff_seg, key=lambda x: x[0])
                return eff_seg

        elif label == "bckg":
            def _get_effseg_fn(row):
                lab = row["lab"] 
                total_dur = row["total_dur"]
                if isinstance(lab, str):
                    lab = ast.literal_eval(lab)
                eff_seg_inv = []
                for vs in lab:
                    if vs[2] != "seiz":
                        continue
                    # the start and end of the effective segment under removal
                    vl = np.maximum(vs[0] - rm_params.rm_len, 0)
                    vu = np.minimum(vs[1] - rm_params.rm_len, total_dur)
                    eff_seg_inv.append((vl, vu))
                eff_seg_inv = sorted(eff_seg_inv, key=lambda x: x[0])
                start_points = [0, ]
                end_points = []
                for vs in eff_seg_inv:
                    start_points.append(vs[1])
                    end_points.append(vs[0])
                end_points.append(total_dur)
                eff_seg = [(start_points[i], end_points[i]) for i in range(len(start_points)) if (end_points[i]-start_points[i])>=winlen]
                eff_seg = sorted(eff_seg, key=lambda x: x[0])
                return eff_seg
        else: 
            raise ValueError(f"Unknown label: {label}. Valid labels are sz, bckg, bckg-str")
            

        eff_segs = all_data.apply(_get_effseg_fn, axis=1).tolist()
        num_sps_persub = []
        for eff_seg in eff_segs:
            num_sps = [self._len2num(
                (vs[1]-vs[0])*pre_params.target_fs, 
                move_params.winsize, 
                move_params.stepsize, 
                move_params.marginsize)[0] 
                               for vs in eff_seg]
            num_sps_persub.append(num_sps)

        self.all_data = all_data
        self.num_sps_persub = num_sps_persub
        self.eff_segs = eff_segs
        self.discrete_k = discrete_k
        self.label = label

    def __len__(self):
        return int(np.concatenate(self.num_sps_persub).sum())
        
    def __getitem__(self, idx):
        """
        Get the data segment with index idx
        """
        if isinstance(idx, (int, np.integer)):
            if idx < 0:
                idx = self.__len__() + idx
            num_sps_persub = [np.sum(idx) for idx in self.num_sps_persub]
            num_cumsum = np.cumsum(num_sps_persub)
            num_cumsum = num_cumsum.astype(int)
            # this way can handle the case when num of seg is 0 for same subject
            sub_idx = np.sum(num_cumsum < (idx+1))
            if sub_idx != 0:
                loc_idx = idx - num_cumsum[sub_idx-1]
            else:
                loc_idx = idx

        elif isinstance(idx, str) and idx.lower().startswith("sub"):
            sub_idx = int(idx.split("sub")[-1])
            if sub_idx < 0:
                sub_idx = len(self.all_data) + sub_idx
            if sub_idx > (self.__len__()-1):
                raise IndexError
            loc_idx = None
            
        else:
            raise NotImplementedError("The index type is not supported")

        data = self.get_pre_data(sub_idx)
        # remove the first and last pts
        data = data[:, int(self.pre_params.target_fs*self.rm_params.rm_len):-int(self.pre_params.target_fs*self.rm_params.rm_len)]
        data = self._robust_EEG_rescale(data, data_max=self.pre_params.scale_fct)

        if loc_idx is not None:
            cur_eff_segs = self.eff_segs[sub_idx]
            low_idxss = []
            for vs in cur_eff_segs:
                low_idxs = self._len2num((vs[1]-vs[0])*self.pre_params.target_fs, 
                                         self.move_params.winsize, 
                                         self.move_params.stepsize, 
                                         self.move_params.marginsize)[1]
                low_idxss.append(low_idxs)
            n_per_effseg = self.num_sps_persub[sub_idx]
            cumsum_per_effseg = np.cumsum(n_per_effseg)
            effseg_idx = np.sum(cumsum_per_effseg < loc_idx+1)
            if effseg_idx != 0:
                loc_idx_in_effseg = loc_idx - cumsum_per_effseg[effseg_idx-1]
            else: 
                loc_idx_in_effseg = loc_idx

            loc_idx_low = int(low_idxss[effseg_idx][loc_idx_in_effseg] + cur_eff_segs[effseg_idx][0]*self.pre_params.target_fs)
            loc_idx_up = int(loc_idx_low + self.move_params.winsize)
            
            data = data[:, loc_idx_low:loc_idx_up]
            
            if data.shape[1] < self.move_params.winsize:
                padding = np.zeros((data.shape[0], self.move_params.winsize-data.shape[1]))
                data = np.concatenate([data, padding], axis=1)
        data = data.transpose(1, 0)
        if self.discrete_k is not None:
            data_dis = self._get_dis_data(data, self.discrete_k)
            return data_dis, data
            #data_rec = self._rec_dis_data(data_dis, self.discrete_k)
            #return data_dis, data_rec
        else:
            return data

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
        name = f"dis_cutoffs_bckg_subset_{self.subset}_{name_p1}_{name_p2}_k{k}_seed{seed}.pkl"
        save_folder = self.root/"dis_cutoffs"
        if not save_folder.exists():
            save_folder.mkdir(exist_ok=True)

        if (not (save_folder/name).exists()) or regen:
            assert self.label == "bckg", "We only use the background data to generate the cutoffs"
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