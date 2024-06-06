# this file include some fns 
# specific to TUH_seizure data
from utils.misc import load_txt
import numpy as np
from constants import DATA_ROOT
import time

def sz_lab_fn(lab_info_cl):
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

def sz_lab_from_row_fn(obs_row, root=None):
    """A convinient wrapper of `sz_lab_fn`
    """
    if root is None:
        root = DATA_ROOT/"EEG_seizure/edf"
    fil_path = root/obs_row["relative_path"]
    lab_info = load_txt(fil_path.with_suffix(".csv_bi"))
    lab_info_cl =  [lab for lab in lab_info if lab.startswith("TERM") and "seiz" in lab]
    return sz_lab_fn(lab_info_cl)

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
    ress["labs"] = []
    for ix, lab in enumerate(txt[6:]):
        #res = dict(
        #st = float(lab.split(",")[1]),
        #et = float(lab.split(",")[2]),
        #typ = lab.split(",")[3], 
        #)
        res = (float(lab.split(",")[1]),float(lab.split(",")[2]),lab.split(",")[3], 
              float(lab.split(",")[4]))
        #ress[f"lab{ix+1}"] = res
        ress["labs"].append(res)
   
    ress["is_seizure"] = bool(np.sum(['seiz' in i for i in txt]))
    return ress
