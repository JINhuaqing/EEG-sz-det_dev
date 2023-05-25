import numpy as np
import pickle
from easydict import EasyDict as edict
import time

def truncated_mean_upper(data, upper_pct=0.2):
    """
    Calculates the mean of the data after removing the upper percentage of data points specified by upper_pct.

    Args:
    data: A list or array of numerical data.
    upper_pct: A float between 0 and 1 representing the percentage of data points to remove from the upper end of the data.

    Returns:
    The mean of the truncated data.
    """
    data = np.array(data)
    assert 0 <= upper_pct <= 1, "Upper percentile should be between 0 and 1."

    upper_threshold = np.percentile(data, 100 * (1 - upper_pct))

    truncated_data = data[data <= upper_threshold]

    return np.mean(truncated_data)

def load_txt(p):
    with open(p, "r") as f:
        txt = f.readlines()
    return txt

def delta_time(t):
    """Return the time diff from t
    """
    delta_t = time.time() - t
    return delta_t


def load_pkl_folder2dict(folder, excluding=[], including=["*"], verbose=True):
    """The function is to load pkl file in folder as an edict
        args:
            folder: the target folder
            excluding: The files excluded from loading
            including: The files included for loading
            Note that excluding override including
    """
    if not isinstance(including, list):
        including = [including]
    if not isinstance(excluding, list):
        excluding = [excluding]
        
    if len(including) == 0:
        inc_fs = []
    else:
        inc_fs = list(set(np.concatenate([list(folder.glob(nam+".pkl")) for nam in including])))
    if len(excluding) == 0:
        exc_fs = []
    else:
        exc_fs = list(set(np.concatenate([list(folder.glob(nam+".pkl")) for nam in excluding])))
    load_fs = np.setdiff1d(inc_fs, exc_fs)
    res = edict()
    for fil in load_fs:
        res[fil.stem] = load_pkl(fil, verbose)                                                                                                                                  
    return res

# save a dict into a folder
def save_pkl_dict2folder(folder, res, is_force=False, verbose=True):
    assert isinstance(res, dict)
    for ky, v in res.items():
        save_pkl(folder/f"{ky}.pkl", v, is_force=is_force, verbose=verbose)

# load file from pkl
def load_pkl(fil, verbose=True):
    if verbose:
        print(f"Load file {fil}")
    with open(fil, "rb") as f:
        result = pickle.load(f)
    return result

# save file to pkl
def save_pkl(fil, result, is_force=False, verbose=True):
    if not fil.parent.exists():
        fil.parent.mkdir()
        if verbose:
            print(fil.parent)
            print(f"Create a folder {fil.parent}")
    if is_force or (not fil.exists()):
        if verbose:
            print(f"Save to {fil}")
        with open(fil, "wb") as f:
            pickle.dump(result, f)
    else:
        if verbose:
            print(f"{fil} exists! Use is_force=True to save it anyway")
        else:
            pass
