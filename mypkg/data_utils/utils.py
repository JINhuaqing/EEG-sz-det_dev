import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from constants import DATA_ROOT
from utils.misc import load_txt, save_pkl, load_pkl


def rec_data(data_dis, k, verbose=False, typ="_rescale_health"):
    """ Reconstruct the digitized data for 2^k levels.
        Approximate inverse operator of digitize_data
    """
    data_dis = convert_to_type(data_dis, "np")
    EEG_ROOT = list(DATA_ROOT.glob("EEG_seizure"))[0]
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
    data = convert_to_type(data, "np")
    EEG_ROOT = list(DATA_ROOT.glob("EEG_seizure"))[0]
    fil = EEG_ROOT/f"discrete_cuts/cuts_2power{k}{typ}.pkl";
    assert fil.exists(), "No cutoff values"
    cuts = load_pkl(fil, verbose=verbose);
    cuts_all = np.sort(np.concatenate([-cuts, [0], cuts]));
    data_discrete = np.digitize(data, cuts_all);
    return data_discrete

def get_triple(X, k, nch=19):
    """Give a X in anyform, class form, prob form or reconstruct form, 
    return all triple in numpy.ndarray
    """
    n_cls = 2**k
    X = convert_to_type(X, "np")
    lastdim = X.shape[-1]
    
    if lastdim == n_cls:
        X_prob = X
        X_cls = np.argmax(X_prob, axis=-1)
        X_rec = rec_data(X_cls, k=k)
    elif lastdim == nch:
        if np.min(X) < 0:
            X_rec = X
            X_cls = digitize_data(X_rec, k=k)
        else:
            X_cls = X
            X_rec = rec_data(X_cls, k=k)
            
        X_prob = nn.functional.one_hot(convert_to_type(X_cls, "cpu"), num_classes=n_cls).double()
        X_prob = convert_to_type(X_prob, "np")
    
    triple = edict()
    triple.cls = X_cls
    triple.prob = X_prob
    triple.rec = X_rec
    return triple

def convert_to_type(X, target_type):
    """
    Convert the input X to the specified target_type.
    
    :param X: Input object, which can be numpy.ndarray, torch.Tensor, or CUDA tensor
    :param target_type: Target type to which X should be converted, can be 'numpy', 'cpu', or 'cuda'
    :return: The converted object
    """

    target_type = target_type.lower()
    if target_type.startswith('np'):
        target_type = "numpy"
    if target_type not in ['numpy', 'cpu', 'cuda']:
        raise ValueError("target_type must be 'numpy', 'cpu', or 'cuda'")

    # Convert to numpy
    if target_type == 'numpy':
        if isinstance(X, np.ndarray):
            return X
        elif isinstance(X, torch.Tensor):
            return X.cpu().detach().numpy()
    
    # Convert to CPU tensor
    if target_type == 'cpu':
        if isinstance(X, np.ndarray):
            return torch.tensor(X)
        elif isinstance(X, torch.Tensor):
            return X.cpu()
    
    # Convert to CUDA tensor
    if target_type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if isinstance(X, np.ndarray):
            return torch.tensor(X).cuda()
        elif isinstance(X, torch.Tensor):
            return X.cuda()

    # If we reach here, it means there's an unhandled type
    raise TypeError("Unknown input type; X is neither numpy.ndarray nor torch.Tensor")