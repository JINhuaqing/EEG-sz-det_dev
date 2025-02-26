# this file contains the utility functions for the model
import torch
import numpy as np
import importlib
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict as ddict
from sklearn.metrics import roc_auc_score

def generate_position_encode(block_size, nfeature):
    """
    Generate positional encoding for a given block size and number of features.

    Args:
        block_size (int): The size of the block.
        nfeature (int): The number of features.

    Returns:
        pos_enc (torch.Tensor): The positional encoding tensor.
    """
    # create a matrix with shape (blocksize, nfeature)
    position = torch.arange(block_size, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, nfeature, 2).float() * (-np.log(10000.0) / nfeature))
    pos_enc = torch.zeros((block_size, nfeature))
    # apply sine to even indices in the array
    pos_enc[:, 0::2] = torch.sin(position * div_term)
    # apply cosine to odd indices in the array
    # to avoid the case when nfeature is odd
    pos_enc[:, 1::2] = torch.cos(position * div_term[:int(nfeature/2)])
    return pos_enc

def snorm_cdf(x, sd=1):
    """cdf of normal dist with sd"""
    sd = sd + 1e-20
    return 0.5 * (1 + torch.erf(x/sd/np.sqrt(2)))

def load_model_class(module_path, module_name=None):
    """Load the model class file from each saved model
    """
    if not isinstance(module_path, Path):
        module_path = Path(module_path)
    if module_name is None:
        module_name = "model_class"
    abs_path = module_path.resolve()
    abs_path = abs_path/(module_name+".py")

    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trans_batch(batch_sz, batch_bckg, config, shuffle=False):
    """transform the batch to make it easy for training
    args: 
        - batch_sz: the seizure data batch from the dataloader
        - batch_bckg: the background data batch from the dataloader
        - config: the configuration of the model
        - shuffle: whether to shuffle the batch
    return: res (both seizure and background data)
        - res[0]: X_rec, the input of the model 
        - res[1]: Y_dis, the output of the model, the discrete value
        - res[2]: labels, the labels of the data, 1 for seizure, 0 for background
    """
    def _trans_batch_single(batch):
        batch_dis, batch_org = batch

        X_org, Y_org = batch_org[:, :-config.move_step], batch_org[:, config.move_step:]
        Y_dis = batch_dis[:, config.move_step:]
        Y_move_dis = batch_dis[:, (config.move_step-1):-1] # use X_t as prediction of X_t+1
        Y_move_prob = nn.functional.one_hot(Y_move_dis, num_classes=2**config.k).double()
        return X_org, Y_dis

    res_sz = _trans_batch_single(batch_sz)
    res_bckg = _trans_batch_single(batch_bckg)
    labels = torch.cat([torch.ones(res_sz[0].size(0)), torch.zeros(res_bckg[0].size(0))], dim=0).long()
    res = []
    if shuffle:
        n_totol = res_sz[0].size(0) + res_bckg[0].size(0)
        idx = torch.randperm(n_totol)
        labels = labels[idx]
    for re1, re2 in zip(res_sz, res_bckg):
        re = torch.cat([re1, re2], dim=0)
        if shuffle:
            re = re[idx]
        res.append(re)
    res.append(labels)
    return res

def eval_model_selflearning(
               trained_model, 
               data_loader,
               loss_fn, 
               df_dtype,
               n_batch=None, 
               random=False, 
               seed=None,
               verbose=False):
    """Test the model
    args:
        - trained_model: the trained model
        - data_loader: the data loader
        - loss_fn: the loss function
        - df_dtype: the data type of the data
        - n_batch: the number of test batch
        - verbose: whether to show the progress bar or not
    return:
        - res: the loss of the model
    """
    pos_enc = generate_position_encode(trained_model.config.block_size,
                                       trained_model.config.nfeature).unsqueeze(0);
    if n_batch is None or n_batch == 0:
        n_batch = len(data_loader)
    assert n_batch > 0 and n_batch <= len(data_loader), "n_batch is too large or too small"
    if random:
        rng = np.random.default_rng(seed)
        idxs = rng.integers(0, len(data_loader), n_batch)
    else:
        idxs = np.arange(n_batch)
    if verbose:
        idxs = tqdm(idxs, total=n_batch)
    
    
    res = []
    for idx in idxs:
        batch = data_loader(idx)
        batch = batch.to(df_dtype)
        batch_wpos = batch + pos_enc
        
        trained_model.eval()
        with torch.no_grad():
            batch_rec = trained_model(batch_wpos)
            loss = loss_fn(batch_rec, batch)
            res.append(loss.item())
    return res

def eval_model(trained_model, 
               data_loader_sz,
               data_loader_bckg,
               cls_loss_fn=None, 
               aux_loss_fn=None,
               n_batch=None, 
               random=False, 
               verbose=False):
    """Test the model
    args:
        - trained_model: the trained model
        - data_loader_sz: the seizure data loader
        - data_loader_bckg: the background data loader
        - cls_loss_fn: the classification loss function
            if None, then no classification loss
        - aux_loss_fn: the auxiliary loss function
            if None, then no auxiliary loss
        - n_batch: the number of test batch
        - verbose: whether to show the progress bar or not
    return:
        - probs_sz: the probability of the seizure data from the model
        - labs_sz: the labels of the seizure data, 1 is seizure, 0 is background
    """
    pos_enc = generate_position_encode(trained_model.config.block_size,
                                       trained_model.config.nfeature).unsqueeze(0);
    if n_batch is None or n_batch == 0:
        n_batch = len(data_loader_sz)
    assert n_batch > 0 and n_batch <= len(data_loader_sz), "n_batch is too large or too small"
    if random:
        idxs = np.random.randint(0, len(data_loader_sz), n_batch)
    else:
        idxs = np.arange(n_batch)
    if verbose:
        idxs = tqdm(idxs, total=n_batch)
    
    
    ress = ddict(list)
    for idx in idxs:
        batch_sz, batch_bckg = data_loader_sz(idx), data_loader_bckg(idx);
        X_org, Y_dis, labels = trans_batch(batch_sz, batch_bckg, config=trained_model.config,
                                           shuffle=False)
        X_org_wpos = X_org + pos_enc;
        
        trained_model.eval()
        with torch.no_grad():
            res = trained_model(X_org_wpos)
            if isinstance(res, tuple):
                probs_aux, log_probs_cls = res
                if aux_loss_fn is not None:
                    loss_aux = aux_loss_fn(probs_aux, Y_dis, num_cls=2**trained_model.config.k)
                    ress["aux_loss"].append(loss_aux.item())
            else:
                log_probs_cls = res
        probs_cls = torch.exp(log_probs_cls)
        ress["probs_sz"].append(probs_cls[:, 1].cpu().numpy())
        ress["labs_sz"].append(labels.cpu().numpy())
        if cls_loss_fn is not None:
            loss_cls = cls_loss_fn(log_probs_cls, labels)
            ress["cls_loss"].append(loss_cls.item())
    ress["probs_sz"] = np.concatenate(ress["probs_sz"])
    ress["labs_sz"] = np.concatenate(ress["labs_sz"])
    ress["auc"] = roc_auc_score(ress["labs_sz"], ress["probs_sz"])
    return ress


def trans_batch_multi(batch_sz, batch_bckg, config, shuffle=False):
    """transform the batch to make it easy for training
        it is for multiple step trianing, i.e., the aux loss is to prediction X_t+1, X+t+k from X_t
    args: 
        - batch_sz: the seizure data batch from the dataloader
        - batch_bckg: the background data batch from the dataloader
        - config: the configuration of the model
        - shuffle: whether to shuffle the batch
    return: res (both seizure and background data)
        - res[0]: X_rec, the input of the model 
        - res[1]: Y_dis, the output of the model, the discrete value
        - res[2]: labels, the labels of the data, 1 for seizure, 0 for background
    """
    max_move_step = np.max(config.move_steps)
    def _trans_batch_single(batch):
        batch_dis, batch_org = batch
        X_org = batch_org[:, :-max_move_step]
        Y_diss = []
        for move_step in config.move_steps:
            Y_dis = batch_dis[:, move_step:(move_step+config.block_size)]
            Y_diss.append(Y_dis)
        return X_org, *Y_diss

    res_sz = _trans_batch_single(batch_sz)
    res_bckg = _trans_batch_single(batch_bckg)
    labels = torch.cat([torch.ones(res_sz[0].size(0)), torch.zeros(res_bckg[0].size(0))], dim=0).long()
    res = []
    if shuffle:
        n_totol = res_sz[0].size(0) + res_bckg[0].size(0)
        idx = torch.randperm(n_totol)
        labels = labels[idx]
    for re1, re2 in zip(res_sz, res_bckg):
        re = torch.cat([re1, re2], dim=0)
        if shuffle:
            re = re[idx]
        res.append(re)
    res.append(labels)
    return res

def eval_model_multi(trained_model, 
               data_loader_sz,
               data_loader_bckg,
               cls_loss_fn=None, 
               aux_loss_fn=None,
               n_batch=None, 
               random=False, 
               verbose=False):
    """Test the model (multiple step)
    args:
        - trained_model: the trained model
        - data_loader_sz: the seizure data loader
        - data_loader_bckg: the background data loader
        - cls_loss_fn: the classification loss function
            if None, then no classification loss
        - aux_loss_fn: the auxiliary loss function
            if None, then no auxiliary loss
        - n_batch: the number of test batch
        - verbose: whether to show the progress bar or not
    return:
        - probs_sz: the probability of the seizure data from the model
        - labs_sz: the labels of the seizure data, 1 is seizure, 0 is background
    """
    pos_enc = generate_position_encode(trained_model.config.block_size,
                                       trained_model.config.nfeature).unsqueeze(0);
    if n_batch is None or n_batch == 0:
        n_batch = len(data_loader_sz)
    assert n_batch > 0 and n_batch <= len(data_loader_sz), "n_batch is too large or too small"
    if random:
        idxs = np.random.randint(0, len(data_loader_sz), n_batch)
    else:
        idxs = np.arange(n_batch)
    if verbose:
        idxs = tqdm(idxs, total=n_batch)
    
    
    ress = ddict(list)
    for idx in idxs:
        batch_sz, batch_bckg = data_loader_sz(idx), data_loader_bckg(idx);
        trans_res = trans_batch_multi(batch_sz, batch_bckg, config=trained_model.config,
                                           shuffle=False)
        X_org = trans_res[0]
        labels = trans_res[-1]
        Y_diss = trans_res[1:-1]
        X_org_wpos = X_org + pos_enc;
        
        trained_model.eval()
        with torch.no_grad():
            res = trained_model(X_org_wpos)
            if isinstance(res, tuple):
                probss_aux, log_probs_cls = res
                if aux_loss_fn is not None:
                    loss_aux = 0
                    for probs_aux, Y_dis in zip(probss_aux, Y_diss):
                        loss_aux += aux_loss_fn(probs_aux, Y_dis, num_cls=2**trained_model.config.k)
                    loss_aux /= len(probss_aux)
                    ress["aux_loss"].append(loss_aux.item())
            else:
                log_probs_cls = res
        probs_cls = torch.exp(log_probs_cls)
        ress["probs_sz"].append(probs_cls[:, 1].cpu().numpy())
        ress["labs_sz"].append(labels.cpu().numpy())
        if cls_loss_fn is not None:
            loss_cls = cls_loss_fn(log_probs_cls, labels)
            ress["cls_loss"].append(loss_cls.item())
    ress["probs_sz"] = np.concatenate(ress["probs_sz"])
    ress["labs_sz"] = np.concatenate(ress["labs_sz"])
    ress["auc"] = roc_auc_score(ress["labs_sz"], ress["probs_sz"])
    return ress
