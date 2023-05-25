import torch
import numpy as np
import importlib
from pathlib import Path

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