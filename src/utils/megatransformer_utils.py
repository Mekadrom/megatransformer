import numpy as np
import os
import random

try:
    import deepspeed
except ImportError:
    deepspeed = None

import torch
import torch.nn as nn

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from transformers import set_seed as hf_set_seed



def set_seed_everywhere(seed: int):
    """Set seed for all random number generators."""
    # Python's built-in random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch (CPU and CUDA)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        
    # Make PyTorch operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # HuggingFace Transformers (this sets seeds for all random generators in transformers)
    hf_set_seed(seed)
    
    # For good measure, set the Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def embedding_weight_init(hidden_size):
    def init_weights(module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=hidden_size ** -0.5)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
    return init_weights


def transformer_weight_init():
    def init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
    return init_weights


def conv2d_weight_init():
    def init_weights(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
    return init_weights


def print_debug_tensor(pre: str, tensor: torch.Tensor):
    # avoid printing on devices other than cuda:0
    if not isinstance(tensor, torch.Tensor):
        print(f"{pre}:\n\tNot a tensor {type(tensor)}")
        return
    if tensor is None:
        print(f"{pre}: None\n\t")
    elif tensor.numel() == 0:
        print(f"{pre}: empty tensor\n\tshape {tensor.shape}")
    elif tensor.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
        print(f"{pre}:\n\tptr: {tensor.data_ptr()}\n\tdtype: {tensor.dtype}\n\tdevice: {tensor.device}\n\tshape: {tensor.shape}\n\tmean: {tensor.mean()}\n\tstd: {tensor.std()}\n\tmin: {tensor.min()}\n\tmax: {tensor.max()}\n\tnorm: {tensor.norm()}\n\tany nan: {tensor.isnan().any()}\n\tany inf: {tensor.isinf().any()}")
        if tensor.numel() < 100:
            print(f"\t{tensor}")
    else:
        # non-float tensors
        print(f"{pre}:\n\tdtype: {tensor.dtype}\n\tdevice: {tensor.device}\n\tshape: {tensor.shape}\n\tmin: {tensor.min()}\n\tmax: {tensor.max()}\n\tany nan: {tensor.isnan().any()}\n\tany inf: {tensor.isinf().any()}")
        if tensor.numel() < 100:
            print(f"\t{tensor}")


def sanitize_model(model):
    if isinstance(model, DistributedDataParallel):
        return sanitize_model(model.module)
    if hasattr(model, '_orig_mod'):
        return sanitize_model(model._orig_mod)
    if isinstance(model, DataParallel):
        return sanitize_model(model.module)
    if deepspeed is not None:
        if isinstance(model, deepspeed.runtime.engine.DeepSpeedEngine):
            return model.module
    return model
