from typing import Optional
import numpy as np
import os
import random

try:
    import deepspeed
except ImportError:
    deepspeed = None

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """Embedding init: N(0, 1/sqrt(hidden_size)).

    This keeps embedding output magnitude ~1 regardless of d_model, so the
    downstream transformer doesn't see embeddings that scale wildly with width.
    """
    def init_weights(module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=hidden_size ** -0.5)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
    return init_weights


def linear_weight_init(gain: float = 1.0):
    """Standard Xavier-normal init for `nn.Linear` layers.

    This is the principled default for q/k/v projections, FFN expand layers,
    and any other linear layer that should preserve activation variance. Uses
    `xavier_normal_(gain=gain)` which gives `std = gain * sqrt(2/(fan_in+fan_out))`.

    For residual-output layers (attention `o_proj`, FFN `condense`) you should
    additionally call `apply_depth_scaled_residual_init` after constructing the
    full stack so the residual stream variance does not grow with depth.
    """
    def init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=gain)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
    return init_weights


# Backwards-compat alias. Old call sites that used `transformer_weight_init()`
# meant "init all linear layers" — now uses the principled gain=1.0 default
# instead of the previous gain=0.02 which was effectively zero.
def transformer_weight_init(gain: float = 1.0):
    return linear_weight_init(gain=gain)


def conv2d_weight_init():
    """Kaiming-normal init for `nn.Conv2d` (and `nn.Conv1d`) layers.

    Uses `fan_out` mode and `relu` nonlinearity, which is the principled choice
    for conv layers feeding into ReLU/GELU activations.
    """
    def init_weights(module):
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
    return init_weights


def _depth_scaled_residual_init_(linear: nn.Linear, n_residual_layers: int):
    """In-place depth-scaled init for a single residual-output Linear.

    Uses `std = sqrt(1 / (n_residual_layers * fan_in))`, derived from Kaiming
    `sqrt(2/fan_in)` divided by `sqrt(2*n_residual_layers)` (the GPT-2/Megatron
    pattern). Keeps the variance of the residual stream stable as depth grows.
    """
    if not isinstance(linear, nn.Linear):
        return
    fan_in = linear.weight.shape[1]
    std = (1.0 / max(n_residual_layers, 1) / fan_in) ** 0.5
    nn.init.normal_(linear.weight, mean=0.0, std=std)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def apply_depth_scaled_residual_init(blocks, n_residual_layers: Optional[int] = None):
    """Re-initialize residual-output layers of every block in a stack.

    Targets per block:
      - `block.self_attn.o_proj`
      - `block.cross_attn.o_proj` (if present)
      - `block.ffn.condense` (if present)

    `n_residual_layers` defaults to `len(blocks)` — the depth of this residual
    stack. Pass an explicit value to handle cases like the recurrent block where
    the effective depth is the number of iterations, not the number of blocks.

    Call this AFTER constructing all blocks (and after their per-block default
    init has run). It overwrites the residual-output layers with depth-scaled
    init while leaving q/k/v/expand and other layers alone.
    """
    if n_residual_layers is None:
        try:
            n_residual_layers = len(blocks)
        except TypeError:
            n_residual_layers = 1
    if n_residual_layers <= 0:
        return
    for block in blocks:
        if hasattr(block, 'self_attn') and hasattr(block.self_attn, 'o_proj'):
            _depth_scaled_residual_init_(block.self_attn.o_proj, n_residual_layers)
        if hasattr(block, 'cross_attn') and hasattr(block.cross_attn, 'o_proj'):
            _depth_scaled_residual_init_(block.cross_attn.o_proj, n_residual_layers)
        if hasattr(block, 'ffn') and hasattr(block.ffn, 'condense'):
            _depth_scaled_residual_init_(block.ffn.condense, n_residual_layers)


def init_transformer_stack(blocks, n_residual_layers: Optional[int] = None, gain: float = 1.0):
    """One-shot principled init for a stack of transformer blocks.

    1. Applies standard Xavier (`linear_weight_init(gain)`) to every Linear in
       every block — overwriting any per-block default init so the whole stack
       is consistent.
    2. Then re-initializes the residual-output layers (o_proj / condense) with
       depth-scaled init so the residual stream variance is preserved.

    This is the recommended init for any encoder/decoder stack inside a parent
    module (preludes, codas, cross-attention decoders).
    """
    init_linear = linear_weight_init(gain=gain)
    for block in blocks:
        block.apply(init_linear)
    apply_depth_scaled_residual_init(blocks, n_residual_layers=n_residual_layers)


def print_debug_tensor(pre: str, tensor: torch.Tensor):
    # avoid printing on devices other than cuda:0
    ind = '\t' * (pre.count('\t') + 1)
    if not isinstance(tensor, torch.Tensor):
        print(f"{pre}:\n{ind}Not a tensor {type(tensor)}")
        return
    if tensor is None:
        print(f"{pre}: None\n")
    elif tensor.numel() == 0:
        print(f"{pre}: empty tensor\n{ind}shape {tensor.shape}")
    elif tensor.numel() == 1:
        print(f"{pre}: single element tensor\n{ind} with value {tensor.item()}")
    elif tensor.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
        print(f"{pre}:\n{ind}ptr: {tensor.data_ptr()}\n{ind}dtype: {tensor.dtype}\n{ind}device: {tensor.device}\n{ind}shape: {tensor.shape}\n{ind}mean: {tensor.mean()}\n{ind}std: {tensor.std()}\n{ind}min: {tensor.min()}\n{ind}max: {tensor.max()}\n{ind}norm: {tensor.norm()}\n{ind}any nan: {tensor.isnan().any()}\n{ind}any inf: {tensor.isinf().any()}")
        if tensor.numel() < 100:
            print(f"{ind}{tensor}")
    else:
        # non-float tensors
        print(f"{pre}:\n{ind}dtype: {tensor.dtype}\n{ind}device: {tensor.device}\n{ind}shape: {tensor.shape}\n{ind}min: {tensor.min()}\n{ind}max: {tensor.max()}\n{ind}any nan: {tensor.isnan().any()}\n{ind}any inf: {tensor.isinf().any()}")
        if tensor.numel() < 100:
            print(f"{ind}{tensor}")


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


def trim(tensor: Optional[torch.Tensor], max_length: int, dim: int = -1) -> Optional[torch.Tensor]:
    """Trim the tensor along the specified dimension to the max_length."""
    if tensor is None:
        return None
    if tensor.shape[dim] > max_length:
        slices = [slice(None)] * tensor.dim()
        slices[dim] = slice(0, max_length)
        return tensor[tuple(slices)]
    return tensor


def pad_and_mask(tensors: list[torch.Tensor], lengths: list[int]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    pad_to_length = max(lengths)
    padded_tensors = []
    masks = []
    for tensor, length in zip(tensors, lengths):
        # Create mask (1 = valid, 0 = padding)
        mask = torch.zeros(pad_to_length, dtype=torch.float32, device=tensor.device)
        mask[:length] = 1.0

        # Truncate to batch max length (along last dimension for both formats)
        tensor = tensor[..., :pad_to_length]

        # Pad if needed (along last dimension)
        if tensor.shape[-1] < pad_to_length:
            tensor = F.pad(tensor, (0, pad_to_length - tensor.shape[-1]), value=0)

        padded_tensors.append(tensor)
        masks.append(mask)
    return padded_tensors, masks
