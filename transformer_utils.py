from positional_encodings.torch_encodings import PositionalEncoding2D
from swiglu import SwiGLU
from torch import nn

import math
import torch

def get_activation_function(activation_function_name):
    if activation_function_name == 'relu':
        return nn.ReLU
    elif activation_function_name == 'gelu':
        return nn.GELU
    elif activation_function_name == 'elu':
        return nn.ELU
    elif activation_function_name == 'selu':
        return nn.SELU
    elif activation_function_name == 'prelu':
        return nn.PReLU
    elif activation_function_name == 'leaky_relu':
        return nn.LeakyReLU
    elif activation_function_name == 'silu':
        return nn.SiLU
    elif activation_function_name == 'none':
        return nn.Identity
    else:
        raise Exception(f"Unknown activation function {activation_function_name}")

def create_activation_function(d_in, activation_function_name):
    if activation_function_name == 'swiglu':
        return SwiGLU(d_in)
    return get_activation_function(activation_function_name)()

def get_buffered_positional_encoding(device, d_model, positional_encoding_dim: int, maxlen=100, num_dims=1):
    if num_dims == 1:
        positional_encoding = torch.zeros((maxlen, d_model)) # (max_length, d_model)
        for i in range(maxlen):
            for k in range(d_model):
                if k % 2 == 0:
                    positional_encoding[i, k] = math.sin(i / math.pow(10000, k / d_model))
                else:
                    positional_encoding[i, k] = math.cos(i / math.pow(10000, (k - 1) / d_model))
        positional_encoding = positional_encoding.unsqueeze(0) # (1, max_length, d_model)
    elif num_dims == 2:
        positional_encoding_2d = PositionalEncoding2D(positional_encoding_dim).to(device)
        positional_encoding = torch.zeros((1, maxlen, maxlen, positional_encoding_dim))
        positional_encoding = positional_encoding_2d(positional_encoding.to(device))
    return positional_encoding  # (1, max_length, d_model) or (1, max_length, max_length, d_model)

def get_tensor_positional_encoding(device, d_model: int, positional_encoding_dim: int, learnable_positional_encoding: bool, maxlen: int):
    positional_encoding = get_buffered_positional_encoding(
        device,
        d_model,
        positional_encoding_dim,
        maxlen=maxlen + 1,
    ).to(device)
    positional_encoding.requires_grad = learnable_positional_encoding
    return positional_encoding
