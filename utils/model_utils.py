import math

import torch
import torch.nn as nn

from model import activations, norms


def get_activation_type(activation_function_name):
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
    elif activation_function_name == 'tanh':
        return nn.Tanh
    elif activation_function_name == 'sigmoid':
        return nn.Sigmoid
    elif activation_function_name == 'swiglu':
        return activations.SwiGLU
    elif activation_function_name == 'snake':
        return activations.Snake
    elif activation_function_name == 'none':
        return nn.Identity
    else:
        raise Exception(f"Unknown activation function {activation_function_name}")


def create_norm(hidden_size, norm_type, norm_eps):
    if norm_type == "layernorm":
        return nn.LayerNorm(hidden_size, eps=norm_eps)
    elif norm_type == "rmsnorm":
        return norms.RMSNorm(hidden_size, eps=norm_eps)
    else:
        raise Exception(f"Unknown normalization type {norm_type}")


def create_sinusoidal_1d_pos_encoding(max_position_embeddings, hidden_size):
    positional_encoding = torch.zeros((max_position_embeddings, hidden_size))  # (max_length, d_model)
    for i in range(max_position_embeddings):
        for k in range(hidden_size):
            if k % 2 == 0:
                positional_encoding[i, k] = math.sin(i / math.pow(10000, k / hidden_size))
            else:
                positional_encoding[i, k] = math.cos(i / math.pow(10000, (k - 1) / hidden_size))
    return positional_encoding.unsqueeze(0)  # (1, max_length, d_model)
