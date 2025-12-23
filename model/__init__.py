import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Union

from . import activations, causal, kv_cache, norms
from utils import megatransformer_utils


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


class Sum(nn.Module):
    def __init__(self):
        super(Sum, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class Mult(nn.Module):
    def __init__(self):
        super(Mult, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y


class SinusoidalPositionEmbeddings(nn.Module):
    """Time step embeddings for diffusion models."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0, dtype=time.dtype)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, dtype=time.dtype, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        # Handle odd dimensions
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1, 0, 0))

        return embeddings


class SimpleBlock(nn.Module):
    def __init__(self, config, name, n_layers: int, dropout: float):
        super().__init__()
        self.config = config
        self.name = name
        self.transformer = nn.ModuleList([causal.MegaTransformerBlock(config) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        past_key_values: list[kv_cache.KVCache]=None,
        use_cache=False,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        all_hidden_states: Optional[list] = [] if output_hidden_states else None
        all_attentions: Optional[list] = [] if output_attentions else None

        for i, block in enumerate(self.transformer):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if not self.training and all_hidden_states is not None:
                all_hidden_states.append(hidden_states)

            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i] if head_mask is not None else None,
                past_key_values=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states
            )

            if not return_dict:
                hidden_states = outputs[0]
                attention_probs = outputs[2]
            else:
                hidden_states = outputs.hidden_states
                attention_probs = outputs.attention_probs

            if not self.training and all_attentions is not None:
                all_attentions.append(attention_probs)

        if not self.training and all_hidden_states is not None:
            all_hidden_states.append(hidden_states)

        hidden_states = self.dropout(hidden_states)

        if not return_dict:
            return (
                hidden_states,
                past_key_values,
                all_hidden_states,
                all_attentions,
            )

        return megatransformer_utils.MegaTransformerCausalOutput(
            logits=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class AvgMaxAdaptivePool2d(nn.Module):
    def __init__(self, output_size: Union[int, tuple[int, int]]=(1, 1)):
        super(AvgMaxAdaptivePool2d, self).__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        elif isinstance(output_size, tuple) and len(output_size) == 1:
            output_size = (output_size[0], output_size[0])
        elif not isinstance(output_size, tuple) or len(output_size) != 2:
            raise ValueError("output_size must be an int or a tuple of two ints.")
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = F.adaptive_avg_pool2d(x, self.output_size)
        max_pool = F.adaptive_max_pool2d(x, self.output_size)
        # Concatenate along the channel dimension
        return torch.cat((avg_pool, max_pool), dim=1)


def create_sinusoidal_1d_pos_encoding(max_position_embeddings, hidden_size):
    positional_encoding = torch.zeros((max_position_embeddings, hidden_size))  # (max_length, d_model)
    for i in range(max_position_embeddings):
        for k in range(hidden_size):
            if k % 2 == 0:
                positional_encoding[i, k] = math.sin(i / math.pow(10000, k / hidden_size))
            else:
                positional_encoding[i, k] = math.cos(i / math.pow(10000, (k - 1) / hidden_size))
    return positional_encoding.unsqueeze(0)  # (1, max_length, d_model)
