from torch import nn

from . import sparse_moe, transformer_utils

import copy
import torch

class Phi3MLP(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        ffn_config = model_config.ffn_config

        self.d_model = model_config.d_model
        self.dropout = model_config.dropout
        self.norm = model_config.norm
        self.norm_eps = model_config.norm_eps

        self.ffn_type = ffn_config.ffn_type
        self.d_inner = ffn_config.d_inner
        self.activation_function = ffn_config.activation_function
        self.ffn_bias = ffn_config.ffn_bias

        self.layer_norm = self.norm(self.d_model, self.norm_eps)
        self.activation = transformer_utils.create_activation_function(self.d_inner, self.activation_function)
        self.dropout = nn.Dropout(self.dropout)
        
        self.expand: nn.Module
        if self.ffn_type == 'sparse':
            sparse_ffn_config = copy.deepcopy(ffn_config)
            sparse_ffn_config.d_inner = 2 * self.d_inner
            self.expand = sparse_moe.SparseMoE(model_config)
        else:
            self.expand = nn.Linear(self.d_model, 2 * self.d_inner, bias=self.ffn_bias)
        self.condense = nn.Linear(self.d_inner, self.d_model, bias=self.ffn_bias)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        sequences = self.layer_norm(sequences)

        expanded = self.expand(sequences)

        gate, expanded = expanded.chunk(2, dim=-1)
        expanded = expanded * self.activation_function(gate)

        condensed = self.condense(expanded)

        condensed = self.dropout(condensed)

        return condensed
