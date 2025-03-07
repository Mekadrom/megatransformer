from torch import nn

from . import sparse_moe, transformer_utils

class PositionWiseFCNetwork(nn.Module):
    def __init__(self, model_config):
        super(PositionWiseFCNetwork, self).__init__()

        ffn_config = model_config.ffn_config

        self.d_model = model_config.d_model
        self.dropout = model_config.dropout

        self.ffn_type = ffn_config.ffn_type
        self.d_inner = ffn_config.d_inner
        self.activation_function = ffn_config.activation_function
        self.ffn_bias = ffn_config.ffn_bias

        self.activation = transformer_utils.create_activation_function(self.d_inner, self.activation_function)
        self.dropout = nn.Dropout(self.dropout)
        
        self.expand: nn.Module
        if self.ffn_type == 'sparse':
            self.expand = sparse_moe.SparseMoE(model_config)
        else:
            self.expand = nn.Linear(self.d_model, self.d_inner, bias=self.ffn_bias)
        self.condense = nn.Linear(self.d_inner, self.d_model, bias=self.ffn_bias)

    def forward(self, sequences, *args):
        if type(self.expand) == nn.Linear:
            sequences = self.expand(sequences)
            gating_variances = None
        else:
            sequences, gating_variances = self.expand(sequences)

        sequences = self.activation(sequences)
        sequences = self.dropout(sequences)

        sequences = self.condense(sequences)

        sequences = self.dropout(sequences)

        return sequences, gating_variances
