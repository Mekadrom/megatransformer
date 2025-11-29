from torch import nn

from model import activations

import megatransformer_utils

class SimpleFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.expand = nn.Linear(config.hidden_size, config.intermediate_size)
        self.condense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        activation_type = megatransformer_utils.get_activation_type(config.intermediate_activation)
        if activation_type == activations.SwiGLU:
            self.activation = activations.SwiGLU(config.intermediate_size)
        else:
            self.activation = activation_type()
    
    def forward(self, hidden_states):
        hidden_states = self.expand(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.condense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
