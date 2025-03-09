from torch import nn

from . import swiglu
from .. import megatransformer_utils


class SimpleFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.expand = nn.Linear(config.hidden_size, config.intermediate_size)
        self.condense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        activation_type = megatransformer_utils.get_activation_function(config.intermediate_activation)
        if activation_type == swiglu.SwiGLU:
            self.activation = swiglu.SwiGLU(config.intermediate_size)
        else:
            self.activation = activation_type()
    
    def forward(self, hidden_states):
        hidden_states = self.expand(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.condense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class GateFFN(nn.Module):
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig):
        super().__init__()
        self.expand = nn.Linear(config.hidden_size, 2*config.intermediate_size, bias=config.use_hidden_bias)
        self.condense = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.use_hidden_bias)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        activation_type = megatransformer_utils.get_activation_function(config.intermediate_activation)
        if activation_type == swiglu.SwiGLU:
            self.activation = swiglu.SwiGLU(config.intermediate_size)
        else:
            self.activation = activation_type()

    def forward(self, hidden_states):
        hidden_states = self.expand(hidden_states)
        gate, hidden_states = hidden_states.chunk(2, dim=-1)
        hidden_states = hidden_states * self.activation(gate)
        hidden_states = self.condense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
