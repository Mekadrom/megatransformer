import torch
import torch.nn as nn
import transformer_utils

class Phi3MLP(nn.Module):
    def __init__(self, model_config, ffn_config):
        super().__init__()

        self.d_model = model_config.d_model
        self.d_inner = ffn_config.d_inner
        self.activation_function = ffn_config.activation_function

        self.gate_up_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=False)
        self.down_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        self.activation_fn = transformer_utils.create_activation_function(self.d_inner, self.activation_function)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        up_states = self.gate_up_proj(hidden_states)

        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)

        return self.down_proj(up_states)
