import torch
import torch.nn as nn
import transformer_utils

class Phi3MLP(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.gate_up_proj = nn.Linear(args.d_model, 2 * args.d_inner, bias=False)
        self.down_proj = nn.Linear(args.d_inner, args.d_model, bias=False)

        self.activation_fn = transformer_utils.create_activation_function(args.d_inner, args.activation_function)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        up_states = self.gate_up_proj(hidden_states)

        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)

        return self.down_proj(up_states)
