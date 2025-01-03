from torch import nn

import sparse_moe
import transformer_utils

class PositionWiseFCNetwork(nn.Module):
    def __init__(self, args, norm=nn.LayerNorm):
        super(PositionWiseFCNetwork, self).__init__()

        self.args = args

        self.layer_norm = norm(args.d_model, args.norm_eps)
        self.activation = transformer_utils.create_activation_function(args.d_inner, args.activation_function)
        self.dropout = nn.Dropout(args.dropout)
        
        if args.fcn_type == 'simple':
            self.expand = sparse_moe.SparseMoE(args)
        else:
            self.expand = nn.Linear(args.d_model, args.d_inner, bias=bool(args.fcn_bias))

        self.condense = nn.Linear(args.d_inner, args.d_model, bias=bool(args.fcn_bias))

    def forward(self, sequences, *args):
        sequences = self.layer_norm(sequences)

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
