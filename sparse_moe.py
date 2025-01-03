from torch import nn

import torch

class SparseMoE(nn.Module):
    def __init__(self, args):
        super(SparseMoE, self).__init__()

        self.args = args

        self.expert_weights = nn.ModuleList([nn.Linear(args.d_model, args.d_inner, bias=bool(args.fcn_bias)) for _ in range(args.moe_n_experts)])
        self.gating = nn.Linear(args.d_model, args.moe_n_experts, bias=bool(args.fcn_bias))
        self.softmax = nn.Softmax(dim=-1)
        
        self.reset_parameters()

    def reset_parameters(self):
        for expert in self.expert_weights:
            nn.init.xavier_uniform_(expert.weight)
            nn.init.zeros_(expert.bias)

    def forward(self, sequences):
        N, P, D = sequences.shape

        # merge batch and sequence dimensions
        flat_sequences = sequences.view(-1, D) # (N * pad_length, d_model)
        gating_scores = self.softmax(self.gating(flat_sequences))

        top_k_indices = torch.topk(gating_scores, self.args.moe_top_k, dim=1).indices

        output = torch.zeros(N*P, self.expert_weights[0].out_features, device=sequences.device)

        for i in range(len(self.expert_weights)):
            expert_mask = top_k_indices == i
            expert_input = flat_sequences[expert_mask.any(dim=1)]
            expert_output = self.expert_weights[i](expert_input)

            output[expert_mask.any(dim=1)] += expert_output

        # record export choices to self.gating_variances for loss calculation to encourage diversity
        if self.training:
            gating_variances = torch.var(gating_scores, dim=0)
        else:
            gating_variances = None

        # normalize
        output /= self.args.moe_top_k

        return output.view(N, P, -1), gating_variances
