from torch import nn

import torch

class SparseMoE(nn.Module):
    def __init__(self, model_config):
        super(SparseMoE, self).__init__()

        ffn_config = model_config.ffn_config

        self.d_model = model_config.d_model

        self.d_inner = ffn_config.d_inner
        self.n_experts = ffn_config.moe_n_experts
        self.top_k = ffn_config.moe_top_k
        self.ffn_bias = ffn_config.ffn_bias

        self.expert_weights = nn.ModuleList([nn.Linear(self.d_model, self.d_inner, bias=self.ffn_bias) for _ in range(self.n_experts)])
        self.gating = nn.Linear(self.d_model, self.n_experts, bias=self.ffn_bias)
        self.softmax = nn.Softmax(dim=-1)
        
        self.init_weights()

    def init_weights(self):
        for expert in self.expert_weights:
            nn.init.xavier_uniform_(expert.weight)
            nn.init.zeros_(expert.bias)

    def forward(self, sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        N, P, D = sequences.shape

        # merge batch and sequence dimensions
        flat_sequences = sequences.view(-1, D) # (N * pad_length, d_model)
        gating_scores = self.softmax(self.gating(flat_sequences))

        top_k_indices = torch.topk(gating_scores, self.top_k, dim=1).indices

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
        output /= self.top_k

        return output.view(N, P, -1), gating_variances
