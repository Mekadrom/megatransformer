from typing import Tuple
from torch import nn

import math
import numpy as np
import torch
import torch.nn.functional as F


"""
Adapted from https://github.com/facebookresearch/XLM/blob/main/PKM-layer.ipynb and https://web3.arxiv.org/abs/2407.04153
"""

def get_uniform_keys(n_keys, dim, seed):
    """
    Generate random uniform keys (same initialization as nn.Linear).
    """
    rng = np.random.RandomState(seed)
    bound = 1 / math.sqrt(dim)
    keys = rng.uniform(-bound, bound, (n_keys, dim))
    return keys.astype(np.float32)

class MillionsMoE(nn.Module):
    def __init__(self, args, activation=nn.ReLU):
        super(MillionsMoE, self).__init__()

        self.args = args

        self.query_cast = nn.Linear(args.d_model, args.millions_moe_n_heads * args.millions_moe_d_keys)

        self.w_down_embed = nn.Embedding(num_embeddings=args.moe_n_experts**2, embedding_dim=args.d_model)
        self.w_up_embed = nn.Embedding(num_embeddings=args.moe_n_experts**2, embedding_dim=args.d_model)

        nn.init.normal_(self.w_down_embed.weight, mean=0, std=args.d_model ** -0.5)
        nn.init.normal_(self.w_up_embed.weight, mean=0, std=args.d_model ** -0.5)

        self.activation = activation()

        self.input_dropout = nn.Dropout(args.millions_moe_input_dropout)
        self.query_dropout = nn.Dropout(args.millions_moe_query_dropout)
        self.value_dropout = nn.Dropout(args.millions_moe_value_dropout)

        self.initialize_keys()

    def initialize_keys(self):
        """
        Create two subkey sets per head.
        `self.keys` is of shape (heads, 2, n_keys, self.args.millions_moe_d_keys // 2)
        """

        half = self.args.millions_moe_d_keys // 2
        keys = nn.Parameter(torch.from_numpy(np.array([
            get_uniform_keys(self.args.moe_n_experts, half, seed=(2 * i + j))
            for i in range(self.args.millions_moe_n_heads)
            for j in range(2)
        ])).view(self.args.millions_moe_n_heads, 2, self.args.moe_n_experts, half))
        self.keys = nn.Parameter(keys)

    def get_indices_for_head(self, query_head: torch.Tensor, sub_keys) -> Tuple[torch.Tensor]:
        assert query_head.dim() == 2 and query_head.size(1) == self.args.millions_moe_d_keys

        N = query_head.size(0)

        half = self.args.millions_moe_d_keys // 2

        # split query for product quantization
        q1 = query_head[:, :half] # (N, half)
        q2 = query_head[:, half:] # (N, half)

        # compute indices with associated scores
        scores1 = F.linear(q1, sub_keys[0], bias=None) # (N, n_keys)
        scores2 = F.linear(q2, sub_keys[1], bias=None) # (N, n_keys)

        scores1, indices1 = scores1.topk(self.args.moe_top_k, dim=1) # (N, moe_top_k)
        scores2, indices2 = scores2.topk(self.args.moe_top_k, dim=1) # (N, moe_top_k)

        # cartesian product on best candidate keys
        all_scores = (
            scores1.view(N, self.args.moe_top_k, 1).expand(N, self.args.moe_top_k, self.args.moe_top_k) +
            scores2.view(N, 1, self.args.moe_top_k).expand(N, self.args.moe_top_k, self.args.moe_top_k)
        ).view(N, -1) # (N, moe_top_k**2)
        all_indices = (
            indices1.view(N, self.args.moe_top_k, 1).expand(N, self.args.moe_top_k, self.args.moe_top_k) * self.args.moe_n_experts +
            indices2.view(N, 1, self.args.moe_top_k).expand(N, self.args.moe_top_k, self.args.moe_top_k)
        ).view(N, -1) # (N, moe_top_k**2)

        # select best scores with associated indices
        scores, best_indices = torch.topk(all_scores, k=self.args.moe_top_k, dim=1) # (N, moe_top_k)
        indices = all_indices.gather(1, best_indices) # (N, moe_top_k)

        assert scores.shape == indices.shape == (N, self.args.moe_top_k)
        
        return scores, indices

    def get_indices(self, queries: torch.Tensor):
        N, T, D = queries.shape

        queries = self.input_dropout(queries) # (N, T, D)

        query_heads = self.query_cast(queries.contiguous().view(-1, D)) # (N*T, H*K)
        query_heads = self.query_dropout(query_heads.view(N*T*self.args.millions_moe_n_heads, self.args.millions_moe_d_keys)) # (N*T*H, K)

        assert query_heads.shape == (N*T*self.args.millions_moe_n_heads, self.args.millions_moe_d_keys) # (N*T*H, K)

        query_heads = query_heads.view(-1, self.args.millions_moe_n_heads, self.args.millions_moe_d_keys)

        N = len(query_heads)

        outputs = [self.get_indices_for_head(query_heads[:, i], self.keys[i]) for i in range(self.args.millions_moe_n_heads)]

        s = torch.cat([s.view(N, 1, self.args.moe_top_k) for s, _ in outputs], 1) # (N, H, topk)
        i = torch.cat([i.view(N, 1, self.args.moe_top_k) for _, i in outputs], 1) # (N, H, topk)

        return s.view(-1, self.args.moe_top_k), i.view(-1, self.args.moe_top_k)

    def peer_forward(self, queries: torch.Tensor) -> torch.Tensor:
        scores, indices = self.get_indices(queries)

        w_down = self.w_down_embed(indices)
        w_up = self.w_up_embed(indices)

        queries = torch.einsum("...d,...hkd->...hk", queries, w_down)
        queries = self.activation(queries)
        values = queries * F.softmax(scores, dim=-1)

        return torch.einsum("...hk,...hkd->...d", values, w_up)
    
    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        assert queries.shape[-1] == self.args.d_model

        N, T, D = queries.shape

        values = self.peer_forward(queries).view(N, T, D)

        return self.value_dropout(values), None
