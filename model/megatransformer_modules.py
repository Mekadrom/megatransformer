from model import megatransformer_blocks
from typing import Optional

import megatransformer_utils
import torch
import torch.nn as nn
import torch.nn.functional as F


class Sum(nn.Module):
    def __init__(self):
        super(Sum, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y

class Mult(nn.Module):
    def __init__(self):
        super(Mult, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_sqr = x**2
        RMS = torch.rsqrt(x_sqr.mean(dim = -1, keepdim = True) + self.eps)
        new_x = x * RMS
        new_x = new_x * self.weight

        return new_x

class SwiGLU(nn.Module):
    def __init__(self, d_in):
        super(SwiGLU, self).__init__()

        self.cast = nn.Linear(d_in // 2, d_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        x = F.silu(gate) * x
        x = self.cast(x)
        return x

class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1, 0, 0))

        return embeddings

class SimpleSelfAttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        h = self.norm(x)
        
        h = h.reshape(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        
        h, _ = self.attention(h, h, h)
        
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        
        return x + h

class SimpleCrossAttentionBlock(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        
        # Projections
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, self.inner_dim, bias=False)
        
        self.to_out = nn.Linear(self.inner_dim, query_dim)
        
    def forward(self, x: torch.Tensor, context):
        batch_size, _, _ = x.shape
        h = self.heads
        
        q = self.to_q(x).reshape(batch_size, -1, h, self.inner_dim // h).permute(0, 2, 1, 3)
        k = self.to_k(context).reshape(batch_size, -1, h, self.inner_dim // h).permute(0, 2, 1, 3)
        v = self.to_v(context).reshape(batch_size, -1, h, self.inner_dim // h).permute(0, 2, 1, 3)
        
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, -1, self.inner_dim)
        
        return self.to_out(out)

class SimpleBlock(nn.Module):
    def __init__(self, config, n_layers: int, dropout: float):
        super().__init__()
        self.config = config
        self.prelude = nn.ModuleList([megatransformer_blocks.MegaTransformerBlock(config) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        past_key_values: list[megatransformer_utils.KVCache]=None,
        use_cache=False,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        all_hidden_states: Optional[list] = [] if output_hidden_states else None
        all_attentions: Optional[list] = [] if output_attentions else None

        for i, block in enumerate(self.prelude):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)

            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i] if head_mask is not None else None,
                past_key_values=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states
            )

            hidden_states = outputs.hidden_states
            attention_probs = outputs.attention_probs

            if all_attentions is not None:
                all_attentions.append(attention_probs)

        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)

        hidden_states = self.dropout(hidden_states)

        if not return_dict:
            return (
                hidden_states,
                past_key_values,
                all_hidden_states,
                all_attentions,
            )

        return megatransformer_utils.MegaTransformerCausalOutput(
            logits=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
