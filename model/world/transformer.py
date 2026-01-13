import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rotary_embedding_torch import RotaryEmbedding

from model import activations
from utils import configuration
from utils.megatransformer_utils import transformer_weight_init
from utils.model_utils import create_norm, get_activation_type


def create_alibi_bias(n_heads, maxlen):
    slopes = torch.pow(2, -torch.arange(1, n_heads + 1) * 8 / n_heads)
    # Create position differences matrix
    pos = torch.arange(maxlen)
    diff = pos.unsqueeze(-1) - pos.unsqueeze(-2)  # [seq_len, seq_len]
    # Calculate bias for each head
    bias = -torch.abs(diff).unsqueeze(0) * slopes.unsqueeze(-1).unsqueeze(-1)
    return bias  # [n_heads, seq_len, seq_len]


class MegaTransformerCausalSelfAttention(nn.Module):
    def __init__(self, config: configuration.TransformerBlockConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.hidden_size = config.d_model
        self.use_grok_scaled_attn = config.use_grok_scaled_attn

        self.d_queries = config.d_queries
        self.d_values = config.d_values
        self.n_query_groups = config.n_query_groups
        self.n_heads = config.n_heads

        # GQA: queries have full n_heads, keys/values have fewer n_query_groups (shared across query heads)
        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.d_queries, bias=config.use_qkv_bias)
        self.k_proj = nn.Linear(config.d_model, self.n_query_groups * self.d_queries, bias=config.use_qkv_bias)
        self.v_proj = nn.Linear(config.d_model, self.n_query_groups * self.d_values, bias=config.use_qkv_bias)

        # Number of times to repeat K/V heads to match Q heads
        assert self.n_heads % self.n_query_groups == 0, \
            f"n_heads ({self.n_heads}) must be divisible by n_query_groups ({self.n_query_groups})"
        self.n_rep = self.n_heads // self.n_query_groups

        self.heads_activation = None
        if config.heads_activation is not None:
            activation_type = get_activation_type(config.heads_activation)
            if activation_type == activations.SwiGLU:
                self.heads_activation = activations.SwiGLU(config.d_values)
            else:
                self.heads_activation = activation_type()

        self.rotary_embedding = None
        if bool(config.use_rotary_embedding):
            self.rotary_embedding = RotaryEmbedding(dim=config.rotary_embedding_dim, learned_freq=config.rotary_embedding_learnable)

        if bool(config.use_alibi_bias):
            self.register_buffer('alibi_bias', create_alibi_bias(n_heads=config.n_heads, maxlen=config.max_position_embeddings))
        else:
            self.register_buffer('alibi_bias', None)
        
        self.o_proj = nn.Linear(self.n_heads * self.d_values, config.d_model)
        
        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.proj_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        max_positions = config.max_position_embeddings

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_positions, max_positions)).view(1, 1, max_positions, max_positions),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor=None,
        head_mask: torch.Tensor=None,
    ) -> torch.Tensor:
        N, _ = hidden_states.shape[:2]

        queries: torch.Tensor = self.q_proj(hidden_states)
        keys: torch.Tensor = self.k_proj(hidden_states)
        values: torch.Tensor = self.v_proj(hidden_states)

        if self.heads_activation is not None:
            queries = self.heads_activation(queries)
            keys = self.heads_activation(keys)
            values = self.heads_activation(values)

        # Reshape to head format before applying RoPE
        # Q: (N, seq, n_heads * d_queries) -> (N, n_heads, seq, d_queries)
        # K: (N, seq, n_query_groups * d_queries) -> (N, n_query_groups, seq, d_queries)
        # V: (N, seq, n_query_groups * d_values) -> (N, n_query_groups, seq, d_values)
        queries = queries.view(N, -1, self.n_heads, self.d_queries).permute(0, 2, 1, 3).contiguous()
        keys = keys.view(N, -1, self.n_query_groups, self.d_queries).permute(0, 2, 1, 3).contiguous()
        values = values.view(N, -1, self.n_query_groups, self.d_values).permute(0, 2, 1, 3).contiguous()

        # Apply RoPE after reshaping to heads (operates on last dim which is d_queries)
        if self.rotary_embedding is not None:
            queries = self.rotary_embedding.rotate_queries_or_keys(queries)
            keys = self.rotary_embedding.rotate_queries_or_keys(keys)

        # Expand K/V heads to match Q heads for GQA
        # (N, n_query_groups, seq, d) -> (N, n_heads, seq, d)
        if self.n_rep > 1:
            keys = keys.repeat_interleave(self.n_rep, dim=1)
            values = values.repeat_interleave(self.n_rep, dim=1)

        attention_scores = torch.matmul(queries, keys.transpose(-1, -2))

        _, _, t = queries.shape[:3]
        _, _, T = keys.shape[:3]

        if self.alibi_bias is not None:
            attention_scores = attention_scores + self.alibi_bias[:, :t, :T].unsqueeze(0).repeat(N, 1, 1, 1)

        attention_scores = attention_scores / math.sqrt(self.d_queries)
        if bool(self.use_grok_scaled_attn):
            attention_scores = 30.0 * torch.tanh(attention_scores / 30.0)

        max_seq_len = max(t, T)
        if max_seq_len > min(self.causal_mask.shape[-1], self.causal_mask.shape[-2]):
            # recalculate causal mask with new longest length
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len),
            )

        causal_mask_slice = self.causal_mask[:, :, :t, :T].to(attention_scores.device)
        attention_scores = attention_scores.masked_fill(causal_mask_slice == 0, float("-inf"))
        
        if attention_mask is not None:
            # HuggingFace uses 1 for tokens to attend to and 0 for masked tokens
            # which is the opposite of the original transformer implementation
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float("-inf"))
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        context_layer = torch.matmul(attention_probs, values)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.d_values*self.n_heads,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Apply output projection
        output = self.o_proj(context_layer)
        output = self.proj_dropout(output)

        return output


class SimpleFFN(nn.Module):
    def __init__(self, config: configuration.TransformerBlockConfig):
        super().__init__()
        self.expand = nn.Linear(config.d_model, config.d_inner)
        self.condense = nn.Linear(config.d_inner, config.d_model)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        activation_type = get_activation_type(config.activation_function)
        if activation_type == activations.SwiGLU:
            self.activation = activations.SwiGLU(config.d_inner)
        else:
            self.activation = activation_type()
    
    def forward(self, hidden_states):
        hidden_states = self.expand(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.condense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class MegaTransformerBlock(nn.Module):
    """Causal Self Attention followed by FFN with optional pre/post layer norms."""
    def __init__(self, config: configuration.TransformerBlockConfig):
        super().__init__()
        self.self_attn = MegaTransformerCausalSelfAttention(config)

        self.ffn: nn.Module
        if config.ffn_type == "mlp":
            self.ffn = SimpleFFN(config)
        else:
            raise ValueError(f"Unknown ffn_type: {config.ffn_type}")

        if config.pre_attn_norm:
            self.pre_attn_norm = create_norm(config.d_model, config.norm_type, config.norm_eps)
        else:
            self.pre_attn_norm = None
        if config.post_attn_norm:
            self.post_attn_norm = create_norm(config.d_model, config.norm_type, config.norm_eps)
        else:
            self.post_attn_norm = None
        if config.pre_ffn_norm:
            self.pre_ffn_norm = create_norm(config.d_model, config.norm_type, config.norm_eps)
        else:
            self.pre_ffn_norm = None
        if config.post_ffn_norm:
            self.post_ffn_norm = create_norm(config.d_model, config.norm_type, config.norm_eps)
        else:
            self.post_ffn_norm = None

        self._init_weights()

    def _init_weights(self):
        self.apply(transformer_weight_init())
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
    ) -> torch.Tensor:
        if self.pre_attn_norm is not None:
            attn_input = self.pre_attn_norm(hidden_states)
        else:
            attn_input = hidden_states

        attn_outputs = self.self_attn(
            attn_input,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )

        hidden_states = hidden_states + attn_outputs

        if self.post_attn_norm is not None:
            hidden_states = self.post_attn_norm(hidden_states)

        if self.pre_ffn_norm is not None:
            ffn_input = self.pre_ffn_norm(hidden_states)
        else:
            ffn_input = hidden_states
        
        ffn_output = self.ffn(ffn_input)

        hidden_states = hidden_states + ffn_output

        if self.post_ffn_norm is not None:
            hidden_states = self.post_ffn_norm(hidden_states)

        return hidden_states
