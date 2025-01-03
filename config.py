from torch import nn
from typing import Literal, Optional

class AttentionConfig:
    def __init__(self,
                 n_heads: int,
                 n_gqa_groups: int,
                 d_queries: int,
                 d_values: int,
                 q_bias: bool = False,
                 k_bias: bool = False,
                 v_bias: bool = False,
                 o_bias: bool = False,
                 heads_activation_function: Optional[str] = None,
                 use_infinite_attention: bool = False,
                 infinite_attention_n_segments: int = 16,
                 infinite_attention_update: Literal["nonlinear", "linear"] = 'linear',
                 use_grok_scaled_attn: bool = True):
        self.n_heads = n_heads
        self.n_gqa_groups = n_gqa_groups
        self.d_queries = d_queries
        self.d_values = d_values
        self.q_bias = q_bias
        self.k_bias = k_bias
        self.v_bias = v_bias
        self.o_bias = o_bias
        self.heads_activation_function = heads_activation_function
        self.use_infinite_attention = use_infinite_attention
        self.infinite_attention_n_segments = infinite_attention_n_segments
        self.infinite_attention_update = infinite_attention_update
        self.use_grok_scaled_attn = use_grok_scaled_attn

class FFNConfig:
    def __init__(self,
                 ffn_type: Literal["millions", "phi3", "simple", "sparse"],
                 d_inner: int,
                 moe_replace,
                 moe_n_experts: int = 32,
                 moe_top_k: int = 2,
                 millions_moe_n_heads: int = 1,
                 millions_moe_d_keys: int = 128,
                 millions_moe_input_dropout: float = 0.0,
                 millions_moe_query_dropout: float = 0.0,
                 millions_moe_value_dropout: float = 0.0,
                 activation_function: str = 'gelu',
                 ffn_bias: bool = False):
        self.ffn_type = ffn_type
        self.d_inner = d_inner
        self.moe_replace = moe_replace
        self.moe_n_experts = moe_n_experts
        self.moe_top_k = moe_top_k
        self.millions_moe_n_heads = millions_moe_n_heads
        self.millions_moe_d_keys = millions_moe_d_keys
        self.millions_moe_input_dropout = millions_moe_input_dropout
        self.millions_moe_query_dropout = millions_moe_query_dropout
        self.millions_moe_value_dropout = millions_moe_value_dropout
        self.activation_function = activation_function
        self.ffn_bias = ffn_bias

class EncoderDecoderConfig:
    def __init__(self,
                 self_attn_config: AttentionConfig,
                 cross_attn_config: Optional[AttentionConfig],
                 ffn_config: FFNConfig,
                 vocab_size: int,
                 n_layers: int,
                 embedding_compression_dim: int,
                 per_lang_embedding_layers: int,
                 embedding_activation: str,
                 param_sharing_type: Literal["none", "all", "cycle-rev", "cycle", "ffn-cycle-rev", "heads-cycle-rev"] = 'none',
                 m_independent_layers: int = 1):
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.embedding_compression_dim = embedding_compression_dim
        self.per_lang_embedding_layers = per_lang_embedding_layers
        self.embedding_activation = embedding_activation

        self.param_sharing_type = param_sharing_type
        self.m_independent_layers = m_independent_layers

        self.self_attn_config = self_attn_config
        self.cross_attn_config = cross_attn_config
        self.ffn_config = ffn_config

class TransformerConfig:
    def __init__(self,
                 encoder_config: EncoderDecoderConfig,
                 decoder_config: EncoderDecoderConfig,
                 maxlen: int,
                 d_model: int,
                 dropout: float,
                 use_admin: bool = False,
                 positional_encoding_type: Literal["sinusoidal", "rotary"] = 'rotary',
                 positional_encoding_dim: int = 64,
                 learnable_positional_encoding: bool = False,
                 tie_embeddings: bool = False,
                 padding_value: int = 0,
                 norm_eps: float = 1e-5,
                 norm = nn.LayerNorm):
        self.maxlen = maxlen
        self.d_model = d_model
        self.dropout = dropout

        self.use_admin = use_admin
        self.positional_encoding_type = positional_encoding_type
        self.positional_encoding_dim = positional_encoding_dim
        self.learnable_positional_encoding = learnable_positional_encoding
        self.tie_embeddings = tie_embeddings
        self.padding_value = padding_value
        self.norm_eps = norm_eps
        self.norm = norm

        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
