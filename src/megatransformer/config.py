from torch import nn
from typing import Literal, Optional

import yaml

class ConfigDict:
    def __repr__(self):
        return str(self.__dict__)

class AttentionConfig(ConfigDict):
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

class FFNConfig(ConfigDict):
    def __init__(self,
                 ffn_type: Literal["millions", "phi3", "simple", "sparse"],
                 d_inner: int,
                 moe_replace: bool = False,
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

class EncoderDecoderConfig(ConfigDict):
    def __init__(self,
                 device,
                 vocab_size: int,
                 n_layers: int,
                 self_attn_config: AttentionConfig,
                 cross_attn_config: Optional[AttentionConfig] = None,
                 embedding_compression_dim: int = 0,
                 per_lang_embedding_layers: int = 0,
                 embedding_activation: str = 'none',
                 param_sharing_type: Literal["none", "all", "cycle-rev", "cycle", "ffn-cycle-rev", "heads-cycle-rev"] = 'none',
                 m_independent_layers: int = 1):
        self.device = device
        self.vocab_size = vocab_size
        self.n_layers = n_layers

        self.embedding_compression_dim = embedding_compression_dim
        self.per_lang_embedding_layers = per_lang_embedding_layers
        self.embedding_activation = embedding_activation
        self.param_sharing_type = param_sharing_type
        self.m_independent_layers = m_independent_layers

        self.self_attn_config = self_attn_config
        self.cross_attn_config = cross_attn_config

    def load_config(self, config):
        self.self_attn_config = AttentionConfig(**config['self_attn_config'])
        if 'cross_attn_config' in config:
            self.cross_attn_config = AttentionConfig(**config['cross_attn_config'])

class TransformerConfig(ConfigDict):
    def __init__(self,
                 encoder_config: Optional[EncoderDecoderConfig] = None,
                 decoder_config: Optional[EncoderDecoderConfig] = None,
                 tokenizer: Optional[str] = None,
                 ffn_config: Optional[FFNConfig] = None,
                 maxlen: Optional[int] = None,
                 d_model: Optional[int] = None,
                 dropout: Optional[float] = None,
                 use_admin: bool = False,
                 positional_encoding_type: Literal["sinusoidal", "rotary"] = 'rotary',
                 positional_encoding_dim: int = 64,
                 learnable_positional_encoding: bool = False,
                 tie_embeddings: bool = False,
                 padding_value: int = 0,
                 norm_eps: float = 1e-5,
                 norm = nn.LayerNorm,
                 init_weights_from: str = 'glorot_uniform',
                 init_weights_gain: float = 1.0):
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
        self.init_weights_from = init_weights_from
        self.init_weights_gain = init_weights_gain

        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.ffn_config = ffn_config
        self.tokenizer = tokenizer
        
    def load_yaml(self, path):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        self.__dict__.update(config)

        self.encoder_config = EncoderDecoderConfig(**config['encoder_config'])
        self.encoder_config.load_config(config['encoder_config'])
        self.decoder_config = EncoderDecoderConfig(**config['decoder_config'])
        self.decoder_config.load_config(config['decoder_config'])

        self.ffn_config = FFNConfig(**config['ffn_config'])

        print(f"Loaded config from {path}")