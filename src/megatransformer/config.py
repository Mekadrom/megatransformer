from torch import nn
from typing import Literal, Optional

import yaml

class ConfigDict:
    def __repr__(self):
        return yaml.dump(self)
    
    @classmethod
    def to_yaml(cls, dumper, data):
        cleaned_data = dict((k, v) for (k, v) in data.__dict__.items() if v is not None)
        return dumper.represent_mapping(cls.yaml_tag, cleaned_data)

class AttentionConfig(ConfigDict, yaml.YAMLObject):
    yaml_tag = u'!AttentionConfig'

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
                 attn_impl: Literal["mha", "gqa", "infinite"] = 'mha',
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
        self.attn_impl = attn_impl
        self.infinite_attention_n_segments = infinite_attention_n_segments
        self.infinite_attention_update = infinite_attention_update
        self.use_grok_scaled_attn = use_grok_scaled_attn

class FFNConfig(ConfigDict, yaml.YAMLObject):
    yaml_tag = u'!FFNConfig'

    def __init__(self,
                 ffn_type: Literal["millions", "phi3", "simple", "sparse"],
                 d_inner: int,
                 moe_replace: bool = False,
                 moe_n_experts: int = 32,
                 moe_top_k: int = 2,
                 millions_moe_n_heads: int = 1,
                 millions_moe_d_keys: int = 128,
                 millions_moe_dropout: float = 0.0,
                 activation_function: str = 'gelu',
                 ffn_bias: bool = False):
        self.ffn_type = ffn_type
        self.d_inner = d_inner

        self.moe_replace = moe_replace
        self.moe_n_experts = moe_n_experts
        self.moe_top_k = moe_top_k
        self.millions_moe_n_heads = millions_moe_n_heads
        self.millions_moe_d_keys = millions_moe_d_keys
        self.millions_moe_dropout = millions_moe_dropout
        self.activation_function = activation_function
        self.ffn_bias = ffn_bias

class EncoderDecoderConfig(ConfigDict, yaml.YAMLObject):
    yaml_tag = u'!EncoderDecoderConfig'

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
                 m_independent_layers: int = 1,
                 embed_scale: float = 1.0,
                 pre_self_attn_norm: bool = False,
                 post_self_attn_norm: bool = False,
                 pre_cross_attn_norm: bool = False,
                 post_cross_attn_norm: bool = False,
                 pre_ffn_norm: bool = False,
                 post_ffn_norm: bool = True,
                 moe_diversity_loss_coefficient: float = 0.0,
                 n_huginn_prelude_layers: Optional[int] = None,
                 n_huginn_thinking_layers: Optional[int] = None,
                 n_huginn_coda_layers: Optional[int] = None,
                 mean_huginn_thinking_steps: Optional[int] = None,
                 mean_huginn_backprop_depth: Optional[int] = None,
                 huginn_thought_initialization_method: Optional[Literal["none", "zero", "normal", "embed", "like-init", "unit"]] = None,
                 huginn_adapter_method: Optional[Literal["add", "gate", "linear"]] = None,
                 huginn_exit_criteria: Optional[Literal["kl_divergence"]] = None,
                 huginn_exit_criteria_threshold: Optional[float] = None):
        self.device = device
        self.vocab_size = vocab_size
        self.n_layers = n_layers

        self.embedding_compression_dim = embedding_compression_dim
        self.per_lang_embedding_layers = per_lang_embedding_layers
        self.embedding_activation = embedding_activation
        self.param_sharing_type = param_sharing_type
        self.m_independent_layers = m_independent_layers
        self.embed_scale = embed_scale
        self.pre_self_attn_norm = pre_self_attn_norm
        self.post_self_attn_norm = post_self_attn_norm
        self.pre_cross_attn_norm = pre_cross_attn_norm
        self.post_cross_attn_norm = post_cross_attn_norm
        self.pre_ffn_norm = pre_ffn_norm
        self.post_ffn_norm = post_ffn_norm

        self.moe_diversity_loss_coefficient = moe_diversity_loss_coefficient

        self.n_huginn_prelude_layers = n_huginn_prelude_layers
        self.n_huginn_thinking_layers = n_huginn_thinking_layers
        self.n_huginn_coda_layers = n_huginn_coda_layers
        self.mean_huginn_thinking_steps = mean_huginn_thinking_steps
        self.mean_huginn_backprop_depth = mean_huginn_backprop_depth
        self.huginn_thought_initialization_method = huginn_thought_initialization_method
        self.huginn_adapter_method = huginn_adapter_method
        self.huginn_exit_criteria = huginn_exit_criteria
        self.huginn_exit_criteria_threshold = huginn_exit_criteria_threshold

        self.self_attn_config = self_attn_config
        self.cross_attn_config = cross_attn_config

    def load_config(self, config):
        self.self_attn_config = AttentionConfig(**config['self_attn_config'])
        if 'cross_attn_config' in config:
            self.cross_attn_config = AttentionConfig(**config['cross_attn_config'])

class TransformerConfig(ConfigDict, yaml.YAMLObject):
    yaml_tag = u'!TransformerConfig'

    def __init__(self,
                 ignore_token_id: int = -100,
                 encoder_config: Optional[EncoderDecoderConfig] = None,
                 decoder_config: Optional[EncoderDecoderConfig] = None,
                 tokenizer: Optional[str] = None,
                 ffn_config: Optional[FFNConfig] = None,
                 maxlen: Optional[int] = None,
                 d_model: Optional[int] = None,
                 dropout: Optional[float] = None,
                 use_admin: bool = False,
                 positional_encoding_type: Literal["sinusoidal", "rotary", "alibi"] = 'rotary',
                 positional_encoding_dim: int = 64,
                 learnable_positional_encoding: bool = False,
                 tie_embeddings: bool = False,
                 label_smoothing: float = 0.0,
                 norm_eps: float = 1e-5,
                 norm = nn.LayerNorm,
                 init_weights_from: str = 'glorot_uniform',
                 init_weights_gain: float = 1.0):
        self.ignore_token_id = ignore_token_id

        self.maxlen = maxlen
        self.d_model = d_model
        self.dropout = dropout

        self.use_admin = use_admin
        self.positional_encoding_type = positional_encoding_type
        self.positional_encoding_dim = positional_encoding_dim
        self.learnable_positional_encoding = learnable_positional_encoding
        self.tie_embeddings = tie_embeddings
        self.label_smoothing = label_smoothing
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

        if 'encoder_config' in config:
            self.encoder_config = EncoderDecoderConfig(**config['encoder_config'])
            self.encoder_config.load_config(config['encoder_config'])
        self.decoder_config = EncoderDecoderConfig(**config['decoder_config'])
        self.decoder_config.load_config(config['decoder_config'])

        self.ffn_config = FFNConfig(**config['ffn_config'])

        print(f"Loaded config from {path}")

if __name__ == '__main__':
    config = TransformerConfig(ignore_token_id=-100)
    config.load_yaml('configs/causal/aiayn.yaml')
    with open('configs/causal/test.yaml', 'w') as f:
        yaml.dump(config, f)
    print(config)
