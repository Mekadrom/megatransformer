from dataclasses import dataclass
from torch import nn
from typing import Any, Literal, Optional

import yaml

class ConfigDict:
    def __repr__(self):
        return yaml.dump(self)
    
    @classmethod
    def to_yaml(cls, dumper, data):
        cleaned_data = dict((k, v) for (k, v) in data.__dict__.items() if v is not None)
        return dumper.represent_mapping(cls.yaml_tag, cleaned_data)

@dataclass
class AttentionConfig(ConfigDict, yaml.YAMLObject):
    yaml_tag = u'!AttentionConfig'

    n_heads: int
    n_gqa_groups: int = 1
    d_queries: int = 64
    d_values: int = 64
    q_bias: bool = False
    k_bias: bool = False
    v_bias: bool = False
    o_bias: bool = False
    heads_activation_function: Optional[str] = None
    attn_impl: Literal["mha", "gqa", "infinite"] = 'mha'
    infinite_attention_n_segments: int = 16
    infinite_attention_update: Literal["nonlinear", "linear"] = 'linear'
    use_grok_scaled_attn: bool = True

@dataclass
class FFNConfig(ConfigDict, yaml.YAMLObject):
    yaml_tag = u'!FFNConfig'

    d_inner: int
    ffn_type: Literal["millions", "phi3", "simple", "sparse"] = "simple"
    moe_replace: bool = False
    moe_n_experts: int = 0
    moe_top_k: int = 0
    millions_moe_n_heads: int = 0
    millions_moe_d_keys: int = 0
    millions_moe_dropout: float = 0.0
    activation_function: str = 'swiglu'
    ffn_bias: bool = False

@dataclass
class EncoderDecoderConfig(ConfigDict, yaml.YAMLObject):
    yaml_tag = u'!EncoderDecoderConfig'

    device: Any
    n_layers: int
    vocab_size: Optional[int] = None # default to None so it can be set later
    self_attn_config: Optional[AttentionConfig] = None
    cross_attn_config: Optional[AttentionConfig] = None
    embedding_compression_dim: int = 0
    per_lang_embedding_layers: int = 0
    embedding_activation: str = 'none'
    param_sharing_type: Literal["none", "all", "cycle-rev", "cycle", "ffn-cycle-rev", "heads-cycle-rev"] = 'none'
    m_independent_layers: int = 1
    embed_scale: float = 1.0
    pre_self_attn_norm: bool = True
    post_self_attn_norm: bool = False
    pre_cross_attn_norm: bool = False
    post_cross_attn_norm: bool = False
    pre_ffn_norm: bool = True
    post_ffn_norm: bool = False
    post_block_norm: bool = True
    moe_diversity_loss_coefficient: float = 0.0
    n_huginn_prelude_layers: Optional[int] = None
    n_huginn_thinking_layers: Optional[int] = None
    n_huginn_coda_layers: Optional[int] = None
    mean_huginn_thinking_steps: Optional[int] = None
    mean_huginn_backprop_depth: Optional[int] = None
    huginn_thought_initialization_method: Optional[Literal["none", "zero", "normal", "embed", "like-init", "unit"]] = None
    huginn_adapter_method: Optional[Literal["add", "gate", "linear"]] = None
    huginn_exit_criteria: Optional[Literal["kl_divergence"]] = None
    huginn_exit_criteria_threshold: Optional[float] = None

    def load_config(self, config):
        self.self_attn_config = AttentionConfig(**config['self_attn_config'])
        if 'cross_attn_config' in config:
            self.cross_attn_config = AttentionConfig(**config['cross_attn_config'])

@dataclass
class TransformerConfig(ConfigDict, yaml.YAMLObject):
    yaml_tag = u'!TransformerConfig'

    ignore_token_id: int = -100
    encoder_config: Optional[EncoderDecoderConfig] = None
    decoder_config: Optional[EncoderDecoderConfig] = None
    tokenizer: Optional[str] = None
    ffn_config: Optional[FFNConfig] = None
    maxlen: Optional[int] = None
    d_model: Optional[int] = None
    dropout: Optional[float] = None
    use_admin: bool = False
    positional_encoding_type: Literal["sinusoidal", "rotary", "alibi"] = 'rotary'
    positional_encoding_dim: int = 64
    learnable_positional_encoding: bool = False
    tie_embeddings: bool = False
    label_smoothing: float = 0.0
    norm_eps: float = 1e-5
    norm: nn.Module = nn.LayerNorm
    init_weights_from: str = 'glorot_uniform'
    init_weights_gain: float = 1.0
        
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
    config = TransformerConfig()
    config.load_yaml('configs/causal/aiayn.yaml')
    with open('configs/causal/test.yaml', 'w') as f:
        yaml.dump(config, f)
    print(config)
