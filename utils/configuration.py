from dataclasses import dataclass
from typing import Optional
from transformers import PretrainedConfig


class MegaTransformerConfig(PretrainedConfig):
    model_type = "megatransformer"
    
    def __init__(
        self,
        include_modes=["text", "audio", "image"],
        vocab_size=50257,
        max_position_embeddings=1024,
        hidden_size=768,
        n_layers=12,
        n_prelude_layers=None,
        n_recurrent_layers=None,
        n_coda_layers=None,
        d_queries=64,
        d_values=64,
        n_query_groups=12,
        n_heads=12,
        heads_activation=None,
        intermediate_size=3072,
        intermediate_activation="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,

        recurrent_mean_thinking_steps=32,
        recurrent_backprop_depth=8,
        recurrent_thought_initialization_method="like-init",
        recurrent_adapter_method="linear",
        recurrent_exit_criteria="kl_divergence",
        recurrent_exit_criteria_threshold=5e-4,
        recurrent_lockstep_n=True,
        recurrent_lockstep_k=True,

        norm_type="layernorm",
        norm_eps=1e-5,
        use_qkv_bias=True,
        use_hidden_bias=True,
        ffn_type="mlp",

        pre_attn_norm=True,
        post_attn_norm=False,
        pre_ffn_norm=True,
        post_ffn_norm=False,
        use_final_norm=True,

        use_positional_embedding=True,

        use_sinusoidal_embedding=False,
        sinusoidal_embedding_learnable=True,

        use_rotary_embedding=None,
        rotary_embedding_learnable=False,
        rotary_embedding_dim=64,

        use_alibi_bias=False,

        use_grok_scaled_attn=False,
        initializer_range=0.02,
        use_cache=True,
        tie_word_embeddings=True,
        pad_token_id=50256,
        bos_token_id=50256,
        eos_token_id=50256,

        # multimodal specific
        begin_audio_token_id=50257,
        end_audio_token_id=50258,
        begin_image_token_id=50259,
        end_image_token_id=50260,
        begin_voice_token_id=50261,
        end_voice_token_id=50262,

        text_prelude_config=None,

        audio_prelude_config=None,

        audio_n_mels=80,
        audio_n_fft=1024,
        audio_hop_length=256,
        audio_max_duration=30.0, # used for trimming data/skipping examples that are too long
        audio_sample_rate=16000,
        image_size=256,

        audio_encoder_base_channels=32,
        audio_encoder_kernel_sizes=[3, 3, 3, 3, 3, 3],
        audio_encoder_norm_type="layernorm",
        audio_encoder_norm_eps=1e-5,
        audio_encoder_activation="relu",
        audio_encoder_dropout=0.1,

        image_prelude_config=None,

        image_encoder_patch_size=16,
        image_encoder_norm_type="layernorm",
        image_encoder_norm_eps=1e-5,
        image_encoder_activation="relu",
        image_encoder_pos_dropout=0.1,

        text_coda_config=None,

        audio_coda_config=None,

        audio_decoder_model_channels=128,
        audio_decoder_time_embedding_dim=128,
        audio_decoder_attention_levels=[False, False, True, True],
        audio_decoder_num_res_blocks=4,
        audio_decoder_activation="silu",
        audio_decoder_dropout=0.1,

        audio_decoder_unet_dropout_p=0.1,
        audio_decoder_betas_schedule="linear",
        audio_decoder_down_block_self_attn_n_heads=8,
        audio_decoder_down_block_self_attn_d_queries=64,
        audio_decoder_down_block_self_attn_d_values=64,
        audio_decoder_down_block_self_attn_use_flash_attention=True,
        audio_decoder_up_block_self_attn_n_heads=8,
        audio_decoder_up_block_self_attn_d_queries=64,
        audio_decoder_up_block_self_attn_d_values=64,
        audio_decoder_up_block_self_attn_use_flash_attention=True,
        audio_decoder_cross_attn_n_heads=8,
        audio_decoder_cross_attn_d_queries=64,
        audio_decoder_cross_attn_d_values=64,
        audio_decoder_cross_attn_use_flash_attention=True,

        audio_vocoder_hidden_channels=2048,
        audio_vocoder_upsample_factors=[8, 8, 4],
        audio_vocoder_n_residual_layers=4,

        image_coda_config=None,

        image_decoder_model_channels=128,
        image_decoder_time_embedding_dim=128,
        image_decoder_attention_levels=[False, False, True, True],
        image_decoder_num_res_blocks=4,
        image_decoder_activation="silu",
        image_decoder_dropout=0.1,

        image_decoder_unet_dropout_p=0.1,
        image_decoder_betas_schedule="linear",
        image_decoder_down_block_self_attn_n_heads=8,
        image_decoder_down_block_self_attn_d_queries=64,
        image_decoder_down_block_self_attn_d_values=64,
        image_decoder_down_block_self_attn_use_flash_attention=True,
        image_decoder_up_block_self_attn_n_heads=8,
        image_decoder_up_block_self_attn_d_queries=64,
        image_decoder_up_block_self_attn_d_values=64,
        image_decoder_up_block_self_attn_use_flash_attention=True,
        image_decoder_cross_attn_n_heads=8,
        image_decoder_cross_attn_d_queries=64,
        image_decoder_cross_attn_d_values=64,
        image_decoder_cross_attn_use_flash_attention=True,
        image_decoder_channel_multipliers=[2, 4, 8],

        **kwargs
    ):
        super().__init__(**kwargs)

        self.include_modes = include_modes
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_prelude_layers = n_prelude_layers
        self.n_recurrent_layers = n_recurrent_layers
        self.n_coda_layers = n_coda_layers
        self.d_queries = d_queries
        self.d_values = d_values
        self.n_query_groups = n_query_groups
        self.n_heads = n_heads
        self.heads_activation = heads_activation
        self.intermediate_size = intermediate_size
        self.intermediate_activation = intermediate_activation
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        # recurrent specific
        self.recurrent_mean_thinking_steps = recurrent_mean_thinking_steps
        self.recurrent_backprop_depth = recurrent_backprop_depth
        self.recurrent_thought_initialization_method = recurrent_thought_initialization_method
        self.recurrent_adapter_method = recurrent_adapter_method
        self.recurrent_exit_criteria = recurrent_exit_criteria
        self.recurrent_exit_criteria_threshold = recurrent_exit_criteria_threshold
        self.recurrent_lockstep_n = recurrent_lockstep_n
        self.recurrent_lockstep_k = recurrent_lockstep_k

        self.norm_type = norm_type
        self.norm_eps = norm_eps
        
        self.use_qkv_bias = use_qkv_bias
        self.use_hidden_bias = use_hidden_bias
        
        self.ffn_type = ffn_type

        self.pre_attn_norm = pre_attn_norm
        self.post_attn_norm = post_attn_norm
        self.pre_ffn_norm = pre_ffn_norm
        self.post_ffn_norm = post_ffn_norm
        self.use_final_norm = use_final_norm

        self.use_positional_embedding = use_positional_embedding

        self.use_sinusoidal_embedding = use_sinusoidal_embedding
        self.sinusoidal_embedding_learnable = sinusoidal_embedding_learnable

        self.use_rotary_embedding = use_rotary_embedding
        self.rotary_embedding_learnable = rotary_embedding_learnable
        self.rotary_embedding_dim = rotary_embedding_dim

        self.use_alibi_bias = use_alibi_bias

        self.use_grok_scaled_attn = use_grok_scaled_attn
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.begin_audio_token_id = begin_audio_token_id
        self.end_audio_token_id = end_audio_token_id
        self.begin_image_token_id = begin_image_token_id
        self.end_image_token_id = end_image_token_id
        self.begin_voice_token_id = begin_voice_token_id
        self.end_voice_token_id = end_voice_token_id

        self.text_prelude_config = text_prelude_config

        self.audio_prelude_config = audio_prelude_config

        self.audio_n_mels = audio_n_mels
        self.audio_n_fft = audio_n_fft
        self.audio_hop_length = audio_hop_length
        self.audio_max_duration = audio_max_duration
        self.audio_max_waveform_length = round(self.audio_max_duration * audio_sample_rate)
        self.audio_max_frames = round(self.audio_max_waveform_length / audio_hop_length)
        self.audio_sample_rate = audio_sample_rate

        self.audio_encoder_base_channels = audio_encoder_base_channels
        self.audio_encoder_kernel_sizes = audio_encoder_kernel_sizes
        self.audio_encoder_norm_type = audio_encoder_norm_type
        self.audio_encoder_norm_eps = audio_encoder_norm_eps
        self.audio_encoder_activation = audio_encoder_activation
        self.audio_encoder_dropout = audio_encoder_dropout

        self.image_size = image_size

        self.image_prelude_config = image_prelude_config

        self.image_encoder_patch_size = image_encoder_patch_size
        self.image_encoder_norm_type = image_encoder_norm_type
        self.image_encoder_norm_eps = image_encoder_norm_eps
        self.image_encoder_activation = image_encoder_activation
        self.image_encoder_pos_dropout = image_encoder_pos_dropout

        self.text_coda_config = text_coda_config

        self.audio_coda_config = audio_coda_config

        self.audio_decoder_activation = audio_decoder_activation
        self.audio_decoder_model_channels = audio_decoder_model_channels
        self.audio_decoder_time_embedding_dim = audio_decoder_time_embedding_dim
        self.audio_decoder_attention_levels = audio_decoder_attention_levels
        self.audio_decoder_num_res_blocks = audio_decoder_num_res_blocks
        self.audio_decoder_dropout = audio_decoder_dropout

        self.audio_decoder_unet_dropout_p = audio_decoder_unet_dropout_p
        self.audio_decoder_betas_schedule = audio_decoder_betas_schedule
        self.audio_decoder_down_block_self_attn_n_heads = audio_decoder_down_block_self_attn_n_heads
        self.audio_decoder_down_block_self_attn_d_queries = audio_decoder_down_block_self_attn_d_queries
        self.audio_decoder_down_block_self_attn_d_values = audio_decoder_down_block_self_attn_d_values
        self.audio_decoder_down_block_self_attn_use_flash_attention = audio_decoder_down_block_self_attn_use_flash_attention
        self.audio_decoder_up_block_self_attn_n_heads = audio_decoder_up_block_self_attn_n_heads
        self.audio_decoder_up_block_self_attn_d_queries = audio_decoder_up_block_self_attn_d_queries
        self.audio_decoder_up_block_self_attn_d_values = audio_decoder_up_block_self_attn_d_values
        self.audio_decoder_up_block_self_attn_use_flash_attention = audio_decoder_up_block_self_attn_use_flash_attention
        self.audio_decoder_cross_attn_n_heads = audio_decoder_cross_attn_n_heads
        self.audio_decoder_cross_attn_d_queries = audio_decoder_cross_attn_d_queries
        self.audio_decoder_cross_attn_d_values = audio_decoder_cross_attn_d_values
        self.audio_decoder_cross_attn_use_flash_attention = audio_decoder_cross_attn_use_flash_attention            

        self.audio_vocoder_hidden_channels = audio_vocoder_hidden_channels
        self.audio_vocoder_upsample_factors = audio_vocoder_upsample_factors
        self.audio_vocoder_n_residual_layers = audio_vocoder_n_residual_layers

        self.image_coda_config = image_coda_config

        self.image_decoder_activation = image_decoder_activation
        self.image_decoder_model_channels = image_decoder_model_channels
        self.image_decoder_time_embedding_dim = image_decoder_time_embedding_dim
        self.image_decoder_attention_levels = image_decoder_attention_levels
        self.image_decoder_num_res_blocks = image_decoder_num_res_blocks
        self.image_decoder_dropout = image_decoder_dropout

        self.image_decoder_unet_dropout_p = image_decoder_unet_dropout_p
        self.image_decoder_betas_schedule = image_decoder_betas_schedule
        self.image_decoder_down_block_self_attn_n_heads = image_decoder_down_block_self_attn_n_heads
        self.image_decoder_down_block_self_attn_d_queries = image_decoder_down_block_self_attn_d_queries
        self.image_decoder_down_block_self_attn_d_values = image_decoder_down_block_self_attn_d_values
        self.image_decoder_down_block_self_attn_use_flash_attention = image_decoder_down_block_self_attn_use_flash_attention
        self.image_decoder_up_block_self_attn_n_heads = image_decoder_up_block_self_attn_n_heads
        self.image_decoder_up_block_self_attn_d_queries = image_decoder_up_block_self_attn_d_queries
        self.image_decoder_up_block_self_attn_d_values = image_decoder_up_block_self_attn_d_values
        self.image_decoder_up_block_self_attn_use_flash_attention = image_decoder_up_block_self_attn_use_flash_attention
        self.image_decoder_cross_attn_n_heads = image_decoder_cross_attn_n_heads
        self.image_decoder_cross_attn_d_queries = image_decoder_cross_attn_d_queries
        self.image_decoder_cross_attn_d_values = image_decoder_cross_attn_d_values
        self.image_decoder_cross_attn_use_flash_attention = image_decoder_cross_attn_use_flash_attention
        self.image_decoder_channel_multipliers = image_decoder_channel_multipliers

        self.current_epoch = 0
        self.current_global_step = 0


@dataclass
class TransformerBlockConfig(PretrainedConfig):
    d_model: int
    n_heads: int
    d_queries: int
    d_values: int
    n_query_groups: int
    d_inner: int
    n_layers: int
    heads_activation: Optional[str] = None
    use_qkv_bias: bool = True
    use_rotary_embedding: bool = False
    rotary_embedding_dim: int = 64
    rotary_embedding_learnable: bool = False
    use_grok_scaled_attn: bool = False
    use_alibi_bias: bool = False
    max_position_embeddings: int = 1024
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    norm_type: str = "layernorm"
    norm_eps: float = 1e-5
    ffn_type: str = "mlp"
    pre_attn_norm: bool = True
    post_attn_norm: bool = False
    pre_ffn_norm: bool = True
    post_ffn_norm: bool = False
    activation_function: str = "gelu"


@dataclass
class AudioConfig:
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    sample_rate: int = 16000
    max_audio_duration: float = 30.0  # in seconds
    latent_channels: int = 8
    latent_compression_factor: tuple[int, int] = (8, 12)


@dataclass
class ImageConfig:
    image_size: int = 256
    latent_channels: int = 4
    latent_compression_factor: int = 8
    latent_patch_size: int = 2
