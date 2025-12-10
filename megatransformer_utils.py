from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from positional_encodings.torch_encodings import PositionalEncodingPermute2D
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from transformers import PretrainedConfig
from transformers import set_seed as hf_set_seed
from transformers.trainer_callback import TrainerCallback
from typing import Optional

import argparse
import deepspeed
import glob
import json
import math
import numpy as np
import os
import psutil
import random
import torch
import torch.distributed as dist
import torch.nn as nn

from model import activations, norms


BEGIN_AUDIO_TOKEN = "<|AUDIO|>"
END_AUDIO_TOKEN = "<|/AUDIO|>"

BEGIN_IMAGE_TOKEN = "<|IMAGE|>"
END_IMAGE_TOKEN = "<|/IMAGE|>"

BEGIN_VOICE_TOKEN = "<|VOICE|>"
END_VOICE_TOKEN = "<|/VOICE|>"

class KVCache:
    def __init__(self):
        self.key: Optional[torch.Tensor] = None
        self.value: Optional[torch.Tensor] = None

    def reset(self):
        self.key: Optional[torch.Tensor] = None
        self.value: Optional[torch.Tensor] = None

    def update(self, key: torch.Tensor, value: torch.Tensor):
        if self.key is None:
            self.key = key
        else:
            self.key = torch.cat([self.key, key], dim=2)
        if self.value is None:
            self.value = value
        else:
            self.value = torch.cat([self.value, value], dim=2)

    def __getitem__(self, idx):
        if idx == 0:
            return self.key
        elif idx == 1:
            return self.value
        else:
            raise IndexError(f"KVCache index out of range: {idx}")
        
    def size(self):
        return {
            "keys_shape": self.key.shape if self.key is not None else None,
            "values_shape": self.value.shape if self.value is not None else None
        }
    
    def __deepspeed_tensor_attributes__(self):
        return ['key', 'value']

class PreAllocatedKVCache:
    def __init__(self, max_length, batch_size, n_heads, d_queries, d_values, dtype=torch.float32, device='cuda'):
        self.key = torch.zeros((batch_size, n_heads, max_length, d_queries), dtype=dtype, device=device)
        self.value = torch.zeros((batch_size, n_heads, max_length, d_values), dtype=dtype, device=device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values

        self.position = 0

    def reset(self):
        self.key = torch.zeros((self.batch_size, self.n_heads, self.max_length, self.d_queries), dtype=self.key.dtype)
        self.value = torch.zeros((self.batch_size, self.n_heads, self.max_length, self.d_values), dtype=self.value.dtype)

    def update(self, key: torch.Tensor, value: torch.Tensor):
        if self.position + key.shape[2] > self.max_length:
            raise ValueError(f"Cannot update KVCache: position {self.position} + key length {key.shape[2]} exceeds max length {self.max_length}")

        self.key[:, :, self.position:self.position + key.shape[2], :] = key
        self.value[:, :, self.position:self.position + value.shape[2], :] = value
        self.position += key.shape[2]

    def __getitem__(self, idx):
        """Return slice that goes until the current position, non-inclusive."""
        if idx == 0:
            return self.key[:, :, :self.position, :]
        elif idx == 1:
            return self.value[:, :, :self.position, :]
        else:
            raise IndexError(f"PreAllocatedKVCache index out of range: {idx}")
        
    def size(self):
        return {
            "keys_shape": self.key.shape,
            "values_shape": self.value.shape
        }
    
    def __deepspeed_tensor_attributes__(self):
        return ['key', 'value']


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

        self.current_epoch = 0
        self.current_global_step = 0

class MegaTransformerCausalOutput(dict):
    def __init__(self,
        loss: Optional[torch.Tensor]=None,
        logits: Optional[torch.Tensor]=None,
        past_key_values: Optional[list[KVCache]]=None,
        hidden_states: Optional[list]=None,
        attentions: Optional[list]=None,
        n_steps_no_grad: Optional[int]=None,
        k_steps_grad: Optional[int]=None,
    ):
        self.loss = loss
        self.logits = logits
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions
        self.n_steps_no_grad = n_steps_no_grad
        self.k_steps_grad = k_steps_grad

        super().__init__(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions,
            n_steps_no_grad=n_steps_no_grad,
            k_steps_grad=k_steps_grad,
        )

class BlockOutput:
    def __init__(self, hidden_states, past_key_values: KVCache=None, attention_probs=None):
        self.hidden_states = hidden_states
        self.past_key_values = past_key_values
        self.attention_probs = attention_probs

class MegaTransformerMultimodalOutput(dict):
    def __init__(self,
        loss: Optional[torch.Tensor]=None,
        logits: Optional[torch.Tensor]=None,
        image_raw_outputs: Optional[torch.Tensor]=None,
        audio_raw_outputs: Optional[torch.Tensor]=None,
        past_key_values: Optional[list[KVCache]]=None,
        hidden_states: Optional[list]=None,
        attentions: Optional[list]=None,
        n_steps_no_grad: Optional[int]=None,
        k_steps_grad: Optional[int]=None,
    ):
        self.loss = loss
        self.logits = logits
        self.image_raw_outputs = image_raw_outputs
        self.audio_raw_outputs = audio_raw_outputs
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions
        self.n_steps_no_grad = n_steps_no_grad
        self.k_steps_grad = k_steps_grad

        super().__init__(
            loss=loss,
            logits=logits,
            image_raw_outputs=image_raw_outputs,
            audio_raw_outputs=audio_raw_outputs,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions,
            n_steps_no_grad=n_steps_no_grad,
            k_steps_grad=k_steps_grad,
        )

class MultimodalGenerationOutput:
    """Output class for multimodal generation results."""
    def __init__(self, sequences=None, audio_outputs=None, audio_mel_specs=None, image_outputs=None, intermediate_image_outputs=None):
        self.sequences = sequences
        self.audio_outputs = audio_outputs
        self.audio_mel_specs = audio_mel_specs
        self.image_outputs = image_outputs
        self.intermediate_image_outputs = intermediate_image_outputs

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, stop_step: int):
        self.stop_step = stop_step

    def on_step_begin(self, args, state, control, **kwargs):
        if self.stop_step > 0 and state.global_step >= self.stop_step:
            print(f"Early stopping at step {state.global_step} as per stop_step={self.stop_step}.")
            control.should_training_stop = True


def create_alibi_bias(n_heads, maxlen):
    slopes = torch.pow(2, -torch.arange(1, n_heads + 1) * 8 / n_heads)
    # Create position differences matrix
    pos = torch.arange(maxlen)
    diff = pos.unsqueeze(-1) - pos.unsqueeze(-2)  # [seq_len, seq_len]
    # Calculate bias for each head
    bias = -torch.abs(diff).unsqueeze(0) * slopes.unsqueeze(-1).unsqueeze(-1)
    return bias  # [n_heads, seq_len, seq_len]

def create_sinusoidal_2d_pos_encoding(channels):
    return PositionalEncodingPermute2D(channels)

def create_sinusoidal_1d_pos_encoding(max_position_embeddings, hidden_size):
    positional_encoding = torch.zeros((max_position_embeddings, hidden_size)) # (max_length, d_model)
    for i in range(max_position_embeddings):
        for k in range(hidden_size):
            if k % 2 == 0:
                positional_encoding[i, k] = math.sin(i / math.pow(10000, k / hidden_size))
            else:
                positional_encoding[i, k] = math.cos(i / math.pow(10000, (k - 1) / hidden_size))
    return positional_encoding.unsqueeze(0) # (1, max_length, d_model)

def get_activation_type(activation_function_name):
    if activation_function_name == 'relu':
        return nn.ReLU
    elif activation_function_name == 'gelu':
        return nn.GELU
    elif activation_function_name == 'elu':
        return nn.ELU
    elif activation_function_name == 'selu':
        return nn.SELU
    elif activation_function_name == 'prelu':
        return nn.PReLU
    elif activation_function_name == 'leaky_relu':
        return nn.LeakyReLU
    elif activation_function_name == 'silu':
        return nn.SiLU
    elif activation_function_name == 'tanh':
        return nn.Tanh
    elif activation_function_name == 'sigmoid':
        return nn.Sigmoid
    elif activation_function_name == 'swiglu':
        return activations.SwiGLU
    elif activation_function_name == 'snake':
        return activations.Snake
    elif activation_function_name == 'none':
        return nn.Identity
    else:
        raise Exception(f"Unknown activation function {activation_function_name}")

def create_norm(hidden_size, norm_type, norm_eps):
    if norm_type == "layernorm":
        return nn.LayerNorm(hidden_size, eps=norm_eps)
    elif norm_type == "rmsnorm":
        return norms.RMSNorm(hidden_size, eps=norm_eps)
    else:
        raise Exception(f"Unknown normalization type {norm_type}")

# function definitions for args/initialization shared by pretrain and finetune scripts
def check_tpu_availability():
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        
        device = xm.xla_device()
        print(f"TPU is available! Device: {device}")
        
        tpu_cores = xm.xrt_world_size()
        print(f"Number of TPU cores: {tpu_cores}")
        print(f"TPU type: {torch_xla._XLAC._get_tpu_type()}")
        
        return True
    except (ImportError, EnvironmentError, RuntimeError) as e:
        print(f"TPU is not available: {e}")
        return False

def set_seed_everywhere(seed: int):
    """Set seed for all random number generators."""
    # Python's built-in random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch (CPU and CUDA)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        
    # Make PyTorch operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # HuggingFace Transformers (this sets seeds for all random generators in transformers)
    hf_set_seed(seed)
    
    # For good measure, set the Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_process_cmdline(pid):
    """
    Retrieves the command line arguments for a process given its PID.
    Returns a list of strings representing the command line, or None if the process is not found.
    """
    try:
        process = psutil.Process(pid)
        return process.cmdline()
    except psutil.NoSuchProcess:
        return None

def parse_args():
    is_tpu_available = check_tpu_availability()
    print(f"TPU available: {is_tpu_available}")

    argparser = argparse.ArgumentParser()

    # meta params
    argparser.add_argument('--seed', type=int, default=42, help='Random seed')
    argparser.add_argument('--logging_base_dir', type=str, default=os.path.join('runs', 'causal'), help='Base directory for logging')
    argparser.add_argument('--run_name', type=str, help='Name of the run', required=True)
    argparser.add_argument('--include_modes', type=str, default='text,audio,image', help='Comma-separated list of modes to include (e.g., text,audio,image or audio,image), order agnostic')
    argparser.add_argument('--dataset_cache_dir', type=str, default='cached_datasets', help='Path to the dataset cache directory')
    argparser.add_argument('--tokenizer_name', type=str, default="mistralai/Mistral-7B-v0.1", help='Tokenizer name')
    argparser.add_argument('--trainer', type=str, default="default", help='Trainer type: grokfast_ema, grokfast_ma, debug, or default')
    argparser.add_argument('--config', type=str, default="modern", help='Model configuration.')
    argparser.add_argument('--max_position_embeddings', type=int, default=4096, help='Max position embeddings (maximum sequence length)')
    argparser.add_argument('--cpu', action='store_true', help='Use CPU for training')
    argparser.add_argument('--log_level', type=str, default='warning', help='Logging level: debug, info, warning, error, critical')
    argparser.add_argument('--resume_from_checkpoint', type=str, help='Resume from checkpoint at this path')
    argparser.add_argument('--start_step', type=int, default=None, help='Start step for training')
    argparser.add_argument('--lr_scheduler_type', type=str, default='cosine', help='Learning rate scheduler type')
    argparser.add_argument('--lr_scheduler_kwargs', type=str, default=None, help='Additional kwargs for LR scheduler as a JSON stringified dict')

    # efficiency params
    argparser.add_argument('--compile_model', action='store_true', help='Whether to compile the model')
    argparser.add_argument('--cudnn_benchmark', action='store_true', help='Whether to enable cuDNN benchmark')
    argparser.add_argument('--use_gradient_checkpointing', action='store_true', help='Whether to use gradient checkpointing')
    argparser.add_argument('--use_xla', action='store_true', default=is_tpu_available, help='Whether to use XLA')

    # generic hyperparams
    argparser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    argparser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    argparser.add_argument('--num_train_epochs', type=int, default=-1, help='Number of training epochs')
    argparser.add_argument('--max_steps', type=int, default=-1, help='Max steps for training')
    argparser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    argparser.add_argument('--gradient_accumulation_steps', type=int, default=32, help='Gradient accumulation steps')
    argparser.add_argument('--warmup_ratio', type=float, default=0.03, help='Warmup ratio')
    argparser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    argparser.add_argument('--fp16', action='store_true', help='Whether to use fp16')
    argparser.add_argument('--bf16', action='store_true', help='Whether to use bf16')

    # grokfast hyperparams
    argparser.add_argument('--grokfast_ema_alpha', type=float, default=0.98, help='Alpha for GrokFast EMA trainer')
    argparser.add_argument('--grokfast_ema_lambda', type=float, default=2.0, help='Lambda for GrokFast EMA trainer')
    argparser.add_argument('--grokfast_ma_window_size', type=int, default=100, help='Window size for GrokFast MA trainer')
    argparser.add_argument('--grokfast_ma_lambda', type=float, default=5.0, help='Lambda for GrokFast MA trainer')
    argparser.add_argument('--grokfast_ma_filter_type', type=str, default='mean', help='Filter type for GrokFast MA trainer')
    argparser.add_argument('--grokfast_ma_warmup', action='store_true', help='Whether to use warmup for GrokFast MA trainer')

    # deepspeed
    argparser.add_argument('--use_deepspeed', action='store_true', help='Whether to use DeepSpeed')
    argparser.add_argument('--deepspeed_config', type=str, default=None, help='DeepSpeed configuration file')
    argparser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')

    # peft lora/int8 training
    argparser.add_argument('--use_int8_peft', action='store_true', help='Use INT8 with PEFT/LoRA')
    argparser.add_argument('--use_int8_deepspeed', action='store_true', help='Use DeepSpeed INT8 quantization')
    argparser.add_argument('--lora_rank', type=int, default=16, help='Rank for LoRA adaptation')
    argparser.add_argument('--lora_alpha', type=int, default=32, help='Alpha for LoRA adaptation')
    argparser.add_argument('--lora_dropout', type=float, default=0.05, help='Dropout for LoRA adaptation')

    # logging
    argparser.add_argument('--logging_steps', type=int, default=100, help='Logging steps')
    argparser.add_argument('--eval_steps', type=int, default=1000, help='Evaluation steps')
    argparser.add_argument('--save_steps', type=int, default=500, help='Save steps')
    argparser.add_argument('--generation_steps', type=int, default=1000, help='Generation steps')

    argparser.add_argument('--adam_beta1', type=float, default=0.9, help='Adam optimizer beta1')
    argparser.add_argument('--adam_beta2', type=float, default=0.999, help='Adam optimizer beta2')

    argparser.add_argument('--stop_step', type=int, default=-1, help='Step to stop training at. For preserving the LR schedule while not training further.')
    argparser.add_argument('--commit_hash', type=str, default='', help='Git commit hash for this run. Logged in tensorboard.')

    args, unk = argparser.parse_known_args()

    current_process_pid = psutil.Process().pid
    setattr(args, 'cmdline', " ".join(get_process_cmdline(current_process_pid)))

    setattr(args, 'include_modes', args.include_modes.split(','))

    if unk and len(unk) > 0:
        print(f"unknown args: {unk}")

    if args.use_xla and args.use_deepspeed:
        raise ValueError("DeepSpeed is not supported with TPU training. Please use CPU or GPU, or disable DeepSpeed.")
    
    if args.use_xla and args.use_int8_peft:
        raise ValueError("INT8 training with PEFT is not supported with TPU training. Please use CPU or GPU, or disable INT8 training.")
    
    if args.use_xla and args.fp16:
        raise ValueError("FP16 training is not supported with TPU training. Please only enable BF16, or disable FP16 training.")
    
    if args.num_train_epochs == -1 and args.max_steps == -1:
        raise ValueError("Either num_train_epochs or max_steps must be specified. Please check your configuration.")
    
    if args.num_train_epochs != -1 and args.max_steps != -1:
        print("Both num_train_epochs and max_steps are specified. max_steps will take precedence.")
        args.num_train_epochs = -1

    set_seed_everywhere(args.seed)

    if args.lr_scheduler_kwargs is None:
        args.lr_scheduler_kwargs = {}
    else:
        args.lr_scheduler_kwargs = json.loads(args.lr_scheduler_kwargs)

    # make sure deepspeed config specified exists
    if args.use_deepspeed:
        if args.deepspeed_config is not None and os.path.exists(args.deepspeed_config):
            print(f"Loading DeepSpeed config from {args.deepspeed_config}")
        else:
            raise FileNotFoundError(f"DeepSpeed config file {args.deepspeed_config} not found.")

    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark enabled.")

    return args, unk

def setup_int8_training(args, model):
    # Method 1: Using PEFT with Bits and Bytes quantization
    if args.use_int8_peft:
        print("Setting up INT8 training with PEFT/LoRA")
        
        model = prepare_model_for_kbit_training(model, args.use_gradient_checkpointing)
        
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["query", "key", "value", "dense"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    # Method 2: Using DeepSpeed ZeroQuant (configured in ds_config.json)
    elif args.use_int8_deepspeed:
        print("Using DeepSpeed for INT8 quantization during training")
        return model
    # No INT8 training
    else:
        return model

def load_model(finetune, model, run_dir):
    # load if model file exists
    if os.path.exists(run_dir):
        globbed_checkpoint_folders = glob.glob(os.path.join(run_dir, "checkpoint-*", "pytorch_model.bin"))
        # sort by step number (format checkpoint-<step>/pytorch_model.bin)
        if globbed_checkpoint_folders:
            sorted_checkpoints = sorted(globbed_checkpoint_folders, key=lambda x: int(x.split("-")[-1].split("/")[0]))
            latest_checkpoint = sorted_checkpoints[-1]
            print(f"Loading model from {latest_checkpoint}")
            try:
                model.load_state_dict(torch.load(latest_checkpoint), strict=False)
                model_loaded = True
            except RuntimeError as e:
                print(f"Error loading model: {e}. This is most likely due to a mismatch in model architecture.")
                model_loaded = False
        else:
            print(f"No checkpoints found in {run_dir}.")
            model_loaded = False
    else:
        print(f"Model directory {run_dir} does not exist.")
        model_loaded = False

    if not model_loaded:
        print("Model not loaded from checkpoint.")
        if finetune:
            raise ValueError("Fine-tuning is enabled but no checkpoint found. Please check the run directory, or your configuration.")

    return model, model_loaded


def get_token_correlation(hidden_states):
    x_c = hidden_states - hidden_states.mean(dim=1, keepdim=True)

    normed_x = x_c / x_c.norm(dim=-1, keepdim=True)

    token_correlation = (normed_x @ normed_x.transpose(1, 2)).mean() - (1 / hidden_states.shape[1])

    return token_correlation

def create_multimodal_optimizer(model, weight_decay):
    audio_decoder_params = set(model.output_transform.audio_decoder.parameters())
    vocoder_params = set(model.output_transform.audio_decoder.vocoder.parameters())

    # Get parameters unique to audio_decoder (excluding vocoder)
    audio_decoder_only_params = [p for p in audio_decoder_params if p not in vocoder_params]

    # Create AdamW optimizer with these groups
    optimizer = torch.optim.AdamW([
        {'params': model.input_transform.parameters(), 'lr': 1e-4},
        {'params': model.world_model.parameters(), 'lr': 5e-5},
        {'params': model.output_transform.text_coda.parameters(), 'lr': 1e-4},
        {'params': model.output_transform.text_decoder.parameters(), 'lr': 2e-4},
        {'params': model.output_transform.audio_coda.parameters(), 'lr': 1e-4},
        {'params': audio_decoder_only_params, 'lr': 2e-5},
        {'params': model.output_transform.audio_decoder.vocoder.parameters(), 'lr': 3e-5},
        {'params': model.output_transform.image_coda.parameters(), 'lr': 1e-4},
        {'params': model.output_transform.image_decoder.parameters(), 'lr': 2e-5},
    ], weight_decay=weight_decay)
    return optimizer

def ensure_all_grads(model):
    """Make sure all parameters have at least zero gradients"""
    for param in model.parameters():
        if param.requires_grad and param.grad is None:
            param.grad = torch.zeros_like(param)

def sync_check(message):
    """Check if all ranks can synchronize"""
    if dist.is_initialized():
        rank = dist.get_rank()
        print(f"[Rank {rank}] Before barrier: {message}")
        dist.barrier()
        print(f"[Rank {rank}] After barrier: {message}")

def log_rank_info(message):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    print(f"[Rank {rank}/{world_size}] {message}")

def debug_cuda_memory():
    """Print CUDA memory usage for current device"""
    if torch.cuda.is_available():
        rank = dist.get_rank() if dist.is_initialized() else 0
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[Rank {rank}] CUDA Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

def embedding_weight_init(hidden_size):
    def init_weights(module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=hidden_size ** -0.5)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
    return init_weights

def transformer_weight_init():
    def init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
    return init_weights

def conv2d_weight_init():
    def init_weights(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
    return init_weights

def print_debug_tensor(pre: str, tensor: torch.Tensor):
    # avoid printing on devices other than cuda:0
    if not isinstance(tensor, torch.Tensor):
        print(f"{pre}:\n\tNot a tensor {type(tensor)}")
        return
    if tensor is None:
        print(f"{pre}: None\n\t")
    elif tensor.numel() == 0:
        print(f"{pre}: empty tensor\n\tshape {tensor.shape}")
    elif tensor.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
        print(f"{pre}:\n\tptr: {tensor.data_ptr()}\n\tdtype: {tensor.dtype}\n\tdevice: {tensor.device}\n\tshape: {tensor.shape}\n\tmean: {tensor.mean()}\n\tstd: {tensor.std()}\n\tmin: {tensor.min()}\n\tmax: {tensor.max()}\n\tnorm: {tensor.norm()}\n\tany nan: {tensor.isnan().any()}\n\tany inf: {tensor.isinf().any()}")
        if tensor.numel() < 100:
            print(f"\t{tensor}")
    else:
        # non-float tensors
        print(f"{pre}:\n\tdtype: {tensor.dtype}\n\tdevice: {tensor.device}\n\tshape: {tensor.shape}\n\tmin: {tensor.min()}\n\tmax: {tensor.max()}\n\tany nan: {tensor.isnan().any()}\n\tany inf: {tensor.isinf().any()}")
        if tensor.numel() < 100:
            print(f"\t{tensor}")

def sanitize_model(model):
    if isinstance(model, DistributedDataParallel):
        return sanitize_model(model.module)
    if hasattr(model, '_orig_mod'):
        return sanitize_model(model._orig_mod)
    if isinstance(model, DataParallel):
        return sanitize_model(model.module)
    if isinstance(model, deepspeed.runtime.engine.DeepSpeedEngine):
        return model.module
    return model
