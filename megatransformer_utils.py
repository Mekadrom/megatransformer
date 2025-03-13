from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import PretrainedConfig
from transformers import set_seed as hf_set_seed
from typing import Optional

from model import rmsnorm, swiglu

import math
import numpy as np
import os
import random
import torch
import torch.nn as nn


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
        intermediate_size=3072,
        intermediate_activation="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,

        huginn_mean_thinking_steps=32,
        huginn_backprop_depth=8,
        huginn_thought_initialization_method="like-init",
        huginn_adapter_method="linear",
        huginn_exit_criteria="kl_divergence",
        huginn_exit_criteria_threshold=5e-4,
        huginn_lockstep_n=True,
        huginn_lockstep_k=True,

        norm_type="layernorm",
        norm_eps=1e-5,
        use_qkv_bias=True,
        use_hidden_bias=True,
        ffn_type="mlp",

        pre_attn_norm=True,
        post_attn_norm=False,
        pre_ffn_norm=True,
        post_ffn_norm=False,

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
        **kwargs
    ):
        super().__init__(**kwargs)
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
        self.intermediate_size = intermediate_size
        self.intermediate_activation = intermediate_activation
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        # huginn specific
        self.huginn_mean_thinking_steps = huginn_mean_thinking_steps
        self.huginn_backprop_depth = huginn_backprop_depth
        self.huginn_thought_initialization_method = huginn_thought_initialization_method
        self.huginn_adapter_method = huginn_adapter_method
        self.huginn_exit_criteria = huginn_exit_criteria
        self.huginn_exit_criteria_threshold = huginn_exit_criteria_threshold
        self.huginn_lockstep_n = huginn_lockstep_n
        self.huginn_lockstep_k = huginn_lockstep_k

        self.norm_type = norm_type
        self.norm_eps = norm_eps
        
        self.use_qkv_bias = use_qkv_bias
        self.use_hidden_bias = use_hidden_bias
        
        self.ffn_type = ffn_type

        self.pre_attn_norm = pre_attn_norm
        self.post_attn_norm = post_attn_norm
        self.pre_ffn_norm = pre_ffn_norm
        self.post_ffn_norm = post_ffn_norm

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

def create_alibi_bias(n_heads, maxlen):
    slopes = torch.pow(2, -torch.arange(1, n_heads + 1) * 8 / n_heads)
    # Create position differences matrix
    pos = torch.arange(maxlen)
    diff = pos.unsqueeze(-1) - pos.unsqueeze(-2)  # [seq_len, seq_len]
    # Calculate bias for each head
    bias = -torch.abs(diff).unsqueeze(0) * slopes.unsqueeze(-1).unsqueeze(-1)
    return bias  # [n_heads, seq_len, seq_len]

def create_sinusoidal_embedding(max_position_embeddings, hidden_size):
    positional_encoding = torch.zeros((max_position_embeddings, hidden_size)) # (max_length, d_model)
    for i in range(max_position_embeddings):
        for k in range(hidden_size):
            if k % 2 == 0:
                positional_encoding[i, k] = math.sin(i / math.pow(10000, k / hidden_size))
            else:
                positional_encoding[i, k] = math.cos(i / math.pow(10000, (k - 1) / hidden_size))
    return positional_encoding.unsqueeze(0) # (1, max_length, d_model)

def get_activation_function(activation_function_name):
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
        return swiglu.SwiGLU
    elif activation_function_name == 'none':
        return nn.Identity
    else:
        raise Exception(f"Unknown activation function {activation_function_name}")

def create_norm(config):
    if config.norm_type == "layernorm":
        return nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
    elif config.norm_type == "rmsnorm":
        return rmsnorm.RMSNorm(config.hidden_size, eps=config.norm_eps)
    else:
        raise Exception(f"Unknown normalization type {config.norm_type}")

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
