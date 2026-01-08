import argparse
import json
import numpy as np
import os
import psutil
import random

try:
    import deepspeed
except ImportError:
    deepspeed = None

import torch
import torch.nn as nn

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from transformers import set_seed as hf_set_seed


BEGIN_AUDIO_TOKEN = "<|AUDIO|>"
END_AUDIO_TOKEN = "<|/AUDIO|>"

BEGIN_IMAGE_TOKEN = "<|IMAGE|>"
END_IMAGE_TOKEN = "<|/IMAGE|>"

BEGIN_VOICE_TOKEN = "<|VOICE|>"
END_VOICE_TOKEN = "<|/VOICE|>"


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
    argparser.add_argument('--warmup_ratio', type=float, default=0.0, help='Warmup ratio')
    argparser.add_argument('--warmup_steps', type=int, default=0, help='Warmup steps')
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

    # audio processing parameters
    argparser.add_argument('--audio_n_fft', type=int, default=None, help='FFT window size for audio processing. Overrides config value if set.')
    argparser.add_argument('--audio_hop_length', type=int, default=None, help='Hop length for audio processing. Overrides config value if set.')
    argparser.add_argument('--audio_sample_rate', type=int, default=None, help='Sample rate for audio processing. Overrides config value if set.')

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

    # overrides default of false in set_seed_everywhere
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark enabled.")

    return args, unk

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
    if deepspeed is not None:
        if isinstance(model, deepspeed.runtime.engine.DeepSpeedEngine):
            return model.module
    return model
