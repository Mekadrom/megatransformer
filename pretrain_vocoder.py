"""
Standalone vocoder pretraining script.

This script trains a vocoder model to convert mel spectrograms to raw audio waveforms.
Uses Common Voice dataset and the existing vocoder architecture with SC, logmag, and l1 losses.

Usage:
    deepspeed --num_gpus=2 pretrain_vocoder.py \
        --use_deepspeed \
        --bf16 \
        --run_name vocoder_pretrain \
        --max_steps 100000 \
        --gradient_accumulation_steps 8 \
        --deepspeed_config ds_config_zero-2.json
"""

import os

os.environ["DEEPSPEED_UNIT_TEST"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ['NCCL_TIMEOUT'] = '1200000'

import argparse
import logging
import random

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, Audio
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed as hf_set_seed
from transformers.integrations import TensorBoardCallback

from model.megatransformer_audio_decoder import AudioVocoder, MultiResolutionSTFTLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class VocoderConfig:
    """
    Configuration for the vocoder model.

    Default values match MegaTransformerConfig and create_small_multimodal_model
    to ensure compatibility when loading pretrained vocoder into world model training.
    """
    def __init__(
        self,
        # Audio parameters (from MegaTransformerConfig defaults)
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 512,
        max_audio_duration: float = 10.0,

        # Vocoder architecture (from MegaTransformerConfig / create_small_multimodal_model)
        # Note: create_small_multimodal_model uses 1536, MegaTransformerConfig default is 2048
        hidden_channels: int = 2048,
        upsample_factors: list = None,  # Default [8, 8, 8] set below
        n_residual_layers: int = 4,
        dilation_cycle: int = 4,
        kernel_size: int = 3,
        leaky_relu_slope: float = 0.1,

        # Loss weights
        waveform_l1_weight: float = 1.0,
        sc_loss_weight: float = 1.5,
        mag_loss_weight: float = 1.5,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_audio_duration = max_audio_duration
        self.max_waveform_length = int(max_audio_duration * sample_rate)
        self.max_mel_frames = int(self.max_waveform_length / hop_length)

        self.hidden_channels = hidden_channels
        self.upsample_factors = upsample_factors or [8, 8, 8]
        self.n_residual_layers = n_residual_layers
        self.dilation_cycle = dilation_cycle
        self.kernel_size = kernel_size
        self.leaky_relu_slope = leaky_relu_slope

        self.waveform_l1_weight = waveform_l1_weight
        self.sc_loss_weight = sc_loss_weight
        self.mag_loss_weight = mag_loss_weight


# ============================================================================
# Dataset
# ============================================================================

def extract_waveform(audio, sr=16000):
    """Extract waveform from audio data."""
    if isinstance(audio, dict) and 'array' in audio:
        y = audio['array']
        orig_sr = audio['sampling_rate']
        if orig_sr != sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    elif isinstance(audio, torch.Tensor):
        y = audio.numpy()
    else:
        y, _ = librosa.load(audio, sr=sr)

    return torch.tensor(y, dtype=torch.float32)


def extract_mel(y, sr=16000, n_mels=128, n_fft=1024, hop_length=512):
    """Extract mel spectrogram from waveform."""
    if isinstance(y, torch.Tensor):
        y = y.numpy()

    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    log_mel_spec = librosa.power_to_db(mel_spec)

    return torch.tensor(log_mel_spec, dtype=torch.float32)


class CommonVoiceVocoderDataset(Dataset):
    """Dataset for vocoder training using Common Voice."""

    def __init__(
        self,
        config: VocoderConfig,
        split: str = "train",
        language: str = "en",
        cache_dir: str = None,
        max_samples: int = None,
    ):
        self.config = config
        self.split = split

        logger.info(f"Loading Common Voice dataset (language={language}, split={split})...")

        # Load Common Voice dataset
        self.dataset = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            language,
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        # Cast audio column to the target sample rate
        self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate=config.sample_rate))

        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

        logger.info(f"Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio = item["audio"]

        # Extract waveform
        waveform = extract_waveform(audio, sr=self.config.sample_rate)

        # Truncate or skip if too long
        max_len = self.config.max_waveform_length
        if len(waveform) > max_len:
            # Random crop
            start = random.randint(0, len(waveform) - max_len)
            waveform = waveform[start:start + max_len]

        # Extract mel spectrogram
        mel_spec = extract_mel(
            waveform,
            sr=self.config.sample_rate,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
        )

        return {
            "mel_spec": mel_spec,  # [n_mels, T_mel]
            "waveform": waveform,   # [T_audio]
        }


class VocoderDataCollator:
    """Data collator that pads mel spectrograms and waveforms."""

    def __init__(self, config: VocoderConfig):
        self.config = config

    def __call__(self, features):
        mel_specs = [f["mel_spec"] for f in features]
        waveforms = [f["waveform"] for f in features]

        # Pad mel spectrograms to max length in batch
        max_mel_len = max(m.shape[1] for m in mel_specs)
        padded_mels = []
        for mel in mel_specs:
            pad_len = max_mel_len - mel.shape[1]
            if pad_len > 0:
                mel = F.pad(mel, (0, pad_len), value=-80.0)  # Pad with silence (low dB)
            padded_mels.append(mel)

        # Pad waveforms to match mel length after upsampling
        expected_waveform_len = max_mel_len * self.config.hop_length
        padded_waveforms = []
        for wf in waveforms:
            pad_len = expected_waveform_len - len(wf)
            if pad_len > 0:
                wf = F.pad(wf, (0, pad_len), value=0.0)
            elif pad_len < 0:
                wf = wf[:expected_waveform_len]
            padded_waveforms.append(wf)

        return {
            "mel_spec": torch.stack(padded_mels),      # [B, n_mels, T_mel]
            "waveform": torch.stack(padded_waveforms), # [B, T_audio]
        }


# ============================================================================
# Model Wrapper
# ============================================================================

class VocoderForTraining(nn.Module):
    """Wrapper around AudioVocoder for HuggingFace Trainer compatibility."""

    def __init__(self, config: VocoderConfig):
        super().__init__()
        self.config = config

        # Create vocoder
        self.vocoder = AudioVocoder(
            hidden_channels=config.hidden_channels,
            in_channels=config.n_mels,
            conditioning_channels=config.hidden_channels,  # Not used, but required
            upsample_factors=config.upsample_factors,
            n_residual_layers=config.n_residual_layers,
            dilation_cycle=config.dilation_cycle,
            kernel_size=config.kernel_size,
            leaky_relu_slope=config.leaky_relu_slope,
            conditioning_enabled=False,  # No conditioning for standalone vocoder
        )

        # Loss functions
        self.stft_loss = MultiResolutionSTFTLoss()

        # Store loss weights
        self.waveform_l1_weight = config.waveform_l1_weight
        self.sc_loss_weight = config.sc_loss_weight
        self.mag_loss_weight = config.mag_loss_weight

    def forward(self, mel_spec, waveform, **kwargs):
        """
        Forward pass.

        Args:
            mel_spec: [B, n_mels, T_mel] - Input mel spectrogram
            waveform: [B, T_audio] - Target waveform

        Returns:
            loss: Combined loss value
            pred_waveform: Predicted waveform
        """
        # Vocoder forward
        pred_waveform = self.vocoder(mel_spec, condition=None)  # [B, T_audio]

        # Ensure same length
        min_len = min(pred_waveform.shape[-1], waveform.shape[-1])
        pred_waveform_aligned = pred_waveform[..., :min_len]
        waveform_aligned = waveform[..., :min_len]

        # L1 waveform loss
        waveform_l1 = F.l1_loss(pred_waveform_aligned, waveform_aligned)

        # Multi-resolution STFT loss (SC and log magnitude)
        # Need to add channel dimension for STFT loss
        pred_for_stft = pred_waveform_aligned.unsqueeze(1)  # [B, 1, T]
        target_for_stft = waveform_aligned.unsqueeze(1)     # [B, 1, T]
        sc_loss, mag_loss = self.stft_loss(pred_for_stft, target_for_stft)

        # Combined loss
        total_loss = (
            self.waveform_l1_weight * waveform_l1 +
            self.sc_loss_weight * sc_loss +
            self.mag_loss_weight * mag_loss
        )

        return total_loss, {
            "loss": total_loss,
            "waveform_l1": waveform_l1,
            "sc_loss": sc_loss,
            "mag_loss": mag_loss,
            "pred_waveform": pred_waveform,
        }


# ============================================================================
# Custom Trainer
# ============================================================================

class VocoderTrainer(Trainer):
    """Custom trainer for vocoder training."""

    def __init__(self, vocoder_config: VocoderConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocoder_config = vocoder_config
        self.writer = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mel_spec = inputs["mel_spec"]
        waveform = inputs["waveform"]

        loss, outputs = model(mel_spec=mel_spec, waveform=waveform)

        # Log individual losses to tensorboard
        if self.state.global_step % self.args.logging_steps == 0:
            self._ensure_tensorboard_writer()
            if self.writer is not None:
                prefix = "train/" if model.training else "eval/"
                self.writer.add_scalar(f"{prefix}waveform_l1", outputs["waveform_l1"].item(), self.state.global_step)
                self.writer.add_scalar(f"{prefix}sc_loss", outputs["sc_loss"].item(), self.state.global_step)
                self.writer.add_scalar(f"{prefix}mag_loss", outputs["mag_loss"].item(), self.state.global_step)

        return (loss, outputs) if return_outputs else loss

    def _ensure_tensorboard_writer(self):
        if self.writer is None:
            for callback in self.callback_handler.callbacks:
                if isinstance(callback, TensorBoardCallback):
                    self.writer = callback.tb_writer
                    break


class VocoderCheckpointCallback(TrainerCallback):
    """
    Callback to save vocoder weights separately for easy loading in world model training.

    Saves to: {output_dir}/vocoder_checkpoint/vocoder.pt

    The saved checkpoint contains:
        - 'vocoder_state_dict': The vocoder model weights
        - 'config': VocoderConfig parameters as a dict
        - 'step': Training step when saved
    """

    def __init__(self, config: VocoderConfig, save_every_n_steps: int = 1000):
        self.config = config
        self.save_every_n_steps = save_every_n_steps
        self.output_dir = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.output_dir = args.output_dir

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.save_every_n_steps != 0:
            return

        self._save_vocoder(model, state.global_step)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        # Always save at end of training
        self._save_vocoder(model, state.global_step, final=True)

    def _save_vocoder(self, model, step, final=False):
        if model is None or self.output_dir is None:
            return

        # Handle DeepSpeed/DDP wrapped models
        if hasattr(model, 'module'):
            vocoder = model.module.vocoder
        else:
            vocoder = model.vocoder

        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.output_dir, "vocoder_checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save vocoder weights and config
        checkpoint = {
            'vocoder_state_dict': vocoder.state_dict(),
            'config': {
                'sample_rate': self.config.sample_rate,
                'n_mels': self.config.n_mels,
                'n_fft': self.config.n_fft,
                'hop_length': self.config.hop_length,
                'hidden_channels': self.config.hidden_channels,
                'upsample_factors': self.config.upsample_factors,
                'n_residual_layers': self.config.n_residual_layers,
                'dilation_cycle': self.config.dilation_cycle,
                'kernel_size': self.config.kernel_size,
                'leaky_relu_slope': self.config.leaky_relu_slope,
            },
            'step': step,
        }

        # Save with step number and also as 'latest'
        save_path = os.path.join(checkpoint_dir, f"vocoder_step_{step}.pt")
        latest_path = os.path.join(checkpoint_dir, "vocoder.pt")

        torch.save(checkpoint, save_path)
        torch.save(checkpoint, latest_path)

        if final:
            final_path = os.path.join(checkpoint_dir, "vocoder_final.pt")
            torch.save(checkpoint, final_path)

        logger.info(f"Saved vocoder checkpoint to {save_path}")


def load_pretrained_vocoder(checkpoint_path: str, device: str = 'cuda') -> AudioVocoder:
    """
    Load a pretrained vocoder from a checkpoint saved by VocoderCheckpointCallback.

    This function is designed to be used when loading a frozen vocoder for world model training.

    Args:
        checkpoint_path: Path to vocoder checkpoint (e.g., 'runs/vocoder/my_run/vocoder_checkpoint/vocoder.pt')
        device: Device to load the model to

    Returns:
        AudioVocoder model with loaded weights, set to eval mode with frozen parameters

    Example usage in world model training:
        ```python
        from pretrain_vocoder import load_pretrained_vocoder

        vocoder = load_pretrained_vocoder('runs/vocoder/my_run/vocoder_checkpoint/vocoder.pt')
        # vocoder is already frozen and in eval mode

        # Use in your audio decoder:
        model.output_transform.audio_decoder.vocoder = vocoder
        ```
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    vocoder = AudioVocoder(
        hidden_channels=config['hidden_channels'],
        in_channels=config['n_mels'],
        conditioning_channels=config['hidden_channels'],  # Not used when conditioning_enabled=False
        upsample_factors=config['upsample_factors'],
        n_residual_layers=config['n_residual_layers'],
        dilation_cycle=config.get('dilation_cycle', 4),
        kernel_size=config.get('kernel_size', 3),
        leaky_relu_slope=config.get('leaky_relu_slope', 0.1),
        conditioning_enabled=False,
    )

    vocoder.load_state_dict(checkpoint['vocoder_state_dict'])

    # Freeze and set to eval mode for inference
    vocoder.eval()
    for param in vocoder.parameters():
        param.requires_grad = False

    vocoder.to(device)

    logger.info(f"Loaded pretrained vocoder from {checkpoint_path} (step {checkpoint['step']})")

    return vocoder


class AudioSampleCallback(TrainerCallback):
    """Callback to log audio samples during training."""

    def __init__(self, dataset, config: VocoderConfig, num_samples: int = 3, log_every_n_steps: int = 1000):
        self.dataset = dataset
        self.config = config
        self.num_samples = num_samples
        self.log_every_n_steps = log_every_n_steps
        self.writer = None
        self.trainer = None

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.log_every_n_steps != 0:
            return

        if self.writer is None:
            return

        model.eval()
        device = next(model.parameters()).device

        with torch.no_grad():
            for i in range(min(self.num_samples, len(self.dataset))):
                sample = self.dataset[i]
                mel_spec = sample["mel_spec"].unsqueeze(0).to(device)
                waveform_gt = sample["waveform"]

                # Generate
                pred_waveform = model.module.vocoder(mel_spec, condition=None) if hasattr(model, 'module') else model.vocoder(mel_spec, condition=None)
                pred_waveform = pred_waveform.squeeze(0).cpu()

                # Log to tensorboard
                self.writer.add_audio(
                    f"sample_{i}/predicted",
                    pred_waveform.numpy(),
                    state.global_step,
                    sample_rate=self.config.sample_rate
                )
                self.writer.add_audio(
                    f"sample_{i}/ground_truth",
                    waveform_gt.numpy(),
                    state.global_step,
                    sample_rate=self.config.sample_rate
                )

        model.train()

    def on_train_begin(self, args, state, control, **kwargs):
        # Get tensorboard writer from trainer
        if self.trainer is not None:
            for callback in self.trainer.callback_handler.callbacks:
                if isinstance(callback, TensorBoardCallback):
                    self.writer = callback.tb_writer
                    break


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain vocoder on Common Voice dataset")

    # Run configuration
    parser.add_argument('--run_name', type=str, required=True, help='Name of the run')
    parser.add_argument('--logging_base_dir', type=str, default=os.path.join('runs', 'vocoder'), help='Base directory for logging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume_from_checkpoint', action='store_true', help='Resume from checkpoint')

    # Data configuration
    parser.add_argument('--language', type=str, default='en', help='Common Voice language code')
    parser.add_argument('--dataset_cache_dir', type=str, default='cached_datasets', help='Dataset cache directory')
    parser.add_argument('--max_train_samples', type=int, default=None, help='Maximum training samples')
    parser.add_argument('--max_eval_samples', type=int, default=1000, help='Maximum evaluation samples')

    # Audio configuration
    parser.add_argument('--sample_rate', type=int, default=16000, help='Audio sample rate')
    parser.add_argument('--n_mels', type=int, default=128, help='Number of mel bands')
    parser.add_argument('--n_fft', type=int, default=1024, help='FFT size')
    parser.add_argument('--hop_length', type=int, default=512, help='Hop length')
    parser.add_argument('--max_audio_duration', type=float, default=10.0, help='Maximum audio duration in seconds')

    # Model configuration (defaults match MegaTransformerConfig)
    # Note: create_small_multimodal_model uses 1536, use --hidden_channels=1536 for that config
    parser.add_argument('--hidden_channels', type=int, default=2048, help='Vocoder hidden channels (MegaTransformerConfig default: 2048, small_multimodal: 1536)')
    parser.add_argument('--n_residual_layers', type=int, default=4, help='Number of residual layers')

    # Loss weights
    parser.add_argument('--waveform_l1_weight', type=float, default=1.0, help='Weight for L1 waveform loss')
    parser.add_argument('--sc_loss_weight', type=float, default=1.5, help='Weight for spectral convergence loss')
    parser.add_argument('--mag_loss_weight', type=float, default=1.5, help='Weight for log magnitude loss')

    # Training configuration
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.03, help='Warmup ratio')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    parser.add_argument('--num_train_epochs', type=int, default=-1, help='Number of training epochs')
    parser.add_argument('--max_steps', type=int, default=100000, help='Max training steps')

    # Precision
    parser.add_argument('--fp16', action='store_true', help='Use FP16')
    parser.add_argument('--bf16', action='store_true', help='Use BF16')

    # Logging
    parser.add_argument('--logging_steps', type=int, default=100, help='Logging frequency')
    parser.add_argument('--save_steps', type=int, default=1000, help='Checkpoint save frequency')
    parser.add_argument('--eval_steps', type=int, default=1000, help='Evaluation frequency')
    parser.add_argument('--audio_log_steps', type=int, default=2000, help='Audio sample logging frequency')

    # DeepSpeed
    parser.add_argument('--use_deepspeed', action='store_true', help='Use DeepSpeed')
    parser.add_argument('--deepspeed_config', type=str, default=None, help='DeepSpeed config file')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')

    # Other
    parser.add_argument('--cpu', action='store_true', help='Use CPU')
    parser.add_argument('--log_level', type=str, default='warning', help='Log level')

    args, _ = parser.parse_known_args()

    # Validation
    if args.num_train_epochs == -1 and args.max_steps == -1:
        raise ValueError("Either num_train_epochs or max_steps must be specified.")

    if args.use_deepspeed and args.deepspeed_config:
        if not os.path.exists(args.deepspeed_config):
            raise FileNotFoundError(f"DeepSpeed config not found: {args.deepspeed_config}")

    return args


def set_seed_everywhere(seed: int):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    hf_set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    set_seed_everywhere(args.seed)

    run_dir = os.path.join(args.logging_base_dir, args.run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    # Create config
    vocoder_config = VocoderConfig(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        max_audio_duration=args.max_audio_duration,
        hidden_channels=args.hidden_channels,
        n_residual_layers=args.n_residual_layers,
        waveform_l1_weight=args.waveform_l1_weight,
        sc_loss_weight=args.sc_loss_weight,
        mag_loss_weight=args.mag_loss_weight,
    )

    # Create model
    model = VocoderForTraining(vocoder_config)

    if args.local_rank == 0 or not args.use_deepspeed:
        print(f"Model structure: {model}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create datasets
    train_dataset = CommonVoiceVocoderDataset(
        config=vocoder_config,
        split="train",
        language=args.language,
        cache_dir=args.dataset_cache_dir,
        max_samples=args.max_train_samples,
    )

    eval_dataset = CommonVoiceVocoderDataset(
        config=vocoder_config,
        split="validation",
        language=args.language,
        cache_dir=args.dataset_cache_dir,
        max_samples=args.max_eval_samples,
    )

    # Create data collator
    data_collator = VocoderDataCollator(vocoder_config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=run_dir,
        overwrite_output_dir=True,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs if args.num_train_epochs > 0 else 1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        weight_decay=args.weight_decay,
        report_to="tensorboard",
        logging_dir=run_dir,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_safetensors=False,
        save_steps=args.save_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        max_grad_norm=args.max_grad_norm,
        deepspeed=args.deepspeed_config if args.use_deepspeed else None,
        use_cpu=args.cpu,
        log_level=args.log_level,
        logging_first_step=True,
        local_rank=args.local_rank,
    )

    # Create trainer
    trainer = VocoderTrainer(
        vocoder_config=vocoder_config,
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Add vocoder checkpoint callback (saves vocoder weights separately for world model training)
    vocoder_checkpoint_callback = VocoderCheckpointCallback(
        config=vocoder_config,
        save_every_n_steps=args.save_steps,
    )
    trainer.add_callback(vocoder_checkpoint_callback)

    # Add audio sample callback
    audio_callback = AudioSampleCallback(
        dataset=eval_dataset,
        config=vocoder_config,
        num_samples=3,
        log_every_n_steps=args.audio_log_steps,
    )
    trainer.add_callback(audio_callback)
    audio_callback.trainer = trainer

    # Train
    print(f"Starting training with {sum(p.numel() for p in model.parameters()):,} parameters")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()