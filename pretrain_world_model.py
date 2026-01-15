#!/usr/bin/env python3
"""
World Model Pretraining Script.

Trains a multimodal autoregressive world model on preprocessed sharded datasets
containing text, audio, voice, and image modalities.

Usage:
    # Single GPU
    python pretrain_world_model.py \
        --run_name my_world_model \
        --config tiny \
        --data_dir cached_datasets/world_model \
        --batch_size 8 \
        --learning_rate 1e-4

    # Multi-GPU with DeepSpeed
    deepspeed --num_gpus=4 pretrain_world_model.py \
        --use_deepspeed \
        --bf16 \
        --run_name my_world_model \
        --config small \
        --data_dir cached_datasets/world_model \
        --batch_size 8 \
        --learning_rate 1e-4 \
        --gradient_accumulation_steps 4 \
        --deepspeed_config ds_config_zero-2.json
"""

import argparse
import os

os.environ["DEEPSPEED_UNIT_TEST"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ['NCCL_TIMEOUT'] = '1200000'

import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

from model.world.world_model import (
    MegatransformerWorldModel,
    MegatransformerWorldModelConfig,
    get_wm_config,
    tiny_world_model_config,
    small_world_model_config,
    medium_world_model_config,
)
from shard_utils import WorldModelShardedDataset, WorldModelDataCollator
from utils.training_utils import EarlyStoppingCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# VAE Loading Utilities
# =============================================================================

def load_audio_vae_decoder(
    checkpoint_path: str,
    config_name: str,
    latent_channels: int = 32,
    speaker_embedding_dim: int = 0,
    device: str = "cpu"
):
    """
    Load an audio VAE decoder from checkpoint.

    Args:
        checkpoint_path: Path to VAE checkpoint directory
        config_name: Config name (tiny_test, medium_no_attn, large, etc.)
        latent_channels: Number of latent channels (must match training)
        speaker_embedding_dim: Speaker embedding dimension (0 for no speaker conditioning)
        device: Device to load to

    Returns:
        AudioVAEDecoder model
    """
    from model.audio.vae import model_config_lookup

    if config_name not in model_config_lookup:
        raise ValueError(f"Unknown audio VAE config: {config_name}. Available: {list(model_config_lookup.keys())}")

    # Create full VAE with the same config as training
    vae = model_config_lookup[config_name](
        latent_channels=latent_channels,
        speaker_embedding_dim=speaker_embedding_dim,
        perceptual_loss_type="none",
    )

    # Load checkpoint weights
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    pytorch_path = os.path.join(checkpoint_path, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
        vae.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded audio VAE from {safetensors_path}")
    elif os.path.exists(pytorch_path):
        state_dict = torch.load(pytorch_path, map_location="cpu")
        vae.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded audio VAE from {pytorch_path}")
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}, using random init")

    # Extract just the decoder
    decoder = vae.decoder
    decoder = decoder.to(device)
    decoder.eval()
    return decoder


def load_image_vae_decoder(
    checkpoint_path: str,
    config_name: str,
    latent_channels: int = 4,
    device: str = "cpu"
):
    """
    Load an image VAE decoder from checkpoint.

    Args:
        checkpoint_path: Path to VAE checkpoint directory
        config_name: Config name (tiny, mini, small, medium, etc.)
        latent_channels: Number of latent channels (must match training)
        device: Device to load to

    Returns:
        ImageVAEDecoder model
    """
    from model.image.vae import model_config_lookup

    if config_name not in model_config_lookup:
        raise ValueError(f"Unknown image VAE config: {config_name}. Available: {list(model_config_lookup.keys())}")

    # Create full VAE with the same config as training
    vae = model_config_lookup[config_name](
        latent_channels=latent_channels,
        perceptual_loss_type="none",
    )

    # Load checkpoint weights
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    pytorch_path = os.path.join(checkpoint_path, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
        vae.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded image VAE from {safetensors_path}")
    elif os.path.exists(pytorch_path):
        state_dict = torch.load(pytorch_path, map_location="cpu")
        vae.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded image VAE from {pytorch_path}")
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}, using random init")

    # Extract just the decoder
    decoder = vae.decoder
    decoder = decoder.to(device)
    decoder.eval()
    return decoder


def load_vocoder(checkpoint_path: str, config_name: str, device: str = "cpu"):
    """
    Load a vocoder from checkpoint.

    Args:
        checkpoint_path: Path to vocoder checkpoint directory
        config_name: Config name
        device: Device to load to

    Returns:
        Vocoder model
    """
    from model.audio.vocoders.vocoders import FrequencyDomainVocoderWithAttention
    from utils.configuration import get_vocoder_config

    config = get_vocoder_config(config_name)
    vocoder = FrequencyDomainVocoderWithAttention(config)

    ckpt_file = os.path.join(checkpoint_path, "pytorch_model.bin")
    if os.path.exists(ckpt_file):
        state_dict = torch.load(ckpt_file, map_location="cpu")
        vocoder.load_state_dict(state_dict)
    else:
        logger.warning(f"Vocoder checkpoint not found: {ckpt_file}")

    vocoder = vocoder.to(device)
    vocoder.eval()
    return vocoder


# =============================================================================
# Decoded Metrics
# =============================================================================

def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.

    Args:
        pred: Predicted image [B, C, H, W] in range [0, max_val]
        target: Target image [B, C, H, W]
        max_val: Maximum pixel value

    Returns:
        PSNR in dB (higher is better)
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(max_val) - 10 * math.log10(mse.item())
    return psnr


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2
) -> float:
    """
    Compute Structural Similarity Index (simplified version).

    Args:
        pred: Predicted image [B, C, H, W]
        target: Target image [B, C, H, W]
        window_size: Size of gaussian window
        C1, C2: Stability constants

    Returns:
        SSIM score (higher is better, max 1.0)
    """
    # Create gaussian window
    def gaussian_window(size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.unsqueeze(0) * g.unsqueeze(1)

    window = gaussian_window(window_size).unsqueeze(0).unsqueeze(0)
    window = window.expand(pred.shape[1], 1, window_size, window_size).to(pred.device)

    # Compute means
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=pred.shape[1])
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=target.shape[1])

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Compute variances and covariance
    sigma1_sq = F.conv2d(pred ** 2, window, padding=window_size // 2, groups=pred.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, window, padding=window_size // 2, groups=target.shape[1]) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=pred.shape[1]) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


def compute_mel_spectrogram_metrics(
    pred_mel: torch.Tensor,
    target_mel: torch.Tensor
) -> Dict[str, float]:
    """
    Compute metrics for mel spectrogram comparison.

    Args:
        pred_mel: Predicted mel spectrogram [B, n_mels, T]
        target_mel: Target mel spectrogram [B, n_mels, T]

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # L1 in mel domain
    metrics["mel_l1"] = F.l1_loss(pred_mel, target_mel).item()

    # L2 in mel domain
    metrics["mel_l2"] = F.mse_loss(pred_mel, target_mel).sqrt().item()

    # Mel Cepstral Distortion (MCD) - simplified
    # MCD measures distance in cepstral domain
    # Apply DCT to convert mel to cepstral
    # Simplified: just use cosine similarity as proxy
    pred_flat = pred_mel.reshape(pred_mel.shape[0], -1)
    target_flat = target_mel.reshape(target_mel.shape[0], -1)
    cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=-1).mean()
    metrics["mel_cosine_similarity"] = cos_sim.item()

    return metrics


# =============================================================================
# TensorBoard Logging Utilities
# =============================================================================

def log_image_to_tensorboard(
    writer: SummaryWriter,
    tag: str,
    image: torch.Tensor,
    step: int,
    normalize: bool = True
):
    """
    Log an image to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        tag: Tag for the image
        image: Image tensor [C, H, W] or [B, C, H, W]
        step: Global step
        normalize: Whether to normalize to [0, 1]
    """
    if image.dim() == 4:
        image = image[0]  # Take first in batch

    if normalize:
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    writer.add_image(tag, image, step)


def log_mel_spectrogram_to_tensorboard(
    writer: SummaryWriter,
    tag: str,
    mel: torch.Tensor,
    step: int
):
    """
    Log a mel spectrogram to TensorBoard as an image.

    Args:
        writer: TensorBoard SummaryWriter
        tag: Tag for the spectrogram
        mel: Mel spectrogram [n_mels, T] or [B, n_mels, T]
        step: Global step
    """
    if mel.dim() == 3:
        mel = mel[0]  # Take first in batch

    # Normalize to [0, 1]
    mel_norm = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)

    # Add channel dimension and flip vertically (low freq at bottom)
    mel_img = mel_norm.unsqueeze(0).flip(1)

    writer.add_image(tag, mel_img, step)


def log_audio_to_tensorboard(
    writer: SummaryWriter,
    tag: str,
    audio: torch.Tensor,
    sample_rate: int,
    step: int
):
    """
    Log audio waveform to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        tag: Tag for the audio
        audio: Audio waveform [T] or [B, T]
        sample_rate: Audio sample rate
        step: Global step
    """
    if audio.dim() == 2:
        audio = audio[0]  # Take first in batch

    # Normalize
    audio = audio / (audio.abs().max() + 1e-8)

    # Add channel dimension
    audio = audio.unsqueeze(0)

    writer.add_audio(tag, audio, step, sample_rate=sample_rate)


# =============================================================================
# Custom Trainer for World Model
# =============================================================================

def compute_latent_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute standard metrics for latent space predictions.

    Args:
        pred: Predicted latents, shape [B, C, ...]
        target: Target latents, same shape as pred
        mask: Optional mask for valid positions

    Returns:
        Dictionary of metric names to values
    """
    metrics = {}

    # Flatten spatial dimensions for easier computation
    pred_flat = pred.reshape(pred.shape[0], -1)  # [B, C*H*W] or [B, C*H*T]
    target_flat = target.reshape(target.shape[0], -1)

    if mask is not None:
        # Apply mask (expand to match flattened shape if needed)
        mask_flat = mask.reshape(mask.shape[0], -1) if mask.dim() > 1 else mask
    else:
        mask_flat = None

    # L1 loss (already computed, but we can add per-sample version)
    l1 = torch.abs(pred_flat - target_flat).mean(dim=-1)  # [B]
    metrics["l1_loss"] = l1.mean().item()

    # L2 loss
    l2 = torch.sqrt(((pred_flat - target_flat) ** 2).mean(dim=-1) + 1e-8)  # [B]
    metrics["l2_loss"] = l2.mean().item()

    # Cosine similarity
    pred_norm = pred_flat / (pred_flat.norm(dim=-1, keepdim=True) + 1e-8)
    target_norm = target_flat / (target_flat.norm(dim=-1, keepdim=True) + 1e-8)
    cos_sim = (pred_norm * target_norm).sum(dim=-1)  # [B]
    metrics["cosine_similarity"] = cos_sim.mean().item()

    # Signal-to-Noise Ratio (SNR) in dB
    signal_power = (target_flat ** 2).mean(dim=-1)
    noise_power = ((pred_flat - target_flat) ** 2).mean(dim=-1)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
    metrics["snr_db"] = snr.mean().item()

    return metrics


def compute_text_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> Dict[str, float]:
    """
    Compute standard metrics for text predictions.

    Args:
        logits: Model logits, shape [B, T, vocab_size]
        targets: Target token IDs, shape [B, T]
        ignore_index: Token ID to ignore in metrics

    Returns:
        Dictionary of metric names to values
    """
    metrics = {}

    # Create mask for valid positions
    valid_mask = targets != ignore_index

    if valid_mask.sum() == 0:
        return {"perplexity": 0.0, "top1_accuracy": 0.0, "top5_accuracy": 0.0}

    # Perplexity from cross-entropy
    ce_loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=ignore_index,
        reduction="mean"
    )
    perplexity = torch.exp(ce_loss).item()
    # Clamp perplexity to avoid inf
    metrics["perplexity"] = min(perplexity, 1e6)

    # Top-1 accuracy
    predictions = logits.argmax(dim=-1)  # [B, T]
    correct_top1 = (predictions == targets) & valid_mask
    metrics["top1_accuracy"] = correct_top1.sum().item() / valid_mask.sum().item()

    # Top-5 accuracy
    top5_preds = logits.topk(5, dim=-1).indices  # [B, T, 5]
    targets_expanded = targets.unsqueeze(-1).expand_as(top5_preds)  # [B, T, 5]
    correct_top5 = (top5_preds == targets_expanded).any(dim=-1) & valid_mask
    metrics["top5_accuracy"] = correct_top5.sum().item() / valid_mask.sum().item()

    return metrics


class WorldModelTrainer(Trainer):
    """
    Custom trainer for the multimodal world model.

    Handles:
    - Combined loss from multiple modalities (text, audio, voice, image)
    - Proper loss weighting across modalities
    - Gradient accumulation with mixed modality batches
    - Standard evaluation metrics (perplexity, accuracy, SNR, cosine similarity)
    - VAE-decoded metrics during evaluation (PSNR, SSIM for images, mel metrics for audio)
    - TensorBoard logging of example predictions
    """

    def __init__(
        self,
        *args,
        text_loss_weight: float = 1.0,
        audio_loss_weight: float = 1.0,
        voice_loss_weight: float = 1.0,
        image_loss_weight: float = 1.0,
        # VAE checkpoints for decoded evaluation
        audio_vae_checkpoint: Optional[str] = None,
        audio_vae_config: str = "large",
        audio_latent_channels: int = 32,
        # Voice VAE (can be separate from audio VAE for speaker conditioning)
        voice_vae_checkpoint: Optional[str] = None,
        voice_vae_config: Optional[str] = None,  # If None, uses audio_vae_config
        voice_latent_channels: Optional[int] = None,  # If None, uses audio_latent_channels
        voice_speaker_embedding_dim: int = 0,  # 0 = no speaker conditioning
        # Image VAE
        image_vae_checkpoint: Optional[str] = None,
        image_vae_config: str = "tiny",
        image_latent_channels: int = 4,
        # Vocoder
        vocoder_checkpoint: Optional[str] = None,
        vocoder_config: str = "tiny",
        # Evaluation settings
        log_dir: Optional[str] = None,
        n_eval_examples: int = 4,
        sample_rate: int = 16000,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.text_loss_weight = text_loss_weight
        self.audio_loss_weight = audio_loss_weight
        self.voice_loss_weight = voice_loss_weight
        self.image_loss_weight = image_loss_weight

        # Audio VAE settings (for general audio, no speaker conditioning)
        self.audio_vae_checkpoint = audio_vae_checkpoint
        self.audio_vae_config = audio_vae_config
        self.audio_latent_channels = audio_latent_channels

        # Voice VAE settings (may have speaker conditioning)
        # Falls back to audio VAE settings if not specified
        self.voice_vae_checkpoint = voice_vae_checkpoint or audio_vae_checkpoint
        self.voice_vae_config = voice_vae_config or audio_vae_config
        self.voice_latent_channels = voice_latent_channels or audio_latent_channels
        self.voice_speaker_embedding_dim = voice_speaker_embedding_dim

        # Image VAE settings
        self.image_vae_checkpoint = image_vae_checkpoint
        self.image_vae_config = image_vae_config
        self.image_latent_channels = image_latent_channels

        # Vocoder settings
        self.vocoder_checkpoint = vocoder_checkpoint
        self.vocoder_config = vocoder_config

        # Evaluation settings
        self.log_dir = log_dir
        self.n_eval_examples = n_eval_examples
        self.sample_rate = sample_rate

        # Lazy-loaded decoders (None until needed)
        self._audio_vae_decoder = None
        self._voice_vae_decoder = None  # Separate from audio for speaker conditioning
        self._image_vae_decoder = None
        self._vocoder = None
        self._eval_writer = None

        # Track modality-specific losses for logging
        self._current_losses = {}
        self._current_metrics = {}

    def _get_audio_vae_decoder(self, device: str):
        """Lazy load audio VAE decoder (no speaker conditioning)."""
        if self._audio_vae_decoder is None and self.audio_vae_checkpoint:
            logger.info(f"Loading audio VAE decoder from {self.audio_vae_checkpoint}")
            self._audio_vae_decoder = load_audio_vae_decoder(
                checkpoint_path=self.audio_vae_checkpoint,
                config_name=self.audio_vae_config,
                latent_channels=self.audio_latent_channels,
                speaker_embedding_dim=0,  # General audio has no speaker conditioning
                device=device,
            )
        return self._audio_vae_decoder

    def _get_voice_vae_decoder(self, device: str):
        """Lazy load voice VAE decoder (may have speaker conditioning)."""
        if self._voice_vae_decoder is None and self.voice_vae_checkpoint:
            logger.info(f"Loading voice VAE decoder from {self.voice_vae_checkpoint}")
            self._voice_vae_decoder = load_audio_vae_decoder(
                checkpoint_path=self.voice_vae_checkpoint,
                config_name=self.voice_vae_config,
                latent_channels=self.voice_latent_channels,
                speaker_embedding_dim=self.voice_speaker_embedding_dim,
                device=device,
            )
        return self._voice_vae_decoder

    def _get_image_vae_decoder(self, device: str):
        """Lazy load image VAE decoder."""
        if self._image_vae_decoder is None and self.image_vae_checkpoint:
            logger.info(f"Loading image VAE decoder from {self.image_vae_checkpoint}")
            self._image_vae_decoder = load_image_vae_decoder(
                checkpoint_path=self.image_vae_checkpoint,
                config_name=self.image_vae_config,
                latent_channels=self.image_latent_channels,
                device=device,
            )
        return self._image_vae_decoder

    def _get_vocoder(self, device: str):
        """Lazy load vocoder."""
        if self._vocoder is None and self.vocoder_checkpoint:
            logger.info(f"Loading vocoder from {self.vocoder_checkpoint}")
            self._vocoder = load_vocoder(
                self.vocoder_checkpoint, self.vocoder_config, device
            )
        return self._vocoder

    def _unload_vae_decoders(self):
        """Move VAE decoders back to CPU to free GPU memory."""
        if self._audio_vae_decoder is not None:
            self._audio_vae_decoder = self._audio_vae_decoder.cpu()
            logger.info("Moved audio VAE decoder to CPU")
        if self._voice_vae_decoder is not None:
            self._voice_vae_decoder = self._voice_vae_decoder.cpu()
            logger.info("Moved voice VAE decoder to CPU")
        if self._image_vae_decoder is not None:
            self._image_vae_decoder = self._image_vae_decoder.cpu()
            logger.info("Moved image VAE decoder to CPU")
        if self._vocoder is not None:
            self._vocoder = self._vocoder.cpu()
            logger.info("Moved vocoder to CPU")
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_eval_writer(self):
        """Get or create TensorBoard writer for evaluation."""
        if self._eval_writer is None and self.log_dir:
            self._eval_writer = SummaryWriter(log_dir=self.log_dir)
        return self._eval_writer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute combined loss from all modalities.

        The world model returns losses for each modality present in the batch:
        - text_classification_loss: Cross-entropy loss for text tokens
        - audio_latent_loss: L1/MSE loss for audio latent reconstruction
        - voice_latent_loss: L1/MSE loss for voice latent reconstruction
        - image_latent_loss: L1/MSE loss for image latent reconstruction
        """
        # Prepare inputs for the model
        model_inputs = {
            "text_input_ids": inputs["text_input_ids"],
            "precomputed_latents": True,
        }

        # Add text targets (shifted for next-token prediction)
        # The text generator handles this internally with the targets parameter
        model_inputs["text_targets"] = inputs["text_input_ids"]

        # Add audio latents if present
        if "audio_mel_spec_latents" in inputs and inputs.get("audio_latent_lengths") is not None:
            audio_latents = inputs["audio_mel_spec_latents"]
            # Shape: [B, C, H, T] -> [B, 1, C, H, T] (add n_audio dimension)
            if audio_latents.dim() == 4:
                audio_latents = audio_latents.unsqueeze(1)
            model_inputs["audio_inputs"] = audio_latents
            model_inputs["audio_latent_labels"] = audio_latents
            model_inputs["audio_lengths"] = inputs["audio_latent_lengths"].unsqueeze(1)

        # Add voice latents if present
        if "voice_mel_spec_latents" in inputs and inputs.get("voice_latent_lengths") is not None:
            voice_latents = inputs["voice_mel_spec_latents"]
            if voice_latents.dim() == 4:
                voice_latents = voice_latents.unsqueeze(1)
            model_inputs["voice_inputs"] = voice_latents
            model_inputs["voice_latent_labels"] = voice_latents
            model_inputs["voice_lengths"] = inputs["voice_latent_lengths"].unsqueeze(1)

        # Add image latents if present
        if "image_latents" in inputs and inputs.get("image_present") is not None:
            image_latents = inputs["image_latents"]
            if image_latents.dim() == 4:
                image_latents = image_latents.unsqueeze(1)
            model_inputs["image_inputs"] = image_latents
            model_inputs["image_latent_labels"] = image_latents

        # Forward pass
        outputs = model(**model_inputs)

        # Combine losses
        total_loss = torch.tensor(0.0, device=model.device, requires_grad=True)
        n_losses = 0

        # Text loss
        if "text_classification_loss" in outputs and outputs["text_classification_loss"] is not None:
            text_loss = outputs["text_classification_loss"] * self.text_loss_weight
            total_loss = total_loss + text_loss
            n_losses += 1
            self._current_losses["text_loss"] = text_loss.detach().item()

        # Audio loss
        if "audio_latent_loss" in outputs and outputs["audio_latent_loss"] is not None:
            audio_loss = outputs["audio_latent_loss"] * self.audio_loss_weight
            total_loss = total_loss + audio_loss
            n_losses += 1
            self._current_losses["audio_loss"] = audio_loss.detach().item()

        # Voice loss
        if "voice_latent_loss" in outputs and outputs["voice_latent_loss"] is not None:
            voice_loss = outputs["voice_latent_loss"] * self.voice_loss_weight
            total_loss = total_loss + voice_loss
            n_losses += 1
            self._current_losses["voice_loss"] = voice_loss.detach().item()

        # Image loss
        if "image_latent_loss" in outputs and outputs["image_latent_loss"] is not None:
            image_loss = outputs["image_latent_loss"] * self.image_loss_weight
            total_loss = total_loss + image_loss
            n_losses += 1
            self._current_losses["image_loss"] = image_loss.detach().item()

        # Average across modalities if multiple present
        if n_losses > 0:
            total_loss = total_loss / n_losses

        self._current_losses["total_loss"] = total_loss.detach().item()
        self._current_losses["n_modalities"] = n_losses

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def log(self, logs: Dict[str, float]) -> None:
        """Override log to include modality-specific losses."""
        # Add current modality losses to logs
        for key, value in self._current_losses.items():
            if key not in logs:
                logs[f"train/{key}"] = value

        # Add current metrics to logs
        for key, value in self._current_metrics.items():
            if key not in logs:
                logs[key] = value

        super().log(logs)

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        """
        Override evaluation loop to compute additional metrics.

        Computes:
        - Text: perplexity, top-1 accuracy, top-5 accuracy
        - Audio/Voice/Image: L1, L2, cosine similarity, SNR (latent space)
        - Decoded metrics: PSNR, SSIM for images; mel L1/L2 for audio (if VAEs available)
        - Logs example predictions to TensorBoard
        """
        model = self.model
        model.eval()
        device = next(model.parameters()).device

        # Get TensorBoard writer
        writer = self._get_eval_writer()
        global_step = self.state.global_step if self.state else 0

        # Accumulate metrics
        all_metrics = {
            "text_perplexity": [],
            "text_top1_accuracy": [],
            "text_top5_accuracy": [],
            "audio_l1_loss": [],
            "audio_l2_loss": [],
            "audio_cosine_similarity": [],
            "audio_snr_db": [],
            "voice_l1_loss": [],
            "voice_l2_loss": [],
            "voice_cosine_similarity": [],
            "voice_snr_db": [],
            "image_l1_loss": [],
            "image_l2_loss": [],
            "image_cosine_similarity": [],
            "image_snr_db": [],
            # Decoded metrics
            "image_psnr": [],
            "image_ssim": [],
            "audio_mel_l1": [],
            "audio_mel_l2": [],
            "audio_mel_cosine": [],
            "voice_mel_l1": [],
            "voice_mel_l2": [],
            "voice_mel_cosine": [],
        }
        total_loss = 0.0
        n_batches = 0
        n_examples_logged = 0

        # Load VAE decoders if available
        audio_decoder = self._get_audio_vae_decoder(str(device)) if self.audio_vae_checkpoint else None
        voice_decoder = self._get_voice_vae_decoder(str(device)) if self.voice_vae_checkpoint else None
        image_decoder = self._get_image_vae_decoder(str(device)) if self.image_vae_checkpoint else None
        vocoder = self._get_vocoder(str(device)) if self.vocoder_checkpoint else None

        for step, inputs in enumerate(dataloader):
            # Move inputs to device
            inputs = self._prepare_inputs(inputs)

            with torch.no_grad():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                total_loss += loss.item()
                n_batches += 1

                # Text metrics
                if "logits" in outputs and outputs["logits"] is not None:
                    text_targets = inputs["text_input_ids"]
                    # Shift for next-token prediction
                    shifted_targets = text_targets[:, 1:].contiguous()
                    shifted_logits = outputs["logits"][:, :-1, :].contiguous()

                    text_metrics = compute_text_metrics(
                        shifted_logits, shifted_targets,
                        ignore_index=self.args.label_pad_token_id if hasattr(self.args, 'label_pad_token_id') else -100
                    )
                    all_metrics["text_perplexity"].append(text_metrics["perplexity"])
                    all_metrics["text_top1_accuracy"].append(text_metrics["top1_accuracy"])
                    all_metrics["text_top5_accuracy"].append(text_metrics["top5_accuracy"])

                # Audio latent metrics
                if "audio_latent_preds" in outputs and outputs["audio_latent_preds"] is not None:
                    if "audio_mel_spec_latents" in inputs:
                        audio_target = inputs["audio_mel_spec_latents"]
                        if audio_target.dim() == 5:
                            audio_target = audio_target[:, 0]
                        audio_pred = outputs["audio_latent_preds"]
                        if audio_pred.dim() == 5:
                            audio_pred = audio_pred[:, 0]

                        # Latent space metrics
                        audio_metrics = compute_latent_metrics(audio_pred, audio_target)
                        all_metrics["audio_l1_loss"].append(audio_metrics["l1_loss"])
                        all_metrics["audio_l2_loss"].append(audio_metrics["l2_loss"])
                        all_metrics["audio_cosine_similarity"].append(audio_metrics["cosine_similarity"])
                        all_metrics["audio_snr_db"].append(audio_metrics["snr_db"])

                        # Decoded metrics (if VAE decoder available)
                        if audio_decoder is not None:
                            pred_mel = audio_decoder(audio_pred)
                            target_mel = audio_decoder(audio_target)
                            mel_metrics = compute_mel_spectrogram_metrics(pred_mel, target_mel)
                            all_metrics["audio_mel_l1"].append(mel_metrics["mel_l1"])
                            all_metrics["audio_mel_l2"].append(mel_metrics["mel_l2"])
                            all_metrics["audio_mel_cosine"].append(mel_metrics["mel_cosine_similarity"])

                            # Log example to TensorBoard
                            if writer and n_examples_logged < self.n_eval_examples:
                                log_mel_spectrogram_to_tensorboard(
                                    writer, f"eval/audio_pred_{n_examples_logged}",
                                    pred_mel[0].cpu(), global_step
                                )
                                log_mel_spectrogram_to_tensorboard(
                                    writer, f"eval/audio_target_{n_examples_logged}",
                                    target_mel[0].cpu(), global_step
                                )

                                # If vocoder available, log audio too
                                if vocoder is not None:
                                    try:
                                        pred_audio = vocoder(pred_mel[:1])
                                        target_audio = vocoder(target_mel[:1])
                                        log_audio_to_tensorboard(
                                            writer, f"eval/audio_pred_wav_{n_examples_logged}",
                                            pred_audio[0].cpu(), self.sample_rate, global_step
                                        )
                                        log_audio_to_tensorboard(
                                            writer, f"eval/audio_target_wav_{n_examples_logged}",
                                            target_audio[0].cpu(), self.sample_rate, global_step
                                        )
                                    except Exception as e:
                                        logger.warning(f"Failed to generate audio with vocoder: {e}")

                # Voice latent metrics
                if "voice_latent_preds" in outputs and outputs["voice_latent_preds"] is not None:
                    if "voice_mel_spec_latents" in inputs:
                        voice_target = inputs["voice_mel_spec_latents"]
                        if voice_target.dim() == 5:
                            voice_target = voice_target[:, 0]
                        voice_pred = outputs["voice_latent_preds"]
                        if voice_pred.dim() == 5:
                            voice_pred = voice_pred[:, 0]

                        voice_metrics = compute_latent_metrics(voice_pred, voice_target)
                        all_metrics["voice_l1_loss"].append(voice_metrics["l1_loss"])
                        all_metrics["voice_l2_loss"].append(voice_metrics["l2_loss"])
                        all_metrics["voice_cosine_similarity"].append(voice_metrics["cosine_similarity"])
                        all_metrics["voice_snr_db"].append(voice_metrics["snr_db"])

                        # Decoded metrics (use voice decoder if available, fall back to audio)
                        decoder_for_voice = voice_decoder if voice_decoder is not None else audio_decoder
                        if decoder_for_voice is not None:
                            # Note: voice decoder may require speaker_embedding=None for unconditional decoding
                            pred_mel = decoder_for_voice(voice_pred)
                            target_mel = decoder_for_voice(voice_target)
                            mel_metrics = compute_mel_spectrogram_metrics(pred_mel, target_mel)
                            all_metrics["voice_mel_l1"].append(mel_metrics["mel_l1"])
                            all_metrics["voice_mel_l2"].append(mel_metrics["mel_l2"])
                            all_metrics["voice_mel_cosine"].append(mel_metrics["mel_cosine_similarity"])

                            # Log example
                            if writer and n_examples_logged < self.n_eval_examples:
                                log_mel_spectrogram_to_tensorboard(
                                    writer, f"eval/voice_pred_{n_examples_logged}",
                                    pred_mel[0].cpu(), global_step
                                )
                                log_mel_spectrogram_to_tensorboard(
                                    writer, f"eval/voice_target_{n_examples_logged}",
                                    target_mel[0].cpu(), global_step
                                )

                # Image latent metrics
                if "image_latent_preds" in outputs and outputs["image_latent_preds"] is not None:
                    if "image_latents" in inputs:
                        image_target = inputs["image_latents"]
                        if image_target.dim() == 5:
                            image_target = image_target[:, 0]
                        image_pred = outputs["image_latent_preds"]
                        if image_pred.dim() == 5:
                            image_pred = image_pred[:, 0]

                        # Latent space metrics
                        image_metrics = compute_latent_metrics(image_pred, image_target)
                        all_metrics["image_l1_loss"].append(image_metrics["l1_loss"])
                        all_metrics["image_l2_loss"].append(image_metrics["l2_loss"])
                        all_metrics["image_cosine_similarity"].append(image_metrics["cosine_similarity"])
                        all_metrics["image_snr_db"].append(image_metrics["snr_db"])

                        # Decoded metrics (if VAE decoder available)
                        if image_decoder is not None:
                            pred_img = image_decoder(image_pred)
                            target_img = image_decoder(image_target)

                            # Normalize to [0, 1] for metrics
                            pred_img_norm = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min() + 1e-8)
                            target_img_norm = (target_img - target_img.min()) / (target_img.max() - target_img.min() + 1e-8)

                            all_metrics["image_psnr"].append(compute_psnr(pred_img_norm, target_img_norm))
                            all_metrics["image_ssim"].append(compute_ssim(pred_img_norm, target_img_norm))

                            # Log example to TensorBoard
                            if writer and n_examples_logged < self.n_eval_examples:
                                log_image_to_tensorboard(
                                    writer, f"eval/image_pred_{n_examples_logged}",
                                    pred_img[0].cpu(), global_step
                                )
                                log_image_to_tensorboard(
                                    writer, f"eval/image_target_{n_examples_logged}",
                                    target_img[0].cpu(), global_step
                                )

                # Increment examples logged
                if n_examples_logged < self.n_eval_examples:
                    n_examples_logged += 1

        model.train()

        # Unload VAE decoders to free GPU memory
        self._unload_vae_decoders()

        # Aggregate metrics
        final_metrics = {f"{metric_key_prefix}_loss": total_loss / max(n_batches, 1)}

        for key, values in all_metrics.items():
            if values:
                final_metrics[f"{metric_key_prefix}_{key}"] = sum(values) / len(values)

        # Log to trainer state
        self._current_metrics = final_metrics

        # Return in format expected by Trainer
        from transformers.trainer_utils import EvalLoopOutput
        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=final_metrics,
            num_samples=n_batches * self.args.per_device_eval_batch_size,
        )


# =============================================================================
# Logging Callbacks
# =============================================================================

class WorldModelLoggingCallback(TrainerCallback):
    """
    Callback for logging world model training metrics.

    Logs:
    - Per-modality losses
    - Sample counts per modality
    - Generated samples periodically
    """

    def __init__(
        self,
        log_dir: str,
        log_every_n_steps: int = 100,
        sample_every_n_steps: int = 1000,
    ):
        self.log_dir = log_dir
        self.log_every_n_steps = log_every_n_steps
        self.sample_every_n_steps = sample_every_n_steps
        self.writer = None

    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize TensorBoard writer."""
        if state.is_world_process_zero:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            logger.info(f"TensorBoard logging to {self.log_dir}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics to TensorBoard."""
        if self.writer is None or not state.is_world_process_zero:
            return

        if logs is None:
            return

        step = state.global_step

        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)

    def on_train_end(self, args, state, control, **kwargs):
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()


# =============================================================================
# Configuration
# =============================================================================

def get_config_by_name(name: str) -> MegatransformerWorldModelConfig:
    """Get a pre-defined config by name."""
    configs = {
        "tiny": tiny_world_model_config,
        "small": small_world_model_config,
        "medium": medium_world_model_config,
    }

    if name not in configs:
        raise ValueError(f"Unknown config '{name}'. Available: {list(configs.keys())}")

    return configs[name]


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pretrain the multimodal world model"
    )

    # Required arguments
    parser.add_argument("--run_name", type=str, required=True, help="Name of the training run")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing preprocessed world model shards")

    # Model configuration
    parser.add_argument("--config", type=str, default="tiny",
                        choices=["tiny", "small", "medium"],
                        help="Model configuration to use")

    # Optional data arguments
    parser.add_argument("--eval_data_dir", type=str, default=None,
                        help="Directory for evaluation shards (optional)")

    # Training hyperparameters
    parser.add_argument("--num_train_epoch", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (-1 for unlimited)")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")

    # Precision
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
    parser.add_argument("--fp16", action="store_true", help="Use float16 precision")

    # Gradient checkpointing
    parser.add_argument("--use_gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory")

    # DeepSpeed
    parser.add_argument("--use_deepspeed", action="store_true", help="Enable DeepSpeed")
    parser.add_argument("--deepspeed_config", type=str, default=None,
                        help="Path to DeepSpeed config file")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")

    # Loss weights
    parser.add_argument("--text_loss_weight", type=float, default=1.0)
    parser.add_argument("--audio_loss_weight", type=float, default=1.0)
    parser.add_argument("--voice_loss_weight", type=float, default=1.0)
    parser.add_argument("--image_loss_weight", type=float, default=1.0)

    # Data collator settings
    parser.add_argument("--pad_token_id", type=int, default=0)
    parser.add_argument("--max_text_length", type=int, default=2048)
    parser.add_argument("--audio_latent_channels", type=int, default=32)
    parser.add_argument("--audio_latent_mel_bins", type=int, default=10)
    parser.add_argument("--voice_latent_channels", type=int, default=32)
    parser.add_argument("--voice_latent_mel_bins", type=int, default=10)
    parser.add_argument("--image_latent_channels", type=int, default=4)
    parser.add_argument("--image_latent_size", type=int, default=32)

    # Logging and saving
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Max checkpoints to keep")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--sample_every_n_steps", type=int, default=1000)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    # Resume
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Early stopping
    parser.add_argument("--early_stopping_patience", type=int, default=None,
                        help="Early stopping patience (steps without improvement)")

    # VAE checkpoints for decoded evaluation metrics
    parser.add_argument("--audio_vae_checkpoint", type=str, default=None,
                        help="Path to audio VAE checkpoint for general audio (no speaker conditioning)")
    parser.add_argument("--audio_vae_config", type=str, default="large",
                        help="Config name for audio VAE")
    # Voice VAE (can be separate from audio VAE for speaker conditioning)
    parser.add_argument("--voice_vae_checkpoint", type=str, default=None,
                        help="Path to voice VAE checkpoint (defaults to audio_vae_checkpoint if not set)")
    parser.add_argument("--voice_vae_config", type=str, default=None,
                        help="Config name for voice VAE (defaults to audio_vae_config if not set)")
    parser.add_argument("--voice_speaker_embedding_dim", type=int, default=0,
                        help="Speaker embedding dimension for voice VAE (0 = no speaker conditioning)")
    # Image VAE
    parser.add_argument("--image_vae_checkpoint", type=str, default=None,
                        help="Path to image VAE checkpoint for decoded eval metrics")
    parser.add_argument("--image_vae_config", type=str, default="tiny",
                        help="Config name for image VAE")
    # Vocoder
    parser.add_argument("--vocoder_checkpoint", type=str, default=None,
                        help="Path to vocoder checkpoint for audio waveform generation")
    parser.add_argument("--vocoder_config", type=str, default="tiny",
                        help="Config name for vocoder")

    # Evaluation settings
    parser.add_argument("--n_eval_examples", type=int, default=4,
                        help="Number of examples to log to TensorBoard per evaluation")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Eval batch size (defaults to train batch_size if not set)")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Audio sample rate for TensorBoard logging")

    args = parser.parse_args()

    # Setup run directory
    run_dir = os.path.join("runs", "world_model", args.run_name)
    log_dir = os.path.join("logs", "world_model", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Log directory: {log_dir}")

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    logger.info(f"Loading {args.config} world model configuration...")
    config = get_config_by_name(args.config)

    model = MegatransformerWorldModel(config)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    logger.info(f"Loading dataset from {args.data_dir}...")
    train_dataset = WorldModelShardedDataset(
        shard_dir=args.data_dir,
        cache_size=3,
    )
    logger.info(f"Training samples: {len(train_dataset):,}")

    eval_dataset = None
    if args.eval_data_dir:
        eval_dataset = WorldModelShardedDataset(
            shard_dir=args.eval_data_dir,
            cache_size=2,
        )
        logger.info(f"Evaluation samples: {len(eval_dataset):,}")

    # Data collator
    collator = WorldModelDataCollator(
        pad_token_id=args.pad_token_id,
        max_text_length=args.max_text_length,
        audio_latent_channels=args.audio_latent_channels,
        audio_latent_mel_bins=args.audio_latent_mel_bins,
        voice_latent_channels=args.voice_latent_channels,
        voice_latent_mel_bins=args.voice_latent_mel_bins,
        image_latent_channels=args.image_latent_channels,
        image_latent_size=args.image_latent_size,
    )

    # -------------------------------------------------------------------------
    # Training Arguments
    # -------------------------------------------------------------------------
    # Use smaller eval batch size if specified (helps with VAE decoding memory)
    eval_batch_size = args.eval_batch_size if args.eval_batch_size else args.batch_size

    training_args = TrainingArguments(
        output_dir=run_dir,
        run_name=args.run_name,

        # Training hyperparameters
        num_train_epochs=args.num_train_epoch,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 0.01,
        warmup_ratio=args.warmup_ratio if hasattr(args, 'warmup_ratio') else 0.1,

        # Precision
        bf16=args.bf16 if hasattr(args, 'bf16') else False,
        fp16=args.fp16 if hasattr(args, 'fp16') else False,

        # Gradient checkpointing
        gradient_checkpointing=args.use_gradient_checkpointing if hasattr(args, 'use_gradient_checkpointing') else False,

        # Logging
        logging_dir=log_dir,
        logging_steps=args.log_every_n_steps,
        report_to=["tensorboard"],

        # Saving
        save_strategy="steps",
        save_steps=args.save_steps if hasattr(args, 'save_steps') else 1000,
        save_total_limit=args.save_total_limit if hasattr(args, 'save_total_limit') else 3,

        # Evaluation
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if hasattr(args, 'eval_steps') else 500,

        # DeepSpeed
        deepspeed=args.deepspeed_config if hasattr(args, 'deepspeed_config') and args.use_deepspeed else None,

        # Other
        dataloader_num_workers=args.dataloader_num_workers if hasattr(args, 'dataloader_num_workers') else 4,
        remove_unused_columns=False,  # Important: we use custom collator
        label_names=[],  # No labels column, losses computed internally
    )

    # -------------------------------------------------------------------------
    # Trainer
    # -------------------------------------------------------------------------
    callbacks = [
        WorldModelLoggingCallback(
            log_dir=log_dir,
            log_every_n_steps=args.log_every_n_steps,
            sample_every_n_steps=args.sample_every_n_steps,
        ),
    ]

    if hasattr(args, 'early_stopping_patience') and args.early_stopping_patience:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
            )
        )

    trainer = WorldModelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
        # Loss weights
        text_loss_weight=args.text_loss_weight,
        audio_loss_weight=args.audio_loss_weight,
        voice_loss_weight=args.voice_loss_weight,
        image_loss_weight=args.image_loss_weight,
        # Audio VAE for decoded evaluation (general audio, no speaker conditioning)
        audio_vae_checkpoint=args.audio_vae_checkpoint,
        audio_vae_config=args.audio_vae_config,
        audio_latent_channels=args.audio_latent_channels,
        # Voice VAE (may have speaker conditioning)
        voice_vae_checkpoint=args.voice_vae_checkpoint,  # Falls back to audio if None
        voice_vae_config=args.voice_vae_config,  # Falls back to audio if None
        voice_latent_channels=args.voice_latent_channels,  # Falls back to audio if None
        voice_speaker_embedding_dim=args.voice_speaker_embedding_dim,
        # Image VAE
        image_vae_checkpoint=args.image_vae_checkpoint,
        image_vae_config=args.image_vae_config,
        image_latent_channels=args.image_latent_channels,
        # Vocoder
        vocoder_checkpoint=args.vocoder_checkpoint,
        vocoder_config=args.vocoder_config,
        # Evaluation settings
        log_dir=log_dir,
        n_eval_examples=args.n_eval_examples,
        sample_rate=args.sample_rate,
    )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    logger.info("Starting training...")

    # Save config
    config_path = os.path.join(run_dir, "world_model_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "config_name": args.config,
            "data_dir": args.data_dir,
            "text_loss_weight": args.text_loss_weight,
            "audio_loss_weight": args.audio_loss_weight,
            "voice_loss_weight": args.voice_loss_weight,
            "image_loss_weight": args.image_loss_weight,
            "total_params": total_params,
            "trainable_params": trainable_params,
        }, f, indent=2)

    # Resume from checkpoint if specified
    resume_from = None
    if hasattr(args, 'resume_from_checkpoint') and args.resume_from_checkpoint:
        resume_from = args.resume_from_checkpoint
        logger.info(f"Resuming from checkpoint: {resume_from}")

    train_result = trainer.train(resume_from_checkpoint=resume_from)

    # Save final model
    trainer.save_model(os.path.join(run_dir, "final"))

    # Log final metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("Training complete!")
    logger.info(f"Final model saved to {os.path.join(run_dir, 'final')}")


if __name__ == "__main__":
    main()