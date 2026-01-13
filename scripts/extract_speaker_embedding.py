#!/usr/bin/env python3
"""
Extract speaker embeddings from an audio file.

Supports three modes:
1. Pretrained speaker encoders (ECAPA-TDNN, WavLM) - uses pretrained models
2. Learned speaker embeddings from VAE encoder - uses a trained VAE checkpoint

Example usage:
    # Using ECAPA-TDNN (default, 192-dim):
    python scripts/extract_speaker_embedding.py \
        --audio_path /path/to/audio.wav \
        --output_path /path/to/speaker_embedding.pt

    # Using WavLM (768-dim, richer features):
    python scripts/extract_speaker_embedding.py \
        --audio_path /path/to/audio.wav \
        --output_path /path/to/speaker_embedding.pt \
        --speaker_encoder_type wavlm

    # Using learned speaker embedding from VAE encoder:
    python scripts/extract_speaker_embedding.py \
        --audio_path /path/to/audio.wav \
        --output_path /path/to/speaker_embedding.pt \
        --speaker_encoder_type vae_encoder \
        --vae_checkpoint runs/audio_vae/my_run/checkpoint-10000 \
        --vae_config small

    # With custom audio settings (ECAPA-TDNN only):
    python scripts/extract_speaker_embedding.py \
        --audio_path /path/to/audio.wav \
        --output_path /path/to/speaker_embedding.pt \
        --sample_rate 16000 \
        --n_mels 80 \
        --n_fft 1024 \
        --hop_length 256
"""

import argparse
import os
import sys

# Add repo root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torchaudio

from dataset_loading.audio_loading import extract_mels
from utils.audio_utils import SharedWindowBuffer
from utils.speaker_encoder import (
    get_speaker_encoder,
    get_speaker_encoder_input_type,
    SpeakerEncoderType,
)


@torch.no_grad()
def extract_speaker_embedding_from_audio(
    audio_path: str,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    speaker_encoder_type: SpeakerEncoderType = "ecapa_tdnn",
    device: str = "cuda",
) -> torch.Tensor:
    """
    Extract speaker embedding from an audio file.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length for spectrogram
        speaker_encoder_type: Type of speaker encoder ("ecapa_tdnn" or "wavlm")
        device: Device to run on

    Returns:
        Speaker embedding tensor of shape [embedding_dim]
    """
    # Load audio
    waveform, sr = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    # Remove channel dimension: [1, T] -> [T]
    waveform = waveform.squeeze(0)

    # Get cached speaker encoder
    encoder = get_speaker_encoder(encoder_type=speaker_encoder_type, device=device)
    input_type = get_speaker_encoder_input_type(speaker_encoder_type)

    if input_type == "waveform":
        # WavLM: use waveform directly
        waveform_gpu = waveform.unsqueeze(0).to(device)  # [1, T]
        speaker_embedding = encoder(
            waveform=waveform_gpu,
            sample_rate=sample_rate,
        ).squeeze(0)  # [embedding_dim]
    else:
        # ECAPA-TDNN: extract mel spectrogram first
        shared_window_buffer = SharedWindowBuffer()
        mel_spec = extract_mels(
            shared_window_buffer,
            waveform,
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )  # [n_mels, T]

        speaker_embedding = encoder(
            mel_spec=mel_spec,
        ).squeeze(0)  # [embedding_dim]

    return speaker_embedding.cpu()


def load_vae_encoder(
    checkpoint_path: str,
    config: str,
    latent_channels: int = 32,
    learned_speaker_dim: int = 256,
    device: str = "cuda",
):
    """
    Load VAE encoder from checkpoint for speaker embedding extraction.

    The VAE must have been trained with learn_speaker_embedding=True.
    """
    from model.audio.vae import model_config_lookup
    from utils.model_loading_utils import load_model

    if config not in model_config_lookup:
        raise ValueError(f"Unknown audio VAE config: {config}. Available: {list(model_config_lookup.keys())}")

    # Create model with learn_speaker_embedding=True
    # Note: speaker_embedding_dim is for the decoder's FiLM conditioning, which should
    # match the learned_speaker_dim when using learned embeddings
    model = model_config_lookup[config](
        latent_channels=latent_channels,
        speaker_embedding_dim=learned_speaker_dim,
        learn_speaker_embedding=True,
        learned_speaker_dim=learned_speaker_dim,
    )

    # Load weights
    model, _ = load_model(False, model, checkpoint_path)

    # Verify that the encoder has the speaker head
    if not hasattr(model.encoder, 'speaker_head'):
        raise ValueError(
            "VAE encoder does not have a speaker_head. "
            "Make sure the checkpoint was trained with learn_speaker_embedding=True"
        )

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_speaker_embedding_from_vae(
    model,
    audio_path: str,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Extract learned speaker embedding from audio using VAE encoder.

    Args:
        model: VAE model with learn_speaker_embedding=True
        audio_path: Path to audio file
        sample_rate: Target sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length for spectrogram
        device: Device to run on

    Returns:
        Speaker embedding tensor of shape [embedding_dim]
    """
    # Load audio
    waveform, sr = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    # Remove channel dimension: [1, T] -> [T]
    waveform = waveform.squeeze(0)

    # Extract mel spectrogram
    shared_window_buffer = SharedWindowBuffer()
    mel_spec = extract_mels(
        shared_window_buffer,
        waveform,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )  # [n_mels, T]

    # Add batch and channel dimensions for VAE: [1, 1, n_mels, T]
    mel_spec = mel_spec.unsqueeze(0).unsqueeze(0).to(device)

    # Encode to get learned speaker embedding
    # VAE encoder returns (mu, logvar, learned_speaker_embedding) when learn_speaker_embedding=True
    encode_result = model.encode(mel_spec)
    if len(encode_result) == 3:
        _, _, speaker_embedding = encode_result
    else:
        raise ValueError(
            "VAE encoder did not return learned speaker embedding. "
            "Make sure the model was trained with learn_speaker_embedding=True"
        )

    return speaker_embedding.squeeze(0).cpu()


def main():
    parser = argparse.ArgumentParser(
        description="Extract speaker embedding from an audio file"
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Path to input audio file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save speaker embedding (.pt file)",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Target sample rate (default: 16000)",
    )
    parser.add_argument(
        "--n_mels",
        type=int,
        default=80,
        help="Number of mel bands (default: 80)",
    )
    parser.add_argument(
        "--n_fft",
        type=int,
        default=1024,
        help="FFT window size (default: 1024)",
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=256,
        help="Hop length (default: 256)",
    )
    parser.add_argument(
        "--speaker_encoder_type",
        type=str,
        default="ecapa_tdnn",
        choices=["ecapa_tdnn", "wavlm", "vae_encoder"],
        help="Speaker encoder type: ecapa_tdnn (192-dim), wavlm (768-dim), or vae_encoder (learned, requires --vae_checkpoint) (default: ecapa_tdnn)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available)",
    )

    # VAE encoder arguments (required when speaker_encoder_type is vae_encoder)
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default=None,
        help="Path to VAE checkpoint directory (required when --speaker_encoder_type=vae_encoder)",
    )
    parser.add_argument(
        "--vae_config",
        type=str,
        default=None,
        help="VAE config name, e.g., 'small', 'medium' (required when --speaker_encoder_type=vae_encoder)",
    )
    parser.add_argument(
        "--latent_channels",
        type=int,
        default=32,
        help="Number of latent channels in VAE (default: 32)",
    )
    parser.add_argument(
        "--learned_speaker_dim",
        type=int,
        default=256,
        help="Dimension of learned speaker embeddings from VAE encoder (default: 256)",
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.audio_path):
        raise FileNotFoundError(f"Audio file not found: {args.audio_path}")

    # Validate VAE arguments
    if args.speaker_encoder_type == "vae_encoder":
        if args.vae_checkpoint is None:
            raise ValueError("--vae_checkpoint is required when --speaker_encoder_type=vae_encoder")
        if args.vae_config is None:
            raise ValueError("--vae_config is required when --speaker_encoder_type=vae_encoder")

    # Create output directory if needed
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Extracting speaker embedding from: {args.audio_path}")
    print(f"  Sample rate: {args.sample_rate}")
    print(f"  Speaker encoder: {args.speaker_encoder_type}")
    if args.speaker_encoder_type == "ecapa_tdnn":
        print(f"  Mel bands: {args.n_mels}")
        print(f"  FFT size: {args.n_fft}")
        print(f"  Hop length: {args.hop_length}")
    elif args.speaker_encoder_type == "vae_encoder":
        print(f"  VAE checkpoint: {args.vae_checkpoint}")
        print(f"  VAE config: {args.vae_config}")
        print(f"  Learned speaker dim: {args.learned_speaker_dim}")
        print(f"  Mel bands: {args.n_mels}")
        print(f"  FFT size: {args.n_fft}")
        print(f"  Hop length: {args.hop_length}")
    print(f"  Device: {args.device}")

    # Extract embedding
    if args.speaker_encoder_type == "vae_encoder":
        # Load VAE model with learned speaker embedding head
        print("\nLoading VAE encoder...")
        model = load_vae_encoder(
            checkpoint_path=args.vae_checkpoint,
            config=args.vae_config,
            latent_channels=args.latent_channels,
            learned_speaker_dim=args.learned_speaker_dim,
            device=args.device,
        )
        print("VAE encoder loaded successfully")

        # Extract learned speaker embedding
        speaker_embedding = extract_speaker_embedding_from_vae(
            model=model,
            audio_path=args.audio_path,
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            device=args.device,
        )
    else:
        # Use pretrained speaker encoder (ECAPA-TDNN or WavLM)
        speaker_embedding = extract_speaker_embedding_from_audio(
            audio_path=args.audio_path,
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            speaker_encoder_type=args.speaker_encoder_type,
            device=args.device,
        )

    # Save embedding
    torch.save(speaker_embedding, args.output_path)

    print(f"\nSpeaker embedding saved to: {args.output_path}")
    print(f"  Shape: {speaker_embedding.shape}")
    print(f"  Dtype: {speaker_embedding.dtype}")


if __name__ == "__main__":
    main()
