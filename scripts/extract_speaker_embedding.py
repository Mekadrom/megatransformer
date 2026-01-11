#!/usr/bin/env python3
"""
Extract speaker embeddings from an audio file.

Uses the centralized speaker encoder utility for consistency across the codebase.
Supports ECAPA-TDNN (192-dim) and WavLM (768-dim) speaker encoders.

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
        choices=["ecapa_tdnn", "wavlm"],
        help="Speaker encoder type: ecapa_tdnn (192-dim) or wavlm (768-dim) (default: ecapa_tdnn)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available)",
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.audio_path):
        raise FileNotFoundError(f"Audio file not found: {args.audio_path}")

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
    print(f"  Device: {args.device}")

    # Extract embedding
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
