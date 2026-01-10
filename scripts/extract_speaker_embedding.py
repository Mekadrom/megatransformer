#!/usr/bin/env python3
"""
Extract speaker embeddings from an audio file using ECAPA-TDNN (mel-spec version).

Uses the same mel spectrogram extraction and speaker embedding computation
as preprocess_audio_vae_dataset.py for consistency.

Example usage:
    python scripts/extract_speaker_embedding.py \
        --audio_path /path/to/audio.wav \
        --output_path /path/to/speaker_embedding.pt

    # With custom audio settings:
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


def load_speaker_encoder(device: str = "cuda"):
    """Load ECAPA-TDNN speaker encoder (mel-spec version) from SpeechBrain."""
    from speechbrain.inference.speaker import EncoderClassifier

    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb-mel-spec",
        savedir="pretrained_models/spkrec-ecapa-voxceleb-mel-spec",
        run_opts={"device": device},
    )
    return encoder


@torch.no_grad()
def extract_speaker_embedding(
    audio_path: str,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
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
        device: Device to run on

    Returns:
        Speaker embedding tensor of shape [192]
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

    # Remove channel dimension for extract_mels: [1, T] -> [T]
    waveform = waveform.squeeze(0)

    # Extract mel spectrogram using the same function as preprocessing
    shared_window_buffer = SharedWindowBuffer()
    mel_spec = extract_mels(
        shared_window_buffer,
        waveform,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )  # [n_mels, T]

    # Load speaker encoder
    speaker_encoder = load_speaker_encoder(device)

    # Prepare mel for ECAPA-TDNN
    # ECAPA-TDNN mel-spec model expects [B, T, n_mels]
    mel_for_ecapa = mel_spec.transpose(0, 1).unsqueeze(0).to(device)  # [1, T, n_mels]

    # Relative length (1.0 for full length)
    rel_lens = torch.tensor([1.0], device=device)

    # Call normalizer and embedding model directly
    # (encode_batch assumes compute_features exists, which mel-spec model lacks)
    normalized = speaker_encoder.mods.normalizer(mel_for_ecapa, rel_lens, epoch=1)
    speaker_embedding = speaker_encoder.mods.embedding_model(
        normalized, rel_lens
    ).squeeze(1).squeeze(0)  # [192]

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
    print(f"  Mel bands: {args.n_mels}")
    print(f"  FFT size: {args.n_fft}")
    print(f"  Hop length: {args.hop_length}")
    print(f"  Device: {args.device}")

    # Extract embedding
    speaker_embedding = extract_speaker_embedding(
        audio_path=args.audio_path,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        device=args.device,
    )

    # Save embedding
    torch.save(speaker_embedding, args.output_path)

    print(f"\nSpeaker embedding saved to: {args.output_path}")
    print(f"  Shape: {speaker_embedding.shape}")
    print(f"  Dtype: {speaker_embedding.dtype}")


if __name__ == "__main__":
    main()
