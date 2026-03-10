#!/usr/bin/env python3
"""
Extract a speaker embedding from an audio file and save it as a .pt tensor.

Usage:
    python -m src.scripts.extract_speaker_embedding --audio_path path/to/reference.wav --output_path path/to/speaker_embedding.pt

The saved tensor can be used with --static_speaker_embedding_path in world model training.
"""
import argparse

import torch
import torchaudio

from utils.speaker_encoder import get_speaker_encoder, SPEAKER_ENCODER_INPUT_TYPES


def main():
    parser = argparse.ArgumentParser(description="Extract speaker embedding from audio file")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to input audio file (wav, flac, etc.)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the speaker embedding .pt file")
    parser.add_argument("--encoder_type", type=str, default="ecapa_tdnn", choices=["ecapa_tdnn", "wavlm"], help="Speaker encoder type")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--n_mels", type=int, default=80, help="Number of mel bands (for ecapa_tdnn)")
    parser.add_argument("--n_fft", type=int, default=400, help="FFT size for mel spectrogram (for ecapa_tdnn)")
    parser.add_argument("--hop_length", type=int, default=160, help="Hop length for mel spectrogram (for ecapa_tdnn)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load audio
    print(f"Loading audio: {args.audio_path}")
    waveform, sr = torchaudio.load(args.audio_path)

    # Resample if needed
    if sr != args.sample_rate:
        print(f"Resampling from {sr}Hz to {args.sample_rate}Hz")
        waveform = torchaudio.functional.resample(waveform, sr, args.sample_rate)

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    print(f"Audio: {waveform.shape[1] / args.sample_rate:.2f}s, {waveform.shape[1]} samples")

    # Load encoder
    encoder = get_speaker_encoder(encoder_type=args.encoder_type, device=device)
    input_type = SPEAKER_ENCODER_INPUT_TYPES[args.encoder_type]

    if input_type == "mel":
        # Compute mel spectrogram for ECAPA-TDNN
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
        )
        mel_spec = mel_transform(waveform)  # (1, n_mels, T)
        mel_spec = torch.log(mel_spec.clamp(min=1e-5))
        print(f"Mel spectrogram: {mel_spec.shape}")
        embedding = encoder(mel_spec=mel_spec.to(device))
    else:
        # Raw waveform for WavLM
        embedding = encoder(waveform=waveform.squeeze(0).to(device))

    # Save as 1D tensor (speaker_dim,)
    embedding = embedding.squeeze(0).cpu()
    print(f"Speaker embedding: shape={embedding.shape}, dtype={embedding.dtype}")

    torch.save(embedding, args.output_path)
    print(f"Saved to: {args.output_path}")


if __name__ == "__main__":
    main()
