"""
Gradio demo for voice cloning using SIVE + CVAE + vocoder.

Uploads a content audio and a speaker reference audio, extracts SIVE features
from the content, ECAPA-TDNN speaker embeddings from the reference, runs the
CVAE decoder to produce a mel spectrogram, and synthesizes audio via vocoder.

Usage:
    python -m src.scripts.eval.audio.cvae.voice_clone --sive_checkpoint_path ./checkpoints/sive/checkpoint-60000 --sive_config tiny_deep --cvae_checkpoint_path ./checkpoints/cvae/checkpoint-50000 --cvae_config small_decoder_only_1d --vocoder_config hifigan
"""
import argparse

import gradio as gr
import numpy as np
import torch
import torchaudio

from model.audio.sive.sive import SpeakerInvariantVoiceEncoder
from model.audio.vae.vae import AudioVAE, AudioCVAEDecoderOnly
from utils.audio_utils import SharedWindowBuffer, extract_mels
from utils.model_loading_utils import load_model, load_vocoder
from utils.speaker_encoder import get_speaker_encoder


def parse_args():
    parser = argparse.ArgumentParser(description="Voice cloning with SIVE + CVAE")

    # SIVE
    parser.add_argument("--sive_checkpoint_path", type=str, required=True, help="Path to SIVE checkpoint directory")
    parser.add_argument("--sive_config", type=str, default="tiny_deep", help="SIVE config name")
    parser.add_argument("--sive_layer", type=int, default=10, help="SIVE all_hiddens index to extract from (0=subsampling frontend, 1-12=encoder layers 1-12). Default 10 = encoder layer 10 = -3 from the end of 12 encoder layers.")
    parser.add_argument("--num_speakers", type=int, default=2338, help="Number of speakers for SIVE model (must match checkpoint)")
    parser.add_argument("--speaker_pooling", type=str, default=None, help="SIVE speaker pooling override (e.g. 'attentive_statistics', 'mean')")

    # CVAE
    parser.add_argument("--cvae_checkpoint_path", type=str, required=True, help="Path to CVAE checkpoint directory")
    parser.add_argument("--cvae_config", type=str, default="small_decoder_only_1d", help="CVAE config name")
    parser.add_argument("--decoder_only", action="store_true", default=True, help="Use AudioCVAEDecoderOnly (default: True)")
    parser.add_argument("--no_decoder_only", action="store_true", help="Use full AudioVAE instead of decoder-only")

    # Vocoder
    parser.add_argument("--vocoder_checkpoint_path", type=str, default=None, help="Path to custom vocoder checkpoint (None for pretrained)")
    parser.add_argument("--vocoder_config", type=str, default="hifigan", help="Vocoder config name (use 'hifigan' for pretrained HiFi-GAN)")

    # Audio
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)

    # Runtime
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860)

    return parser.parse_args()


def load_audio(path, sample_rate):
    """Load audio file, resample to target rate, convert to mono."""
    waveform, sr = torchaudio.load(path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0)  # [T]


def main():
    args = parse_args()
    decoder_only = args.decoder_only and not args.no_decoder_only
    device = args.device
    shared_window_buffer = SharedWindowBuffer()

    print("Loading models...")

    # Load SIVE
    print(f"Loading SIVE ({args.sive_config}) from {args.sive_checkpoint_path}")
    sive = load_model(
        SpeakerInvariantVoiceEncoder,
        args.sive_config,
        checkpoint_path=args.sive_checkpoint_path,
        device=device,
        overrides={k: v for k, v in {"num_speakers": args.num_speakers, "speaker_pooling": args.speaker_pooling}.items() if v is not None},
    )
    sive.eval()
    print(f"SIVE loaded ({sum(p.numel() for p in sive.parameters()):,} params)")

    # Load CVAE — auto-wire latent_channels from SIVE encoder_dim
    sive_encoder_dim = sive.config.encoder_dim
    cvae_cls = AudioCVAEDecoderOnly if decoder_only else AudioVAE
    print(f"Loading {'AudioCVAEDecoderOnly' if decoder_only else 'AudioVAE'} ({args.cvae_config}) from {args.cvae_checkpoint_path}")
    print(f"  Auto-wiring latent_channels={sive_encoder_dim} from SIVE encoder_dim")
    cvae = load_model(
        cvae_cls,
        args.cvae_config,
        checkpoint_path=args.cvae_checkpoint_path,
        device=device,
        overrides={"latent_channels": sive_encoder_dim},
    )
    cvae.eval()
    print(f"CVAE loaded ({sum(p.numel() for p in cvae.parameters()):,} params)")

    # Load speaker encoder
    print("Loading ECAPA-TDNN speaker encoder...")
    speaker_encoder = get_speaker_encoder(encoder_type="ecapa_tdnn", device=device)

    # Load vocoder
    print(f"Loading vocoder ({args.vocoder_config})...")
    vocoder = load_vocoder(args.vocoder_checkpoint_path, args.vocoder_config, shared_window_buffer)
    if vocoder is not None:
        vocoder = vocoder.to(device)
    else:
        raise RuntimeError("Failed to load vocoder")

    print("All models loaded.\n")

    @torch.no_grad()
    def voice_clone(content_audio_path, speaker_audio_path):
        if content_audio_path is None or speaker_audio_path is None:
            return None

        # Load audio files
        content_waveform = load_audio(content_audio_path, args.sample_rate).to(device)
        speaker_waveform = load_audio(speaker_audio_path, args.sample_rate).to(device)

        # Extract mel spectrograms
        content_mel = extract_mels(shared_window_buffer, content_waveform, sr=args.sample_rate, n_mels=args.n_mels, n_fft=args.n_fft, hop_length=args.hop_length)
        speaker_mel = extract_mels(shared_window_buffer, speaker_waveform, sr=args.sample_rate, n_mels=args.n_mels, n_fft=args.n_fft, hop_length=args.hop_length)
        # content_mel: [n_mels, T], speaker_mel: [n_mels, T]

        # Extract speaker embedding from reference audio
        # ECAPA-TDNN expects [B, n_mels, T]
        speaker_embedding = speaker_encoder(mel_spec=speaker_mel.unsqueeze(0))  # [1, 192]

        # Extract SIVE features from content audio
        # SIVE expects [B, n_mels, T]
        sive_features, feature_lengths = sive.extract_features(
            content_mel.unsqueeze(0),
            lengths=torch.tensor([content_mel.shape[-1]], device=device),
            layer=args.sive_layer,
        )
        # sive_features: [1, T', encoder_dim]

        # CVAE expects channel-first: [B, D, T']
        sive_features_cf = sive_features.permute(0, 2, 1)

        # Run CVAE decoder
        if decoder_only:
            mel_recon = cvae.decode(
                z=sive_features_cf,
                speaker_embedding=speaker_embedding,
                features=sive_features_cf,
            )
        else:
            # Full VAE: encode to latent, then decode
            mu, logvar = cvae.encode(sive_features_cf)
            mel_recon = cvae.decode(
                z=mu,
                speaker_embeddings=speaker_embedding,
                features=sive_features_cf,
            )

        # Handle 2D decoder output [B, 1, n_mels, T] -> [B, n_mels, T]
        if mel_recon.dim() == 4:
            mel_recon = mel_recon.squeeze(1)

        # Vocoder: mel -> waveform
        result = vocoder(mel_recon)
        pred_waveform = result["pred_waveform"]

        # Ensure [T] shape
        if pred_waveform.dim() > 1:
            pred_waveform = pred_waveform.squeeze(0)

        # Normalize to [-1, 1]
        peak = pred_waveform.abs().max()
        if peak > 0:
            pred_waveform = pred_waveform / peak

        waveform_np = pred_waveform.cpu().float().numpy()
        return (args.sample_rate, waveform_np)

    demo = gr.Interface(
        fn=voice_clone,
        inputs=[
            gr.Audio(type="filepath", label="Content Audio (what to say)"),
            gr.Audio(type="filepath", label="Speaker Reference (whose voice to use)"),
        ],
        outputs=gr.Audio(type="numpy", label="Voice Cloned Output"),
        title="Voice Cloning with SIVE + CVAE",
        description=(
            "Upload a content audio file and a speaker reference audio file. "
            "The system extracts linguistic features from the content audio using SIVE "
            "and speaker characteristics from the reference using ECAPA-TDNN, "
            "then generates speech in the target speaker's voice via the CVAE decoder."
        ),
    )
    demo.launch(share=args.share, server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
