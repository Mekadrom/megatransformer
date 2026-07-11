"""
Gradio demo for voice cloning using a content encoder + SMG + vocoder.

Uploads a content audio and a speaker reference audio, extracts content features
(SIVE from the mel, or off-the-shelf ContentVec from the waveform), ECAPA-TDNN
speaker embeddings from the reference, runs the SMG decoder to produce a mel
spectrogram, and synthesizes audio via vocoder. If the SMG mel hop differs from
the vocoder's (e.g. a 50 Hz ContentVec SMG @hop320 vs a 62.5 Hz @hop256 vocoder),
the mel is resampled to the vocoder's rate before synthesis.

Usage (SIVE):
    python -m megatransformer.scripts.eval.smg.voice_clone --content_encoder sive --sive_checkpoint_path ./checkpoints/sive/checkpoint-60000 --sive_config tiny_deep --smg_checkpoint_path ./checkpoints/smg/checkpoint-50000 --smg_config small_decoder_only_1d --vocoder_config hifigan
Usage (ContentVec vec256, 50 Hz — matches the smg_..._1d1x_contentvec run):
    python -m megatransformer.scripts.eval.smg.voice_clone --content_encoder contentvec --contentvec_dim 256 --hop_length 320 --smg_checkpoint_path runs/smg/smg_libritts_r_1d1x_contentvec_baseline_nogan_0/checkpoint-27000 --smg_config medium_decoder_only_1d_1x --vocoder_config hifigan
"""
import argparse

import gradio as gr
import numpy as np
import torch
import torchaudio

from megatransformer.model.voice.sive.sive import SpeakerInvariantVoiceEncoder
from megatransformer.model.smg.smg import SMG
from megatransformer.utils.audio_utils import SharedWindowBuffer, extract_mels
from megatransformer.utils.model_loading_utils import load_model, load_vocoder
from megatransformer.utils.speaker_encoder import get_speaker_encoder


def parse_args():
    parser = argparse.ArgumentParser(description="Voice cloning with SIVE + SMG")

    # Content encoder — must match how the SMG was trained
    parser.add_argument("--content_encoder", type=str, default="sive", choices=["sive", "contentvec"],
                        help="Content features source: 'sive' (custom checkpoint) or 'contentvec' (off-the-shelf HuBERT, on the raw waveform).")
    parser.add_argument("--contentvec_model", type=str, default="lengyue233/content-vec-best",
                        help="HF model id for ContentVec (transformers.HubertModel), used when --content_encoder contentvec.")
    parser.add_argument("--contentvec_dim", type=int, default=768, choices=[256, 768],
                        help="ContentVec feature width — MUST match how the SMG was trained. 768 = "
                             "last_hidden_state; 256 = final_proj head (vec256). Matches the preprocessor's "
                             "--contentvec_dim and the SMG's --sive_encoder_dim.")

    # SIVE (used when --content_encoder sive)
    parser.add_argument("--sive_checkpoint_path", type=str, default=None, help="Path to SIVE checkpoint directory (required for --content_encoder sive)")
    parser.add_argument("--sive_config", type=str, default="tiny_deep", help="SIVE config name")
    parser.add_argument("--sive_layer", type=int, default=10, help="SIVE all_hiddens index to extract from (0=subsampling frontend, 1-12=encoder layers 1-12). Default 10 = encoder layer 10 = -3 from the end of 12 encoder layers.")
    parser.add_argument("--num_speakers", type=int, default=2338, help="Number of speakers for SIVE model (must match checkpoint)")
    parser.add_argument("--speaker_pooling", type=str, default=None, help="SIVE speaker pooling override (e.g. 'attentive_statistics', 'mean')")

    # SMG
    parser.add_argument("--smg_checkpoint_path", type=str, required=True, help="Path to SMG checkpoint directory")
    parser.add_argument("--smg_config", type=str, default="small_decoder_only_1d", help="SMG config name")

    # Vocoder
    parser.add_argument("--vocoder_checkpoint_path", type=str, default=None, help="Path to custom vocoder checkpoint (None for pretrained)")
    parser.add_argument("--vocoder_config", type=str, default="hifigan", help="Vocoder config name (use 'hifigan' for pretrained HiFi-GAN)")

    # Audio
    parser.add_argument("--speaker_embedding_dim", type=int, default=192,
                        help="ECAPA speaker embedding width (must match the SMG's training).")
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


def load_embedding_file(path, dim):
    """Load an ECAPA speaker embedding from a .pt file for speaker-space probing.

    Accepts a [dim] or [1, dim] tensor (or a dict holding one under a common key).
    Returns a [1, dim] float CPU tensor. Raises gr.Error on a shape/format mismatch."""
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for k in ("speaker_embedding", "speaker_embeddings", "embedding", "emb", "ecapa"):
            if k in obj:
                obj = obj[k]
                break
        else:
            raise gr.Error(f".pt is a dict with no embedding key (keys: {list(obj)[:6]})")
    t = torch.as_tensor(obj).float().reshape(-1)
    if t.numel() != dim:
        raise gr.Error(f"embedding has {t.numel()} values, expected {dim} (ECAPA)")
    return t.reshape(1, dim)


def main():
    args = parse_args()
    device = args.device
    shared_window_buffer = SharedWindowBuffer()

    print("Loading models...")

    # Load the content encoder — SIVE (custom) or ContentVec (off-the-shelf HuBERT).
    sive = None
    contentvec = None
    if args.content_encoder == "sive":
        assert args.sive_checkpoint_path, "--sive_checkpoint_path is required for --content_encoder sive"
        print(f"Loading SIVE ({args.sive_config}) from {args.sive_checkpoint_path}")
        sive = load_model(
            SpeakerInvariantVoiceEncoder,
            args.sive_config,
            checkpoint_path=args.sive_checkpoint_path,
            device=device,
            overrides={k: v for k, v in {"num_speakers": args.num_speakers, "speaker_pooling": args.speaker_pooling}.items() if v is not None},
        ).eval()
        content_dim = sive.config.encoder_dim
        print(f"SIVE loaded ({sum(p.numel() for p in sive.parameters()):,} params)")
    else:
        # Match the preprocessor EXACTLY: load via load_contentvec (adds the final_proj
        # head for dim=256) and extract last_hidden_state (-> final_proj if 256).
        from megatransformer.utils.contentvec_features import load_contentvec
        print(f"Loading ContentVec: {args.contentvec_model} (dim={args.contentvec_dim})")
        contentvec = load_contentvec(args.contentvec_model, device, dim=args.contentvec_dim)
        content_dim = args.contentvec_dim
        print(f"ContentVec loaded ({sum(p.numel() for p in contentvec.parameters()):,} params), dim={content_dim}")

    # Load SMG — wire sive_encoder_dim from the content encoder's dim, and hop_length
    # so the F0-embedding phase matches the mel rate (e.g. 320 for a 50 Hz ContentVec SMG).
    print(f"Loading SMG ({args.smg_config}) from {args.smg_checkpoint_path}")
    print(f"  Wiring sive_encoder_dim={content_dim}, hop_length={args.hop_length}")
    smg = load_model(
        SMG,
        args.smg_config,
        checkpoint_path=args.smg_checkpoint_path,
        device=device,
        overrides={"sive_encoder_dim": content_dim, "hop_length": args.hop_length},
    )
    smg.eval()
    print(f"SMG loaded ({sum(p.numel() for p in smg.parameters()):,} params)")

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

    import os
    import tempfile
    emb_dim = args.speaker_embedding_dim

    @torch.no_grad()
    def voice_clone(content_audio_path, speaker_audio_path, speaker_source,
                    emb_pt_file, blend_alpha, scale, noise_std, noise_seed, l2_normalize):
        if content_audio_path is None:
            raise gr.Error("Content audio is required.")

        # --- Resolve the base speaker embedding: from reference audio, an uploaded
        #     .pt, or a blend — so the embedding space can be probed directly. ---
        emb_audio = emb_pt = None
        if speaker_audio_path is not None:
            speaker_waveform = load_audio(speaker_audio_path, args.sample_rate).to(device)
            speaker_mel = extract_mels(shared_window_buffer, speaker_waveform, sr=args.sample_rate, n_mels=args.n_mels, n_fft=args.n_fft, hop_length=args.hop_length)
            emb_audio = speaker_encoder(mel_spec=speaker_mel.unsqueeze(0)).reshape(1, emb_dim).to(device)
        if emb_pt_file is not None:
            emb_pt = load_embedding_file(emb_pt_file, emb_dim).to(device)

        if speaker_source == "uploaded .pt":
            if emb_pt is None:
                raise gr.Error("Upload a speaker embedding .pt (or switch source).")
            base_emb = emb_pt
        elif speaker_source == "blend":
            if emb_audio is None or emb_pt is None:
                raise gr.Error("Blend needs BOTH a speaker reference audio and a .pt.")
            a = float(blend_alpha)
            base_emb = (1.0 - a) * emb_audio + a * emb_pt
        else:  # "reference audio"
            if emb_audio is None:
                raise gr.Error("Provide a speaker reference audio (or switch source).")
            base_emb = emb_audio

        # --- Deterministic probe modifiers ---
        emb = base_emb.clone()
        if float(scale) != 1.0:
            emb = emb * float(scale)
        if float(noise_std) > 0:
            g = torch.Generator(device="cpu").manual_seed(int(noise_seed))
            d = torch.randn(emb.shape, generator=g).to(emb.device)
            d = d / (d.norm(dim=-1, keepdim=True) + 1e-8)  # unit direction
            # Scale to a FRACTION of the embedding's own norm. ECAPA here is ||e||~286 and
            # the SMG L2-normalizes internally, so only the ANGLE matters: fraction f rotates
            # the embedding by ~arctan(f) (f=1 -> ~45 deg; different speakers sit ~76 deg apart).
            # A raw std (old behavior) was ~2% of ||e|| at the slider max => inaudible.
            emb = emb + (float(noise_std) * base_emb.norm(dim=-1, keepdim=True)) * d
        if l2_normalize:
            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        speaker_embedding = emb  # [1, emb_dim]

        # --- Export the EXACT embedding used, as [emb_dim], for download+re-upload ---
        fd, emb_out_path = tempfile.mkstemp(suffix="_ecapa.pt"); os.close(fd)
        torch.save(speaker_embedding.squeeze(0).cpu().clone(), emb_out_path)
        stats = (f"source={speaker_source} | scale={float(scale):g} "
                 f"noise={float(noise_std):g}(seed {int(noise_seed)}) l2norm={bool(l2_normalize)} | "
                 f"norm={speaker_embedding.norm().item():.3f} mean={speaker_embedding.mean().item():+.4f} "
                 f"min={speaker_embedding.min().item():+.3f} max={speaker_embedding.max().item():+.3f}")

        content_waveform = load_audio(content_audio_path, args.sample_rate).to(device)

        # Content features from the content audio -> [1, D, T'] (channel-first for the SMG).
        if sive is not None:
            # SIVE takes the mel; extract a specific encoder layer.
            content_mel = extract_mels(shared_window_buffer, content_waveform, sr=args.sample_rate, n_mels=args.n_mels, n_fft=args.n_fft, hop_length=args.hop_length)
            feats, _ = sive.extract_features(
                content_mel.unsqueeze(0),
                lengths=torch.tensor([content_mel.shape[-1]], device=device),
                layer=args.sive_layer,
            )  # [1, T', D]
            content_features = feats.permute(0, 2, 1)
        else:
            # ContentVec takes the raw waveform -> [T@50Hz, D] -> [1, D, T]. Match the
            # preprocessor: last_hidden_state (layer=-1), + final_proj when dim=256.
            from megatransformer.utils.contentvec_features import contentvec_hidden
            hidden = contentvec_hidden(contentvec, content_waveform, layer=-1,
                                       final_proj=(args.contentvec_dim == 256))  # [T', D]
            content_features = hidden.permute(1, 0).unsqueeze(0)  # [1, D, T']

        # Run SMG decoder
        mel_recon = smg.decode(
            z=content_features,
            speaker_embedding=speaker_embedding,
            features=content_features,
        )

        # Handle 2D decoder output [B, 1, n_mels, T] -> [B, n_mels, T]
        if mel_recon.dim() == 4:
            mel_recon = mel_recon.squeeze(1)

        # Resample the mel to the vocoder's frame rate if the SMG mel hop differs
        # (e.g. a 50 Hz ContentVec mel @hop320 -> a 62.5 Hz @hop256 vocoder). No-op
        # when they match, so SIVE runs at the vocoder's hop are unaffected.
        voc_hop = getattr(getattr(vocoder, "config", None), "hop_length", args.hop_length)
        if args.hop_length != voc_hop:
            new_T = max(1, round(mel_recon.shape[-1] * args.hop_length / voc_hop))
            mel_recon = torch.nn.functional.interpolate(mel_recon.float(), size=new_T, mode="linear", align_corners=False)

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
        return (args.sample_rate, waveform_np), emb_out_path, stats

    demo = gr.Interface(
        fn=voice_clone,
        inputs=[
            gr.Audio(type="filepath", label="Content Audio (what to say)"),
            gr.Audio(type="filepath", label="Speaker Reference (optional if using a .pt)"),
            gr.Radio(["reference audio", "uploaded .pt", "blend"], value="reference audio",
                     label="Speaker source"),
            gr.File(type="filepath", file_types=[".pt"],
                    label=f"Speaker embedding .pt  ([{emb_dim}] or [1,{emb_dim}])"),
            gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Blend alpha (0=audio → 1=.pt)"),
            gr.Number(value=1.0, label="Scale (× embedding) — NO-OP if the model L2-normalizes"),
            gr.Slider(0.0, 1.0, value=0.0, step=0.02,
                      label="Add noise (fraction of ‖emb‖, random dir; f≈angle: 1→~45°, speakers ~76° apart)"),
            gr.Number(value=0, precision=0, label="Noise seed"),
            gr.Checkbox(value=False, label="L2-normalize embedding — NO-OP if the model normalizes internally"),
        ],
        outputs=[
            gr.Audio(type="numpy", label="Voice Cloned Output"),
            gr.File(label=f"Embedding used (.pt, [{emb_dim}]) — download, modify, re-upload"),
            gr.Textbox(label="Embedding stats"),
        ],
        title=f"Voice Cloning / speaker-space probe — {args.content_encoder} + SMG",
        description=(
            "Content features come from "
            f"{'SIVE' if args.content_encoder == 'sive' else 'ContentVec'}; speaker identity from a "
            f"[{emb_dim}]-dim ECAPA embedding. Speaker source = a reference audio, an uploaded .pt "
            "embedding, or a blend; then probe the space with scale / additive noise / L2-normalize. "
            "NOTE: if the SMG L2-normalizes the embedding internally (this run does), only the "
            "DIRECTION matters — scale and L2-normalize are no-ops, and noise is scaled as a fraction "
            "of ‖emb‖ so it maps to a real rotation angle. Meaningful moves = large angles: blend "
            "toward another speaker's .pt, or noise fraction ≳0.3. The exact embedding used is exported "
            "as a .pt so you can download it, edit it, and re-upload."
        ),
    )
    demo.launch(share=args.share, server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
