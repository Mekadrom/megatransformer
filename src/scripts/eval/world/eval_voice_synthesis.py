"""Evaluate voice synthesis (text → voice) quality.

For each text sample, generates voice via the world model, decodes through
CVAE + vocoder, and computes:
1. Mel Cepstral Distortion (MCD) against ground-truth mel spectrogram
2. Speaker embedding cosine similarity (if speaker encoder available)

Usage:
    python -m src.scripts.eval.world.eval_voice_synthesis --checkpoint_path runs/my_run/checkpoint-3000 --config small_sum_dit --cache_dir ../cached_datasets/sive --include_modes text,voice --voice_cvae_checkpoint_path ./runs/audio_cvae/.../checkpoint-300000 --voice_cvae_config medium_decoder_only_1d_3x --voice_cvae_latent_channels 128 --vocoder_config hifigan --static_speaker_embedding_path ./logs/speaker_embedding_1.pt --max_samples 100 --bf16
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.amp import autocast

from model.world.world_model import MegaTransformerWorldModel
from utils import model_loading_utils
from utils.constants import BOV_TOKEN_ID


def parse_args():
    p = argparse.ArgumentParser(description="Voice synthesis eval (MCD + speaker similarity)")
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--config", type=str, default="small_sum_dit")
    p.add_argument("--include_modes", type=str, default="text,voice")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--text_cache_dir", type=str, default=None)
    p.add_argument("--voice_cache_dir", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--use_memorization_dataset", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--tie_word_embeddings", action="store_true")
    # Decoders
    p.add_argument("--voice_cvae_checkpoint_path", type=str, default=None)
    p.add_argument("--voice_cvae_config", type=str, default="small")
    p.add_argument("--voice_cvae_latent_channels", type=int, default=None)
    p.add_argument("--vocoder_config", type=str, default="hifigan")
    p.add_argument("--vocoder_checkpoint_path", type=str, default=None)
    p.add_argument("--static_speaker_embedding_path", type=str, default=None)
    p.add_argument("--save_audio", type=str, default=None,
                   help="Directory to save generated .wav files")
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--split", type=str, default="val", help="Dataset split (train/val)")
    p.add_argument("--log_dir", type=str, default=None, help="TensorBoard log dir for metrics")
    p.add_argument("--step", type=int, default=None, help="Step number (inferred from checkpoint path if omitted)")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_world_model(args, device):
    include_modes = [m.strip() for m in args.include_modes.split(",")]
    overrides = {"include_modes": include_modes}
    if args.tie_word_embeddings:
        overrides["tie_word_embeddings"] = True
    return model_loading_utils.load_model(
        MegaTransformerWorldModel, args.config,
        checkpoint_path=args.checkpoint_path,
        overrides=overrides, device=device,
    )


def load_dataset(args, split="val"):
    include_modes = [m.strip() for m in args.include_modes.split(",")]

    def resolve(specific, base, s):
        d = specific or base
        if d is None:
            return None
        for candidate in [d + "_" + s, d]:
            if os.path.isdir(candidate):
                return candidate
        return None

    text_dir = resolve(args.text_cache_dir, args.cache_dir, split) if "text" in include_modes else None
    voice_dir = resolve(args.voice_cache_dir, args.cache_dir, split) if "voice" in include_modes else None

    if args.use_memorization_dataset:
        from scripts.data.world.memorization_dataset import MultimodalMemorizationDataset
        return MultimodalMemorizationDataset(
            text_shard_dir=text_dir, voice_shard_dir=voice_dir,
            max_samples=args.max_samples,
        )
    else:
        from scripts.data.world.dataset import MultimodalShardedDataset
        return MultimodalShardedDataset(
            text_shard_dir=text_dir, voice_shard_dir=voice_dir,
            cache_size=32, max_samples=args.max_samples,
        )


def decode_sive_to_mel(cvae_decoder, latent, speaker_embedding):
    """Decode SIVE latent (C, T) → mel spectrogram (n_mels, T)."""
    device = next(cvae_decoder.parameters()).device
    dtype = next(cvae_decoder.parameters()).dtype
    z = latent.to(device=device, dtype=dtype).unsqueeze(0)
    spk = speaker_embedding.to(device=device, dtype=dtype).unsqueeze(0)
    with torch.no_grad():
        mel = cvae_decoder.decode(z=z, speaker_embedding=spk, features=z)
    if isinstance(mel, tuple):
        mel = mel[0]
    if isinstance(mel, dict):
        mel = mel.get("reconstructed", next(iter(mel.values())))
    return mel[0].float().cpu()


def mel_to_waveform(vocoder, mel):
    """Decode mel (n_mels, T) → waveform (samples,)."""
    from utils.visualization import render_vocoder_audio
    return render_vocoder_audio(vocoder, mel)


def compute_mcd(pred_mel, target_mel, n_mfcc=13):
    """Mel Cepstral Distortion between two mel spectrograms.

    Lower is better. Uses DCT to convert mel to cepstral coefficients.
    """
    # Align lengths
    T = min(pred_mel.shape[-1], target_mel.shape[-1])
    pred = pred_mel[:, :T].float()
    target = target_mel[:, :T].float()

    # Log mel (add eps for stability)
    pred_log = torch.log(pred.clamp_min(1e-7))
    target_log = torch.log(target.clamp_min(1e-7))

    # DCT-II to get MFCCs (approximate via matrix multiply)
    N = pred_log.shape[0]  # n_mels
    n = torch.arange(N, dtype=torch.float32)
    k = torch.arange(n_mfcc, dtype=torch.float32)
    dct_matrix = torch.cos(torch.pi / N * (n.unsqueeze(1) + 0.5) * k.unsqueeze(0))  # (N, n_mfcc)
    dct_matrix *= (2.0 / N) ** 0.5

    pred_mfcc = dct_matrix.T @ pred_log  # (n_mfcc, T)
    target_mfcc = dct_matrix.T @ target_log  # (n_mfcc, T)

    # Skip c0 (energy), use c1..c12
    diff = pred_mfcc[1:] - target_mfcc[1:]
    mcd = (10.0 / torch.log(torch.tensor(10.0))) * (2.0 * (diff ** 2).sum(dim=0)).sqrt().mean()
    return mcd.item()


def encode_static_prompt(text, suffix_tokens, tokenizer, max_new_tokens, max_seq_len, device):
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    max_prompt = max_seq_len - max_new_tokens - len(suffix_tokens)
    token_ids = token_ids[:max(1, max_prompt)]
    all_ids = token_ids + suffix_tokens
    return torch.tensor([all_ids], dtype=torch.long, device=device)


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    # Load CVAE decoder
    cvae_decoder = None
    if args.voice_cvae_checkpoint_path:
        from model.audio.vae.vae import AudioCVAEDecoderOnly
        cvae_overrides = {}
        if args.voice_cvae_latent_channels is not None:
            cvae_overrides["latent_channels"] = args.voice_cvae_latent_channels
        cvae_decoder = model_loading_utils.load_model(
            AudioCVAEDecoderOnly, args.voice_cvae_config,
            checkpoint_path=args.voice_cvae_checkpoint_path,
            strict=False, overrides=cvae_overrides,
        )
        cvae_decoder.eval()
        print("Loaded CVAE decoder")

    # Load vocoder
    vocoder = None
    if args.vocoder_config or args.vocoder_checkpoint_path:
        try:
            from utils.audio_utils import SharedWindowBuffer
            vocoder = model_loading_utils.load_vocoder(
                args.vocoder_checkpoint_path, args.vocoder_config, SharedWindowBuffer(),
            )
            print("Loaded vocoder")
        except Exception as e:
            print(f"Warning: Failed to load vocoder: {e}")

    # Load speaker embedding
    static_speaker_emb = None
    if args.static_speaker_embedding_path:
        static_speaker_emb = torch.load(
            args.static_speaker_embedding_path, map_location="cpu", weights_only=True,
        )
        print(f"Loaded speaker embedding: {static_speaker_emb.shape}")

    # Load world model
    print(f"Loading world model from {args.checkpoint_path}...")
    model = load_world_model(args, device)
    model.to(device)
    model.eval()

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args, split=args.split)
    print(f"Dataset: {len(dataset)} samples")

    if args.save_audio:
        os.makedirs(args.save_audio, exist_ok=True)

    mcd_scores = []
    spk_similarities = []

    for i in range(len(dataset)):
        sample = dataset[i]
        voice_features = sample.get("voice_features")
        if voice_features is None:
            continue

        # Get text prompt
        text = sample.get("text_text", "")
        if not text:
            text_token_ids = sample.get("text_token_ids")
            if text_token_ids is not None:
                text_length = sample.get("text_text_length", len(text_token_ids))
                if isinstance(text_length, torch.Tensor):
                    text_length = text_length.item()
                text_ids = [t for t in text_token_ids[:text_length].tolist() if t < 32000 and t != 0]
                text = tokenizer.decode(text_ids, skip_special_tokens=True)
        if isinstance(text, list):
            text = text[0] if text else ""
        text = str(text).strip()
        if not text:
            continue

        # Build synthesis prompt: [text] [BOV]
        prompt = encode_static_prompt(
            text[:500], [BOV_TOKEN_ID], tokenizer,
            args.max_new_tokens, 1024, device,
        )

        # Generate voice
        with torch.no_grad():
            with autocast(device, dtype=dtype, enabled=args.bf16):
                outputs = model.generate(
                    text_input_ids=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )

        voice_preds = outputs.get("voice_latent_preds")
        if voice_preds is None or voice_preds.numel() == 0:
            print(f"[{i}] No voice generated, skipping")
            continue

        gen_latent = voice_preds[0, 0]  # (C, T)

        # Get speaker embedding (from sample or static)
        speaker_emb = sample.get("voice_speaker_embedding", static_speaker_emb)
        if speaker_emb is None:
            print(f"[{i}] No speaker embedding, skipping MCD/audio")
            continue

        metrics_str = []

        # MCD: compare generated vs target mel spectrograms
        if cvae_decoder is not None:
            target_mel = sample.get("voice_mel_spec")  # (n_mels, T)
            gen_mel = decode_sive_to_mel(cvae_decoder, gen_latent, speaker_emb)

            if target_mel is not None:
                mcd = compute_mcd(gen_mel, target_mel)
                mcd_scores.append(mcd)
                metrics_str.append(f"MCD={mcd:.2f}")

            # Save audio
            if args.save_audio and vocoder is not None:
                try:
                    import torchaudio
                    gen_wav = mel_to_waveform(vocoder, gen_mel)
                    if gen_wav is not None:
                        if gen_wav.dim() == 1:
                            gen_wav = gen_wav.unsqueeze(0)
                        torchaudio.save(
                            os.path.join(args.save_audio, f"gen_{i}.wav"),
                            gen_wav.cpu(), args.sample_rate,
                        )
                    if target_mel is not None:
                        tgt_wav = mel_to_waveform(vocoder, target_mel)
                        if tgt_wav is not None:
                            if tgt_wav.dim() == 1:
                                tgt_wav = tgt_wav.unsqueeze(0)
                            torchaudio.save(
                                os.path.join(args.save_audio, f"target_{i}.wav"),
                                tgt_wav.cpu(), args.sample_rate,
                            )
                except Exception as e:
                    print(f"  Warning: audio save failed: {e}")

        # Speaker similarity via cosine of SIVE embeddings (crude but free)
        target_features = voice_features  # (C, T)
        feat_len = sample.get("voice_feature_length", target_features.shape[-1])
        if isinstance(feat_len, torch.Tensor):
            feat_len = feat_len.item()
        target_flat = target_features[:, :feat_len].flatten()
        gen_flat = gen_latent[:, :min(gen_latent.shape[-1], feat_len)].flatten()
        # Pad shorter to match
        max_len = max(target_flat.shape[0], gen_flat.shape[0])
        target_padded = F.pad(target_flat, (0, max_len - target_flat.shape[0]))
        gen_padded = F.pad(gen_flat, (0, max_len - gen_flat.shape[0]))
        spk_sim = F.cosine_similarity(
            gen_padded.unsqueeze(0), target_padded.unsqueeze(0),
        ).item()
        spk_similarities.append(spk_sim)
        metrics_str.append(f"SIVE_cos={spk_sim:.4f}")

        print(f"[{i}] {', '.join(metrics_str)}  prompt: {text[:80]}")

    print(f"\n{'='*60}")
    print(f"Voice Synthesis Results ({max(len(mcd_scores), len(spk_similarities))} samples)")

    if mcd_scores:
        scores = torch.tensor(mcd_scores)
        print(f"  Mean MCD:         {scores.mean():.2f} dB")
        print(f"  Std:              {scores.std():.2f}")
        print(f"  Min:              {scores.min():.2f}")
        print(f"  Max:              {scores.max():.2f}")

    if spk_similarities:
        sims = torch.tensor(spk_similarities)
        print(f"  Mean SIVE cosine: {sims.mean():.4f}")
        print(f"  Std:              {sims.std():.4f}")
        print(f"  Min:              {sims.min():.4f}")
        print(f"  Max:              {sims.max():.4f}")

    # Log to TensorBoard
    from scripts.eval.world.eval_utils import infer_step_from_checkpoint, init_eval_metrics, log_eval_scalars
    step = args.step if args.step is not None else infer_step_from_checkpoint(args.checkpoint_path)
    init_eval_metrics(args.log_dir, args.checkpoint_path)
    metrics_dict = {}
    if mcd_scores:
        metrics_dict["eval/voice_synthesis_mcd_mean"] = torch.tensor(mcd_scores).mean().item()
    if spk_similarities:
        metrics_dict["eval/voice_synthesis_sive_cosine_mean"] = torch.tensor(spk_similarities).mean().item()
    if metrics_dict:
        log_eval_scalars(metrics_dict, step)

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
