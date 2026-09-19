"""Evaluate voice synthesis (text → voice) quality.

For each text sample, generates voice via the world model, decodes through
SMG + vocoder, and computes:
1. Mel Cepstral Distortion (MCD) against ground-truth mel spectrogram
2. Speaker embedding cosine similarity (if speaker encoder available)

Usage:
    python -m megatransformer.scripts.eval.world.eval_voice_synthesis --checkpoint_path runs/my_run/checkpoint-3000 --config small_sum_dit --cache_dir ../cached_datasets/sive --include_modes text,voice --voice_smg_checkpoint_path ./runs/smg/.../checkpoint-300000 --voice_smg_config medium_decoder_only_1d_3x --voice_smg_sive_encoder_dim 256 --vocoder_config hifigan --static_speaker_embedding_path ./logs/speaker_embedding_1.pt --max_samples 100 --bf16
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.amp import autocast

from megatransformer.utils.tokenizer_resolution import resolve_tokenizer_name
from megatransformer.model.world.world_model import MegaTransformerWorldModel
from megatransformer.utils import model_loading_utils
from megatransformer.utils.constants import BOV_TOKEN_ID


def parse_args():
    p = argparse.ArgumentParser(description="Voice synthesis eval (MCD + speaker similarity)")
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--text_tokenizer", type=str, default=None,
                   help="Corpus tokenizer, as training was given it. Composes with either "
                        "prelude; --text_encoder_model wins when both are set. Without this a "
                        "non-Mistral corpus is encoded/decoded with Mistral, silently.")
    p.add_argument("--text_encoder_model", type=str, default=None,
                   help="Set only if training used a PRETRAINED text prelude.")
    p.add_argument("--config", type=str, default="small_sum_dit")
    p.add_argument("--include_modes", type=str, default="text,voice")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--text_cache_dir", type=str, default=None)
    p.add_argument("--voice_cache_dir", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--use_memorization_dataset", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8,
                   help="TEXT sampling temperature (does not affect the voice path -- use "
                        "--voice_temperature for that).")
    p.add_argument("--voice_temperature", type=float, default=0.0,
                   help="Voice sampling temperature. 0.0 = greedy/argmax (default). The discrete "
                        "(Mimi/VQ) coda softmaxes its unit logits at this temperature, so this is "
                        "the knob to sweep for the machine-gun-vs-coherence question. Higher = more "
                        "random token switching.")
    p.add_argument("--voice_token_budget", type=int, default=None,
                   help="Hard cap on generated content frames before force-closing with EOV. "
                        "None => generate()'s default (209). For 12.5 Hz Mimi, 125 ~= 10 s.")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--tie_word_embeddings", action="store_true")
    # Decoders
    p.add_argument("--voice_smg_checkpoint_path", type=str, default=None)
    p.add_argument("--voice_smg_config", type=str, default="small")
    p.add_argument("--voice_smg_sive_encoder_dim", type=int, default=None)
    p.add_argument("--voice_smg_speaker_embedding_dim", type=int, default=None,
                   help="Speaker-embedding width the SMG was trained at (192 ECAPA vs 768 WavLM). "
                        "Required for a WavLM SMG or the speaker_proj weights load as random.")
    # World-model build overrides -- must match the values the run was TRAINED with, or the
    # rebuilt architecture won't match the checkpoint (feature_channels defaults to 128 in
    # small_sum; a Mimi run uses 256). Mirror the training --voice_* args.
    p.add_argument("--voice_feature_channels", type=int, default=None,
                   help="Voice prelude/coda feature width the world model was trained at "
                        "(256 for the Mimi/WavLM run; small_sum defaults to 128).")
    p.add_argument("--voice_predict_f0", action="store_true",
                   help="The world run trained a coda F0 head (--voice_predict_f0). Required to "
                        "rebuild the matching coda + emit the SMG's F0 contour.")
    p.add_argument("--voice_codebook_path", type=str, default=None,
                   help="Discrete (Mimi/VQ) codebook the run used. Sizes the coda's unit_vocab "
                        "head (K+1) and is installed as the model's centroid buffer so generate() "
                        "can map sampled unit ids -> centroids.")
    p.add_argument("--vocoder_config", type=str, default="hifigan")
    p.add_argument("--vocoder_checkpoint_path", type=str, default=None)
    p.add_argument("--static_speaker_embedding_path", type=str, default=None)
    p.add_argument("--save_audio", type=str, default=None,
                   help="Directory to save generated .wav files")
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--mel_hop_length", type=int, default=320,
                   help="Hop the SMG's mel output is at. 320 = 50 Hz (ContentVec-rate SMG, "
                        "the 1x decoder configs); 256 = 62.5 Hz (SIVE-rate SMG). Resampled to "
                        "the vocoder's own rate before synthesis when the two differ.")
    p.add_argument("--split", type=str, default="val", help="Dataset split (train/val)")
    p.add_argument("--log_dir", type=str, default=None, help="TensorBoard log dir for metrics")
    p.add_argument("--step", type=int, default=None, help="Step number (inferred from checkpoint path if omitted)")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_world_model(args, device):
    import copy
    from megatransformer.model.world.world_model import WORLD_MODEL_CONFIGS

    include_modes = [m.strip() for m in args.include_modes.split(",")]
    overrides = {"include_modes": include_modes}
    if args.tie_word_embeddings:
        overrides["tie_word_embeddings"] = True
    if getattr(args, "voice_cfg_enabled", False):
        overrides["voice_cfg_enabled"] = True  # create null_text_embed to load a CFG checkpoint

    # Rebuild the voice prelude/coda sub-configs to match the TRAINED architecture before the
    # checkpoint loads (mirrors world/training.py's load_model). small_sum's defaults differ
    # from a Mimi run (feature_channels 128 vs 256, no F0 head, default unit vocab), which is a
    # hard state_dict size mismatch otherwise. Deep-copy so the shared global config is untouched.
    base = WORLD_MODEL_CONFIGS.get(args.config)
    if base is not None and (args.voice_feature_channels is not None
                             or args.voice_predict_f0
                             or args.voice_codebook_path is not None):
        prelude_cfg = copy.deepcopy(base.voice_prelude_config)
        coda_cfg = copy.deepcopy(base.voice_coda_config)
        if args.voice_feature_channels is not None:
            prelude_cfg.feature_channels = args.voice_feature_channels
            coda_cfg.feature_channels = args.voice_feature_channels
        if args.voice_predict_f0:
            coda_cfg.predict_f0 = True
        if args.voice_codebook_path:
            from megatransformer.utils.codebook import load_codebook
            K = int(load_codebook(args.voice_codebook_path).shape[0])
            coda_cfg.unit_vocab_size = K + 1  # +1 for the terminal EOV unit
        overrides["voice_prelude_config"] = prelude_cfg
        overrides["voice_coda_config"] = coda_cfg

    # Pretrained-LLM text encoder: match the trained architecture (frozen LLM body + translators +
    # special extension). from_config reconstructs the config, so __post_init__ re-derives the
    # interleaver placeholder ids from the overridden special_token_base.
    if getattr(args, "text_encoder_model", None):
        from transformers import AutoConfig, AutoTokenizer
        from megatransformer.utils import constants
        _llm_cfg = AutoConfig.from_pretrained(args.text_encoder_model)
        _eos = _llm_cfg.eos_token_id
        if _eos is None:
            _eos = AutoTokenizer.from_pretrained(args.text_encoder_model).eos_token_id
        overrides["special_token_base"] = int(_llm_cfg.vocab_size)
        overrides["eos_token_id"] = int(_eos)
        overrides["text_encoder"] = {
            "model": args.text_encoder_model, "freeze": True,
            "translator_hidden_mult": 2.0, "n_special_tokens": constants.N_SPECIAL_TOKENS,
        }

    model = model_loading_utils.load_model(
        MegaTransformerWorldModel, args.config,
        checkpoint_path=args.checkpoint_path,
        overrides=overrides, device=device,
    )
    # Install the centroids so generate()'s discrete path can map unit ids -> centroid frames.
    if args.voice_codebook_path:
        from megatransformer.utils.codebook import load_codebook
        model.set_voice_codebook(load_codebook(args.voice_codebook_path))
    return model


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
        from megatransformer.scripts.data.world.memorization_dataset import MultimodalMemorizationDataset
        return MultimodalMemorizationDataset(
            text_shard_dir=text_dir, voice_shard_dir=voice_dir,
            max_samples=args.max_samples,
        )
    else:
        from megatransformer.scripts.data.world.dataset import MultimodalShardedDataset
        # Pass the codebook so a pre-quantized (Mimi/VQ) cache -- which stores only unit_ids --
        # expands into continuous centroid `voice_features`. Without it every sample has no
        # voice_features and the eval loop skips all of them (0 samples). Must be the SAME
        # codebook the run trained with.
        return MultimodalShardedDataset(
            text_shard_dir=text_dir, voice_shard_dir=voice_dir,
            cache_size=32, max_samples=args.max_samples,
            voice_codebook=args.voice_codebook_path,
        )


def decode_sive_to_mel(smg_decoder, latent, speaker_embedding, f0_contour=None):
    """Decode SIVE latent (C, T) → mel spectrogram (n_mels, T).

    f0_contour (T,) is the world model's SPEAKER-NORMALIZED contour, in sigma units
    rather than log Hz. It goes to the SMG's F0 predictor, which denormalizes it with
    ECAPA -- that denormalization is the SMG's job precisely because the world model
    only sees text and cannot know the speaker's pitch offset. Handing this straight to
    f0_embedding() instead would read ~1.1 sigma as ~1.1 log Hz, i.e. 3 Hz.
    """
    device = next(smg_decoder.parameters()).device
    dtype = next(smg_decoder.parameters()).dtype
    z = latent.to(device=device, dtype=dtype).unsqueeze(0)
    spk = speaker_embedding.to(device=device, dtype=dtype).unsqueeze(0)
    kwargs = {}
    if f0_contour is not None:
        contour = f0_contour.to(device=device, dtype=dtype)
        if contour.shape[-1] != z.shape[-1]:
            # Same generation loop appends both, so this should not drift; trim rather
            # than interpolate so a real desync shows up as a length mismatch instead of
            # being silently resampled into plausible-looking prosody.
            T = min(contour.shape[-1], z.shape[-1])
            contour, z = contour[..., :T], z[..., :T]
        kwargs["f0_contour"] = contour.unsqueeze(0)  # (1, T)
    with torch.no_grad():
        mel = smg_decoder.decode(z=z, speaker_embedding=spk, features=z, **kwargs)
    if isinstance(mel, tuple):
        mel = mel[0]
    if isinstance(mel, dict):
        mel = mel.get("reconstructed", next(iter(mel.values())))
    return mel[0].float().cpu()


def mel_to_waveform(vocoder, mel, mel_hop_length=None):
    """Decode mel (n_mels, T) → waveform (samples,).

    mel_hop_length is the rate the SMG emits at; render_vocoder_audio resamples to the
    vocoder's own rate when they differ. Without it a 50 Hz mel plays 1.25x too fast
    through the 62.5 Hz HiFi-GAN, which also skews every metric computed downstream.
    """
    from megatransformer.utils.visualization import render_vocoder_audio
    return render_vocoder_audio(vocoder, mel, mel_hop_length=mel_hop_length)


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
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_tokenizer_name(args=args))

    # Load SMG decoder
    smg_decoder = None
    if args.voice_smg_checkpoint_path:
        from megatransformer.model.smg.smg import SMG
        smg_overrides = {}
        if args.voice_smg_sive_encoder_dim is not None:
            smg_overrides["sive_encoder_dim"] = args.voice_smg_sive_encoder_dim
        # F0 conditioning derives its harmonic phase step from hop/sample_rate, so both must
        # match the mel rate the SMG was trained at (256/24000 for the 24 kHz Vocos path).
        smg_overrides["hop_length"] = args.mel_hop_length
        smg_overrides["sample_rate"] = args.sample_rate
        # Speaker-embedding width must match the checkpoint (192 ECAPA vs 768 WavLM) or the
        # FiLM / F0 speaker_proj weights load as random under strict=False.
        if args.voice_smg_speaker_embedding_dim is not None:
            smg_overrides["speaker_embedding_dim"] = args.voice_smg_speaker_embedding_dim
        # A discrete-unit SMG (num_codes>0, e.g. Mimi) needs a code_embed_init at build; the
        # trained unit embedding overwrites it, so learned_random avoids requiring the codebook.
        smg_overrides["code_embed_init"] = "learned_random"
        smg_decoder = model_loading_utils.load_model(
            SMG, args.voice_smg_config,
            checkpoint_path=args.voice_smg_checkpoint_path,
            strict=False, overrides=smg_overrides,
        )
        smg_decoder.eval()
        print("Loaded SMG decoder")

    # Load vocoder
    vocoder = None
    if args.vocoder_config or args.vocoder_checkpoint_path:
        try:
            from megatransformer.utils.audio_utils import SharedWindowBuffer
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

    # The SMG and the world model have to agree about who supplies F0, and the two are
    # loaded from independent checkpoints with nothing forcing them to match. Check once
    # here rather than discovering it after generating every sample.
    smg_needs_contour = (
        smg_decoder is not None
        and getattr(smg_decoder, "f0_predictor_input", "features") == "contour"
    )
    world_predicts_f0 = getattr(getattr(model, "voice_generator", None), "f0_head", None) is not None
    if smg_needs_contour and not world_predicts_f0:
        raise SystemExit(
            f"--voice_smg_config {args.voice_smg_config} reads a speaker-normalized F0 "
            "contour (f0_predictor_input='contour'), but this world model has no F0 head "
            "and cannot supply one. Its units are prosody-free, so the result would be "
            "flat regardless. Train the world model with --voice_predict_f0, or pair this "
            "checkpoint with an f0_predictor_input='features' SMG."
        )
    if world_predicts_f0 and not smg_needs_contour:
        print("Note: world model predicts an F0 contour but this SMG infers F0 from "
              "content features — the predicted contour will be ignored.")

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
                text_ids = [t for t in text_token_ids[:text_length].tolist()
                            if t < constants.vocab_bound(tokenizer) and t != 0]
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
                gen_kwargs = dict(
                    text_input_ids=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    voice_temperature=args.voice_temperature,
                )
                if args.voice_token_budget is not None:
                    gen_kwargs["voice_token_budget"] = args.voice_token_budget
                outputs = model.generate(**gen_kwargs)

        voice_preds = outputs.get("voice_latent_preds")
        if voice_preds is None or voice_preds.numel() == 0:
            print(f"[{i}] No voice generated, skipping")
            continue

        gen_latent = voice_preds[0, 0]  # (C, T)

        # The contour the coda emitted for these same frames (see generate()). Padded by
        # the same helper with the same time_dim, so it lines up frame-for-frame.
        gen_contour = None
        f0_preds = outputs.get("voice_f0_preds")
        if f0_preds is not None and f0_preds.numel() > 0:
            gen_contour = f0_preds[0, 0]  # (T,)
        if smg_needs_contour and gen_contour is None:
            print(f"[{i}] Voice generated but no F0 contour came back, skipping")
            continue

        # Get speaker embedding (from sample or static)
        speaker_emb = sample.get("voice_speaker_embedding", static_speaker_emb)
        if speaker_emb is None:
            print(f"[{i}] No speaker embedding, skipping MCD/audio")
            continue

        metrics_str = []

        # MCD: compare generated vs target mel spectrograms
        if smg_decoder is not None:
            target_mel = sample.get("voice_mel_spec")  # (n_mels, T)
            gen_mel = decode_sive_to_mel(smg_decoder, gen_latent, speaker_emb, f0_contour=gen_contour)

            if target_mel is not None:
                mcd = compute_mcd(gen_mel, target_mel)
                mcd_scores.append(mcd)
                metrics_str.append(f"MCD={mcd:.2f}")

            # Save audio
            if args.save_audio and vocoder is not None:
                try:
                    import torchaudio

                    def _save_wav(wav, path):
                        # render_vocoder_audio returns a numpy array; torchaudio needs a
                        # (channels, samples) float tensor.
                        if wav is None:
                            return
                        if not isinstance(wav, torch.Tensor):
                            wav = torch.as_tensor(wav)
                        wav = wav.float().cpu()
                        if wav.dim() == 1:
                            wav = wav.unsqueeze(0)
                        torchaudio.save(path, wav, args.sample_rate)

                    _save_wav(mel_to_waveform(vocoder, gen_mel, mel_hop_length=args.mel_hop_length),
                              os.path.join(args.save_audio, f"gen_{i}.wav"))
                    if target_mel is not None:
                        _save_wav(mel_to_waveform(vocoder, target_mel, mel_hop_length=args.mel_hop_length),
                                  os.path.join(args.save_audio, f"target_{i}.wav"))
                except Exception as e:
                    print(f"  Warning: audio save failed: {e}")

        # Speaker similarity via cosine of SIVE embeddings (crude but free)
        target_features = voice_features  # (C, T)
        feat_len = sample.get("voice_feature_length", target_features.shape[-1])
        if isinstance(feat_len, torch.Tensor):
            feat_len = feat_len.item()
        # gen_latent is on the model device; target_features comes off the CPU dataset --
        # bring both to CPU before the cosine or it's a cross-device RuntimeError.
        target_flat = target_features[:, :feat_len].flatten().cpu()
        gen_flat = gen_latent[:, :min(gen_latent.shape[-1], feat_len)].flatten().float().cpu()
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
    from megatransformer.scripts.eval.world.eval_utils import infer_step_from_checkpoint, init_eval_metrics, log_eval_scalars
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
