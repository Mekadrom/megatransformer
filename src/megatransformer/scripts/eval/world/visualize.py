"""Standalone world model visualization eval script.

Loads a checkpoint, runs the full visualization callback (same scenarios as
during training), and writes results to a TensorBoard log directory.

Usage:
    python -m megatransformer.scripts.eval.world.visualize --checkpoint_path runs/my_run/checkpoint-3000 --config small_sum_dit --log_dir runs/eval_viz --cache_dir ../cached_datasets/sive --include_modes text,voice,image [--vocoder_checkpoint_path ...] [--voice_smg_checkpoint_path ...] [--image_vae_decoder_config litevae] [--use_memorization_dataset --max_samples 50]
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch

from megatransformer.model.world.world_model import MegaTransformerWorldModel
from megatransformer.scripts.data.world.data_collator import MultimodalDataCollator
from megatransformer.scripts.data.world.dataset import MultimodalShardedDataset
from megatransformer.scripts.train.world.visualization_callback import WorldModelVisualizationCallback
from megatransformer.utils import metrics, model_loading_utils, constants


@dataclass
class FakeState:
    """Minimal stand-in for transformers TrainerState."""
    global_step: int = 0
    is_world_process_zero: bool = True


@dataclass
class FakeArgs:
    """Minimal stand-in for training args needed by the callback."""
    bf16: bool = False
    fp16: bool = False


class FakeTrainer:
    """Minimal stand-in for the Trainer — only needs datasets and collator."""
    def __init__(self, train_dataset, eval_dataset, data_collator):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator


def parse_args():
    p = argparse.ArgumentParser(description="Standalone world model visualization eval")

    # Required
    p.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint directory")
    p.add_argument("--config", type=str, default="small_sum_dit", help="World model config name")
    p.add_argument("--text_encoder_model", type=str, default=None,
                   help="Pretrained-LLM text encoder id (e.g. HuggingFaceTB/SmolLM2-135M). Must match "
                        "the checkpoint; sets special_token_base to the LLM vocab and eos to its native eos.")
    p.add_argument("--log_dir", type=str, required=True, help="TensorBoard log output directory")

    # Dataset
    p.add_argument("--cache_dir", type=str, default=None, help="Base cache dir (appends _train/_val)")
    p.add_argument("--text_cache_dir", type=str, default=None)
    p.add_argument("--audio_cache_dir", type=str, default=None)
    p.add_argument("--voice_cache_dir", type=str, default=None)
    p.add_argument("--image_cache_dir", type=str, default=None)
    p.add_argument("--include_modes", type=str, default="text,voice,image")
    p.add_argument("--include_tasks", type=str, default=None,
                   help="Comma-separated tasks to visualize (e.g. voice_synthesis). Gates out "
                        "transcription / cross-modal / OOD generated-text. None = all applicable.")
    p.add_argument("--use_memorization_dataset", action="store_true")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--num_eval_samples", type=int, default=4)

    # Collator
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--voice_max_seconds", type=float, default=10.0)
    p.add_argument("--voice_sample_rate", type=int, default=16000)
    p.add_argument("--voice_hop_length", type=int, default=256)
    p.add_argument("--voice_n_mels", type=int, default=80)
    p.add_argument("--voice_n_fft", type=int, default=1024)
    p.add_argument("--audio_max_seconds", type=float, default=10.0)
    p.add_argument("--audio_sample_rate", type=int, default=16000)
    p.add_argument("--audio_hop_length", type=int, default=256)
    p.add_argument("--audio_n_mels", type=int, default=80)
    p.add_argument("--audio_n_fft", type=int, default=1024)
    p.add_argument("--sive_total_stride", type=int, default=3)

    # Optional decoders
    p.add_argument("--vocoder_checkpoint_path", type=str, default=None)
    p.add_argument("--vocoder_config", type=str, default="hifigan")
    p.add_argument("--image_vae_decoder_path", type=str, default=None)
    p.add_argument("--image_vae_decoder_config", type=str, default=None)
    p.add_argument("--voice_smg_checkpoint_path", type=str, default=None)
    p.add_argument("--voice_smg_config", type=str, default="small")
    p.add_argument("--voice_smg_sive_encoder_dim", type=int, default=None)
    # VQ-voice config (mirror world/training.py) — the checkpoint saves no config.json, so
    # these training-time overrides must be re-applied or the coda's unit head / feature
    # projections won't match the saved weights.
    p.add_argument("--voice_codebook_path", type=str, default=None,
                   help="Sizes the coda's unit head (unit_vocab_size = K+1: K content units "
                        "from this codebook plus the EOV terminal token).")
    p.add_argument("--voice_feature_channels", type=int, default=None,
                   help="SIVE feature width (256 for VQ-SIVE); sizes prelude/coda feature I/O.")
    p.add_argument("--voice_predict_f0", action="store_true",
                   help="Enable the coda's F0 head (must match the trained checkpoint).")
    p.add_argument("--static_speaker_embedding_path", type=str, default=None)

    # Model overrides
    p.add_argument("--gen_query_mode", type=str, default=None)
    p.add_argument("--n_image_gen_positions", type=int, default=None)
    p.add_argument("--iteration_norm", type=str, default=None)
    p.add_argument("--share_block_weights", action="store_true")
    p.add_argument("--tie_word_embeddings", action="store_true")

    # Eval
    p.add_argument("--step", type=int, default=0, help="Step number for TensorBoard logging")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--device", type=str, default=None, help="Device (default: cuda if available)")

    return p.parse_args()


def resolve_shard_dir(cache_dir, split):
    if cache_dir is None:
        return None
    # Accept, in order: a dir already pointing AT the split (…/val passed for split "val");
    # the subdir layout <base>/<split> (the standard preprocess output); and the legacy
    # suffix layout <base>_<split>.
    base = cache_dir.rstrip("/")
    candidates = []
    if os.path.basename(base) == split:
        candidates.append(cache_dir)
    candidates.append(os.path.join(cache_dir, split))
    candidates.append(cache_dir + "_" + split)
    for c in candidates:
        if os.path.isdir(c):
            return c
    print(f"Warning: no shard dir for split '{split}' under {cache_dir} "
          f"(tried /{split} and _{split}), skipping")
    return None


def load_dataset(args, split):
    include_modes = [m.strip() for m in args.include_modes.split(",")]
    text_dir = resolve_shard_dir(args.text_cache_dir or args.cache_dir, split) if "text" in include_modes else None
    audio_dir = resolve_shard_dir(args.audio_cache_dir or args.cache_dir, split) if "audio" in include_modes else None
    voice_dir = resolve_shard_dir(args.voice_cache_dir or args.cache_dir, split) if "voice" in include_modes else None
    image_dir = resolve_shard_dir(args.image_cache_dir or args.cache_dir, split) if "image" in include_modes else None

    use_memorization = args.use_memorization_dataset
    max_samples = args.max_samples

    if use_memorization and max_samples is not None:
        from megatransformer.scripts.data.world.memorization_dataset import MultimodalMemorizationDataset
        return MultimodalMemorizationDataset(
            text_shard_dir=text_dir,
            audio_shard_dir=audio_dir,
            voice_shard_dir=voice_dir,
            image_shard_dir=image_dir,
            max_samples=max_samples,
        )
    else:
        return MultimodalShardedDataset(
            text_shard_dir=text_dir,
            audio_shard_dir=audio_dir,
            voice_shard_dir=voice_dir,
            image_shard_dir=image_dir,
            cache_size=32,
            max_samples=max_samples,
            # Same codebook the world model trained with: quantizes features to unit ids AND
            # loads per-speaker F0 stats so samples carry voice_f0_contour — which the
            # contour-mode SMG needs for target/input decodes.
            voice_codebook=getattr(args, "voice_codebook_path", None),
            # Match training's task filter so the dataset's round-robin only yields tasks
            # whose modalities are loaded (e.g. voice_synthesis, not text_continuation).
            include_tasks=[t.strip() for t in args.include_tasks.split(",")] if getattr(args, "include_tasks", None) else None,
        )


def load_world_model(args, device):
    include_modes = [m.strip() for m in args.include_modes.split(",")]
    overrides = {"include_modes": include_modes}
    if args.tie_word_embeddings:
        overrides["tie_word_embeddings"] = True
    if args.gen_query_mode is not None:
        overrides["gen_query_mode"] = args.gen_query_mode
    if args.n_image_gen_positions is not None:
        overrides["n_image_gen_positions"] = args.n_image_gen_positions

    config_name = args.config
    _text_enc = getattr(args, "text_encoder_model", None)
    _voice_over = (getattr(args, "voice_codebook_path", None)
                   or getattr(args, "voice_feature_channels", None)
                   or getattr(args, "voice_predict_f0", False))
    _mrope = (model_loading_utils.detect_world_mrope(args.checkpoint_path)
              if getattr(args, "checkpoint_path", None) else False)
    _nar = bool(getattr(args, "voice_nar", False)) or (
        model_loading_utils.detect_world_nar(args.checkpoint_path)
        if getattr(args, "checkpoint_path", None) else False)
    if (args.iteration_norm is not None or args.share_block_weights or _voice_over
            or _text_enc or _mrope or _nar):
        import copy
        from megatransformer.config.world.world_model import WORLD_MODEL_CONFIGS
        config = copy.deepcopy(WORLD_MODEL_CONFIGS[args.config])
        if args.iteration_norm is not None:
            config.recurrent_block_config.iteration_norm = args.iteration_norm
        if args.share_block_weights:
            config.recurrent_block_config.share_block_weights = True
        if _nar:
            # Both halves, or the head is silently crippled: the MASK parameter AND the
            # bidirectional coda (a causal coda cannot see units revealed to its right).
            config.voice_nar = True
            config.voice_coda_config.coda_config.causal = False
            print(f"  NAR voice ({'--voice_nar' if getattr(args, 'voice_nar', False) else 'detected in checkpoint'})"
                  " -> voice_nar=True, coda bidirectional")
        if _mrope and str(getattr(args, "mrope_scale_side", None)) == "off":
            print("  [ablation] M-RoPE checkpoint being evaluated with M-RoPE DISABLED "
                  "(single-axis RoPE) -- numbers are NOT comparable to a matched-geometry read")
            _mrope = False
        if _mrope:
            # Detected from the weights (rotary_global/local exist only under M-RoPE).
            # Mirrors training.py: the TRUNK only -- preludes/codas keep single-axis RoPE.
            side = getattr(args, "mrope_scale_side", None)
            if side is None:
                raise ValueError(
                    f"{args.checkpoint_path} was trained with M-RoPE, but --mrope_scale_side "
                    "was not supplied. It carries no weights, so it cannot be detected, and "
                    "guessing it evaluates the checkpoint under the wrong coordinate system. "
                    "Pass 'text' or 'voice' (matching the training CLI).")
            config.recurrent_block_config.block_config.use_mrope = True
            config.mrope_voice_rate = float(getattr(args, "mrope_voice_rate", None) or 6.0)
            config.mrope_scale_side = str(side)
            print(f"  Detected M-RoPE in checkpoint -> use_mrope=True, "
                  f"scale_side={config.mrope_scale_side}, rate={config.mrope_voice_rate}")
        # VQ-voice: re-apply the training-time voice config so the loaded architecture
        # matches the checkpoint (no config.json to recover it from).
        cbp = getattr(args, "voice_codebook_path", None)
        if cbp:
            from megatransformer.utils.codebook import load_codebook
            # +1 for the EOV terminal token (must match training: unit head is K+1-way,
            # EOV = class K). Loading with only K would shape-mismatch the checkpoint's
            # unit head and drop the discrete stop mechanism.
            config.voice_coda_config.unit_vocab_size = int(load_codebook(cbp).shape[0]) + 1
        vfc = getattr(args, "voice_feature_channels", None)
        if vfc:
            config.voice_prelude_config.feature_channels = vfc
            config.voice_coda_config.feature_channels = vfc
        if getattr(args, "voice_predict_f0", False):
            config.voice_coda_config.predict_f0 = True
        if _text_enc:
            # Pretrained-LLM text encoder: rebuild the same config training used so the loaded
            # architecture (frozen LLM body + translators + special extension) matches the ckpt.
            from transformers import AutoConfig, AutoTokenizer
            _llm_cfg = AutoConfig.from_pretrained(_text_enc)
            _eos = _llm_cfg.eos_token_id
            if _eos is None:
                _eos = AutoTokenizer.from_pretrained(_text_enc).eos_token_id
            config.special_token_base = int(_llm_cfg.vocab_size)
            config.eos_token_id = int(_eos)
            config.__post_init__()
            config.text_encoder = {
                "model": _text_enc,
                "freeze": True,
                "translator_hidden_mult": 2.0,
                "n_special_tokens": (model_loading_utils.detect_n_special_tokens(
                                         args.checkpoint_path)
                                     or constants.N_SPECIAL_TOKENS),
            }
        for k, v in overrides.items():
            setattr(config, k, v)
        WORLD_MODEL_CONFIGS[config_name + "_eval"] = config
        config_name = config_name + "_eval"

    model = model_loading_utils.load_model(
        MegaTransformerWorldModel,
        config_name,
        checkpoint_path=args.checkpoint_path,
        overrides=overrides,
        device=device,
    )
    return model


def load_decoders(args):
    vocoder = None
    if args.vocoder_checkpoint_path or args.vocoder_config:
        try:
            from megatransformer.utils.audio_utils import SharedWindowBuffer
            vocoder = model_loading_utils.load_vocoder(
                args.vocoder_checkpoint_path,
                args.vocoder_config,
                SharedWindowBuffer(),
            )
        except Exception as e:
            print(f"Warning: Failed to load vocoder: {e}")

    image_vae_decoder = None
    if args.image_vae_decoder_config == "litevae":
        try:
            from megatransformer.scripts.data.image.vae.preprocess import _load_litevae
            image_vae_decoder = _load_litevae("litevae", device="cpu")
            image_vae_decoder.eval()
            print("Loaded LiteVAE decoder")
        except Exception as e:
            print(f"Warning: Failed to load LiteVAE decoder: {e}")
    elif args.image_vae_decoder_path:
        try:
            from megatransformer.model.image.vae.vae import ImageVAEDecoder
            image_vae_decoder = model_loading_utils.load_model(
                ImageVAEDecoder,
                args.image_vae_decoder_config or "small",
                checkpoint_path=args.image_vae_decoder_path,
                strict=False,
            )
            image_vae_decoder.eval()
        except Exception as e:
            print(f"Warning: Failed to load image VAE decoder: {e}")

    voice_smg_decoder = None
    if args.voice_smg_checkpoint_path:
        try:
            from megatransformer.model.smg.smg import SMG
            smg_overrides = {}
            if args.voice_smg_sive_encoder_dim is not None:
                smg_overrides["sive_encoder_dim"] = args.voice_smg_sive_encoder_dim
            voice_smg_decoder = model_loading_utils.load_model(
                SMG,
                args.voice_smg_config,
                checkpoint_path=args.voice_smg_checkpoint_path,
                strict=False,
                overrides=smg_overrides,
            )
            voice_smg_decoder.eval()
        except Exception as e:
            print(f"Warning: Failed to load voice SMG decoder: {e}")

    static_speaker_embedding = None
    if args.static_speaker_embedding_path:
        try:
            static_speaker_embedding = torch.load(
                args.static_speaker_embedding_path, map_location="cpu", weights_only=True,
            )
            print(f"Loaded static speaker embedding: shape={static_speaker_embedding.shape}")
        except Exception as e:
            print(f"Warning: Failed to load static speaker embedding: {e}")

    return vocoder, image_vae_decoder, voice_smg_decoder, static_speaker_embedding


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.log_dir, exist_ok=True)

    # Metrics
    from megatransformer.utils.metrics_backend import TensorBoardBackend
    metrics.init_metrics(TensorBoardBackend(log_dir=args.log_dir))

    # Model
    print(f"Loading model from {args.checkpoint_path} (config={args.config})...")
    model = load_world_model(args, device)
    # Install the discrete-voice codebook (mirrors training). Without it voice_codebook is
    # None, so generate() falls out of the unit/EOV branch into the continuous branch, which
    # references the removed stop head -> KeyError 'voice_stop_logits'. Sizing unit_vocab_size
    # (above) is NOT enough; the codebook is what routes generation through the discrete path.
    _cbp = getattr(args, "voice_codebook_path", None)
    if _cbp:
        from megatransformer.utils.codebook import load_codebook
        model.set_voice_codebook(load_codebook(_cbp))
        print(f"Installed voice codebook ({int(load_codebook(_cbp).shape[0])} units) for discrete generation")
    model.to(device)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Dataset
    print("Loading datasets...")
    train_dataset = load_dataset(args, "train")
    try:
        eval_dataset = load_dataset(args, "val")
    except Exception:
        print("No val split found, reusing train as eval")
        eval_dataset = train_dataset
    print(f"Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

    # Collator
    voice_max_frames = int(args.voice_max_seconds * args.voice_sample_rate // args.voice_hop_length)
    # Discrete voice: append the EOV terminal unit (id == K) so teacher-forced
    # reconstructions match the training layout. None on the continuous path.
    _cbp = getattr(args, "voice_codebook_path", None)
    if _cbp:
        from megatransformer.utils.codebook import load_codebook
        voice_eov_id = int(load_codebook(_cbp).shape[0])
    else:
        voice_eov_id = None
    # Control-token base + eos come from the loaded model's config (authoritative): default
    # (Mistral 32000 / eos 2) or the pretrained LLM's native vocab/eos. Must match the data.
    collator = MultimodalDataCollator(
        max_seq_len=args.max_seq_len,
        max_waveforms=int(args.voice_max_seconds * args.voice_sample_rate),
        max_mel_spec_frames=voice_max_frames,
        max_sive_feature_frames=math.ceil(voice_max_frames / args.sive_total_stride),
        voice_eov_id=voice_eov_id,
        special_token_base=getattr(model.config, "special_token_base", constants.SPECIAL_TOKEN_BASE),
        eos_token_id=getattr(model.config, "eos_token_id", constants.EOS_TOKEN_ID),
    )

    # Decoders
    vocoder, image_vae_decoder, voice_smg_decoder, static_speaker_embedding = load_decoders(args)

    # Visualization callback. Control-token base + tokenizer must match the model, else the
    # text->voice prompts inject base-32000 BO* ids the pretrained model never sees.
    callback = WorldModelVisualizationCallback(
        tokenizer=None,
        special_token_base=getattr(model.config, "special_token_base", constants.SPECIAL_TOKEN_BASE),
        tokenizer_name=(getattr(args, "text_encoder_model", None) or "mistralai/Mistral-7B-v0.1"),
        vocoder=vocoder,
        image_vae_decoder=image_vae_decoder,
        voice_smg_decoder=voice_smg_decoder,
        static_speaker_embedding=static_speaker_embedding,
        num_eval_samples=args.num_eval_samples,
        step_offset=0,
        voice_sample_rate=args.voice_sample_rate,
        voice_n_mels=args.voice_n_mels,
        voice_n_fft=args.voice_n_fft,
        voice_hop_length=args.voice_hop_length,
        include_modes=[m.strip() for m in args.include_modes.split(",")],
        include_tasks=[t.strip() for t in args.include_tasks.split(",")] if args.include_tasks else None,
    )

    # Wire up the fake trainer
    callback.trainer = FakeTrainer(train_dataset, eval_dataset, collator)

    # Run
    state = FakeState(global_step=args.step)
    fake_args = FakeArgs(bf16=args.bf16)

    print(f"Running visualization at step={args.step}, logging to {args.log_dir}")
    callback.on_evaluate(fake_args, state, None, model=model)

    metrics.flush()
    print("Done.")


if __name__ == "__main__":
    main()
