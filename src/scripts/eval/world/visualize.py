"""Standalone world model visualization eval script.

Loads a checkpoint, runs the full visualization callback (same scenarios as
during training), and writes results to a TensorBoard log directory.

Usage:
    python -m src.scripts.eval.world.visualize --checkpoint_path runs/my_run/checkpoint-3000 --config small_sum_dit --log_dir runs/eval_viz --cache_dir ../cached_datasets/sive --include_modes text,voice,image [--vocoder_checkpoint_path ...] [--voice_smg_checkpoint_path ...] [--image_vae_decoder_config litevae] [--use_memorization_dataset --max_samples 50]
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch

from model.world.world_model import MegaTransformerWorldModel
from scripts.data.world.data_collator import MultimodalDataCollator
from scripts.data.world.dataset import MultimodalShardedDataset
from scripts.train.world.visualization_callback import WorldModelVisualizationCallback
from utils import metrics, model_loading_utils


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
    p.add_argument("--log_dir", type=str, required=True, help="TensorBoard log output directory")

    # Dataset
    p.add_argument("--cache_dir", type=str, default=None, help="Base cache dir (appends _train/_val)")
    p.add_argument("--text_cache_dir", type=str, default=None)
    p.add_argument("--audio_cache_dir", type=str, default=None)
    p.add_argument("--voice_cache_dir", type=str, default=None)
    p.add_argument("--image_cache_dir", type=str, default=None)
    p.add_argument("--include_modes", type=str, default="text,voice,image")
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
    p.add_argument("--sive_total_stride", type=int, default=4)

    # Optional decoders
    p.add_argument("--vocoder_checkpoint_path", type=str, default=None)
    p.add_argument("--vocoder_config", type=str, default="hifigan")
    p.add_argument("--image_vae_decoder_path", type=str, default=None)
    p.add_argument("--image_vae_decoder_config", type=str, default=None)
    p.add_argument("--voice_smg_checkpoint_path", type=str, default=None)
    p.add_argument("--voice_smg_config", type=str, default="small")
    p.add_argument("--voice_smg_latent_channels", type=int, default=None)
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
    candidate = cache_dir + "_" + split
    if os.path.isdir(candidate):
        return candidate
    print(f"Warning: {candidate} not found, skipping")
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
        from scripts.data.world.memorization_dataset import MultimodalMemorizationDataset
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
    if args.iteration_norm is not None or args.share_block_weights:
        import copy
        from config.world.world_model import WORLD_MODEL_CONFIGS
        config = copy.deepcopy(WORLD_MODEL_CONFIGS[args.config])
        if args.iteration_norm is not None:
            config.recurrent_block_config.iteration_norm = args.iteration_norm
        if args.share_block_weights:
            config.recurrent_block_config.share_block_weights = True
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
            from utils.audio_utils import SharedWindowBuffer
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
            from scripts.data.image.vae.preprocess import _load_litevae
            image_vae_decoder = _load_litevae("litevae", device="cpu")
            image_vae_decoder.eval()
            print("Loaded LiteVAE decoder")
        except Exception as e:
            print(f"Warning: Failed to load LiteVAE decoder: {e}")
    elif args.image_vae_decoder_path:
        try:
            from model.image.vae.vae import ImageVAEDecoder
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
            from model.smg.smg import SMG
            smg_overrides = {}
            if args.voice_smg_latent_channels is not None:
                smg_overrides["latent_channels"] = args.voice_smg_latent_channels
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
    from utils.metrics_backend import TensorBoardBackend
    metrics.init_metrics(TensorBoardBackend(log_dir=args.log_dir))

    # Model
    print(f"Loading model from {args.checkpoint_path} (config={args.config})...")
    model = load_world_model(args, device)
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
    collator = MultimodalDataCollator(
        max_seq_len=args.max_seq_len,
        max_waveforms=int(args.voice_max_seconds * args.voice_sample_rate),
        max_mel_spec_frames=voice_max_frames,
        max_sive_feature_frames=math.ceil(voice_max_frames / args.sive_total_stride),
    )

    # Decoders
    vocoder, image_vae_decoder, voice_smg_decoder, static_speaker_embedding = load_decoders(args)

    # Visualization callback
    callback = WorldModelVisualizationCallback(
        tokenizer=None,
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
