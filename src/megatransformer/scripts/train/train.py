import argparse
import math
import numpy as np
import os
import psutil

import matplotlib

from megatransformer.model.voice.sive.ctc_vocab import CTCVocab
from megatransformer.scripts.train.audio.sive.visualization_callback import SIVEVisualizationCallback
matplotlib.use('Agg')
import torch
import torch.nn as nn

import megatransformer.scripts.train.smg.training as smg_training
import megatransformer.scripts.train.audio.sive.training as audio_sive_training
import megatransformer.scripts.train.audio.vocoder.training as audio_vocoder_training
import megatransformer.scripts.train.image.vae.training as image_vae_training
import megatransformer.scripts.train.world.training as world_training


from typing import Dict, Optional, List

from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from megatransformer.scripts.train.optimizers import MuonAdamW, create_muon_adamw_optimizer

from megatransformer.model.ema import EMAModel
from megatransformer.scripts.data.image.vae.data_collator import ImageVAEDataCollator
from megatransformer.scripts.data.image.vae.dataset import ImageVAEShardedDataset
from megatransformer.scripts.data.voice.data_collator import VoiceDataCollator
from megatransformer.scripts.data.voice.dataset import VoiceShardedDataset
from megatransformer.scripts.data.world.data_collator import MultimodalDataCollator
from megatransformer.scripts.data.world.dataset import MultimodalShardedDataset
from megatransformer.scripts.data.world.memorization_dataset import MultimodalMemorizationDataset
from megatransformer.scripts.data.data_collator import DataCollator
from megatransformer.scripts.train.smg.visualization_callback import SMGVisualizationCallback
from megatransformer.scripts.train.audio.vocoder.visualization_callback import VocoderVisualizationCallback
from megatransformer.scripts.train.ema_callback import EMAUpdateCallback
from megatransformer.scripts.train.image.vae.visualization_callback import ImageVAEVisualizationCallback
from megatransformer.scripts.train.trainer import CommonTrainer
from megatransformer.scripts.train.visualization_callback import VisualizationCallback
from megatransformer.utils import megatransformer_utils, metrics, model_loading_utils
from megatransformer.utils.audio_utils import SharedWindowBuffer
from megatransformer.utils.train_utils import EarlyStoppingCallback


_np_multiarray = getattr(np, "_core", np.core).multiarray
torch.serialization.add_safe_globals([_np_multiarray._reconstruct, np.ndarray, np.dtype])


def create_or_load_model(args, shared_window_buffer: Optional[SharedWindowBuffer]) -> nn.Module:
    if args.command in ["smg"]:
        return smg_training.load_model(args)
    elif args.command in ["audio-sive", "sive"]:
        return audio_sive_training.load_model(args)
    elif args.command in ["vocoder"]:
        return audio_vocoder_training.load_model(args, shared_window_buffer=shared_window_buffer)
    elif args.command in ["image-vae"]:
        return image_vae_training.load_model(args)
    elif args.command in ["world"]:
        return world_training.load_model(args)
    else:
        raise ValueError(f"Unknown command: {args.command}. Available: smg, vocoder, image-vae, world")


def get_training_args(args, run_dir) -> TrainingArguments:
    os.environ["TENSORBOARD_LOGGING_DIR"] = run_dir
    backend = getattr(args, 'metrics_backend', 'tensorboard')
    report_to = backend if backend in ("tensorboard", "wandb") else "none"
    return TrainingArguments(
        output_dir=run_dir,
        lr_scheduler_type=args.lr_scheduler_type,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size if args.eval_batch_size > 0 else args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs if args.num_train_epochs > 0 else 1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        weight_decay=args.weight_decay,
        report_to=report_to,
        logging_dir=run_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        gradient_checkpointing=args.use_gradient_checkpointing,
        bf16=args.bf16,
        fp16=args.fp16,
        max_grad_norm=args.max_grad_norm,
        torch_compile=args.compile_model and not args.use_deepspeed,
        deepspeed=args.deepspeed_config if args.use_deepspeed else None,
        use_cpu=args.cpu,
        log_level=args.log_level,
        logging_first_step=True,
        local_rank=args.local_rank,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        ignore_data_skip=False,
        remove_unused_columns=False,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.dataloader_num_workers > 0,
        dataloader_prefetch_factor=6 if args.dataloader_num_workers > 0 else None,
    )


def get_data_collator(command: str, args) -> Optional[DataCollator]:
    if command in ["smg", 'vocoder', 'audio-sive', 'sive']:
        voice_max_frames = int(args.voice_max_seconds * args.voice_sample_rate // args.voice_hop_length)
        collator =  VoiceDataCollator(
            max_waveforms=int(args.voice_max_seconds * args.voice_sample_rate),
            max_mel_spec_frames=voice_max_frames,
            max_sive_feature_frames=math.ceil(voice_max_frames / args.sive_total_stride),
            speaker_embedding_dim=args.speaker_embedding_dim if hasattr(args, 'speaker_embedding_dim') else 192,
        )
    elif command in ["image-vae"]:
        collator = ImageVAEDataCollator()
    elif command in ["world"]:
        voice_max_frames = int(args.voice_max_seconds * args.voice_sample_rate // args.voice_hop_length)
        collator = MultimodalDataCollator(
            max_seq_len=args.max_seq_len,
            max_waveforms=int(args.voice_max_seconds * args.voice_sample_rate),
            max_mel_spec_frames=voice_max_frames,
            max_sive_feature_frames=math.ceil(voice_max_frames / args.sive_total_stride),
        )
    return collator


def _resolve_single_shard_dir(args, split: str) -> Optional[str]:
    """Resolve the shard directory for a non-world command. Prefers the
    explicit per-split arg (--train_cache_dir / --val_cache_dir) when
    present, otherwise appends _train/_val to --cache_dir.
    """
    explicit = getattr(args, f"{split}_cache_dir", None)
    if explicit is not None:
        return explicit
    base = getattr(args, "cache_dir", None)
    if base is None:
        return None
    return base + "_" + split


def get_dataset(command: str, args, split: str):
    shard_dir = _resolve_single_shard_dir(args, split)
    if command in ["smg"]:
        dataset = VoiceShardedDataset(
            shard_dir=shard_dir,
            cache_size=args.shard_cache_size,
            columns=[
                "features",  # includes lengths
                "mel_specs",  # includes lengths
                "speaker_embeddings",
                "speaker_ids",
                "f0",
                "vuv",
                "text",
            ]
        )
    elif command in ['vocoder']:
        dataset = VoiceShardedDataset(
            shard_dir=shard_dir,
            cache_size=3,
            columns=[
                "waveforms",  # includes lengths
                "mel_specs",  # includes lengths
                "conditions",  # includes lengths
                "text",
            ]
        )
    elif command in ['audio-sive', 'sive']:
        dataset = VoiceShardedDataset(
            shard_dir=shard_dir,
            cache_size=args.shard_cache_size,
            columns=[
                "features",  # includes lengths
                "mel_specs",  # includes lengths
                "waveforms",  # includes lengths; used for waveform-level aug / GPU mel extraction when present
                "speaker_embeddings",
                "f0",
                "vuv",
                "speaker_ids",
                "ctc_tokens",  # includes lengths
                "text",
            ]
        )
    elif command in ["image-vae"]:
        dataset = ImageVAEShardedDataset(
            shard_dir=shard_dir,
            cache_size=32,
            image_size=args.image_size,
        )
    elif command in ["world"]:

        def _resolve_shard_dir(modality, split):
            """Resolve shard directory for a requested modality. Prefers the
            explicit per-split arg (--<mod>_<split>_cache_dir) when present,
            otherwise appends _train/_val to --<mod>_cache_dir. Hard-fails if
            the resolved directory is missing — silently skipping an explicitly
            requested modality hides data corruption and lets training proceed
            without the modality it was supposed to include.
            """
            explicit = getattr(args, f"{modality}_{split}_cache_dir", None)
            if explicit is not None:
                candidate = explicit
                source = f"--{modality}_{split}_cache_dir={explicit}"
            else:
                base = getattr(args, f"{modality}_cache_dir", None)
                if base is None:
                    raise ValueError(
                        f"Modality '{modality}' is in --include_modes but no "
                        f"cache_dir was provided. Pass --{modality}_cache_dir "
                        f"or both --{modality}_train_cache_dir and "
                        f"--{modality}_val_cache_dir."
                    )
                candidate = base + "_" + split
                source = f"--{modality}_cache_dir={base}, split={split}"
            if not os.path.isdir(candidate):
                raise FileNotFoundError(
                    f"Shard directory for modality '{modality}' not found: "
                    f"{candidate} (from {source})."
                )
            return candidate

        include_modes = [m.strip() for m in args.include_modes.split(",")]
        text_dir = _resolve_shard_dir("text", split) if "text" in include_modes else None
        audio_dir = _resolve_shard_dir("audio", split) if "audio" in include_modes else None
        voice_dir = _resolve_shard_dir("voice", split) if "voice" in include_modes else None
        image_dir = _resolve_shard_dir("image", split) if "image" in include_modes else None
        max_samples = getattr(args, 'max_samples', None)
        use_memorization = getattr(args, 'use_memorization_dataset', False)

        if use_memorization and max_samples is not None:
            dataset = MultimodalMemorizationDataset(
                text_shard_dir=text_dir,
                audio_shard_dir=audio_dir,
                voice_shard_dir=voice_dir,
                image_shard_dir=image_dir,
                max_samples=max_samples,
            )
        else:
            dataset = MultimodalShardedDataset(
                text_shard_dir=text_dir,
                audio_shard_dir=audio_dir,
                voice_shard_dir=voice_dir,
                image_shard_dir=image_dir,
                cache_size=args.shard_cache_size,
                max_samples=max_samples,
            )
    return dataset


def get_visualization_callback(args, command: str, model: nn.Module, shared_window_buffer=None, legacy_vocoder: bool = False) -> VisualizationCallback:
    if command in ["smg"]:
        vocoder = model_loading_utils.load_vocoder(args.vocoder_checkpoint_path, args.vocoder_config, shared_window_buffer, is_wrapped=legacy_vocoder)
        callback = SMGVisualizationCallback(
            shared_window_buffer=shared_window_buffer,
            step_offset=args.start_step,
            voice_sample_rate=args.voice_sample_rate,
            voice_n_mels=args.voice_n_mels,
            voice_n_fft=args.voice_n_fft,
            voice_hop_length=args.voice_hop_length,
            vocoder_checkpoint_path=args.vocoder_checkpoint_path,
            vocoder=vocoder,
            vocoder_config=args.vocoder_config,
            speaker_encoder_type=args.speaker_encoder_type,
            free_bits=args.free_bits,
        )
    elif command in ['audio-sive', 'sive']:
        vocoder = model_loading_utils.load_vocoder(args.vocoder_checkpoint_path, args.vocoder_config, shared_window_buffer, is_wrapped=legacy_vocoder)
        callback = SIVEVisualizationCallback(
            vocoder=vocoder,
            vocab=CTCVocab(),
            voice_sample_rate=args.voice_sample_rate,
            voice_n_mels=args.voice_n_mels,
            voice_n_fft=args.voice_n_fft,
            voice_hop_length=args.voice_hop_length,
            num_audio_samples=args.num_audio_samples,
            # LM decoder settings
            kenlm_model_path=args.kenlm_model_path,
            lm_alpha=args.lm_alpha,
            lm_beta=args.lm_beta,
            beam_width=args.beam_width,
        )
    elif command in ["vocoder"]:
        callback = VocoderVisualizationCallback(
            shared_window_buffer=shared_window_buffer,
            voice_sample_rate=args.voice_sample_rate,
            voice_n_mels=args.voice_n_mels,
            voice_n_fft=args.voice_n_fft,
            voice_hop_length=args.voice_hop_length,
        )
    elif command in ["image-vae"]:
        callback = ImageVAEVisualizationCallback(
            image_size=args.image_size,
            step_offset=args.start_step,
            num_eval_samples=8,
        )
    elif command in ["world"]:
        from megatransformer.scripts.train.world.visualization_callback import WorldModelVisualizationCallback

        vocoder = None
        if getattr(args, 'vocoder_checkpoint_path', None) or getattr(args, 'vocoder_config', None):
            try:
                vocoder = model_loading_utils.load_vocoder(
                    getattr(args, 'vocoder_checkpoint_path', None),
                    getattr(args, 'vocoder_config', 'hifigan'),
                    shared_window_buffer,
                    is_wrapped=legacy_vocoder,
                )
            except Exception as e:
                print(f"Warning: Failed to load vocoder for world model visualization: {e}")

        image_vae_decoder = None
        image_vae_decoder_config = getattr(args, 'image_vae_decoder_config', 'small')
        if image_vae_decoder_config == 'litevae':
            try:
                from megatransformer.scripts.data.image.vae.preprocess import _load_litevae
                image_vae_decoder = _load_litevae("litevae", device="cpu")
                image_vae_decoder.eval()
                print(f"Loaded LiteVAE decoder for visualization")
            except Exception as e:
                print(f"Warning: Failed to load LiteVAE decoder: {e}")
        elif getattr(args, 'image_vae_decoder_path', None):
            try:
                from megatransformer.model.image.vae.vae import ImageVAEDecoder
                image_vae_decoder = model_loading_utils.load_model(
                    ImageVAEDecoder,
                    getattr(args, 'image_vae_decoder_config', 'small'),
                    checkpoint_path=args.image_vae_decoder_path,
                    strict=False,
                )
                image_vae_decoder.eval()
            except Exception as e:
                print(f"Warning: Failed to load image VAE decoder for world model visualization: {e}")

        voice_smg_decoder = None
        if getattr(args, 'voice_smg_checkpoint_path', None):
            try:
                from megatransformer.model.smg.smg import SMG
                smg_overrides = {}
                if getattr(args, 'voice_smg_latent_channels', None) is not None:
                    smg_overrides["latent_channels"] = args.voice_smg_latent_channels
                voice_smg_decoder = model_loading_utils.load_model(
                    SMG,
                    getattr(args, 'voice_smg_config', 'small'),
                    checkpoint_path=args.voice_smg_checkpoint_path,
                    strict=False,
                    overrides=smg_overrides,
                )
                voice_smg_decoder.eval()
            except Exception as e:
                print(f"Warning: Failed to load voice SMG decoder for world model visualization: {e}")

        static_speaker_embedding = None
        if getattr(args, 'static_speaker_embedding_path', None):
            try:
                static_speaker_embedding = torch.load(
                    args.static_speaker_embedding_path, map_location="cpu", weights_only=True,
                )
                print(f"Loaded static speaker embedding: shape={static_speaker_embedding.shape}")
            except Exception as e:
                print(f"Warning: Failed to load static speaker embedding: {e}")

        callback = WorldModelVisualizationCallback(
            tokenizer=None,  # Will be set up in callback if needed
            vocoder=vocoder,
            image_vae_decoder=image_vae_decoder,
            voice_smg_decoder=voice_smg_decoder,
            static_speaker_embedding=static_speaker_embedding,
            num_eval_samples=getattr(args, 'num_eval_samples', 4),
            step_offset=args.start_step,
            voice_sample_rate=getattr(args, 'voice_sample_rate', 16000),
            voice_n_mels=getattr(args, 'voice_n_mels', 80),
            voice_n_fft=getattr(args, 'voice_n_fft', 1024),
            voice_hop_length=getattr(args, 'voice_hop_length', 256),
        )
    return callback


def get_ema_callback(args) -> Optional[EMAUpdateCallback]:
    if args.use_ema:
        ema = EMAModel(
            model,
            decay=args.ema_decay,
            update_after_step=args.ema_update_after_step,
        )
        ema_callback = EMAUpdateCallback(ema=ema)

        # Load EMA state if resuming from checkpoint
        if args.resume_from_checkpoint is not None:
            model_loading_utils.load_ema_state(ema, args.resume_from_checkpoint)

        if args.local_rank == 0 or not args.use_deepspeed:
            print(f"EMA enabled: decay={args.ema_decay}, update_after_step={args.ema_update_after_step}")
            
        return ema_callback
    return None


def get_trainer(command: str, args, run_dir, model: nn.Module, optimizer: Optional[MuonAdamW], device, shared_window_buffer=None) -> Trainer:
    training_args = get_training_args(args, run_dir=run_dir)
    data_collator = get_data_collator(command, args)
    train_dataset = get_dataset(command, args, split="train")
    eval_dataset = get_dataset(command, args, split="val")
    visualization_callback = get_visualization_callback(args, command, model, shared_window_buffer=shared_window_buffer, legacy_vocoder=args.legacy_vocoder)
    ema = get_ema_callback(args)
    
    if command in ["smg"]:
        trainer: Trainer = smg_training.create_trainer(
            args,
            model,
            optimizer,
            training_args,
            data_collator,
            train_dataset,
            eval_dataset,
            shared_window_buffer,
            vocoder=visualization_callback.vocoder,  # obtain vocoder loaded in visualization callback
            device=device,
        )
    elif command in ['audio-sive', 'sive']:
        trainer: Trainer = audio_sive_training.create_trainer(
            args,
            model,
            optimizer,
            training_args,
            data_collator,
            train_dataset,
            eval_dataset,
            shared_window_buffer=shared_window_buffer,
        )
    elif command in ["vocoder"]:
        trainer: Trainer = audio_vocoder_training.create_trainer(
            args,
            model,
            optimizer,
            training_args,
            data_collator,
            train_dataset,
            eval_dataset,
            shared_window_buffer,
        )
    elif command in ["image-vae"]:
        trainer: Trainer = image_vae_training.create_trainer(
            args,
            model,
            optimizer,
            training_args,
            data_collator,
            train_dataset,
            eval_dataset,
            device=device,
        )
    elif command in ["world"]:
        trainer: Trainer = world_training.create_trainer(
            args,
            model,
            optimizer,
            training_args,
            data_collator,
            train_dataset,
            eval_dataset,
        )
    else:
        raise ValueError(f"Unknown command: {command}. Available: smg, vocoder, image-vae, world")

    if visualization_callback is not None:
        trainer.add_callback(visualization_callback)
        visualization_callback.trainer = trainer

    if ema is not None:
        trainer.add_callback(ema)

    if args.stop_step > 0:
        early_stopping_callback = EarlyStoppingCallback(stop_step=args.stop_step)
        trainer.add_callback(early_stopping_callback)

    # Under DDP + reentrant gradient checkpointing (use_reentrant=True), each
    # checkpointed region's backward re-enters the autograd engine, which
    # causes DDP's per-parameter gradient hooks to fire more than once per
    # iteration. DDP rejects this by default ("marked ready twice"). Static
    # graph mode tells DDP the graph is stable across iterations so repeated
    # hook firing is allowed. Must be set before the first forward pass.
    if not args.use_deepspeed and args.use_gradient_checkpointing:
        from transformers.trainer_callback import TrainerCallback

        class DDPStaticGraphCallback(TrainerCallback):
            def __init__(self, trainer_ref):
                self.trainer_ref = trainer_ref

            def on_train_begin(self, cb_args, state, control, **kwargs):
                wrapped = self.trainer_ref.model_wrapped
                if hasattr(wrapped, '_set_static_graph'):
                    wrapped._set_static_graph()

        trainer.add_callback(DDPStaticGraphCallback(trainer))

    return trainer


def get_optimizer(args, model: nn.Module) -> Optional[MuonAdamW]:
    """
    Create optimizer based on args.

    If --use_muon is set, creates a MuonAdamW optimizer that routes:
    - 2D+ params (Linear weights, Conv filters) -> Muon
    - 1D params (biases, norms, embeddings) -> AdamW
    - First/last layer params (specified via args) -> AdamW

    Returns:
        MuonAdamW optimizer if --use_muon is set, None otherwise
    """
    if not getattr(args, 'use_muon', False):
        return None

    # Parse comma-separated layer name patterns
    first_layer_names = [s.strip() for s in args.muon_first_layer_names.split(',') if s.strip()]
    last_layer_names = [s.strip() for s in args.muon_last_layer_names.split(',') if s.strip()]

    # Trainer-specific param-group overrides. Currently used by SIVE so that
    # --grl_lr / --grl_lr_muon route the speaker_classifier (the GRL branch)
    # to its own LRs distinct from the encoder's lr_muon / lr_adamw.
    param_groupers = None
    if args.command in ('audio-sive', 'sive'):
        spk_overrides: Dict[str, float] = {}
        grl_lr = getattr(args, 'grl_lr', None)
        grl_lr_muon = getattr(args, 'grl_lr_muon', None)
        if grl_lr is not None:
            spk_overrides['lr_adamw'] = grl_lr
        if grl_lr_muon is not None:
            spk_overrides['lr_muon'] = grl_lr_muon
        if spk_overrides:
            param_groupers = [('spk', 'speaker_classifier', spk_overrides)]

    optimizer = create_muon_adamw_optimizer(
        model=model,
        lr_muon=args.lr_muon,
        lr_adamw=args.lr_adamw,
        weight_decay_muon=args.weight_decay_muon,
        weight_decay_adamw=args.weight_decay,  # Use existing weight_decay for AdamW part
        momentum_muon=args.momentum_muon,
        ns_steps=args.ns_steps,
        first_layer_names=first_layer_names if first_layer_names else None,
        last_layer_names=last_layer_names if last_layer_names else None,
        param_groupers=param_groupers,
        verbose=args.muon_verbose,
    )

    if args.local_rank == 0 or not args.use_deepspeed:
        print(f"Created MuonAdamW optimizer:")
        print(f"  Muon params: {len(optimizer.muon_params)} tensors, lr={args.lr_muon}")
        print(f"  AdamW params: {len(optimizer.adamw_params)} tensors, lr={args.lr_adamw}")
        if first_layer_names:
            print(f"  First layer patterns (AdamW): {first_layer_names}")
        if last_layer_names:
            print(f"  Last layer patterns (AdamW): {last_layer_names}")
        if param_groupers:
            for group_name, pattern, overrides in param_groupers:
                print(f"  Param group override '{group_name}' (pattern '{pattern}'): {overrides}")

    return optimizer


training_modules: list = [
    smg_training,
    audio_sive_training,
    audio_vocoder_training,
    image_vae_training,
    world_training,
]


def add_args(parser: argparse.ArgumentParser):
    subparsers = parser.add_subparsers(dest="command")
    for training_module in training_modules:
        # trainer specific args
        sub_parser = training_module.add_cli_args(subparsers)

        # args common to all training scripts
        sub_parser.add_argument('--seed', type=int, default=42, help='Random seed')
        sub_parser.add_argument('--logging_base_dir', type=str, default=os.path.join('runs', 'causal'), help='Base directory for logging')
        sub_parser.add_argument('--run_name', type=str, help='Name of the run', required=True)
        sub_parser.add_argument('--include_modes', type=str, default='text,audio,image', help='Comma-separated list of modes to include (e.g., text,audio,image or audio,image), order agnostic')
        sub_parser.add_argument('--dataset_cache_dir', type=str, default='cached_datasets', help='Path to the dataset cache directory')
        sub_parser.add_argument('--config', type=str, default="small", help='Model configuration.')
        sub_parser.add_argument('--cpu', action='store_true', help='Use CPU for training')
        sub_parser.add_argument('--log_level', type=str, default='warning', help='Logging level: debug, info, warning, error, critical')
        sub_parser.add_argument('--resume_from_checkpoint', type=str, help='Resume from checkpoint at this path')
        sub_parser.add_argument('--fresh_schedule', action='store_true',
            help='Warm restart: load model weights from --resume_from_checkpoint but build a fresh '
                 'optimizer + scheduler + step counter. Use with a new --learning_rate, --warmup_steps, '
                 'and --max_steps. Pair with --start_step <orig_total> for log continuity.')
        sub_parser.add_argument('--start_step', type=int, default=None, help='Start step for training')
        sub_parser.add_argument('--lr_scheduler_type', type=str, default='cosine', help='Learning rate scheduler type')
        sub_parser.add_argument('--lr_scheduler_kwargs', type=str, default=None, help='Additional kwargs for LR scheduler as a JSON stringified dict')

        # efficiency params
        sub_parser.add_argument('--compile_model', action='store_true', help='Whether to compile the model')
        sub_parser.add_argument('--cudnn_benchmark', action='store_true', help='Whether to enable cuDNN benchmark')
        sub_parser.add_argument('--use_gradient_checkpointing', action='store_true', help='Whether to use gradient checkpointing')

        # generic hyperparams
        sub_parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
        sub_parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
        sub_parser.add_argument('--adam_beta1', type=float, default=0.9, help='Adam optimizer beta1')
        sub_parser.add_argument('--adam_beta2', type=float, default=0.999, help='Adam optimizer beta2')
        sub_parser.add_argument('--num_train_epochs', type=int, default=-1, help='Number of training epochs')
        sub_parser.add_argument('--max_steps', type=int, default=-1, help='Max steps for training')
        sub_parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
        sub_parser.add_argument('--eval_batch_size', type=int, default=0, help='Eval batch size (0 = match --batch_size)')
        sub_parser.add_argument('--gradient_accumulation_steps', type=int, default=32, help='Gradient accumulation steps')
        sub_parser.add_argument('--dataloader_num_workers', type=int, default=4, help='Number of dataloader worker processes')
        sub_parser.add_argument('--shard_cache_size', type=int, default=8, help='Per-modality shard cache size (multiplied across workers/GPUs)')
        sub_parser.add_argument('--warmup_steps', type=int, default=0, help='Warmup steps')
        sub_parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
        sub_parser.add_argument('--fp16', action='store_true', help='Whether to use fp16')
        sub_parser.add_argument('--bf16', action='store_true', help='Whether to use bf16')
        sub_parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

        # muon optimizer params
        sub_parser.add_argument('--use_muon', action='store_true', help='Use Muon+AdamW optimizer instead of AdamW')
        sub_parser.add_argument('--lr_muon', type=float, default=0.02, help='Learning rate for Muon (2D+ params)')
        sub_parser.add_argument('--lr_adamw', type=float, default=1e-4, help='Learning rate for AdamW (1D params) when using Muon')
        sub_parser.add_argument('--momentum_muon', type=float, default=0.95, help='Momentum for Muon optimizer')
        sub_parser.add_argument('--weight_decay_muon', type=float, default=0.0, help='Weight decay for Muon params')
        sub_parser.add_argument('--ns_steps', type=int, default=5, help='Newton-Schulz iterations for Muon')
        sub_parser.add_argument('--muon_first_layer_names', type=str, default='', help='Comma-separated substrings to match first layer params (kept on AdamW)')
        sub_parser.add_argument('--muon_last_layer_names', type=str, default='', help='Comma-separated substrings to match last layer params (kept on AdamW)')
        sub_parser.add_argument('--muon_verbose', action='store_true', help='Print parameter routing info for Muon optimizer')

        # ema params
        sub_parser.add_argument('--use_ema', action='store_true', help='Whether to use EMA for model weights')
        sub_parser.add_argument('--ema_decay', type=float, default=0.9999, help='EMA decay rate')
        sub_parser.add_argument('--ema_update_after_step', type=int, default=2000, help='Number of steps to wait before starting EMA updates')

        # grokfast hyperparams
        sub_parser.add_argument('--grokfast_ema_alpha', type=float, default=0.98, help='Alpha for GrokFast EMA trainer')
        sub_parser.add_argument('--grokfast_ema_lambda', type=float, default=2.0, help='Lambda for GrokFast EMA trainer')
        sub_parser.add_argument('--grokfast_ma_window_size', type=int, default=100, help='Window size for GrokFast MA trainer')
        sub_parser.add_argument('--grokfast_ma_lambda', type=float, default=5.0, help='Lambda for GrokFast MA trainer')
        sub_parser.add_argument('--grokfast_ma_filter_type', type=str, default='mean', help='Filter type for GrokFast MA trainer')
        sub_parser.add_argument('--grokfast_ma_warmup', action='store_true', help='Whether to use warmup for GrokFast MA trainer')

        # deepspeed
        sub_parser.add_argument('--use_deepspeed', action='store_true', help='Whether to use DeepSpeed')
        sub_parser.add_argument('--deepspeed_config', type=str, default=None, help='DeepSpeed configuration file')
        sub_parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')

        # peft lora/int8 training
        sub_parser.add_argument('--use_int8_peft', action='store_true', help='Use INT8 with PEFT/LoRA')
        sub_parser.add_argument('--use_int8_deepspeed', action='store_true', help='Use DeepSpeed INT8 quantization')
        sub_parser.add_argument('--lora_rank', type=int, default=16, help='Rank for LoRA adaptation')
        sub_parser.add_argument('--lora_alpha', type=int, default=32, help='Alpha for LoRA adaptation')
        sub_parser.add_argument('--lora_dropout', type=float, default=0.05, help='Dropout for LoRA adaptation')

        # logging
        sub_parser.add_argument('--logging_steps', type=int, default=100, help='Logging steps')
        sub_parser.add_argument('--eval_strategy', type=str, default='epoch', help='Evaluation strategy: steps or epoch')
        sub_parser.add_argument('--eval_steps', type=int, default=0, help='Evaluation steps')
        sub_parser.add_argument('--save_steps', type=int, default=500, help='Save steps')
        sub_parser.add_argument('--metrics_backend', type=str, default='tensorboard', choices=['tensorboard', 'wandb'], help='Metrics logging backend')

        sub_parser.add_argument('--stop_step', type=int, default=-1, help='Step to stop training at. For preserving the LR schedule while not training further.')
        sub_parser.add_argument('--commit_hash', type=str, default='', help='Git commit hash for this run. Logged in tensorboard.')
        sub_parser.add_argument('--legacy_vocoder', action='store_true', help='Use legacy vocoder wrapper')


def get_process_cmdline(pid):
    """
    Retrieves the command line arguments for a process given its PID.
    Returns a list of strings representing the command line, or None if the process is not found.
    """
    try:
        process = psutil.Process(pid)
        return process.cmdline()
    except psutil.NoSuchProcess:
        return None


command_to_module = {
    "smg": smg_training,
    "audio-sive": audio_sive_training,
    "vocoder": audio_vocoder_training,
    "image-vae": image_vae_training,
    "world": world_training,
}


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train a model")

    add_args(argparser)

    args, unk = argparser.parse_known_args()

    megatransformer_utils.set_seed_everywhere(args.seed)

    print(unk)

    # Parse extra arguments
    unk_dict = {}
    for i in range(0, len(unk), 2):
        if '=' in unk[i]:
            key, value = unk[i].split('=', 1)
            unk_dict[key.lstrip('-')] = value
            i -= 1
        else:
            unk_dict[unk[i].lstrip('-')] = unk[i+1]

    print(f"Unknown args: {unk_dict}")

    current_process_pid = psutil.Process().pid
    setattr(args, 'cmdline', " ".join(get_process_cmdline(current_process_pid)))

    run_dir = os.path.join(args.logging_base_dir, args.run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir, exist_ok=True)
    setattr(args, 'run_dir', run_dir)

    module = command_to_module.get(args.command, None)
    if module is None:
        raise ValueError(f"Unknown command: {args.command}. Available: smg, image-vae")

    if module in [smg_training, audio_vocoder_training, world_training, audio_sive_training]:
        shared_window_buffer = SharedWindowBuffer()
    else:
        shared_window_buffer = None

    model = create_or_load_model(args, shared_window_buffer=shared_window_buffer)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    model.to(device)

    # Create optimizer if using Muon (not passed to trainer yet)
    optimizer = get_optimizer(args, model)

    trainer: CommonTrainer = get_trainer(args.command, args, run_dir, model, optimizer, device, shared_window_buffer=shared_window_buffer)

    # Initialize centralized metrics logger
    is_main_process = args.local_rank <= 0 or not args.use_deepspeed
    if is_main_process:
        backend_type = getattr(args, 'metrics_backend', 'tensorboard')
        if backend_type == 'wandb':
            from megatransformer.utils.wandb_backend import WandBBackend
            metrics.init_metrics(WandBBackend(project="megatransformer", run_name=args.run_name, log_dir=run_dir))
        else:
            from megatransformer.utils.metrics_backend import TensorBoardBackend
            metrics.init_metrics(TensorBoardBackend(log_dir=run_dir))
    else:
        from megatransformer.utils.metrics_backend import NoOpBackend
        metrics.init_metrics(NoOpBackend())

    if args.local_rank == 0 or not args.use_deepspeed:
        trainer.start_train_print(args)

    # Log scheduler info
    if hasattr(trainer, 'deepspeed') and trainer.deepspeed is not None:
        scheduler = trainer.deepspeed.lr_scheduler
        if scheduler is not None:
            print(f"DeepSpeed scheduler step: {scheduler.last_epoch}")
            print(f"Current LR: {scheduler.get_last_lr()}")
        else:
            print("No DeepSpeed LR scheduler found.")
    elif trainer.lr_scheduler is not None:
        print(f"Scheduler last_epoch: {trainer.lr_scheduler.last_epoch}")
        print(f"Current LR: {trainer.lr_scheduler.get_last_lr()}")
    else:
        print("No LR scheduler found in trainer.")

    checkpoint_path = args.resume_from_checkpoint
    if checkpoint_path is not None:
        print(f"Rank {trainer.args.local_rank} Checkpoint exists: {os.path.exists(checkpoint_path)}")
        print(f"Rank {trainer.args.local_rank} Checkpoint contents: {os.listdir(checkpoint_path) if os.path.exists(checkpoint_path) else 'N/A'}")

    print(f"Starting training with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    # --fresh_schedule: model weights still came from the checkpoint via load_model
    # earlier; passing None here makes HF Trainer build a fresh optimizer + scheduler
    # against the new --learning_rate / --warmup_steps / --max_steps instead of
    # restoring the saved (near-zero, end-of-cosine) LR. Adam moments are reset too.
    resume_path = None if args.fresh_schedule else args.resume_from_checkpoint
    if args.fresh_schedule:
        print("--fresh_schedule: HF Trainer optimizer/scheduler/global_step will start fresh; "
              "model weights loaded from --resume_from_checkpoint via load_model.")
    trainer.train(resume_from_checkpoint=resume_path)
