import argparse
import os

import torch
import torch.nn as nn

from model.audio.vae.vae import AudioVAE
import scripts.train.audio.trainer as audio_trainer


from typing import Optional

from transformers import Trainer, TrainingArguments
from scripts.data.audio.vae.data_collator import SIVEFeatureDataCollator
from scripts.data.audio.vae.dataset import SIVEFeatureShardedDataset
from scripts.data.data_collator import DataCollator
from scripts.train.audio.visualization_callback import AudioCVAEVisualizationCallback
from scripts.train.trainer import CommonTrainer
from utils.audio_utils import SharedWindowBuffer
from utils.model_loading_utils import load_model, load_vocoder
from utils.train_utils import EarlyStoppingCallback


def create_or_load_model(args, module, overrides={}) -> nn.Module:
    if args.resume_from_checkpoint is not None:
        checkpoint_path = args.resume_from_checkpoint
    else:
        checkpoint_path = None
    return load_model(module._model_cls, args.config,  checkpoint_path=checkpoint_path, overrides=overrides)


def get_training_args(args, run_dir) -> TrainingArguments:
    return TrainingArguments(
        output_dir=run_dir,
        overwrite_output_dir=True,
        lr_scheduler_type=args.lr_scheduler_type,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs if args.num_train_epochs > 0 else 1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        weight_decay=args.weight_decay,
        report_to="tensorboard",
        logging_dir=run_dir,
        logging_steps=args.logging_steps,
        save_safetensors=False,
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
    )


def get_data_collator(command: str, args) -> Optional[DataCollator]:
    if command in ["audio-cvae", 'audio-cvae-decoder']:
        collator =  SIVEFeatureDataCollator(
            max_feature_frames=args.audio_max_frames // 4,
            speaker_embedding_dim=args.speaker_embedding_dim,
        )
    return collator


def get_dataset(command: str, args, split: str):
    if command in ["audio-cvae", 'audio-cvae-decoder']:
        dataset = SIVEFeatureShardedDataset(
            shard_dir=args.cache_dir + "_" + split,
            cache_size=32,
            max_feature_frames=args.audio_max_frames // 4,
        )
    return dataset


def get_visualization_callback(command: str, shared_window_buffer, args, vocoder=None):
    if command in ["audio-cvae", 'audio-cvae-decoder']:
        vocoder = load_vocoder(args.vocoder_checkpoint_path, args.vocoder_config, shared_window_buffer)
        callback = AudioCVAEVisualizationCallback(
            shared_window_buffer=shared_window_buffer,
            step_offset=args.start_step,
            generation_steps=args.generation_steps,
            audio_sample_rate=args.audio_sample_rate,
            audio_n_mels=args.audio_n_mels,
            audio_n_fft=args.audio_n_fft,
            audio_hop_length=args.audio_hop_length,
            audio_max_frames=args.audio_max_frames,
            vocoder_checkpoint_path=args.vocoder_checkpoint_path,
            vocoder_config=args.vocoder_config,
            vocoder=vocoder,  # Use shared vocoder instead of loading a separate one
            speaker_encoder_type=args.speaker_encoder_type,
            free_bits=args.free_bits,
        )
    return callback


def get_trainer(command: str, args, run_dir, model: nn.Module, shared_window_buffer=None) -> Trainer:
    training_args = get_training_args(args, run_dir=run_dir)
    data_collator = get_data_collator(command, args)
    train_dataset = get_dataset(command, args, split="train")
    eval_dataset = get_dataset(command, args, split="val")
    visualization_callback = get_visualization_callback(command, shared_window_buffer, args)
    
    if command in ["audio-cvae", "audio-cvae-decoder"]:
        trainer: Trainer = audio_trainer.create_trainer(
            args,
            model,
            training_args,
            data_collator,
            train_dataset,
            eval_dataset,
            shared_window_buffer,
            vocoder=visualization_callback.vocoder,
            device=model.device,
        )
    else:
        raise ValueError(f"Unknown command: {command}. Available: audio-cvae, audio-cvae-decoder")
    
    trainer.add_callback(visualization_callback)
    visualization_callback.trainer = trainer

    if args.stop_step > 0:
        early_stopping_callback = EarlyStoppingCallback(stop_step=args.stop_step)
        trainer.add_callback(early_stopping_callback)

    return trainer


def add_common_args(parser: argparse.ArgumentParser):
    argparser.add_argument('--seed', type=int, default=42, help='Random seed')
    argparser.add_argument('--logging_base_dir', type=str, default=os.path.join('runs', 'causal'), help='Base directory for logging')
    argparser.add_argument('--run_name', type=str, help='Name of the run', required=True)
    argparser.add_argument('--include_modes', type=str, default='text,audio,image', help='Comma-separated list of modes to include (e.g., text,audio,image or audio,image), order agnostic')
    argparser.add_argument('--dataset_cache_dir', type=str, default='cached_datasets', help='Path to the dataset cache directory')
    argparser.add_argument('--config', type=str, default="small", help='Model configuration.')
    argparser.add_argument('--cpu', action='store_true', help='Use CPU for training')
    argparser.add_argument('--log_level', type=str, default='warning', help='Logging level: debug, info, warning, error, critical')
    argparser.add_argument('--resume_from_checkpoint', type=str, help='Resume from checkpoint at this path')
    argparser.add_argument('--start_step', type=int, default=None, help='Start step for training')
    argparser.add_argument('--lr_scheduler_type', type=str, default='cosine', help='Learning rate scheduler type')
    argparser.add_argument('--lr_scheduler_kwargs', type=str, default=None, help='Additional kwargs for LR scheduler as a JSON stringified dict')

    # efficiency params
    argparser.add_argument('--compile_model', action='store_true', help='Whether to compile the model')
    argparser.add_argument('--cudnn_benchmark', action='store_true', help='Whether to enable cuDNN benchmark')
    argparser.add_argument('--use_gradient_checkpointing', action='store_true', help='Whether to use gradient checkpointing')

    # generic hyperparams
    argparser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    argparser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    argparser.add_argument('--adam_beta1', type=float, default=0.9, help='Adam optimizer beta1')
    argparser.add_argument('--adam_beta2', type=float, default=0.999, help='Adam optimizer beta2')
    argparser.add_argument('--num_train_epochs', type=int, default=-1, help='Number of training epochs')
    argparser.add_argument('--max_steps', type=int, default=-1, help='Max steps for training')
    argparser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    argparser.add_argument('--gradient_accumulation_steps', type=int, default=32, help='Gradient accumulation steps')
    argparser.add_argument('--warmup_ratio', type=float, default=0.0, help='Warmup ratio')
    argparser.add_argument('--warmup_steps', type=int, default=0, help='Warmup steps')
    argparser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    argparser.add_argument('--fp16', action='store_true', help='Whether to use fp16')
    argparser.add_argument('--bf16', action='store_true', help='Whether to use bf16')

    # grokfast hyperparams
    argparser.add_argument('--grokfast_ema_alpha', type=float, default=0.98, help='Alpha for GrokFast EMA trainer')
    argparser.add_argument('--grokfast_ema_lambda', type=float, default=2.0, help='Lambda for GrokFast EMA trainer')
    argparser.add_argument('--grokfast_ma_window_size', type=int, default=100, help='Window size for GrokFast MA trainer')
    argparser.add_argument('--grokfast_ma_lambda', type=float, default=5.0, help='Lambda for GrokFast MA trainer')
    argparser.add_argument('--grokfast_ma_filter_type', type=str, default='mean', help='Filter type for GrokFast MA trainer')
    argparser.add_argument('--grokfast_ma_warmup', action='store_true', help='Whether to use warmup for GrokFast MA trainer')

    # deepspeed
    argparser.add_argument('--use_deepspeed', action='store_true', help='Whether to use DeepSpeed')
    argparser.add_argument('--deepspeed_config', type=str, default=None, help='DeepSpeed configuration file')
    argparser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')

    # peft lora/int8 training
    argparser.add_argument('--use_int8_peft', action='store_true', help='Use INT8 with PEFT/LoRA')
    argparser.add_argument('--use_int8_deepspeed', action='store_true', help='Use DeepSpeed INT8 quantization')
    argparser.add_argument('--lora_rank', type=int, default=16, help='Rank for LoRA adaptation')
    argparser.add_argument('--lora_alpha', type=int, default=32, help='Alpha for LoRA adaptation')
    argparser.add_argument('--lora_dropout', type=float, default=0.05, help='Dropout for LoRA adaptation')

    # logging
    argparser.add_argument('--logging_steps', type=int, default=100, help='Logging steps')
    argparser.add_argument('--eval_strategy', type=str, default='epoch', help='Evaluation strategy: steps or epoch')
    argparser.add_argument('--eval_steps', type=int, default=0, help='Evaluation steps')
    argparser.add_argument('--save_steps', type=int, default=500, help='Save steps')
    argparser.add_argument('--generation_steps', type=int, default=1000, help='Generation steps')

    argparser.add_argument('--stop_step', type=int, default=-1, help='Step to stop training at. For preserving the LR schedule while not training further.')
    argparser.add_argument('--commit_hash', type=str, default='', help='Git commit hash for this run. Logged in tensorboard.')


custom_trainers = [
    audio_trainer
]


def add_args(parser: argparse.ArgumentParser):
    subparsers = parser.add_subparsers(dest="command")
    for custom_trainer_module in custom_trainers:
        # trainer specific args
        sub_parser = custom_trainer_module.add_cli_args(subparsers)

        # args common to all training scripts
        add_common_args(sub_parser)


command_to_module = {
    "audio-cvae": audio_trainer,
}


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train a model")

    add_args(argparser)

    args = argparser.parse_args()

    run_dir = os.path.join(args.logging_base_dir, args.run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir, exist_ok=True)

    module = command_to_module.get(args.command, None)
    if module is None:
        raise ValueError(f"Unknown command: {args.command}. Available: audio-cvae")

    if module == audio_trainer:
        shared_window_buffer = SharedWindowBuffer()
    else:
        shared_window_buffer = None

    model = create_or_load_model(args, module, overrides={})

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    model.to(device)

    trainer: CommonTrainer = get_trainer(args.command, args, run_dir, model, shared_window_buffer=shared_window_buffer)

    if args.local_rank == 0 or not args.use_deepspeed:
        trainer.start_train_print()

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
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
