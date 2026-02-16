from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler
from transformers.integrations.integration_utils import TensorBoardCallback
from transformers.trainer import Trainer
from model.audio.sive.ctc_vocab import CTCVocab
from model.audio.sive.sive import SpeakerInvariantVoiceEncoder
from utils import model_loading_utils
from utils.megatransformer_utils import print_debug_tensor


class GRLAlphaScheduler:
    """
    Schedule GRL alpha from 0 to max_alpha over warmup steps.

    Follows the original DANN paper recommendation:
    alpha = 2 / (1 + exp(-gamma * p)) - 1
    where p progresses from 0 to 1.

    Args:
        warmup_steps: Number of steps to ramp alpha from 0 to max_alpha
        max_alpha: Maximum alpha value (gradient reversal strength)
        gamma: Steepness of the sigmoid ramp
    """

    def __init__(
        self,
        warmup_steps: int = 5000,
        max_alpha: float = 1.0,
        gamma: float = 10.0,
    ):
        self.warmup_steps = warmup_steps
        self.max_alpha = max_alpha
        self.gamma = gamma

    def get_alpha(self, step: int) -> float:
        """Get alpha for a given step. Expects step to already include any offset."""
        if self.warmup_steps == 0:
            return self.max_alpha

        p = min(step / self.warmup_steps, 1.0)
        alpha = 2.0 / (1.0 + np.exp(-self.gamma * p)) - 1.0
        return float(alpha * self.max_alpha)


class SIVETrainer(Trainer):
    """
    Custom trainer for SIVE with CTC + GRL losses.

    Supports:
    - Separate optimizer/LR for speaker classifier (grl_lr)
    - GRL pre-training phase (grl_start_step) where classifier learns without adversarial pressure
    """

    def __init__(
        self,
        *args,
        vocab: CTCVocab,
        grl_alpha_scheduler: GRLAlphaScheduler,
        ctc_weight: float = 1.0,
        grl_weight: float = 0.1,
        grl_start_step: int = 0,  # Step at which GRL kicks in (before this, classifier trains freely)
        grl_lr: float = None,  # Separate LR for speaker classifier (None = use base LR)
        cmdline: str = "",
        git_commit_hash: str = "",
        step_offset: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.vocab = vocab
        self.grl_alpha_scheduler = grl_alpha_scheduler
        self.ctc_weight = ctc_weight
        self.grl_weight = grl_weight
        self.grl_start_step = grl_start_step
        self.grl_lr = grl_lr
        self.cmdline = cmdline
        self.git_commit_hash = git_commit_hash
        self.step_offset = step_offset if step_offset is not None else 0
        self.has_logged_cli = False

        # CTC loss
        self.ctc_criterion = nn.CTCLoss(blank=vocab.blank_idx, reduction="mean", zero_infinity=True)
        self.speaker_criterion = nn.CrossEntropyLoss()

        # Metrics tracking
        self._step_metrics = {}

        # Set up shard-aware sampler if dataset supports it
        self._shard_sampler = None
        if hasattr(self.train_dataset, 'get_sampler'):
            self._shard_sampler = self.train_dataset.get_sampler(shuffle=True, seed=42)

        # TensorBoard writer (lazily initialized)
        self.writer = None

    def create_optimizer(self):
        """
        Override to create separate parameter groups with different learning rates.
        Speaker classifier gets grl_lr (or base_lr if None).
        """
        if self.optimizer is not None:
            return self.optimizer

        model = self.model
        base_lr = self.args.learning_rate
        speaker_lr = self.grl_lr if self.grl_lr is not None else base_lr

        # Separate speaker classifier parameters
        speaker_classifier_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'speaker_classifier' in name:
                speaker_classifier_params.append(param)
            else:
                other_params.append(param)

        # Create parameter groups
        optimizer_grouped_parameters = [
            {
                "params": other_params,
                "lr": base_lr,
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": speaker_classifier_params,
                "lr": speaker_lr,
                "weight_decay": self.args.weight_decay,
            },
        ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, model)

        # Remove lr from kwargs since we set it per-group
        optimizer_kwargs.pop("lr", None)

        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def _ensure_tensorboard_writer(self):
        """Get TensorBoard writer from callback."""
        if self.writer is not None:
            return
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, TensorBoardCallback):
                self.writer = callback.tb_writer
                return

    def _get_train_sampler(self) -> Optional[Sampler]:
        """Override to use shard-aware sampler for efficient shard loading."""
        if self._shard_sampler is not None:
            epoch = int(self.state.epoch) if self.state and self.state.epoch else 0
            self._shard_sampler.set_epoch(epoch)
            return self._shard_sampler
        return super()._get_train_sampler()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        global_step = self.state.global_step + self.step_offset
        self._ensure_tensorboard_writer()

        # Log CLI and git hash on first call (logs at resumed step if resuming)
        if not self.has_logged_cli and self.writer is not None:
            self.writer.add_text("training/command_line", self.cmdline, global_step)
            self.writer.add_text("training/git_commit_hash", self.git_commit_hash, global_step)
            self.has_logged_cli = True

        mel_specs = inputs["mel_specs"]
        mel_lengths = inputs["mel_lengths"]
        ctc_tokens = inputs["ctc_tokens"]
        ctc_lengths = inputs["ctc_lengths"]
        speaker_ids = inputs["speaker_ids"]

        # GRL pre-training phase:
        # Before grl_start_step, classifier trains freely (no gradient reversal)
        # After grl_start_step, GRL kicks in with alpha ramping from that point
        in_pretraining = global_step < self.grl_start_step
        if in_pretraining:
            # Pre-training phase: classifier learns without adversarial pressure
            grl_alpha = 0.0
        else:
            # GRL active: alpha scheduler starts from grl_start_step
            effective_step = global_step - self.grl_start_step
            grl_alpha = self.grl_alpha_scheduler.get_alpha(effective_step)

        # Forward pass
        result = model(mel_specs, lengths=mel_lengths, grl_alpha=grl_alpha)

        asr_logits = result["asr_logits"]  # [B, T, vocab]
        speaker_logits = result["speaker_logits"]  # [B, num_speakers]
        # Use ctc_lengths for CTC loss (accounts for upsampling if enabled)
        output_ctc_lengths = result.get("ctc_lengths", result["feature_lengths"])  # [B]

        # CTC loss
        # CTC expects [T, B, vocab] and log probabilities
        log_probs = F.log_softmax(asr_logits, dim=-1).permute(1, 0, 2)  # [T, B, vocab]

        ctc_loss = self.ctc_criterion(log_probs, ctc_tokens, output_ctc_lengths, ctc_lengths)

        # GRL speaker classification loss
        # We want the classifier to FAIL (be at chance level)
        # But we train it normally - GRL reverses gradients to encoder
        speaker_loss = self.speaker_criterion(speaker_logits, speaker_ids)

        # Speaker accuracy and diagnostics (for logging)
        with torch.no_grad():
            speaker_preds = speaker_logits.argmax(dim=-1)
            speaker_acc = (speaker_preds == speaker_ids).float().mean().item()

            # Diagnostic: check for mode collapse
            pred_probs = F.softmax(speaker_logits, dim=-1)
            pred_entropy = -(pred_probs * torch.log(pred_probs + 1e-8)).sum(dim=-1).mean().item()
            unique_preds = speaker_preds.unique().numel()

            # Max probability (confidence) - high values with low accuracy = overconfident
            max_prob = pred_probs.max(dim=-1).values.mean().item()

        # Combined loss
        # During pre-training phase, speaker loss still contributes but doesn't affect encoder
        # (because grl_alpha=0 means no gradient reversal, but classifier still learns)
        total_loss = self.ctc_weight * ctc_loss + self.grl_weight * speaker_loss

        # Log to TensorBoard
        if self.writer is not None and global_step % self.args.logging_steps == 0:
            self.writer.add_scalar("train/ctc_loss", ctc_loss.item(), global_step)
            self.writer.add_scalar("train/speaker_loss", speaker_loss.item(), global_step)
            self.writer.add_scalar("train/speaker_accuracy", speaker_acc, global_step)
            self.writer.add_scalar("train/grl_alpha", grl_alpha, global_step)
            self.writer.add_scalar("train/total_loss", total_loss.item(), global_step)
            self.writer.add_scalar("train/grl_pretraining", float(in_pretraining), global_step)
            # Diagnostics for speaker classifier behavior
            self.writer.add_scalar("train/speaker_pred_entropy", pred_entropy, global_step)
            self.writer.add_scalar("train/speaker_unique_preds", unique_preds, global_step)
            self.writer.add_scalar("train/speaker_max_prob", max_prob, global_step)

        if return_outputs:
            return total_loss, result
        return total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to handle SIVE inputs correctly during evaluation."""
        model.eval()

        with torch.no_grad():
            mel_specs = inputs["mel_specs"]
            mel_lengths = inputs["mel_lengths"]
            ctc_tokens = inputs["ctc_tokens"]
            ctc_lengths = inputs["ctc_lengths"]
            speaker_ids = inputs["speaker_ids"]

            # Forward pass (no GRL during eval)
            result = model(mel_specs, lengths=mel_lengths, grl_alpha=0.0)

            asr_logits = result["asr_logits"]
            speaker_logits = result["speaker_logits"]
            # Use ctc_lengths for CTC loss (accounts for upsampling if enabled)
            ctc_lengths = result.get("ctc_lengths", result["feature_lengths"])

            # CTC loss
            log_probs = F.log_softmax(asr_logits, dim=-1).permute(1, 0, 2)
            ctc_loss = self.ctc_criterion(log_probs, ctc_tokens, ctc_lengths, ctc_lengths)

            # Speaker loss
            speaker_loss = self.speaker_criterion(speaker_logits, speaker_ids)

            # Combined loss
            total_loss = self.ctc_weight * ctc_loss + self.grl_weight * speaker_loss

        return (total_loss, None, None)

    def start_train_print(self, args):
        print(f"SIVE Pretraining")
        print(f"==================")
        print(f"Config: {args.config}")
        print(f"Run dir: {args.run_dir}")
        print(f"Data cache: {args.cache_dir}")
        if args.ctc_upsample_factor > 1:
            print(f"CTC upsampling: ENABLED")
            print(f"  ctc_upsample_factor: {args.ctc_upsample_factor} ({args.ctc_upsample_factor}x more CTC frames)")
        if args.conv_dropout > 0 or args.feature_dropout > 0 or args.head_dropout > 0 or args.attention_head_drop > 0:
            print(f"Dropout regularization: ENABLED")
            print(f"  conv_dropout: {args.conv_dropout} (Dropout1d in conv frontend)")
            print(f"  feature_dropout: {args.feature_dropout}")
            print(f"  head_dropout: {args.head_dropout} (prediction head)")
            print(f"  attention_head_drop: {args.attention_head_drop} (DropHead on attention)")
        if args.use_spec_augment:
            print(f"SpecAugment: ENABLED")
            print(f"  time_mask_param: {args.spec_time_mask_param}, freq_mask_param: {args.spec_freq_mask_param}")
            print(f"  num_time_masks: {args.spec_num_time_masks}, num_freq_masks: {args.spec_num_freq_masks}")
        if args.drop_path_rate > 0:
            print(f"Stochastic Depth: ENABLED (max drop_path_rate={args.drop_path_rate})")
        if args.activation != "gelu":
            print(f"Architectural options:")
            if args.activation != "gelu":
                print(f"  Activation: {args.activation}")
        if args.vocoder_checkpoint_path:
            print(f"Vocoder (for audio visualization): {args.vocoder_config}")
            print(f"  checkpoint: {args.vocoder_checkpoint_path}")
            print(f"  sample_rate: {args.audio_sample_rate}, n_fft: {args.audio_n_fft}, hop_length: {args.audio_hop_length}")
            print(f"  num_audio_samples: {args.num_audio_samples}")
        print(f"CTC decoding: beam_width={args.beam_width}")
        if args.kenlm_model_path:
            print(f"  LM: {args.kenlm_model_path}")
            print(f"  alpha={args.lm_alpha}, beta={args.lm_beta}")
        else:
            print(f"  No language model (greedy fallback or beam search without LM)")

        print(f"  Train samples: {len(self.train_dataset):,}")
        print(f"  Val samples: {len(self.eval_dataset):,}")
        print(f"  Num speakers: {args.num_speakers}")

        num_params = self.model.get_num_params()
        print(f"Model: {self.model}")
        print(f"Total Parameters: {num_params:,}")

        conv_upsample_params = sum(p.numel() for p in self.model.conv_subsample.parameters())
        encoder_blocks_params = sum(p.numel() for p in self.model.encoder_blocks.parameters())
        final_norm_params = sum(p.numel() for p in self.model.final_norm.parameters())
        head_params = sum(p.numel() for p in self.model.asr_head.parameters())
        print(f"SIVE Parameters: {conv_upsample_params + encoder_blocks_params + final_norm_params + head_params:,}")
        print(f"GRL Parameters: {sum(p.numel() for p in self.model.speaker_classifier.parameters()):,}")

        # Log configuration
        print("Training configuration:")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  CTC weight: {args.ctc_weight}")
        print(f"  GRL weight: {args.grl_weight}")
        print(f"  GRL warmup steps: {args.grl_warmup_steps}")
        print(f"  GRL max alpha: {args.grl_max_alpha}")
        print(f"  GRL start step: {args.grl_start_step}" + (" (pre-training phase)" if args.grl_start_step > 0 else ""))
        print(f"  GRL LR: {args.grl_lr if args.grl_lr is not None else 'same as base LR'}")
        print(f"  Speaker pooling: {args.speaker_pooling}")


def load_model(args):
    return model_loading_utils.load_model(SpeakerInvariantVoiceEncoder, args.config,  checkpoint_path=args.resume_from_checkpoint, overrides={
        'num_speakers': args.num_speakers,
        'audio_n_mels': args.audio_n_mels,
        # CTC upsampling (relaxes CTC length constraint)
        'ctc_upsample_factor': args.ctc_upsample_factor,
        # Dropout regularization
        'conv_dropout': args.conv_dropout,
        'feature_dropout': args.feature_dropout,
        'head_dropout': args.head_dropout,
        'attention_head_drop': args.attention_head_drop,
        # Architectural options
        'conformer_kernel_size': args.conformer_kernel_size,
        'activation': args.activation,
        # Speaker classifier pooling strategy
        'speaker_pooling': args.speaker_pooling,
        # SpecAugment
        'use_spec_augment': args.use_spec_augment,
        'spec_time_mask_param': args.spec_time_mask_param,
        'spec_freq_mask_param': args.spec_freq_mask_param,
        'spec_num_time_masks': args.spec_num_time_masks,
        'spec_num_freq_masks': args.spec_num_freq_masks,
        # Stochastic Depth
        'drop_path_rate': args.drop_path_rate,
    })


def create_trainer(
    args,
    model,
    optimizer,
    training_args,
    data_collator,
    train_dataset,
    eval_dataset,
):
    # Create GRL scheduler
    grl_scheduler = GRLAlphaScheduler(
        warmup_steps=args.grl_warmup_steps,
        max_alpha=args.grl_max_alpha,
    )

    return SIVETrainer(
        model=model,
        optimizers=(optimizer, None),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        vocab=CTCVocab(),
        grl_alpha_scheduler=grl_scheduler,
        ctc_weight=args.ctc_weight,
        grl_weight=args.grl_weight,
        grl_start_step=args.grl_start_step,
        grl_lr=args.grl_lr,
        cmdline=args.cmdline,
        git_commit_hash=args.commit_hash or "",
        step_offset=args.start_step,
    )


def add_cli_args(subparsers):
    sub_parser = subparsers.add_parser("audio-sive", help="Train a Speaker-Invariant Voice Encoder (SIVE) model with CTC + GRL losses")

    # Audio settings
    sub_parser.add_argument("--audio_max_seconds", type=float, default=10.0,
                            help="Maximum audio length in seconds")
    sub_parser.add_argument("--audio_n_mels", type=int, default=80,
                            help="Number of mel filterbanks")
    sub_parser.add_argument("--audio_sample_rate", type=int, default=16000,
                            help="Audio sample rate")
    sub_parser.add_argument("--audio_n_fft", type=int, default=16000,
                            help="FFT size for audio processing")
    sub_parser.add_argument("--audio_hop_length", type=int, default=256,
                            help="Hop length for audio processing")

    # GRL settings
    sub_parser.add_argument("--grl_warmup_steps", type=int, default=5000,
                            help="Number of steps to ramp GRL alpha from 0 to max_alpha")
    sub_parser.add_argument("--grl_max_alpha", type=float, default=1.0,
                            help="Maximum GRL alpha (gradient reversal strength)")
    sub_parser.add_argument("--grl_weight", type=float, default=0.1,
                            help="Weight for GRL speaker classification loss")
    sub_parser.add_argument("--grl_start_step", type=int, default=0,
                            help="Pre-training phase before GRL kicks in")
    sub_parser.add_argument("--grl_lr", type=float, default=None,
                            help="Separate learning rate for speaker classifier (default: use base LR)")
    
    sub_parser.add_argument("--speaker_pooling", type=str, default="attentive_statistics",
                            help="Pooling strategy for speaker classifier (e.g., 'attentive_statistics')")

    # CTC-specific settings
    sub_parser.add_argument("--ctc_weight", type=float, default=1.0,
                            help="Weight for CTC loss in total loss")

    # Dropout settings for regularization (helps prevent memorization)
    sub_parser.add_argument("--conv_dropout", type=float, default=0.05,
                            help="Dropout1d in conv frontend")
    sub_parser.add_argument("--feature_dropout", type=float, default=0.0,
                            help="Feature dropout")
    sub_parser.add_argument("--head_dropout", type=float, default=0.0,
                            help="Dropout in prediction head")
    sub_parser.add_argument("--attention_head_drop", type=float, default=0.0,
                            help="DropHead on attention")

    # Architectural options
    sub_parser.add_argument("--conformer_kernel_size", type=int, default=31,
                            help="Kernel size for conformer convolution modules")
    sub_parser.add_argument("--activation", type=str, default="gelu",
                            help="Activation function: 'gelu' or 'swiglu'")

    # CTC upsampling (relaxes CTC length constraint without increasing transformer cost)
    sub_parser.add_argument("--ctc_upsample_factor", type=int, default=1,
                            help="CTC upsampling factor (e.g., 2 = double CTC frames)")

    # SpecAugment (data augmentation)
    sub_parser.add_argument("--use_spec_augment", action="store_true",
                            help="Enable SpecAugment data augmentation")
    sub_parser.add_argument("--spec_time_mask_param", type=int, default=50,
                            help="Max time mask width for SpecAugment")
    sub_parser.add_argument("--spec_freq_mask_param", type=int, default=20,
                            help="Max frequency mask width for SpecAugment")
    sub_parser.add_argument("--spec_num_time_masks", type=int, default=2,
                            help="Number of time masks for SpecAugment")
    sub_parser.add_argument("--spec_num_freq_masks", type=int, default=2,
                            help="Number of frequency masks for SpecAugment")

    # Stochastic Depth (drop entire residual paths for regularization)
    sub_parser.add_argument("--drop_path_rate", type=float, default=0.0,
                            help="Max drop path rate for stochastic depth (linearly scaled per layer, 0=disabled)")

    # Vocoder settings (for audio generation in TensorBoard)
    sub_parser.add_argument("--vocoder_checkpoint_path", type=str, default=None,
                            help="Path to pretrained vocoder checkpoint for audio visualization")
    sub_parser.add_argument("--vocoder_config", type=str, default="tiny",
                            help="Vocoder config name (e.g., 'tiny_attention_freq_domain_vocoder')")
    sub_parser.add_argument("--num_audio_samples", type=int, default=4,
                            help="Number of audio samples to generate for visualization")

    # LM decoder settings (for CTC mode - beam search with optional language model)
    sub_parser.add_argument("--kenlm_model_path", type=str, default="./pretrained_models/KenLM-4-gram/4-gram.arpa",
                            help="Path to KenLM language model for CTC decoding (if not provided, greedy or beam search without LM is used)")
    sub_parser.add_argument("--lm_alpha", type=float, default=0.5,
                            help="Language model weight for CTC decoding")
    sub_parser.add_argument("--lm_beta", type=float, default=1.0,
                            help="Word insertion bonus for CTC decoding")
    sub_parser.add_argument("--beam_width", type=int, default=100,
                            help="Beam width for CTC beam search decoding")
    
    sub_parser.add_argument("--num_speakers", type=int, default=921,
                            help="Number of speakers for speaker embedding classifier")

    sub_parser.add_argument("--cache_dir", type=str, default="../cached_datasets/audio_sive",
                           help="Directory for cached datasets")

    return sub_parser
