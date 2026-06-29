from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler
from transformers.trainer import Trainer
from megatransformer.model.voice.sive.ctc_vocab import CTCVocab
from megatransformer.model.voice.sive.sive import SpeakerInvariantVoiceEncoder
from megatransformer.model.voice.sive.waveform_augment import WaveformAugment
from megatransformer.scripts.train.trainer import CommonTrainer
from megatransformer.utils import metrics, model_loading_utils
from megatransformer.utils.audio_utils import SharedWindowBuffer, extract_mels
from megatransformer.utils.megatransformer_utils import print_debug_tensor


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


class SIVETrainer(CommonTrainer):
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
        pad_blank_weight: float = 0.05,  # Auxiliary CE pushing pad-region asr_logits toward blank
        cmdline: str = "",
        git_commit_hash: str = "",
        step_offset: int = 0,
        waveform_augment: Optional[WaveformAugment] = None,
        shared_window_buffer: Optional[SharedWindowBuffer] = None,
        mel_sample_rate: int = 16000,
        mel_n_mels: int = 80,
        mel_n_fft: int = 1024,
        mel_hop_length: int = 256,
        max_mel_frames: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.vocab = vocab
        self.grl_alpha_scheduler = grl_alpha_scheduler
        self.ctc_weight = ctc_weight
        self.grl_weight = grl_weight
        self.grl_start_step = grl_start_step
        self.grl_lr = grl_lr
        self.pad_blank_weight = pad_blank_weight
        self.cmdline = cmdline
        self.git_commit_hash = git_commit_hash
        self.step_offset = step_offset if step_offset is not None else 0
        self.has_logged_cli = False

        # Waveform-level augmentation + on-GPU mel extraction. When a dataset
        # surfaces raw waveforms (no precomputed mels), or when waveform aug is
        # requested, mel specs are derived per step in _prepare_mel_inputs.
        self.waveform_augment = waveform_augment
        self.shared_window_buffer = shared_window_buffer or SharedWindowBuffer()
        self.mel_sample_rate = mel_sample_rate
        self.mel_n_mels = mel_n_mels
        self.mel_n_fft = mel_n_fft
        self.mel_hop_length = mel_hop_length
        self.max_mel_frames = max_mel_frames

        # CTC loss
        self.ctc_criterion = nn.CTCLoss(blank=vocab.blank_idx, reduction="mean", zero_infinity=True)
        self.speaker_criterion = nn.CrossEntropyLoss()

        # Metrics tracking
        self._step_metrics = {}

        # Set up shard-aware sampler if dataset supports it
        self._shard_sampler = None
        if hasattr(self.train_dataset, 'get_sampler'):
            self._shard_sampler = self.train_dataset.get_sampler(shuffle=True, seed=42)


    def create_optimizer(self):
        """
        Build optimizer with two-axis parameter grouping:
          axis 1: speaker_classifier (gets grl_lr) vs everything else (gets base_lr)
          axis 2: decay (linear/conv weights) vs no_decay (norms, biases)

        Yields up to 4 param groups so weight_decay isn't applied to LayerNorm
        gain/bias or other 1D parameters — those would otherwise be slowly
        pulled toward zero, hurting convergence on transformer-shaped models.

        Uses HF Trainer's `get_decay_parameter_names` to determine which params
        belong in the decay set (matches the convention HF's default optimizer
        and DeepSpeed both expect).
        """
        if self.optimizer is not None:
            return self.optimizer

        model = self.model
        base_lr = self.args.learning_rate
        speaker_lr = self.grl_lr if self.grl_lr is not None else base_lr

        # CommonTrainer.get_decay_parameter_names already excludes nn.LayerNorm
        # plus BatchNorm/InstanceNorm/GroupNorm gains. No extra filter needed.
        decay_names = set(self.get_decay_parameter_names(model))

        groups = {
            "main_decay":     {"params": [], "lr": base_lr,    "weight_decay": self.args.weight_decay},
            "main_no_decay":  {"params": [], "lr": base_lr,    "weight_decay": 0.0},
            "spk_decay":      {"params": [], "lr": speaker_lr, "weight_decay": self.args.weight_decay},
            "spk_no_decay":   {"params": [], "lr": speaker_lr, "weight_decay": 0.0},
        }

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            in_speaker = 'speaker_classifier' in name
            in_decay = name in decay_names
            key = (
                "spk_decay"     if in_speaker and in_decay else
                "spk_no_decay"  if in_speaker else
                "main_decay"    if in_decay else
                "main_no_decay"
            )
            groups[key]["params"].append(param)

        optimizer_grouped_parameters = [g for g in groups.values() if g["params"]]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, model)
        optimizer_kwargs.pop("lr", None)  # per-group lr takes precedence

        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if self.args.local_rank in (-1, 0):
            counts = {k: sum(p.numel() for p in g["params"]) for k, g in groups.items()}
            print(
                "[SIVE create_optimizer] param group counts:\n"
                f"  main_decay     ({counts['main_decay']:>10,}) wd={self.args.weight_decay} lr={base_lr}\n"
                f"  main_no_decay  ({counts['main_no_decay']:>10,}) wd=0.0 lr={base_lr}\n"
                f"  spk_decay      ({counts['spk_decay']:>10,}) wd={self.args.weight_decay} lr={speaker_lr}\n"
                f"  spk_no_decay   ({counts['spk_no_decay']:>10,}) wd=0.0 lr={speaker_lr}"
            )

        return self.optimizer

    def _get_train_sampler(self, dataset=None) -> Optional[Sampler]:
        """Override to use shard-aware sampler for efficient shard loading."""
        if self._shard_sampler is not None:
            epoch = int(self.state.epoch) if self.state and self.state.epoch else 0
            self._shard_sampler.set_epoch(epoch)
            return self._shard_sampler
        return super()._get_train_sampler(dataset)

    def _waveform_to_mel(self, waveforms: torch.Tensor, wav_lengths: torch.Tensor):
        """Batched on-GPU waveform -> log-mel, returning (mel, mel_lengths).

        Mirrors the offline preprocessing convention: STFT frame count is
        ``1 + L // hop_length``, and mel lengths are clamped to the configured
        frame budget so a slowed-down (longer) clip can't overrun the rest of
        the pipeline.
        """
        mel = extract_mels(
            self.shared_window_buffer,
            waveforms,  # [B, T] -> [B, n_mels, T']
            sr=self.mel_sample_rate,
            n_mels=self.mel_n_mels,
            n_fft=self.mel_n_fft,
            hop_length=self.mel_hop_length,
        )
        # extract_mels squeezes a leading dim of size 1, so a batch of 1 comes
        # back as [n_mels, T']; restore the batch dim.
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)

        mel_lengths = 1 + wav_lengths // self.mel_hop_length

        if self.max_mel_frames is not None and mel.size(-1) > self.max_mel_frames:
            mel = mel[..., : self.max_mel_frames]
        mel_lengths = mel_lengths.clamp(max=mel.size(-1))

        # Precomputed mels arrive already cast to the model dtype by the
        # Trainer; mirror that here (e.g. bf16 under DeepSpeed).
        mel = mel.to(dtype=next(self.model.parameters()).dtype)
        return mel, mel_lengths

    def _prepare_mel_inputs(self, inputs: dict, augment: bool):
        """Resolve (mel_specs, mel_lengths) for a batch.

        Priority:
          1. augment requested + WaveformAugment enabled + waveforms present:
             augment waveforms, extract mel on GPU. Pitch/F0 shift is a
             waveform-domain op, so this wins even if mels are also present.
          2. precomputed mel_specs present: use them (original path).
          3. waveforms present: extract mel on GPU, no augmentation.
          4. otherwise: error.
        """
        waveforms = inputs.get("waveforms", None)
        do_aug = (
            augment
            and self.waveform_augment is not None
            and self.waveform_augment.enabled
            and waveforms is not None
        )

        if do_aug:
            # External module: model.train()/eval() doesn't toggle it, so force
            # train mode here (the `augment` flag is the real on/off switch).
            self.waveform_augment.train()
            waveforms, wav_lengths = self.waveform_augment(waveforms, inputs["waveform_lengths"])
            return self._waveform_to_mel(waveforms, wav_lengths)

        if inputs.get("mel_specs", None) is not None:
            return inputs["mel_specs"], inputs["mel_lengths"]

        if waveforms is not None:
            return self._waveform_to_mel(waveforms, inputs["waveform_lengths"])

        raise KeyError("SIVE batch has neither 'mel_specs' nor 'waveforms'")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        global_step = self.state.global_step + self.step_offset

        # Log CLI and git hash on first call (logs at resumed step if resuming)
        if not self.has_logged_cli and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            metrics.log_text("training/command_line", self.cmdline, global_step)
            metrics.log_text("training/git_commit_hash", self.git_commit_hash, global_step)
            metrics.log_text("training/model_architecture", str(model), global_step)
            metrics.log_text("training/model_param_count", f"{sum(p.numel() for p in model.parameters()):,}", global_step)
            self.has_logged_cli = True

        mel_specs, mel_lengths = self._prepare_mel_inputs(inputs, augment=True)
        ctc_tokens = inputs["ctc_tokens"]
        ctc_lengths = inputs["ctc_lengths"]
        speaker_ids = inputs["speaker_ids"]

        # Guard against speaker-count drift (e.g. a forgotten --num_speakers
        # leaving the stale config default against a larger dataset). Checked
        # once: an out-of-range label would otherwise feed the speaker CE and
        # trip an opaque device-side assert, or silently weaken the adversary.
        if not getattr(self, "_speaker_id_range_checked", False):
            n_spk = self.model.config.num_speakers
            mx = int(speaker_ids.max().item())
            mn = int(speaker_ids.min().item())
            if mx >= n_spk or mn < 0:
                raise ValueError(
                    f"Speaker id out of range for the speaker classifier: saw "
                    f"[{mn}, {mx}] but num_speakers={n_spk}. The dataset has more "
                    f"speakers than the model's head — pass --num_speakers >= {mx + 1} "
                    f"(it defaults to a stale value if omitted)."
                )
            self._speaker_id_range_checked = True

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

        # CTC underflow tracking. nn.CTCLoss(zero_infinity=True) silently zeros
        # the loss for samples where input_length < target_length, so a too-
        # aggressive speed perturb (or just tight transcripts) will quietly
        # drop samples from the gradient instead of erroring. Count them so we
        # know whether --wav_speed_perturb needs to come down or
        # --ctc_upsample_factor needs to go up.
        #
        # These counters call .item(), each a GPU->CPU sync, and are only read in
        # the logging block below — so compute them only on logging steps to
        # avoid stalling the pipeline every step. should_log gates the speaker
        # diagnostics and the TensorBoard block too.
        should_log = (global_step % self.args.logging_steps == 0)
        ctc_underflow_count = ctc_underflow_frac = ctc_margin_min = None
        if should_log:
            with torch.no_grad():
                margin = output_ctc_lengths - ctc_lengths  # [B]; negative => infeasible
                ctc_underflow_count = (margin < 0).sum().item()
                ctc_underflow_frac = ctc_underflow_count / margin.numel()
                ctc_margin_min = margin.min().item()

        # CTC loss
        # CTC expects [T, B, vocab] and log probabilities
        log_probs = F.log_softmax(asr_logits, dim=-1).permute(1, 0, 2)  # [T, B, vocab]

        ctc_loss = self.ctc_criterion(log_probs, ctc_tokens, output_ctc_lengths, ctc_lengths)

        # Pad-blank pressure: CTC loss is masked beyond output_ctc_lengths, so
        # asr_logits at padded frames receive no gradient and drift to arbitrary
        # classes — leaking nonsense into greedy decode and downstream consumers
        # that read the full feature tensor. This auxiliary CE pushes pad-region
        # asr_logits toward the blank class so the encoder learns "silent past
        # audio." Set pad_blank_weight=0 to disable.
        pad_blank_loss = torch.zeros((), device=asr_logits.device)
        if self.pad_blank_weight > 0 and output_ctc_lengths is not None:
            B, T_ctc, V = asr_logits.shape
            frame_idx = torch.arange(T_ctc, device=asr_logits.device).unsqueeze(0)  # [1, T]
            pad_mask = frame_idx >= output_ctc_lengths.unsqueeze(1)  # [B, T] True where padded
            if pad_mask.any():
                pad_logits = asr_logits[pad_mask]  # [N_pad, V]
                blank_targets = torch.full(
                    (pad_logits.size(0),),
                    self.vocab.blank_idx,
                    device=asr_logits.device,
                    dtype=torch.long,
                )
                pad_blank_loss = F.cross_entropy(pad_logits, blank_targets)

        # GRL speaker classification loss
        # We want the classifier to FAIL (be at chance level)
        # But we train it normally - GRL reverses gradients to encoder
        speaker_loss = self.speaker_criterion(speaker_logits, speaker_ids)

        # Speaker accuracy and diagnostics (logging only). Every line here ends
        # in .item()/.numel() — each a GPU->CPU sync — so gate on logging steps.
        speaker_acc = speaker_acc_top5 = pred_entropy = unique_preds = max_prob = None
        if should_log:
            with torch.no_grad():
                speaker_preds = speaker_logits.argmax(dim=-1)
                speaker_acc = (speaker_preds == speaker_ids).float().mean().item()

                # Top-5 accuracy: true speaker among top-5 logits
                top5_preds = speaker_logits.topk(min(5, speaker_logits.size(-1)), dim=-1).indices
                speaker_acc_top5 = (top5_preds == speaker_ids.unsqueeze(-1)).any(dim=-1).float().mean().item()

                # Diagnostic: check for mode collapse
                pred_probs = F.softmax(speaker_logits, dim=-1)
                pred_entropy = -(pred_probs * torch.log(pred_probs + 1e-8)).sum(dim=-1).mean().item()
                unique_preds = speaker_preds.unique().numel()

                # Max probability (confidence) - high values with low accuracy = overconfident
                max_prob = pred_probs.max(dim=-1).values.mean().item()

        # Feature-space regularization losses (zero unless --use_std_hinge or
        # --use_covariance_reg are set; weights are baked into the model-side
        # computation, so no further multiplier is applied here).
        std_hinge_loss = result.get("std_hinge_loss", torch.zeros((), device=ctc_loss.device))
        cov_loss = result.get("cov_loss", torch.zeros((), device=ctc_loss.device))

        # Combined loss
        # During pre-training phase, speaker loss still contributes but doesn't affect encoder
        # (because grl_alpha=0 means no gradient reversal, but classifier still learns)
        total_loss = (
            self.ctc_weight * ctc_loss
            + self.grl_weight * speaker_loss
            + self.pad_blank_weight * pad_blank_loss
            + std_hinge_loss
            + cov_loss
        )

        # Log to TensorBoard
        if should_log:
            metrics.log_scalar("train/ctc_loss", ctc_loss, global_step)
            metrics.log_scalar("train/ctc_underflow_count", ctc_underflow_count, global_step)
            metrics.log_scalar("train/ctc_underflow_frac", ctc_underflow_frac, global_step)
            metrics.log_scalar("train/ctc_margin_min", ctc_margin_min, global_step)
            metrics.log_scalar("train/pad_blank_loss", pad_blank_loss, global_step)
            metrics.log_scalar("train/speaker_loss", speaker_loss, global_step)
            metrics.log_scalar("train/speaker_accuracy", speaker_acc, global_step)
            metrics.log_scalar("train/speaker_accuracy_top5", speaker_acc_top5, global_step)
            metrics.log_scalar("train/grl_alpha", grl_alpha, global_step)
            metrics.log_scalar("train/total_loss", total_loss, global_step)
            metrics.log_scalar("train/grl_pretraining", float(in_pretraining), global_step)
            # Diagnostics for speaker classifier behavior
            metrics.log_scalar("train/speaker_pred_entropy", pred_entropy, global_step)
            metrics.log_scalar("train/speaker_unique_preds", unique_preds, global_step)
            metrics.log_scalar("train/speaker_max_prob", max_prob, global_step)

            # Feature regularization losses (zero when respective flags are off).
            metrics.log_scalar("train/std_hinge_loss", std_hinge_loss, global_step)
            metrics.log_scalar("train/cov_loss", cov_loss, global_step)

            # Per-dim std diagnostics for spotting dead/blown-out feature dims.
            # Compares post-LN ("features") vs pre-LN ("features_unnorm"): if a
            # dim is dead in post-LN but healthy in pre-LN, the final norm's γ
            # (or LN's dim-axis mean/std) is the culprit.
            self._log_feature_dim_stats(
                result["features"],
                result["features_unnorm"],
                result["feature_lengths"],
                global_step,
            )

        if return_outputs:
            return total_loss, result
        return total_loss

    @torch.no_grad()
    def _log_feature_dim_stats(
        self,
        features: torch.Tensor,
        features_unnorm: torch.Tensor,
        feature_lengths: Optional[torch.Tensor],
        global_step: int,
        dead_std_threshold: float = 0.05,
    ) -> None:
        """
        Log per-dim std of SIVE output features (post-LN) and pre-LN features.
        Padded positions are masked out using feature_lengths so they don't
        artificially deflate the variance estimate.
        """
        B, T, D = features.shape
        if feature_lengths is not None:
            valid_mask = (
                torch.arange(T, device=features.device).unsqueeze(0)
                < feature_lengths.unsqueeze(1)
            )  # [B, T]
            flat_mask = valid_mask.reshape(-1)
            feat_flat = features.reshape(-1, D)[flat_mask].float()
            feat_un_flat = features_unnorm.reshape(-1, D)[flat_mask].float()
        else:
            feat_flat = features.reshape(-1, D).float()
            feat_un_flat = features_unnorm.reshape(-1, D).float()

        if feat_flat.size(0) < 2:
            return

        for tag, flat in (("features", feat_flat), ("features_unnorm", feat_un_flat)):
            dim_std = flat.std(dim=0)  # [D]
            dim_mean = flat.mean(dim=0)  # [D]
            dim_absmean = dim_mean.abs()
            dead_count = (dim_std < dead_std_threshold).sum().item()

            metrics.log_scalar(f"feat_dim_std/{tag}/min", dim_std.min().item(), global_step)
            metrics.log_scalar(f"feat_dim_std/{tag}/max", dim_std.max().item(), global_step)
            metrics.log_scalar(f"feat_dim_std/{tag}/mean", dim_std.mean().item(), global_step)
            metrics.log_scalar(f"feat_dim_std/{tag}/median", dim_std.median().item(), global_step)
            metrics.log_scalar(f"feat_dim_std/{tag}/dead_count", dead_count, global_step)
            # Per-dim absolute mean — a "blown out" dead dim shows large |mean|
            # alongside small std (constant high-magnitude output).
            metrics.log_scalar(f"feat_dim_absmean/{tag}/max", dim_absmean.max().item(), global_step)
            metrics.log_scalar(f"feat_dim_absmean/{tag}/mean", dim_absmean.mean().item(), global_step)
            metrics.log_histogram(f"feat_dim_std/{tag}/hist", dim_std.detach().cpu(), global_step)
            metrics.log_histogram(f"feat_dim_absmean/{tag}/hist", dim_absmean.detach().cpu(), global_step)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to handle SIVE inputs correctly during evaluation."""
        model.eval()

        with torch.no_grad():
            # No augmentation during eval; falls back to precomputed mels when
            # present, else extracts mel from waveforms.
            mel_specs, mel_lengths = self._prepare_mel_inputs(inputs, augment=False)
            ctc_tokens = inputs["ctc_tokens"]
            ctc_lengths = inputs["ctc_lengths"]
            speaker_ids = inputs["speaker_ids"]

            # Forward pass (no GRL during eval)
            result = model(mel_specs, lengths=mel_lengths, grl_alpha=0.0)

            asr_logits = result["asr_logits"]
            speaker_logits = result["speaker_logits"]
            # Input frame lengths after CTC upsampling (accounts for upsampling if enabled).
            # NOTE: do not shadow ctc_lengths — that's the target token length from inputs.
            output_ctc_lengths = result.get("ctc_lengths", result["feature_lengths"])

            # CTC loss
            log_probs = F.log_softmax(asr_logits, dim=-1).permute(1, 0, 2)
            ctc_loss = self.ctc_criterion(log_probs, ctc_tokens, output_ctc_lengths, ctc_lengths)

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
        # Read from the loaded model rather than args — CLI no longer overrides
        # this; the config value is the source of truth.
        ctc_upsample_factor = self.model.config.ctc_upsample_factor
        if ctc_upsample_factor > 1:
            print(f"CTC upsampling: ENABLED")
            print(f"  ctc_upsample_factor: {ctc_upsample_factor} ({ctc_upsample_factor}x more CTC frames)")
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
        if args.use_mel_noise:
            print(f"Mel noise: ENABLED")
            print(f"  SNR range: [{args.mel_noise_snr_min_db}, {args.mel_noise_snr_max_db}] dB, prob={args.mel_noise_prob}")
        if args.use_mel_freq_response:
            print(f"Mel freq-response modulation: ENABLED")
            print(f"  strength={args.mel_freq_response_strength}, prob={args.mel_freq_response_prob}, smoothing={args.mel_freq_response_smoothing}")
        if args.use_mel_vtlp:
            print(f"Post-hoc VTLP: ENABLED")
            print(f"  strength={args.mel_vtlp_strength} (alpha in 1 +/- {args.mel_vtlp_strength}), prob={args.mel_vtlp_prob}, boundary_frac={args.mel_vtlp_boundary_frac}")
        if getattr(args, "use_waveform_aug", False):
            print(f"Waveform augmentation: ENABLED (mel recomputed on GPU per step)")
            print(f"  pitch_shift: +/-{args.wav_pitch_shift_semitones} semitones, prob={args.wav_pitch_shift_prob}, quantize_step={args.wav_pitch_quantize_step}")
            print(f"  speed_perturb: 1 +/-{args.wav_speed_perturb}, prob={args.wav_speed_perturb_prob}, quantize_step={args.wav_speed_quantize_step}")
        if args.drop_path_rate > 0:
            print(f"Stochastic Depth: ENABLED (max drop_path_rate={args.drop_path_rate})")
        if args.activation != "gelu":
            print(f"Architectural options:")
            if args.activation != "gelu":
                print(f"  Activation: {args.activation}")
        if args.vocoder_checkpoint_path:
            print(f"Vocoder (for audio visualization): {args.vocoder_config}")
            print(f"  checkpoint: {args.vocoder_checkpoint_path}")
            print(f"  sample_rate: {args.voice_sample_rate}, n_fft: {args.voice_n_fft}, hop_length: {args.voice_hop_length}")
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
    overrides = {
        'num_speakers': args.num_speakers,
        'voice_n_mels': args.voice_n_mels,
        # ctc_upsample_factor is intentionally NOT overridden here: the CLI
        # default would silently clobber whatever the config specifies. The
        # config is the source of truth; change it there if you want a
        # different value.
        # Dropout regularization
        'dropout': args.dropout,
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
        # Mel-space noise / EQ augmentation
        'use_mel_noise': args.use_mel_noise,
        'mel_noise_snr_min_db': args.mel_noise_snr_min_db,
        'mel_noise_snr_max_db': args.mel_noise_snr_max_db,
        'mel_noise_prob': args.mel_noise_prob,
        'use_mel_freq_response': args.use_mel_freq_response,
        'mel_freq_response_strength': args.mel_freq_response_strength,
        'mel_freq_response_prob': args.mel_freq_response_prob,
        'mel_freq_response_smoothing': args.mel_freq_response_smoothing,
        # Post-hoc VTLP
        'use_mel_vtlp': args.use_mel_vtlp,
        'mel_vtlp_strength': args.mel_vtlp_strength,
        'mel_vtlp_prob': args.mel_vtlp_prob,
        'mel_vtlp_boundary_frac': args.mel_vtlp_boundary_frac,
        # Stochastic Depth
        'drop_path_rate': args.drop_path_rate,
        # Std hinge regularization (disabled unless --use_std_hinge)
        'use_std_hinge': args.use_std_hinge,
        'dim_std_min': args.dim_std_min,
        'dim_std_weight': args.dim_std_weight,
        'temporal_std_min': args.temporal_std_min,
        'temporal_std_weight': args.temporal_std_weight,
        # Covariance/decorrelation regularization (disabled unless --use_covariance_reg)
        'use_covariance_reg': args.use_covariance_reg,
        'cov_weight': args.cov_weight,
    }
    # Norm levers (frontend / block pre-norm / conformer conv / final norm).
    # Override the config ONLY when a value is explicitly passed (CLI default is
    # None), so the config stays the source of truth and a CLI default can't
    # silently clobber it (the ctc_upsample_factor footgun noted above — which
    # final_norm_type previously had).
    _norm_overrides = {
        'downsample_norm_type': args.downsample_norm_type,
        'block_norm_type': args.block_norm_type,
        'conv_norm_type': args.conv_norm_type,
        'final_norm_type': args.final_norm_type,
    }
    overrides.update({k: v for k, v in _norm_overrides.items() if v is not None})
    return model_loading_utils.load_model(
        SpeakerInvariantVoiceEncoder, args.config,
        checkpoint_path=args.resume_from_checkpoint, overrides=overrides,
    )


def create_trainer(
    args,
    model,
    optimizer,
    training_args,
    data_collator,
    train_dataset,
    eval_dataset,
    shared_window_buffer=None,
):
    # Create GRL scheduler
    grl_scheduler = GRLAlphaScheduler(
        warmup_steps=args.grl_warmup_steps,
        max_alpha=args.grl_max_alpha,
    )

    waveform_augment = None
    if getattr(args, "use_waveform_aug", False):
        waveform_augment = WaveformAugment(
            sample_rate=args.voice_sample_rate,
            pitch_shift_semitones=args.wav_pitch_shift_semitones,
            pitch_shift_prob=args.wav_pitch_shift_prob,
            speed_perturb=args.wav_speed_perturb,
            speed_perturb_prob=args.wav_speed_perturb_prob,
            pitch_quantize_step=args.wav_pitch_quantize_step,
            speed_quantize_step=args.wav_speed_quantize_step,
        )

    max_mel_frames = int(args.voice_max_seconds * args.voice_sample_rate // args.voice_hop_length)

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
        pad_blank_weight=args.pad_blank_weight,
        cmdline=args.cmdline,
        git_commit_hash=args.commit_hash or "",
        step_offset=args.start_step,
        waveform_augment=waveform_augment,
        shared_window_buffer=shared_window_buffer,
        mel_sample_rate=args.voice_sample_rate,
        mel_n_mels=args.voice_n_mels,
        mel_n_fft=args.voice_n_fft,
        mel_hop_length=args.voice_hop_length,
        max_mel_frames=max_mel_frames,
    )


def add_cli_args(subparsers):
    sub_parser = subparsers.add_parser("audio-sive", help="Train a Speaker-Invariant Voice Encoder (SIVE) model with CTC + GRL losses")

    # Voice settings
    sub_parser.add_argument("--voice_max_seconds", type=float, default=10.0,
                            help="Maximum voice clip length in seconds")
    sub_parser.add_argument("--voice_n_mels", type=int, default=80,
                            help="Number of mel filterbanks")
    sub_parser.add_argument("--voice_sample_rate", type=int, default=16000,
                            help="Voice sample rate")
    sub_parser.add_argument("--voice_n_fft", type=int, default=1024,
                            help="FFT size for voice processing")
    sub_parser.add_argument("--voice_hop_length", type=int, default=256,
                            help="Hop length for voice processing")
    sub_parser.add_argument("--sive_total_stride", type=int, default=4,
                            help="Total temporal downsampling stride of the SIVE encoder (e.g. 4 for 4x, 3 for 3x)")

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
                            help="Separate learning rate for speaker classifier (default: use base LR). "
                                 "When --use_muon is set, controls the AdamW LR for speaker_classifier "
                                 "(biases, norms, and matmul params kept on AdamW via --muon_last_layer_names).")
    sub_parser.add_argument("--grl_lr_muon", type=float, default=None,
                            help="Separate Muon LR for speaker_classifier matmul params (default: use --lr_muon). "
                                 "Only takes effect when --use_muon is set AND speaker_classifier matmul "
                                 "params are routed to Muon (i.e. 'speaker_classifier' is NOT in "
                                 "--muon_last_layer_names).")
    
    sub_parser.add_argument("--speaker_pooling", type=str, default="attentive_statistics",
                            help="Pooling strategy for speaker classifier (e.g., 'attentive_statistics')")

    # CTC-specific settings
    sub_parser.add_argument("--ctc_weight", type=float, default=1.0,
                            help="Weight for CTC loss in total loss")
    sub_parser.add_argument("--pad_blank_weight", type=float, default=0.05,
                            help="Auxiliary CE loss pushing pad-region asr_logits toward blank "
                                 "(keeps SIVE features clean past audio end). 0 disables.")

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
                            help="DEPRECATED / IGNORED: CTC upsampling factor is sourced from the "
                                 "model config. Edit src/config/voice/sive/sive.py to change it. "
                                 "Kept here only so existing scripts that pass this flag don't error.")

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

    # Mel-space Gaussian noise injection (waveform-free noise augmentation).
    sub_parser.add_argument("--use_mel_noise", action="store_true",
                            help="Add Gaussian noise to mel at a random SNR (training-only)")
    sub_parser.add_argument("--mel_noise_snr_min_db", type=float, default=5.0,
                            help="Lower bound on sampled target SNR in dB")
    sub_parser.add_argument("--mel_noise_snr_max_db", type=float, default=20.0,
                            help="Upper bound on sampled target SNR in dB")
    sub_parser.add_argument("--mel_noise_prob", type=float, default=0.5,
                            help="Per-utterance probability of applying mel noise")

    # Post-hoc VTLP (mel-bin axis warp; cheaper but approximate vs
    # filter-bank-level VTLP). Disabled by default.
    sub_parser.add_argument("--use_mel_vtlp", action="store_true",
                            help="Enable post-hoc VTLP on log-mel (piecewise-linear warp of the mel-bin axis)")
    sub_parser.add_argument("--mel_vtlp_strength", type=float, default=0.1,
                            help="Half-width of warp factor range (alpha drawn from 1 +/- this; ~0.1 is conventional)")
    sub_parser.add_argument("--mel_vtlp_prob", type=float, default=0.5,
                            help="Per-utterance probability of applying VTLP")
    sub_parser.add_argument("--mel_vtlp_boundary_frac", type=float, default=0.7,
                            help="Fraction of the mel-bin axis under the linear-with-slope-1/alpha region")

    # Mel-space frequency-response modulation (simulates mic/channel EQ).
    sub_parser.add_argument("--use_mel_freq_response", action="store_true",
                            help="Apply random smooth per-band gain to simulate mic/channel EQ")
    sub_parser.add_argument("--mel_freq_response_strength", type=float, default=0.3,
                            help="Std of pre-smoothing per-band gain noise (0.3 = ~±30% swings)")
    sub_parser.add_argument("--mel_freq_response_prob", type=float, default=0.5,
                            help="Per-utterance probability of applying EQ modulation")
    sub_parser.add_argument("--mel_freq_response_smoothing", type=int, default=7,
                            help="Smoothing kernel width across mel bands (odd int; larger = smoother EQ)")

    # Waveform-level augmentation (pitch / speed). Only usable on datasets that
    # surface raw waveforms; the perturbed waveform is converted to a mel on the
    # GPU each step, so the same example is augmented differently per epoch.
    sub_parser.add_argument("--use_waveform_aug", action="store_true",
                            help="Enable waveform-level augmentation (pitch shift + speed perturb). "
                                 "Requires a dataset with raw waveforms; mel is recomputed on GPU per step.")
    sub_parser.add_argument("--wav_pitch_shift_semitones", type=float, default=4.0,
                            help="Half-width of uniform pitch-shift range in semitones (per-sample draw in +/- this)")
    sub_parser.add_argument("--wav_pitch_shift_prob", type=float, default=0.5,
                            help="Per-sample probability of applying pitch shift")
    sub_parser.add_argument("--wav_speed_perturb", type=float, default=0.1,
                            help="Half-width of uniform speed range (per-sample factor in 1 +/- this; >1 = faster). 0 disables.")
    sub_parser.add_argument("--wav_speed_perturb_prob", type=float, default=0.5,
                            help="Per-sample probability of applying speed perturbation")
    sub_parser.add_argument("--wav_pitch_quantize_step", type=float, default=1.0,
                            help="Quantize per-sample n_steps to multiples of this (semitones). "
                                 "Samples sharing a quantized value share a single batched "
                                 "AF.pitch_shift call. 0 disables quantization (per-sample calls). "
                                 "Default 1.0 = Kaldi-style integer semitones.")
    sub_parser.add_argument("--wav_speed_quantize_step", type=float, default=0.05,
                            help="Quantize per-sample speed factor to multiples of this. "
                                 "Same grouping semantics as --wav_pitch_quantize_step. "
                                 "0 disables. Default 0.05.")

    # Stochastic Depth (drop entire residual paths for regularization)
    sub_parser.add_argument("--drop_path_rate", type=float, default=0.0,
                            help="Max drop path rate for stochastic depth (linearly scaled per layer, 0=disabled)")

    # Norm levers. All default to None = use the config's value (override only
    # when explicitly passed, so a CLI default can't silently clobber the config).
    # The four SIVE norm sites: frontend conv subsampling, transformer block
    # pre-norms, conformer depthwise-conv, and the final output norm.
    sub_parser.add_argument("--downsample_norm_type", type=str, default=None,
                            choices=["batchnorm", "instancenorm", "groupnorm", "layernorm", "rmsnorm", "none"],
                            help="Frontend conv-subsampling norm (config default: instancenorm).")
    sub_parser.add_argument("--block_norm_type", type=str, default=None,
                            choices=["layernorm", "rmsnorm", "none"],
                            help="Transformer encoder pre-norms incl. conformer input norm (config default: layernorm).")
    sub_parser.add_argument("--conv_norm_type", type=str, default=None,
                            choices=["batchnorm", "instancenorm", "groupnorm", "layernorm", "rmsnorm", "none"],
                            help="Conformer depthwise-conv norm (config default: instancenorm).")
    sub_parser.add_argument("--final_norm_type", type=str, default=None,
                            choices=["layernorm", "rmsnorm", "none"],
                            help="Final norm on encoder output features (config default: layernorm). "
                                 "'rmsnorm' avoids LN's dim-axis competition; 'none' skips entirely.")

    # Std-based hinge on per-dim feature std (disabled by default)
    sub_parser.add_argument("--use_std_hinge", action="store_true",
                            help="Enable std-hinge regularization on encoder output features. "
                                 "Penalizes per-dim std falling below --dim_std_min with constant gradient.")
    sub_parser.add_argument("--dim_std_min", type=float, default=0.5,
                            help="Target minimum per-dim std for std hinge (default: 0.5)")
    sub_parser.add_argument("--dim_std_weight", type=float, default=1.0,
                            help="Weight on dim-std hinge loss (default: 1.0)")
    sub_parser.add_argument("--temporal_std_min", type=float, default=0.1,
                            help="Target minimum frame-to-frame std (only used if --temporal_std_weight > 0)")
    sub_parser.add_argument("--temporal_std_weight", type=float, default=0.0,
                            help="Weight on temporal-std hinge loss (default: 0.0 = disabled even if --use_std_hinge)")

    # Covariance / decorrelation regularization (VICReg-style, disabled by default)
    sub_parser.add_argument("--use_covariance_reg", action="store_true",
                            help="Enable VICReg-style covariance regularization on encoder features. "
                                 "Penalizes the squared off-diagonal of the per-batch feature covariance matrix.")
    sub_parser.add_argument("--cov_weight", type=float, default=0.04,
                            help="Weight on covariance loss (default: 0.04, VICReg paper)")

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
                           help="Base dir for cached shards (code appends _train/_val)")
    sub_parser.add_argument("--train_cache_dir", type=str, default=None,
                           help="Explicit train shard dir (overrides --cache_dir)")
    sub_parser.add_argument("--val_cache_dir", type=str, default=None,
                           help="Explicit val shard dir (overrides --cache_dir)")

    return sub_parser
