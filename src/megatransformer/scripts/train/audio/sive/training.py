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
        consistency_weight: float = 0.0,  # ContentVec-style perturbation-consistency loss (0=off); non-adversarial timbre strip
        consistency_rampup_steps: int = 5000,  # ramp the consistency weight 0->max
        grl_start_step: int = 0,  # Step at which GRL kicks in (before this, classifier trains freely)
        grl_lr: float = None,  # Separate LR for speaker classifier (None = use base LR)
        pad_blank_weight: float = 0.05,  # Auxiliary CE pushing pad-region asr_logits toward blank
        recon_weight: float = 0.0,  # Speaker-conditioned mel-recon aux (0=off); forces per-frame content, kills VQ tail-collapse
        recon_ramp_steps: int = 5000,  # ramp recon weight 0->max over THIS run's first N steps (protects encoder while head warms up)
        recon_freeze_encoder_steps: int = 0,  # opt-in: for the first N steps only the recon head trains (no DeepSpeed)
        recon_dom_weight: float = 0.0,  # weight on the anti-domination penalty (needs config.recon_dom_cap>0)
        recon_decay_start: int = 0,  # 0=off. After this step, anneal recon_alpha 1.0->recon_floor (fix tail high, then release content)
        recon_decay_steps: int = 10000,  # anneal duration
        recon_floor: float = 0.25,  # recon_alpha floor after decay (set == 1.0 => a lower static peak with no decay)
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
        speaker_adversary_target: str = "speaker_id",
        gender_grl_weight: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.vocab = vocab
        self.grl_alpha_scheduler = grl_alpha_scheduler
        self.ctc_weight = ctc_weight
        self.grl_weight = grl_weight
        self.consistency_weight = consistency_weight
        self.consistency_rampup_steps = consistency_rampup_steps
        self.grl_start_step = grl_start_step
        self.grl_lr = grl_lr
        self.pad_blank_weight = pad_blank_weight
        self.recon_weight = recon_weight
        self.recon_ramp_steps = recon_ramp_steps
        self.recon_freeze_encoder_steps = recon_freeze_encoder_steps
        self.recon_dom_weight = recon_dom_weight
        self.recon_decay_start = recon_decay_start
        self.recon_decay_steps = recon_decay_steps
        self.recon_floor = recon_floor
        self.recon_head_present = getattr(getattr(self.model, "config", None), "use_recon_aux", False)
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
        self.speaker_adversary_target = speaker_adversary_target
        self.gender_grl_weight = gender_grl_weight

        # CTC loss
        self.ctc_criterion = nn.CTCLoss(blank=vocab.blank_idx, reduction="mean", zero_infinity=True)
        self.speaker_criterion = nn.CrossEntropyLoss()
        # Gender adversary CE — ignore_index=-1 skips unknown-gender utterances.
        self.gender_criterion = nn.CrossEntropyLoss(ignore_index=-1)

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
            # Both adversary heads (speaker + gender) get grl_lr — same adversarial
            # role, same LR treatment. (Under --use_muon, add 'gender_classifier' to
            # --muon_last_layer_names too so its matmuls route like speaker_classifier.)
            in_adversary = ('speaker_classifier' in name) or ('gender_classifier' in name)
            in_decay = name in decay_names
            key = (
                "spk_decay"     if in_adversary and in_decay else
                "spk_no_decay"  if in_adversary else
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

    def _speaker_adversary_loss(self, speaker_out, inputs, speaker_ids):
        """Adversary loss to be gradient-reversed into the encoder.

        speaker_id mode: cross-entropy on speaker logits.
        ecapa_embedding mode: cosine loss regressing the stored ECAPA embedding
        (1 - cos, so minimizing it => predicting speaker well => reversal pushes
        the encoder to make features un-predictive of the embedding).
        """
        if self.speaker_adversary_target == "ecapa_embedding":
            target_emb = inputs["speaker_embeddings"].to(speaker_out.dtype)
            return (1.0 - F.cosine_similarity(speaker_out, target_emb, dim=-1)).mean()
        return self.speaker_criterion(speaker_out, speaker_ids)

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

        # Recon-aux freeze warmup (optional): for the first N steps only the recon
        # head trains, on the (good, from-stdhinge11) features, so its random init
        # can't disrupt the encoder before it's competent. Ramp alone usually
        # suffices; this is opt-in (recon_freeze_encoder_steps>0). Toggling
        # requires_grad is incompatible with DeepSpeed ZeRO param partitioning —
        # only use with plain DDP/single-GPU.
        if self.recon_freeze_encoder_steps > 0 and self.recon_head_present:
            freeze = global_step < self.recon_freeze_encoder_steps
            for n, p in model.named_parameters():
                want = ("recon_head" in n) if freeze else True
                if p.requires_grad != want:
                    p.requires_grad_(want)

        # Forward pass. speaker_embeddings (ECAPA) drives the recon-aux head's FiLM;
        # it's a no-op when use_recon_aux is off (head is None).
        result = model(
            mel_specs, lengths=mel_lengths, grl_alpha=grl_alpha,
            speaker_embeddings=inputs.get("speaker_embeddings"),
        )

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

        # GRL speaker-adversary loss. We train the adversary to succeed; the GRL
        # reverses gradients so the encoder is pushed to make it FAIL. Target is
        # the speaker id (CE) or the ECAPA embedding (cosine regression).
        speaker_loss = self._speaker_adversary_loss(speaker_logits, inputs, speaker_ids)

        # Gender adversary (parallel GRL head): trains to predict gender; the shared
        # GRL reversal pushes the encoder to make features gender-un-predictive.
        # -1 (unknown) is skipped by the CE; an all-unknown batch would give NaN, so
        # guard on there being at least one labelled sample.
        gender_logits = result.get("gender_logits")
        gender_ids = inputs.get("gender_ids")
        gender_loss = torch.zeros((), device=ctc_loss.device)
        if gender_logits is not None:
            if gender_ids is None:
                raise KeyError(
                    "use_gender_grl is set but the batch has no 'gender_ids'. The SIVE "
                    "shards were preprocessed without --gender_column (or the dataset "
                    "column list omits 'gender_ids'). Re-preprocess with gender labels "
                    "or drop --use_gender_grl."
                )
            if (gender_ids != -1).any():
                gender_loss = self.gender_criterion(gender_logits, gender_ids.long())

        # Speaker accuracy and diagnostics (logging only). Every line here ends
        # in .item()/.numel() — each a GPU->CPU sync — so gate on logging steps.
        speaker_acc = speaker_acc_top5 = pred_entropy = unique_preds = max_prob = None
        speaker_cos = None
        if should_log and self.speaker_adversary_target == "speaker_id":
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
        elif should_log:
            # Embedding-regression adversary: cosine sim to the ECAPA target is the
            # adversary-strength readout (higher => features still predict speaker,
            # so the GRL has more to remove).
            with torch.no_grad():
                target_emb = inputs["speaker_embeddings"].to(speaker_logits.dtype)
                speaker_cos = F.cosine_similarity(speaker_logits, target_emb, dim=-1).mean().item()

        # Feature-space regularization losses (zero unless --use_std_hinge or
        # --use_covariance_reg are set; weights are baked into the model-side
        # computation, so no further multiplier is applied here).
        std_hinge_loss = result.get("std_hinge_loss", torch.zeros((), device=ctc_loss.device))
        cov_loss = result.get("cov_loss", torch.zeros((), device=ctc_loss.device))
        # VQ commitment (None unless use_vq). Already scaled by commitment_weight inside the
        # VQ layer; the codebook itself is EMA-updated, so there is no codebook-loss term.
        vq_commitment_loss = result.get("vq_commitment_loss")
        if vq_commitment_loss is None:
            vq_commitment_loss = torch.zeros((), device=ctc_loss.device)

        # ContentVec-style perturbation-consistency (non-adversarial timbre strip). A SECOND
        # independently speaker-perturbed view — pitch shift / VTLP / EQ all preserve duration,
        # so its features are frame-aligned with the first — is forced to match view 1. This
        # removes whatever the perturbations vary (pitch + formants = timbre) without an
        # adversary, so it can't hit the GRL floor. CTC on view 1 is the anti-collapse anchor:
        # the features can't go constant (min consistency) without failing phoneme decode.
        # Off unless --consistency_weight>0; expects the GRL off (--grl_weight 0) for the clean test.
        consistency_loss = torch.zeros((), device=ctc_loss.device)
        consistency_alpha = 0.0
        if self.consistency_weight > 0:
            mel2, mel2_lengths = self._prepare_mel_inputs(inputs, augment=True)  # independent aug draw
            result2 = model(mel2, lengths=mel2_lengths, grl_alpha=0.0)
            f1, f2 = result["features"], result2["features"]
            flen = result["feature_lengths"]
            T = min(f1.shape[1], f2.shape[1])
            fmask = (torch.arange(T, device=f1.device).unsqueeze(0)
                     < flen.clamp(max=T).unsqueeze(1)).unsqueeze(-1)  # [B,T,1] valid frames only
            diff = (f1[:, :T] - f2[:, :T]).pow(2) * fmask
            consistency_loss = diff.sum() / (fmask.sum() * f1.shape[-1]).clamp(min=1)
            # Ramp over the first rampup_steps of THIS run, so it warms up on a
            # fine-tune too. self.state.global_step is relative (0 at start);
            # `global_step` above is absolute (start_step-offset), which on a
            # resumed base (e.g. 300k) already exceeds rampup and would pin the
            # ramp to 1.0 instantly — no warmup, full-weight consistency shock.
            consistency_alpha = min(1.0, self.state.global_step / max(1, self.consistency_rampup_steps))
            consistency_loss = consistency_loss * consistency_alpha

        # Recon-aux (speaker-conditioned mel reconstruction): forces per-frame
        # renderable content into the features so the CTC-blank/VQ tail-collapse
        # can't form. Ramped over recon_ramp_steps of THIS run (relative step, so it
        # warms up on a fine-tune) — a low early weight lets the random-init head
        # become competent before its gradients strongly shape the encoder.
        # recon_alpha schedule: ramp 0->1 over recon_ramp_steps (establish the tail-fix,
        # which scales with weight), HOLD at 1, then optionally anneal 1->recon_floor over
        # [recon_decay_start, +recon_decay_steps] to RELEASE the content-degrading pressure
        # once the tail is structurally learned (the tail-fix's maintenance cost is lower
        # than its establishment cost -- the whole bet). recon_decay_start=0 disables the
        # decay (ramp+hold at 1); recon_floor==1.0 also makes it a no-op. All keyed on the
        # RELATIVE step (self.state.global_step) so it behaves the same on a fresh run or a
        # true resume. Both ramp and decay are min()'d so a misordered config degrades safely.
        recon_loss_raw = result.get("recon_loss")
        recon_loss = torch.zeros((), device=ctc_loss.device)
        recon_alpha = 0.0
        if recon_loss_raw is not None:
            s = self.state.global_step
            ramp = min(1.0, s / max(1, self.recon_ramp_steps))
            decay = 1.0
            if self.recon_decay_start > 0 and s > self.recon_decay_start:
                frac = min(1.0, (s - self.recon_decay_start) / max(1, self.recon_decay_steps))
                decay = 1.0 - frac * (1.0 - self.recon_floor)
            recon_alpha = min(ramp, decay)
            recon_loss = recon_loss_raw

        # Anti-domination penalty (caps single-dim massive activations the recon head
        # otherwise induces). Not ramped -- it's a guardrail we want active from step 0.
        dom_loss_raw = result.get("dom_loss")
        dom_loss = dom_loss_raw if dom_loss_raw is not None else torch.zeros((), device=ctc_loss.device)

        # Combined loss
        # During pre-training phase, speaker loss still contributes but doesn't affect encoder
        # (because grl_alpha=0 means no gradient reversal, but classifier still learns)
        total_loss = (
            self.ctc_weight * ctc_loss
            + self.grl_weight * speaker_loss
            + self.gender_grl_weight * gender_loss
            + self.pad_blank_weight * pad_blank_loss
            + self.consistency_weight * consistency_loss
            + self.recon_weight * recon_alpha * recon_loss
            + self.recon_dom_weight * dom_loss
            + std_hinge_loss
            + cov_loss
            + vq_commitment_loss
        )

        # Log to TensorBoard
        if should_log:
            metrics.log_scalar("train/ctc_loss", ctc_loss, global_step)
            metrics.log_scalar("train/ctc_underflow_count", ctc_underflow_count, global_step)
            metrics.log_scalar("train/ctc_underflow_frac", ctc_underflow_frac, global_step)
            metrics.log_scalar("train/ctc_margin_min", ctc_margin_min, global_step)
            metrics.log_scalar("train/pad_blank_loss", pad_blank_loss, global_step)
            if recon_loss_raw is not None:
                metrics.log_scalar("train/recon_loss", recon_loss_raw, global_step)
                metrics.log_scalar("train/recon_alpha", recon_alpha, global_step)
            if dom_loss_raw is not None:
                metrics.log_scalar("train/recon_dom_loss", dom_loss_raw, global_step)
            metrics.log_scalar("train/speaker_loss", speaker_loss, global_step)
            metrics.log_scalar("train/grl_alpha", grl_alpha, global_step)
            if self.consistency_weight > 0:
                metrics.log_scalar("train/consistency_loss", consistency_loss, global_step)
                metrics.log_scalar("train/consistency_alpha", consistency_alpha, global_step)
            metrics.log_scalar("train/total_loss", total_loss, global_step)
            metrics.log_scalar("train/grl_pretraining", float(in_pretraining), global_step)
            # VQ health: commitment loss + code-usage perplexity. Perplexity collapsing
            # toward 1 (of vq_num_codes) is codebook collapse; dead-code reset fights it,
            # but a perplexity stuck low means the codebook is not being used and the
            # bottleneck is degenerate. Watch this the way you watch CTC accuracy.
            if result.get("vq_perplexity") is not None:
                metrics.log_scalar("train/vq_commitment_loss", vq_commitment_loss, global_step)
                metrics.log_scalar("train/vq_perplexity", result["vq_perplexity"], global_step)
            # Adversary diagnostics — id-mode logs classifier accuracy/collapse;
            # embedding-mode logs cosine sim to the ECAPA target.
            if self.speaker_adversary_target == "speaker_id":
                metrics.log_scalar("train/speaker_accuracy", speaker_acc, global_step)
                metrics.log_scalar("train/speaker_accuracy_top5", speaker_acc_top5, global_step)
                metrics.log_scalar("train/speaker_pred_entropy", pred_entropy, global_step)
                metrics.log_scalar("train/speaker_unique_preds", unique_preds, global_step)
                metrics.log_scalar("train/speaker_max_prob", max_prob, global_step)
            elif speaker_cos is not None:
                metrics.log_scalar("train/speaker_emb_cosine", speaker_cos, global_step)

            # Gender adversary diagnostics (only when the head is active). Accuracy
            # is the leakage readout — HIGH gender accuracy => features still carry
            # gender for the GRL to remove; it should fall as the adversary bites.
            if gender_logits is not None:
                metrics.log_scalar("train/gender_loss", gender_loss, global_step)
                with torch.no_grad():
                    valid_g = gender_ids != -1
                    if valid_g.any():
                        gender_preds = gender_logits.argmax(dim=-1)
                        gender_acc = (gender_preds[valid_g] == gender_ids[valid_g]).float().mean().item()
                        metrics.log_scalar("train/gender_accuracy", gender_acc, global_step)

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

            # Speaker loss (id CE or ECAPA-embedding cosine, matching training)
            speaker_loss = self._speaker_adversary_loss(speaker_logits, inputs, speaker_ids)

            # Gender adversary loss (matches training; skips unknown -1 labels).
            gender_logits = result.get("gender_logits")
            gender_ids = inputs.get("gender_ids")
            gender_loss = torch.zeros((), device=ctc_loss.device)
            if gender_logits is not None and gender_ids is not None and (gender_ids != -1).any():
                gender_loss = self.gender_criterion(gender_logits, gender_ids.long())

            # Combined loss
            total_loss = (
                self.ctc_weight * ctc_loss
                + self.grl_weight * speaker_loss
                + self.gender_grl_weight * gender_loss
            )

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
        # Speaker classifier pooling strategy + adversary target
        'speaker_pooling': args.speaker_pooling,
        'speaker_adversary_target': args.speaker_adversary_target,
        'speaker_embedding_dim': args.speaker_embedding_dim,
        'speaker_classifier_num_heads': args.speaker_classifier_num_heads,
        # Gender GRL adversary (off by default; num_genders always safe to pass).
        'use_gender_grl': args.use_gender_grl,
        'num_genders': args.num_genders,
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
        # VQ bottleneck (disabled unless --use_vq). Post-final-norm discretization.
        'use_vq': args.use_vq,
        'vq_num_codes': args.vq_num_codes,
        'vq_commitment_weight': args.vq_commitment_weight,
        'vq_ema_decay': args.vq_ema_decay,
        'vq_dead_code_threshold': args.vq_dead_code_threshold,
        'vq_codebook_init_path': args.vq_codebook_init_path,
        'vq_cosine': args.vq_cosine,
        'vq_code_dim': args.vq_code_dim,
        # Speaker-conditioned mel-recon aux head (disabled unless --use_recon_aux).
        'use_recon_aux': args.use_recon_aux,
        'recon_aux_width': args.recon_aux_width,
        'recon_aux_blocks': args.recon_aux_blocks,
        'recon_aux_kernel': args.recon_aux_kernel,
        'recon_aux_speaker_dim': args.speaker_embedding_dim,
        'recon_target_cmn': args.recon_target_cmn,
        'recon_dom_cap': args.recon_dom_cap,
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
        # GRL attachment layer — same safe-override discipline (None = keep config).
        'grl_layer': args.grl_layer,
        # Gender adversary pooling — None keeps the config (which mirrors speaker_pooling).
        'gender_pooling': args.gender_pooling,
        # Gender adversary tap layer — None keeps the config (default 10 = SMG tap).
        'gender_grl_layer': args.gender_grl_layer,
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
        consistency_weight=getattr(args, 'consistency_weight', 0.0),
        consistency_rampup_steps=getattr(args, 'consistency_rampup_steps', 5000),
        grl_start_step=args.grl_start_step,
        grl_lr=args.grl_lr,
        pad_blank_weight=args.pad_blank_weight,
        recon_weight=args.recon_weight,
        recon_ramp_steps=args.recon_ramp_steps,
        recon_freeze_encoder_steps=args.recon_freeze_encoder_steps,
        recon_dom_weight=args.recon_dom_weight,
        recon_decay_start=args.recon_decay_start,
        recon_decay_steps=args.recon_decay_steps,
        recon_floor=args.recon_floor,
        speaker_adversary_target=args.speaker_adversary_target,
        gender_grl_weight=args.gender_grl_weight,
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
    sub_parser.add_argument("--sive_total_stride", type=int, default=3,
                            help="Total temporal downsampling stride of the SIVE encoder (default 3 = 3x; 4x was found to over-compress SIVE)")

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
    sub_parser.add_argument("--grl_layer", type=int, default=None,
                            help="Encoder layer the GRL speaker adversary attaches to. Default (None) = "
                                 "config value (-1 = final layer, current behavior). Set e.g. 10 to align "
                                 "the adversary with the SMG's layer-10 tap (0 = conv frontend, 1..N = "
                                 "blocks); CTC stays on the final layer regardless.")
    
    sub_parser.add_argument("--speaker_pooling", type=str, default="attentive_statistics",
                            help="Pooling strategy for speaker classifier: mean | max | statistics | "
                                 "attention | attentive_statistics | multi_head_attention | mhasp "
                                 "(mhasp = multi-head attentive statistics pooling)")
    sub_parser.add_argument("--speaker_adversary_target", type=str, default="speaker_id",
                            choices=["speaker_id", "ecapa_embedding"],
                            help="GRL adversary target: 'speaker_id' (classify id, cross-entropy) or "
                                 "'ecapa_embedding' (cosine-regress the stored ECAPA embedding — richer, "
                                 "generalizes to unseen speakers, enforces features orthogonal to the SMG's "
                                 "speaker vector). Requires speaker_embeddings in the shards (SIVE has them).")
    sub_parser.add_argument("--speaker_embedding_dim", type=int, default=192,
                            help="ECAPA embedding dim; the adversary head output size in ecapa_embedding mode.")
    sub_parser.add_argument("--speaker_classifier_num_heads", type=int, default=4,
                            help="Attention heads for multi-head poolings ('mhasp', 'multi_head_attention').")

    # Gender GRL adversary (parallel to the speaker adversary). Off by default.
    sub_parser.add_argument("--use_gender_grl", action="store_true",
                            help="Enable a gender adversary (GRL) alongside the speaker adversary. A binary "
                                 "male/female pooled head is gradient-reversed into its own tap "
                                 "(--gender_grl_layer, default layer 10 = the SMG tap), scrubbing the gender "
                                 "direction the speaker GRL leaves largely intact (gender otherwise leaks ~0.91 "
                                 "balanced acc). Shares the speaker GRL alpha ramp / start step. Requires "
                                 "gender_ids in the shards.")
    sub_parser.add_argument("--num_genders", type=int, default=2,
                            help="Gender classes for the gender adversary (default 2: 0=male, 1=female; -1=unknown ignored).")
    sub_parser.add_argument("--gender_pooling", type=str, default=None,
                            help="Pooling for the gender adversary head (default: mirror --speaker_pooling).")
    sub_parser.add_argument("--gender_grl_layer", type=int, default=None,
                            help="Encoder layer the gender adversary attaches to (like --grl_layer, but "
                                 "independent). Default (None) = config value (10 = the SMG's layer-10 tap). "
                                 "-1 = final layer; 0 = conv frontend; 1..N = block outputs.")
    sub_parser.add_argument("--gender_grl_weight", type=float, default=0.1,
                            help="Weight on the gender adversary loss in the total (default 0.1, matching --grl_weight). "
                                 "Only active with --use_gender_grl.")

    # CTC-specific settings
    sub_parser.add_argument("--ctc_weight", type=float, default=1.0,
                            help="Weight for CTC loss in total loss")
    # ContentVec-style perturbation-consistency (non-adversarial timbre strip). Needs the
    # speaker-perturbing augments ON (pitch shift + --use_mel_vtlp + --use_mel_freq_response)
    # and length-CHANGING augments OFF (no speed perturb) so the two views stay frame-aligned.
    # Run with --grl_weight 0 (no adversary) for the clean ContentVec-analog test.
    sub_parser.add_argument("--consistency_weight", type=float, default=0.0,
                            help="Weight for the perturbation-consistency loss (0=off). Two independently "
                                 "pitch/formant/EQ-perturbed views are forced to the same features; CTC is "
                                 "the anti-collapse anchor. The non-adversarial alternative to the GRL.")
    sub_parser.add_argument("--consistency_rampup_steps", type=int, default=5000,
                            help="Ramp the consistency weight 0->max over this many steps.")
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
                            choices=["layernorm", "rmsnorm", "batchnorm", "none"],
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

    # VQ bottleneck (post-final-norm discretization; the capacity constraint that makes the
    # GRL's speaker-invariance objective bite). Off by default.
    sub_parser.add_argument("--use_vq", action="store_true",
                            help="Discretize the post-final-norm features through an EMA VQ codebook. "
                                 "CTC + GRL then supervise the codes: content is forced in, speaker "
                                 "is forced out, and the K-code budget is the constraint continuous "
                                 "SIVE lacked. Watch train/vq_perplexity for codebook collapse.")
    sub_parser.add_argument("--vq_num_codes", type=int, default=512,
                            help="Codebook size K. Post-hoc k-means suggested K~100-500 is the "
                                 "interesting range; a trained codebook is more efficient so it can "
                                 "go lower. Sweep once the pipeline works.")
    sub_parser.add_argument("--vq_commitment_weight", type=float, default=0.25,
                            help="Weight on the commitment loss (encoder->code pull). VQ-VAE default 0.25.")
    sub_parser.add_argument("--vq_ema_decay", type=float, default=0.99,
                            help="EMA decay for the codebook (higher = slower/steadier codes).")
    sub_parser.add_argument("--vq_dead_code_threshold", type=float, default=1.0,
                            help="Re-seed codes whose EMA usage falls below this from live encoder "
                                 "outputs (dead-code reset; fights codebook collapse).")
    sub_parser.add_argument("--vq_cosine", action="store_true",
                            help="Cosine-distance VQ: L2-normalize both features and codebook (ViT-VQGAN "
                                 "codebook-collapse fix). Improves codebook utilization (~55%% of K used -> "
                                 "much higher). Cheap; composes with --vq_code_dim.")
    sub_parser.add_argument("--vq_code_dim", type=int, default=0,
                            help="Quantize in a LOW-dim space (learned proj encoder_dim->this, and back). "
                                 "Low-dim codes fill more fully (under-util is partly curse-of-dim). "
                                 "0 = same as encoder_dim (no projection). Try ~32. NOTE: incompatible with "
                                 "--vq_codebook_init_path (kmeans) -- use data-dependent init.")
    sub_parser.add_argument("--vq_codebook_init_path", type=str, default=None,
                            help="Seed the EMA codebook from a precomputed k-means codebook "
                                 "(fit_codebook.py output). Fit it on the SAME features the warm-start "
                                 "encoder produces (i.e. this checkpoint's own feature cache), else the "
                                 "centroids sit outside the encoder's output distribution. None = random "
                                 "data-dependent init from the first batch.")

    # Speaker-conditioned mel-reconstruction aux head (a tiny mini-SMG). Forces
    # per-frame renderable content into the features so a front-loaded CTC alignment
    # can't leave a blank tail for the VQ to collapse (the SMG-flatline root cause).
    sub_parser.add_argument("--use_recon_aux", action="store_true",
                            help="Add a tiny speaker-conditioned (ECAPA-FiLM) mel-recon head that "
                                 "reconstructs the mel from the post-VQ features. Its masked L1 forces "
                                 "EVERY real frame to carry renderable content, structurally preventing "
                                 "the CTC-blank/VQ tail-collapse. Small ON PURPOSE (a big head "
                                 "hallucinates around content-thin features). Needs speaker_embeddings "
                                 "in the shards (ECAPA).")
    sub_parser.add_argument("--recon_weight", type=float, default=1.0,
                            help="Max weight on the recon-aux L1 (ramped in over --recon_ramp_steps). "
                                 "Only active with --use_recon_aux.")
    sub_parser.add_argument("--recon_ramp_steps", type=int, default=5000,
                            help="Ramp recon weight 0->max over the first N steps of THIS run, so the "
                                 "random-init head becomes competent before its gradients strongly "
                                 "shape the encoder (relative step, so it warms up on a fine-tune).")
    sub_parser.add_argument("--recon_freeze_encoder_steps", type=int, default=0,
                            help="Opt-in warmup: for the first N steps only the recon head trains "
                                 "(encoder requires_grad off). Ramp usually suffices; toggling "
                                 "requires_grad is INCOMPATIBLE with DeepSpeed ZeRO — plain DDP/1-GPU only.")
    sub_parser.add_argument("--recon_aux_width", type=int, default=256,
                            help="Recon head conv width (probe-sized; keep small).")
    sub_parser.add_argument("--recon_aux_blocks", type=int, default=6,
                            help="Recon head FiLM conv blocks (probe-sized; keep small).")
    sub_parser.add_argument("--recon_aux_kernel", type=int, default=5,
                            help="Recon head conv kernel size.")
    sub_parser.add_argument("--recon_lr_muon", type=float, default=None,
                            help="Separate (higher) Muon LR for the recon head so the random-init "
                                 "decoder converges fast under --use_muon. None = share encoder lr_muon. "
                                 "Suggest ~0.01-0.02.")
    sub_parser.add_argument("--recon_lr_adamw", type=float, default=None,
                            help="Separate (higher) AdamW LR for the recon head (norms/biases). The "
                                 "encoder's 1.5e-4 is far too slow for a fresh decoder. None = share "
                                 "encoder lr_adamw. Suggest ~5e-4-1e-3.")
    sub_parser.add_argument("--recon_target_cmn", action="store_true",
                            help="Reconstruct the per-utterance mean-normalized mel (cepstral mean "
                                 "normalization) instead of the raw mel. Strips the static spectral "
                                 "envelope (bulk of speaker/VTL timbre) from the target so the head "
                                 "has no reason to pull speaker into the features -> relieves the "
                                 "recon-induced leakage spike. Removes STATIC speaker only; dynamic "
                                 "residual remains.")
    sub_parser.add_argument("--recon_dom_cap", type=float, default=0.0,
                            help="Anti-domination cap: penalize post-norm feature magnitudes above "
                                 "this (relu(|f|-cap)). Stops the recon head from blowing up a single "
                                 "carrier dim (dim-91 -> |156|) that dominates the per-frame norm and "
                                 "squashes content. Symmetric to std_hinge (collapse-only). 0=off; "
                                 "~15-20 typical (normal p99.9 ~6). Needs --recon_dom_weight>0.")
    sub_parser.add_argument("--recon_dom_weight", type=float, default=0.0,
                            help="Weight on the anti-domination penalty (with --recon_dom_cap). "
                                 "Not ramped -- a guardrail active from step 0. Suggest ~0.1-1.0.")
    sub_parser.add_argument("--recon_decay_start", type=int, default=0,
                            help="0=off. After this RELATIVE step, anneal recon_alpha from 1.0 down to "
                                 "--recon_floor over --recon_decay_steps. Rationale: the tail-fix needs "
                                 "HIGH weight to ESTABLISH (scales with weight) but sustained high weight "
                                 "degrades phonetic content (L1-recon is phoneme-confusion-tolerant); "
                                 "decaying releases that pressure once the tail is structurally learned. "
                                 "Set past --recon_ramp_steps (e.g. 5000).")
    sub_parser.add_argument("--recon_decay_steps", type=int, default=10000,
                            help="Anneal duration for the recon_alpha decay.")
    sub_parser.add_argument("--recon_floor", type=float, default=0.25,
                            help="recon_alpha floor after the decay. THE knob to tune: too low and the "
                                 "tail may creep back, too high and content stays degraded. ==1.0 makes "
                                 "the decay a no-op (i.e. a lower static peak is just --recon_weight).")

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
