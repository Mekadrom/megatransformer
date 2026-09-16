import math
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatransformer.model.world.world_model import MegaTransformerWorldModel
from megatransformer.model.image.sdxl_adapter import SDXLConditioningAdapter
from megatransformer.model.image.zimage_adapter import ZImageConditioningAdapter
from megatransformer.config.image.decoder import ZImageAdapterConfig
from megatransformer.scripts.train.trainer import CommonTrainer
from megatransformer.utils import model_loading_utils, megatransformer_utils, metrics, constants


# Target value for text positions the CE must not supervise (batch padding). Cannot be 0:
# the collator pads token ids with 0, which is a real vocab entry (<unk> for Mistral).
TEXT_LOSS_IGNORE_INDEX = -100


def _build_text_targets(full_ids, non_ph_mask_full, valid_full, non_ph_input,
                        ignore_index=TEXT_LOSS_IGNORE_INDEX):
    """Build per-item text targets: strip placeholders from the full sequence, causal-shift
    by one, mark pad positions ignore, and truncate/pad each row to its non-placeholder
    INPUT count K (the number of logits that item produces). Returns (B, max_K).

    Vectorized replacement for the former per-batch loop, which cost ~3*B GPU->CPU syncs
    per step (two boolean-index reads + a `.sum().item()` per row). BIT-IDENTICAL: the
    boolean index that produced `clean`/`clean_valid` returns ascending-position order,
    which is exactly what the cumsum left-pack assigns. Two host syncs total (the two max
    lengths), read together.
    """
    B, T = full_ids.shape
    n_clean = non_ph_mask_full.sum(1)                    # (B,) non-PH count in full seq
    K = non_ph_input.sum(1)                              # (B,) logits per item
    mc, W = (int(x) for x in torch.stack([n_clean.max(), K.max()]).tolist())  # one sync

    targets = full_ids.new_full((B, max(W, 0)), ignore_index)
    if W == 0 or mc < 2:
        return targets                                  # no shifted target survives

    # Left-pack the non-PH tokens (and their validity) per row: destination slot = running
    # count of kept elements; one scatter each; unselected positions go to a throwaway col.
    slots = non_ph_mask_full.long().cumsum(1) - 1
    dst = torch.where(non_ph_mask_full, slots, slots.new_full((), mc))
    clean = full_ids.new_zeros(B, mc + 1)
    clean.scatter_(1, dst, full_ids)
    valid = full_ids.new_zeros(B, mc + 1)
    valid.scatter_(1, dst, valid_full.long())

    shifted = clean[:, 1:mc]                             # (B, mc-1) causal shift (drop col 0)
    valid_shifted = valid[:, 1:mc].bool()
    L = (n_clean - 1).clamp(min=0)                       # (B,) shifted length per row

    use_w = min(W, mc - 1)
    if use_w > 0:
        j = torch.arange(use_w, device=full_ids.device).unsqueeze(0)   # (1, use_w)
        # keep col j iff within this row's logit count (j<K), within its shifted length
        # (j<L), and the shifted token is a valid (non-pad) position.
        keep = (j < K.unsqueeze(1)) & (j < L.unsqueeze(1)) & valid_shifted[:, :use_w]
        targets[:, :use_w] = torch.where(keep, shifted[:, :use_w], targets[:, :use_w])
    return targets


class WorldModelTrainer(CommonTrainer):
    """
    Trainer for the multimodal world model.

    Handles the full forward pass through:
    1. Modality-specific feature extractors
    2. Token interleaving (text as driver with media placeholders)
    3. Recurrent transformer
    4. Token uninterleaving
    5. Modality-specific codas (predictions only)
    6. Loss computation per modality (in this trainer, not the model)

    Each modality is optional per-batch — the model gracefully handles batches
    where only some modalities are present.
    """

    def __init__(
        self,
        *args,
        cmdline: str = "",
        git_commit_hash: str = "",
        step_offset: int = 0,
        # Modality emphasis multipliers (applied AFTER loss whitening, so 1.0
        # means "treat this modality with the same weight as the others"; set
        # higher only when you genuinely want to over-emphasize a modality).
        text_loss_weight: float = 1.0,
        audio_latent_loss_weight: float = 1.0,
        voice_latent_loss_weight: float = 1.0,
        image_latent_loss_weight: float = 1.0,
        # SDXL-adapter path only: weight on the CLIP-conditioning regression loss.
        image_clip_loss_weight: float = 1.0,
        # Z-Image-adapter path: put the Qwen3-4B target encoder on a SEPARATE device
        # (e.g. "cuda:1") to free the training GPU's memory + compute. Optionally load
        # it bf16 there (faster than 4-bit; only fits with headroom).
        image_target_device: Optional[str] = None,
        image_target_bf16: bool = False,
        # Tier-0 whitened MSE (Z-Image adapter): per-dim Qwen3 mean/std stats file to
        # inject into the adapter's whitening buffers (compute via
        # scripts_local/compute_qwen_whiten_stats.py). Requires the small_sum_zimage_whiten
        # config (whiten_target=True). None = naive (un-whitened) MSE.
        image_whiten_stats_path: Optional[str] = None,
        # Tier-1 InfoNCE ramp: linearly ramp the Z-Image adapter's contrastive_weight from 0
        # to its config max over this many steps (measured from the phase start).
        image_contrastive_ramp_steps: int = 2000,
        # Variance-matching aux loss weights (per modality). Penalizes
        # collapsed predictions whose std doesn't match the label std.
        # See WorldModelTrainer._compute_modality_recon_loss for details.
        audio_var_loss_weight: float = 1.0,
        voice_var_loss_weight: float = 1.0,
        image_var_loss_weight: float = 1.0,
        # Variance barrier weights (per modality). Adds a `-log(std_ratio)`
        # barrier that explodes as predictions approach collapse, providing
        # strong anti-collapse pressure that the bounded var_loss cannot.
        # Default 0.0 = disabled. Set to e.g. 0.5–1.0 to enable.
        audio_var_barrier_weight: float = 0.0,
        voice_var_barrier_weight: float = 0.0,
        image_var_barrier_weight: float = 0.0,
        # Stop loss weights for voice/audio autoregressive stop prediction.
        # Whitened by log(2) (BCE baseline) so 1.0 = equal weight to other losses.
        audio_stop_loss_weight: float = 1.0,
        voice_stop_loss_weight: float = 1.0,
        # beta for the voice heteroscedastic beta-NLL (only used when the voice
        # coda is stochastic, i.e. emits a log-variance). 0.5 = Seitzer default.
        voice_beta_nll: float = 0.5,
        # Per-frame probability of feeding the model's own prediction instead of ground
        # truth on the AR path, ramped from 0. 0 = off (pure teacher forcing).
        # F0 contour head weight. Voicing-weighted L1 on the speaker-normalized log-F0
        # contour. There is no voicing head here -- see the loss for why.
        voice_f0_loss_weight: float = 1.0,
        # Deduped (unit, duration) path. When on, the voice coda runs on segments instead
        # of 50Hz frames; the duration head's L1-on-log-frames gets this weight.
        voice_dedup: bool = False,
        voice_duration_loss_weight: float = 1.0,
        voice_scheduled_sampling_prob: float = 0.0,
        voice_scheduled_sampling_ramp_steps: int = 10000,
        voice_scheduled_sampling_start_step: int = 0,
        voice_onpolicy_distill: bool = False,
        voice_onpolicy_temperature: float = 0.8,
        # NAR→AR voice-attention curriculum (Variant B). Down-scales voice→voice
        # attention in the prelude + recurrent trunk + coda by a schedulable alpha,
        # starving the voice-history crutch so the text pathway is forced to carry
        # content. Replaces scheduled sampling (mutually exclusive — see __init__).
        # Schedule: alpha=floor for the first mask_steps (NAR phase), then ramps
        # floor→cap over ramp_steps, then holds at cap. Off when mask_steps==ramp_steps==0.
        voice_ar_attn_mask_steps: int = 0,
        voice_ar_attn_ramp_steps: int = 0,
        voice_ar_attn_floor: float = 0.0,
        voice_ar_attn_cap: float = 1.0,
        voice_ar_attn_ramp_power: float = 1.0,
        # NAR→AR prenet curriculum. Ramps Tacotron-2 prenet dropout on the AR voice path
        # (the shifted-input crutch voice_ar_attn leaves intact) 0→voice_prenet_dropout over
        # ramp_steps, held at 0 for the first start_step. ramp_steps==0 => the static
        # --voice_prenet_dropout (config field) is used unchanged.
        voice_prenet_dropout: float = 0.0,
        voice_cfg_text_dropout_prob: float = 0.0,
        voice_early_text_weight_alpha: float = 1.0,
        voice_early_text_weight_frames: int = 0,
        voice_prenet_dropout_ramp_steps: int = 0,
        voice_prenet_dropout_start_step: int = 0,
        voice_distill_teacher=None,
        voice_distill_weight: float = 0.0,
        voice_distill_temperature: float = 1.0,
        # Modality flags
        include_text: bool = True,
        include_audio: bool = True,
        include_voice: bool = False,
        include_image: bool = True,
        # Whether data provides precomputed VAE latents
        precomputed_latents: bool = True,
        # Text loss label smoothing
        text_label_smoothing: float = 0.0,
        # DiT-specific LR override. When not None, parameters under
        # model.image_generator (the DiffusionBridgeImageDecoder / ImageDecoder)
        # get this LR instead of args.learning_rate. Useful when the DiT path
        # is the destabilizing module and a lower LR keeps training on-rails.
        lr_dit: Optional[float] = None,
        lr_flow: Optional[float] = None,
        # Differential LR schedule (opt-in): give the DiT param group a different
        # LR *schedule* from the trunk/main group. HF's single scheduler otherwise
        # applies the SAME decay curve to every group, so a cosine --lr_scheduler_type
        # silently decays the DiT too. With this on (requires lr_dit set), the DiT
        # follows `lr_dit_schedule` and everything else follows `lr_trunk_schedule`.
        # Grounding: diffusion/flow-matching heads (DiT, SD3, Flux) train best at a
        # flat LR; a settling trunk (cosine/WSD tail) under a constant DiT removes the
        # documented DiT-adaptation spikes (trunk shifts destabilize the DiT).
        differential_lr_schedule: bool = False,
        lr_dit_schedule: str = "constant",      # constant | cosine | wsd
        lr_trunk_schedule: str = "cosine",      # cosine | wsd | constant
        lr_min_ratio: float = 0.1,              # LR floor as a fraction of base (cosine/wsd)
        lr_wsd_decay_frac: float = 0.2,         # fraction of total steps in the WSD decay tail
        # If True, skip text loss in image_synthesis / voice_synthesis batches
        # where text is conditioning rather than target. Default False for
        # backward compatibility with existing runs. Standard practice in
        # multimodal LMs (Flamingo, GIT, BLIP) is True — applying text loss
        # during synthesis competes with the generation objective.
        mask_text_loss_in_synthesis: bool = False,
        emit_duration_token: bool = False,
        # Bistream: keep the text loss alive on chunk-interleaved rows even when
        # --mask_text_loss_in_synthesis is set. Required for inner monologue (the model
        # generating its own transcript); optional for plain bistream TTS, where the
        # transcript is still fed at inference.
        bistream_text_loss: bool = False,
        # Keep the text loss alive on the EOS target that follows a synthesis media block,
        # even under --mask_text_loss_in_synthesis. See the exemption in compute_loss.
        unmask_eos_in_synthesis: bool = False,
        voice_nar: bool = False,
        voice_nar_mask_schedule: str = "cosine",
        voice_nar_mask_ratio_min: float = 0.85,
        voice_nar_mask_anneal_steps: int = 0,
        voice_nar_mask_ratio_floor: float = 0.25,
        voice_nar_trunk_text_only: bool = False,
        # Group within-modality samples by shard when shuffling. Default
        # True. Set False to reproduce the legacy uniform shuffle order
        # when resuming a checkpoint from a pre-shard-aware run.
        shard_aware_sampler: bool = True,
        # Length bucketing (opt-in): group similar-length samples per task into a batch.
        bucket_by_length: bool = False,
        bucket_mega_factor: int = 25,
        # Length curriculum (opt-in): cap voice utterances to <= this many feature frames
        # for THIS run stage (0 = off). A STATIC cap advanced via manual resumes.
        voice_curriculum_max_frames: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.lr_dit = lr_dit
        self.lr_flow = lr_flow
        self.differential_lr_schedule = differential_lr_schedule
        self.lr_dit_schedule = lr_dit_schedule
        self.lr_trunk_schedule = lr_trunk_schedule
        self.lr_min_ratio = lr_min_ratio
        self.lr_wsd_decay_frac = lr_wsd_decay_frac
        self.mask_text_loss_in_synthesis = mask_text_loss_in_synthesis
        self.emit_duration_token = bool(emit_duration_token)
        self.bistream_text_loss = bool(bistream_text_loss)
        self.unmask_eos_in_synthesis = bool(unmask_eos_in_synthesis)
        self.voice_nar = bool(voice_nar)
        self.voice_nar_mask_schedule = str(voice_nar_mask_schedule)
        self.voice_nar_mask_ratio_min = float(voice_nar_mask_ratio_min)
        self.voice_nar_mask_anneal_steps = int(voice_nar_mask_anneal_steps)
        self.voice_nar_mask_ratio_floor = float(voice_nar_mask_ratio_floor)
        self.voice_nar_trunk_text_only = bool(voice_nar_trunk_text_only)
        self.shard_aware_sampler = shard_aware_sampler
        self.bucket_by_length = bucket_by_length
        self.bucket_mega_factor = bucket_mega_factor
        self.voice_curriculum_max_frames = voice_curriculum_max_frames

        self.cmdline = cmdline
        self.git_commit_hash = git_commit_hash
        self.step_offset = step_offset or 0

        self.text_loss_weight = text_loss_weight
        self.audio_latent_loss_weight = audio_latent_loss_weight
        self.voice_latent_loss_weight = voice_latent_loss_weight
        self.image_latent_loss_weight = image_latent_loss_weight
        self.image_clip_loss_weight = image_clip_loss_weight
        self._sdxl_text_encoder = None  # lazy: SDXL CLIP text encoders for adapter targets
        self._zimage_text_encoder = None  # lazy: Qwen3-4B for Z-Image adapter targets
        self.image_target_device = image_target_device
        self.image_target_bf16 = image_target_bf16
        self.image_whiten_stats_path = image_whiten_stats_path
        self._whiten_injected = False
        self.image_contrastive_ramp_steps = int(image_contrastive_ramp_steps)
        self._image_contrastive_phase_start = None

        self.audio_var_loss_weight = audio_var_loss_weight
        self.voice_var_loss_weight = voice_var_loss_weight
        self.image_var_loss_weight = image_var_loss_weight

        self.audio_var_barrier_weight = audio_var_barrier_weight
        self.voice_var_barrier_weight = voice_var_barrier_weight
        self.image_var_barrier_weight = image_var_barrier_weight

        self.audio_stop_loss_weight = audio_stop_loss_weight
        self.voice_stop_loss_weight = voice_stop_loss_weight
        self.voice_beta_nll = voice_beta_nll
        self.voice_f0_loss_weight = voice_f0_loss_weight
        self.voice_dedup = voice_dedup
        self.voice_duration_loss_weight = voice_duration_loss_weight
        self.voice_scheduled_sampling_prob = voice_scheduled_sampling_prob
        self.voice_scheduled_sampling_ramp_steps = voice_scheduled_sampling_ramp_steps
        self.voice_scheduled_sampling_start_step = int(voice_scheduled_sampling_start_step or 0)
        self.voice_onpolicy_distill = bool(voice_onpolicy_distill)
        self.voice_onpolicy_temperature = float(voice_onpolicy_temperature)
        if self.voice_onpolicy_distill and voice_scheduled_sampling_prob <= 0.0:
            raise ValueError(
                "--voice_onpolicy_distill needs --voice_scheduled_sampling_prob > 0: the "
                "on-policy states come from the scheduled-sampling substitution. With prob 0 "
                "the student is fed pure ground truth and the flag would silently no-op.")
        if self.voice_onpolicy_distill and voice_distill_weight <= 0.0:
            raise ValueError(
                "--voice_onpolicy_distill needs --voice_distill_weight > 0; it changes WHICH "
                "history the teacher scores, and does nothing without a distillation term.")

        # NAR→AR voice-attention curriculum (Variant B).
        self.voice_ar_attn_mask_steps = voice_ar_attn_mask_steps
        self.voice_ar_attn_ramp_steps = voice_ar_attn_ramp_steps
        self.voice_ar_attn_floor = voice_ar_attn_floor
        self.voice_ar_attn_cap = voice_ar_attn_cap
        self.voice_ar_attn_ramp_power = voice_ar_attn_ramp_power
        self._voice_ar_attn_enabled = (voice_ar_attn_mask_steps > 0 or voice_ar_attn_ramp_steps > 0)

        # NAR→AR prenet curriculum. Ramp is active only when a positive target AND a positive
        # ramp length are given; otherwise the static config prenet_dropout is used as-is (the
        # forward gets no override), preserving the pre-ramp constant behavior exactly.
        self.voice_prenet_dropout = voice_prenet_dropout
        self.voice_cfg_text_dropout_prob = voice_cfg_text_dropout_prob
        self.voice_early_text_weight_alpha = voice_early_text_weight_alpha
        self.voice_early_text_weight_frames = voice_early_text_weight_frames
        self.voice_prenet_dropout_ramp_steps = voice_prenet_dropout_ramp_steps
        self.voice_prenet_dropout_start_step = voice_prenet_dropout_start_step
        # Frozen distillation teacher (not a submodule: it must never be optimized, saved,
        # or wrapped by the accelerator).
        object.__setattr__(self, "voice_distill_teacher", voice_distill_teacher)
        self.voice_distill_weight = voice_distill_weight
        self.voice_distill_temperature = max(1e-3, voice_distill_temperature)
        self._voice_prenet_ramp_enabled = (voice_prenet_dropout > 0.0 and voice_prenet_dropout_ramp_steps > 0)
        # Both attack the teacher-forcing crutch; stacking them confounds the ablation and
        # the curriculum is the stronger, decisive lever, so it REPLACES scheduled sampling.
        if self._voice_ar_attn_enabled and voice_scheduled_sampling_prob > 0.0:
            raise ValueError(
                "The NAR→AR voice-attention curriculum (--voice_ar_attn_mask_steps/"
                "--voice_ar_attn_ramp_steps) replaces scheduled sampling and is mutually "
                "exclusive with --voice_scheduled_sampling_prob > 0. Disable one."
            )

        self.include_text = include_text
        self.include_audio = include_audio
        self.include_voice = include_voice
        self.include_image = include_image
        self.precomputed_latents = precomputed_latents

        self.has_logged_cli = False

        # Pre-compute log(vocab_size) for text loss whitening. Cross-entropy
        # at uniform predictions equals log(V), so dividing by log(V) puts the
        # loss on a [0, 1] scale where 1 = uniform baseline, 0 = perfect.
        # Makes the text loss commensurable with whitened image/voice/audio.
        model_for_config = self.model.module if hasattr(self.model, 'module') else self.model
        # Control-token ids for THIS model's base (32000 default / native LLM vocab in pretrained
        # mode). Used to strip placeholders when building text targets (below); must match the data.
        self._sp = constants.special_token_ids(
            getattr(model_for_config.config, 'special_token_base', constants.SPECIAL_TOKEN_BASE))
        # Same id the collator appends after a media block, so the EOS exemption below targets
        # the position generation actually samples at.
        self._eos_id = int(getattr(model_for_config.config, 'eos_token_id', constants.EOS_TOKEN_ID))
        try:
            if getattr(model_for_config.config, 'text_encoder', None) is not None:
                # Pretrained mode: the text head spans the LLM's native vocab + the 9 control tokens.
                vocab_size = model_for_config.config.special_token_base + constants.N_SPECIAL_TOKENS
            else:
                vocab_size = model_for_config.config.text_prelude_config.vocab_size
        except AttributeError:
            vocab_size = None
        self.log_vocab_size = math.log(vocab_size) if vocab_size and vocab_size > 1 else 1.0

        # Shard-aware sampler for efficient data loading
        self._shard_sampler = None
        if hasattr(self.train_dataset, 'get_sampler'):
            import torch.distributed as dist
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            self._shard_sampler = self.train_dataset.get_sampler(
                shuffle=True, seed=42,
                batch_size=self.args.per_device_train_batch_size,
                world_size=world_size,
                shard_aware=self.shard_aware_sampler,
                bucket_by_length=self.bucket_by_length,
                bucket_mega_factor=self.bucket_mega_factor,
                curriculum_max_frames=self.voice_curriculum_max_frames,
            )

        # Eval sampler — mirrors the train sampler structure to ensure eval
        # batches are homogeneous by task type. Without this, HF Trainer's
        # default SequentialSampler produces mixed-modality eval batches,
        # which trigger the batch-size-mismatch null-out in world_model.py
        # forward and collapse all eval task_type classification to
        # text_continuation (because voice/image get nulled for non-text-heavy
        # batches). Result: only `eval/text_continuation/*` metrics
        # get logged, and eval_loss ends up being text-only rather than a
        # true average across task types.
        self._eval_shard_sampler = None
        if self.eval_dataset is not None and hasattr(self.eval_dataset, 'get_sampler'):
            import torch.distributed as dist
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            self._eval_shard_sampler = self.eval_dataset.get_sampler(
                shuffle=False,  # deterministic eval for reproducibility across runs
                seed=42,
                batch_size=self.args.per_device_eval_batch_size,
                world_size=world_size,
                shard_aware=self.shard_aware_sampler,
            )

        # GAN support stubs (required by CommonTrainer.is_gan_enabled)
        self.discriminator = None
        self.gan_already_started = False
        self.gan_start_condition_key = None
        self.gan_start_condition_value = None

        # Loss functions
        self.text_loss_fn = nn.CrossEntropyLoss(
            label_smoothing=text_label_smoothing,
            ignore_index=TEXT_LOSS_IGNORE_INDEX,
        )
        # Voice unit CE. Same ignore_index as text: the collator pads unit ids with -100
        # because 0 is a REAL unit id, and padding supervised as unit 0 is exactly the bug
        # the text targets had.
        self.voice_unit_loss_fn = nn.CrossEntropyLoss(ignore_index=TEXT_LOSS_IGNORE_INDEX)
        self.latent_l1_loss = nn.L1Loss()
        self.latent_mse_loss = nn.MSELoss()

        # Numerical stability constant for whitening divisions
        self._loss_eps = 1e-8

        # Per-module gradient norm tracking — built lazily in _get_module_groups()
        self._module_groups = None

        # Per-task eval loss accumulator. Populated during evaluation (see
        # evaluate() / prediction_step); None outside of eval so compute_loss
        # doesn't spend cycles on it during training.
        self._eval_task_accumulator = None
        self._last_loss_components = None
        self._last_task_type = None

    def _compute_modality_recon_loss(
        self,
        name: str,
        preds: torch.Tensor,
        labels: torch.Tensor,
        var_loss_weight: float,
        var_barrier_weight: float = 0.0,
        lengths: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
        beta_nll: float = 0.5,
    ):
        """Whitened L1+MSE reconstruction loss + variance-matching auxiliary loss.

        Whitening:
            Dividing L1 by std(labels) and MSE by var(labels) makes both terms
            dimensionless and ~1 at the "predict the mean" trivial baseline,
            independent of the latent's natural scale. This puts every modality
            on a comparable [0, 1+] range so loss weights become honest emphasis
            multipliers rather than scale-correction hacks.

        Variance-matching aux loss (`var_loss_weight`):
            `|std(preds)/std(labels) - 1|` per sample, mean over batch.
            Dimensionless. Equals 0 when matched, 1 when preds collapse to a
            constant. Symmetric: penalizes both under- and over-shoot. Gradient
            magnitude is bounded, so it provides smooth pressure but cannot
            escape a fully-collapsed degenerate point on its own.

        Variance barrier (`var_barrier_weight`):
            `-log(std(preds)/std(labels))`, clamped to be ≥ 0 (i.e., the
            "prevent collapse" half of a Gaussian KL on the output distribution).
            Has gradient `-1/std(preds)` w.r.t. `std(preds)`, which BLOWS UP
            as the model approaches collapse — providing strong anti-collapse
            pressure that the bounded var_loss above cannot. Equals 0 when
            std(preds) ≥ std(labels). Combine with var_loss to also penalize
            over-shoot.

            This is the variance term of a Gaussian KL between
            `N(μ_pred, σ_pred²)` and `N(μ_label, σ_label²)`, retaining the
            `log(σ_label/σ_pred)` part that makes it act as a barrier function.

        Args:
            lengths: Optional per-sample lengths for variable-length sequences.
                Shape (B,) or (B, 1). When provided, only the first `lengths[b]`
                positions along the last dim of each sample contribute to the
                loss — padding beyond that is masked out. This prevents gradient
                dilution from silence-padded regions in voice/audio features.

        Returns:
            (modality_total_loss, components_dict).
        """
        eps = self._loss_eps

        # Build a mask if lengths are provided. Works for both (B, C, T) voice
        # features and (B, C, H, W) image latents (where masking doesn't apply).
        if lengths is not None:
            lengths = lengths.view(-1)  # (B,)
            T = preds.shape[-1]  # last dim is the time/sequence dim for voice/audio
            # (B, T) mask: True for real positions, False for padding
            mask = torch.arange(T, device=preds.device).unsqueeze(0) < lengths.unsqueeze(1)
            # Expand to match preds shape. For (B, C, T): mask → (B, 1, T)
            while mask.dim() < preds.dim():
                mask = mask.unsqueeze(1)
            # Apply mask: zero out padding in both preds and labels so they
            # don't contribute to any statistic (loss, std, var).
            preds_masked = preds * mask
            labels_masked = labels * mask
            # Count of real elements for mean reduction. Expand the mask to
            # the full data shape before counting so the channel dimension is
            # included (mask is (B, 1, T) but data is (B, C, T)).
            n_real = mask.expand_as(preds).sum().clamp_min(1).float()
        else:
            preds_masked = preds
            labels_masked = labels
            n_real = preds.numel()

        # Heteroscedastic Gaussian (beta-)NLL branch. When the coda emits a
        # per-frame log-variance, `preds` is the Gaussian MEAN and this replaces
        # the whitened recon + variance-matching terms entirely (the NLL already
        # trades mean accuracy against variance calibration). beta-NLL (Seitzer
        # et al. 2022) reweights each term by var**beta (stop-grad) so the mean's
        # gradient isn't down-weighted by 1/var and left to underfit; beta=0.5 is
        # the recommended middle ground (beta=0 => plain NLL, beta=1 => ~MSE).
        if logvar is not None:
            lv = logvar
            inv_var = torch.exp(-lv)
            sq = (labels - preds) ** 2
            nll = 0.5 * (lv + sq * inv_var)  # drop the 0.5*log(2pi) constant
            if beta_nll and beta_nll > 0.0:
                nll = nll * torch.exp(lv * beta_nll).detach()
            if lengths is not None:
                nll_loss = (nll * mask).sum() / n_real
                l1_raw = ((labels - preds).abs() * mask).sum() / n_real
            else:
                nll_loss = nll.mean()
                l1_raw = (labels - preds).abs().mean()
            with torch.no_grad():
                if lengths is not None:
                    sel = mask.expand_as(lv)
                    lv_v = lv[sel]
                else:
                    lv_v = lv.flatten()
                pred_std_mean = torch.exp(0.5 * lv_v).mean()
            return nll_loss, {
                f"{name}_nll_loss": nll_loss.detach(),
                f"{name}_l1_loss_raw": l1_raw.detach(),
                f"{name}_logvar_mean": lv_v.mean().detach(),
                f"{name}_logvar_std": lv_v.std().detach(),
                f"{name}_pred_std_mean": pred_std_mean.detach(),
            }

        # Per-batch label statistics on real positions only, detached so they
        # can't backprop (fixed normalizers, not optimization targets).
        if lengths is not None:
            label_std_global = labels_masked.detach().pow(2).sum().div(n_real).sqrt().clamp_min(eps)
        else:
            label_std_global = labels.detach().std().clamp_min(eps)
        label_var_global = label_std_global * label_std_global

        # Reconstruction terms (raw, then whitened). With masking, we compute
        # element-wise differences and reduce only over real positions.
        if lengths is not None:
            diff = preds_masked - labels_masked
            l1_raw = diff.abs().sum() / n_real
            mse_raw = diff.pow(2).sum() / n_real
        else:
            l1_raw = self.latent_l1_loss(preds, labels)
            mse_raw = self.latent_mse_loss(preds, labels)
        l1_norm = l1_raw / label_std_global
        mse_norm = mse_raw / label_var_global
        recon = l1_norm + mse_norm

        # Per-sample variance matching on real positions only.
        if lengths is not None:
            # Compute per-sample std only over real positions.
            pred_stds = []
            label_stds = []
            for b in range(preds.shape[0]):
                L = int(lengths[b].item())
                if L > 0:
                    pred_stds.append(preds[b, :, :L].flatten().std())
                    label_stds.append(labels[b, :, :L].detach().flatten().std())
                else:
                    pred_stds.append(torch.tensor(0.0, device=preds.device))
                    label_stds.append(torch.tensor(1.0, device=preds.device))
            pred_std_per = torch.stack(pred_stds)
            label_std_per = torch.stack(label_stds).clamp_min(eps)
        else:
            pred_std_per = preds.flatten(1).std(dim=1)
            label_std_per = labels.detach().flatten(1).std(dim=1).clamp_min(eps)
        std_ratio = pred_std_per / label_std_per

        # Symmetric L1 variance loss: bounded gradient, smooth pressure.
        var_loss = (std_ratio - 1.0).abs().mean()

        # Asymmetric log-barrier: blows up as std_ratio → 0, zero for ratio ≥ 1.
        # `-log(ratio).clamp_min(0)` is the "prevent collapse" half of Gaussian KL.
        # We clamp the ratio with eps before logging to avoid numerical -inf.
        var_barrier = (-torch.log(std_ratio.clamp_min(eps))).clamp_min(0.0).mean()

        modality_total = (
            recon
            + var_loss_weight * var_loss
            + var_barrier_weight * var_barrier
        )

        return modality_total, {
            f"{name}_l1_loss_raw": l1_raw.detach(),
            f"{name}_mse_loss_raw": mse_raw.detach(),
            f"{name}_l1_loss_norm": l1_norm.detach(),
            f"{name}_mse_loss_norm": mse_norm.detach(),
            f"{name}_var_loss": var_loss.detach(),
            f"{name}_var_barrier_loss": var_barrier.detach(),
            f"{name}_label_std": label_std_global.detach(),
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        global_step = self.state.global_step + self.step_offset

        # A single-process run never initializes torch.distributed, and is trivially rank 0.
        # Requiring is_initialized() here meant these tags silently never logged outside
        # torchrun/DeepSpeed. Mirrors smg/training.py:330.
        is_main_process = (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
        # `and model.training`: unlike sive/smg, this trainer's prediction_step routes EVAL
        # through compute_loss. Without the training guard, an eval-time call on resume (world
        # evaluates on resume) sets has_logged_cli=True first -- and that eval write does NOT
        # land in the training event file -- so the first training step then skips the block and
        # the run logs no command_line/model_architecture. Only ever consume the flag in training.
        if not self.has_logged_cli and model.training and is_main_process:
            metrics.log_text("training/command_line", self.cmdline, global_step)
            metrics.log_text("training/git_commit_hash", self.git_commit_hash, global_step)
            metrics.log_text("training/model_architecture", str(model), global_step)
            metrics.log_text("training/model_param_count", f"{sum(p.numel() for p in model.parameters()):,}", global_step)
            self.has_logged_cli = True

        # ── Prepare inputs for world model forward ──────────────────────

        # Text: the collator produces text_token_ids [B, T] which include
        # boundary tokens (BOV, EOV, etc.) and placeholder tokens (VOICE_PH, etc.).
        # The interleaver replaces placeholder positions with media embeddings and
        # marks everything else as text. The text coda sees all tokens EXCEPT
        # placeholders.
        #
        # To build aligned targets: remove placeholders from the full sequence,
        # then do the standard causal shift. This ensures e.g. the target for BOV
        # is EOV (the next text token), not VOICE_PH.
        #
        # The model still receives the full text_input_ids (with placeholders)
        # because the interleaver needs them to locate media insertion points.
        text_input_ids = inputs.get("text_token_ids")  # [B, T]
        if text_input_ids is None:
            print(f"[data_debug] text_token_ids is None! Batch keys: {sorted(inputs.keys())}")
            # Return zero loss to skip this batch gracefully
            return torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)
        text_targets = None
        # `or self.emit_duration_token`: a voice-only run has include_text=False, so text
        # targets are never built -- and the NAR duration token lives in the TEXT stream, so
        # it would receive no gradient at all. Caught by a smoke run: nar_mask_frac logged
        # correctly while voice_duration_acc never appeared, i.e. the head was silently dead.
        #
        # `or self.bistream_text_loss`: the SAME trap, hit again on 2026-08-24. The bistream
        # text exemption in compute_loss was correct and completely inert, because targets
        # were never built for it to un-mask -- train/text_loss_norm simply never appeared
        # while every other metric looked healthy. Any future flag that supervises the TEXT
        # stream from a voice-only run has to be added here too.
        if text_input_ids is not None and (self.include_text or self.emit_duration_token
                                           or self.bistream_text_loss
                                           or self.unmask_eos_in_synthesis):
            placeholder_ids = {self._sp.AUDIO_PLACEHOLDER, self._sp.VOICE_PLACEHOLDER, self._sp.IMAGE_PLACEHOLDER}

            # The model sees text_input_ids[:, :-1] as input (standard causal shift).
            # The interleaver removes placeholder positions, so the text coda
            # produces logits only at non-placeholder positions.
            #
            # For targets: remove placeholders from the FULL sequence, then shift.
            # This way BOV's target is EOV (next text token), not VOICE_PH.
            #
            # Example: full = [hello, world, BOV, VOICE_PH, EOV]
            #   Model input (:-1): [hello, world, BOV, VOICE_PH] → 3 text logits
            #   Full no-PH: [hello, world, BOV, EOV] → shift → [world, BOV, EOV] = 3 targets ✓

            full_ids = text_input_ids  # [B, T_full] before :-1

            # Remove placeholders from full sequence, then shift by 1
            non_ph_mask_full = torch.ones_like(full_ids, dtype=torch.bool)
            for pid in placeholder_ids:
                non_ph_mask_full &= (full_ids != pid)

            # The collator right-pads text_token_ids with token id 0 (pad_and_mask), which is a
            # real vocab entry (<unk> for Mistral) — not a reserved pad slot. Pad targets must
            # therefore become -100 (the CE ignore_index) rather than staying 0, or the model is
            # supervised to emit <unk> after EOS. text_token_masks marks valid positions (1.0)
            # vs padding (0.0); it rides the same shifts as the ids so the two stay aligned.
            token_masks = inputs.get("text_token_masks")
            if token_masks is not None:
                valid_full = token_masks.bool()
            else:
                valid_full = torch.ones_like(full_ids, dtype=torch.bool)

            # Model input: keep placeholders for the interleaver
            text_input_ids = full_ids[:, :-1].contiguous()

            # Count non-PH positions in model input (= number of logits per item)
            non_ph_input = torch.ones_like(text_input_ids, dtype=torch.bool)
            for pid in placeholder_ids:
                non_ph_input &= (text_input_ids != pid)

            # Vectorized, bit-identical to the former per-item loop (see _build_text_targets).
            text_targets = _build_text_targets(
                full_ids, non_ph_mask_full, valid_full, non_ph_input, TEXT_LOSS_IGNORE_INDEX)

        # Audio inputs: SIVE features shaped [B, C, T].
        # The world model expects (B, n_audio, C, T) where n_audio is the number
        # of audio clips per batch item. Since the dataset provides one clip per
        # item we unsqueeze to n_audio=1.
        audio_inputs = None
        audio_lengths = None
        audio_latent_labels = None
        audio_loss_lengths = None  # (B,) raw feature lengths for masked loss
        if self.include_audio:
            audio_data = inputs.get("audio_features")  # [B, C, T]
            if audio_data is not None:
                audio_inputs = audio_data.unsqueeze(1)  # [B, 1, C, T]
                # Labels: squeeze n_audio dim to match coda output shape [B, C, T]
                audio_latent_labels = audio_data.clone()

                # Lengths: per-clip lengths, shape (B, n_audio=1)
                feat_lengths = inputs.get("audio_feature_lengths")
                if feat_lengths is not None:
                    audio_lengths = feat_lengths.unsqueeze(1)  # [B, 1]
                    audio_loss_lengths = feat_lengths  # (B,) for masked loss

        # Deduped path: swap the 50Hz voice streams for the (unit, duration) segment
        # streams BEFORE the rest of the voice code runs, so the model input, the unit CE,
        # F0, VUV and stop all transparently operate on segments. The model input becomes
        # the segment centroids (codebook[dedup_unit_ids]); the only genuinely new term is
        # the duration loss below. Everything else is unchanged, just shorter sequences.
        if self.voice_dedup and inputs.get("voice_dedup_unit_ids") is not None:
            unwrapped = model.module if hasattr(model, "module") else model
            cb = getattr(unwrapped, "voice_codebook", None)
            if cb is None:
                raise ValueError("--voice_dedup requires a codebook; call model.set_voice_codebook().")
            d_ids = inputs["voice_dedup_unit_ids"]                    # (B, M) with -100 pad
            cb = cb.to(d_ids.device)
            seg_feats = cb[d_ids.clamp(min=0)].permute(0, 2, 1)      # (B, C, M); pad rows harmless (masked by length)
            inputs = dict(inputs)                                    # shallow copy; don't mutate caller's batch
            inputs["voice_features"] = seg_feats
            inputs["voice_feature_lengths"] = inputs["voice_dedup_lengths"]
            inputs["voice_unit_ids"] = d_ids
            # F0/VUV are NOT swapped to segment rate: the contour stays 50Hz and is
            # predicted from the duration-expanded hidden state below, so within-segment
            # pitch survives. Keep voice_f0_contour / voice_vuv as the collator emitted
            # them. Stash the 50Hz frame length (lost when feature_lengths became segment
            # counts) for the expansion.
            inputs["_voice_frame_lengths"] = voice_frame_lengths_50hz = inputs.get("voice_mel_lengths")

        # Voice: same shape contract as audio (SIVE features). Separate placeholder token.
        voice_inputs = None
        voice_lengths = None
        voice_latent_labels = None
        voice_loss_lengths = None  # (B,) raw feature lengths for masked loss
        voice_chunk_map = None
        if self.include_voice:
            voice_data = inputs.get("voice_features")  # [B, C, T]
            if voice_data is not None:
                voice_inputs = voice_data.unsqueeze(1)  # [B, 1, C, T]
                voice_latent_labels = voice_data.clone()

                voice_feat_lengths = inputs.get("voice_feature_lengths")
                if voice_feat_lengths is not None:
                    voice_lengths = voice_feat_lengths.unsqueeze(1)
                    voice_loss_lengths = voice_feat_lengths  # (B,) for masked loss

                # BISTREAM: (B, M, 3) = (utt_idx, start, length) per voice placeholder. The
                # collator emits it only when at least one row is chunked, so its absence is
                # the unistream path and costs nothing.
                _cs = inputs.get("voice_chunk_starts")
                if _cs is not None:
                    voice_chunk_map = torch.stack(
                        [inputs["voice_chunk_utts"], _cs, inputs["voice_chunk_lengths"]], dim=-1)

        # Image inputs: collator provides image_images [B, C, H, W] (raw or latent).
        # World model expects (B, n_images, ...).
        image_inputs = None
        image_latent_labels = None
        if self.include_image:
            image_data = inputs.get("image_images")  # [B, C, H, W]
            if image_data is not None:
                image_inputs = image_data.unsqueeze(1)  # [B, 1, C, H, W]
                # Labels: keep without n_images dim to match coda output [B, C, H, W]
                image_latent_labels = image_data.clone()

        # ── Forward pass (predictions only, no loss) ────────────────────

        is_synthesis = inputs.get("is_synthesis")
        if is_synthesis is not None:
            is_synthesis = is_synthesis.to(text_input_ids.device if text_input_ids is not None else next(model.parameters()).device)

        # Mixed eval batches: modality tensors may have fewer samples than text.
        # Null out mismatched modalities to avoid assertion errors in the interleaver.
        if text_input_ids is not None:
            B = text_input_ids.shape[0]
            if voice_inputs is not None and voice_inputs.shape[0] != B:
                voice_inputs = None
                voice_lengths = None
                voice_latent_labels = None
                voice_chunk_map = None
            if audio_inputs is not None and audio_inputs.shape[0] != B:
                audio_inputs = None
                audio_lengths = None
                audio_latent_labels = None
            if image_inputs is not None and image_inputs.shape[0] != B:
                image_inputs = None
                image_latent_labels = None

        # print("World model inputs:")
        # megatransformer_utils.print_debug_tensor("\ttext_input_ids", text_input_ids)
        # if audio_inputs is not None:
        #     megatransformer_utils.print_debug_tensor("\taudio_inputs", audio_inputs)
        # if audio_latent_labels is not None:
        #     megatransformer_utils.print_debug_tensor("\taudio_latent_labels", audio_latent_labels)
        # if voice_inputs is not None:
        #     megatransformer_utils.print_debug_tensor("\tvoice_inputs", voice_inputs)
        # if voice_latent_labels is not None:
        #     megatransformer_utils.print_debug_tensor("\tvoice_latent_labels", voice_latent_labels)
        # if image_inputs is not None:
        #     megatransformer_utils.print_debug_tensor("\timage_inputs", image_inputs)
        # if image_latent_labels is not None:
        #     megatransformer_utils.print_debug_tensor("\timage_latent_labels", image_latent_labels)

        self._log_precision_once(model)

        # SDXL-adapter path: compute the CLIP(caption) regression targets from the
        # batch captions (lazy-load the two frozen SDXL CLIP text encoders once).
        # No-op for DiT/direct decoders (image_generator isn't an SDXLConditioningAdapter).
        image_clip_seq_labels = None
        image_clip_pooled_labels = None
        unwrapped_model = model.module if hasattr(model, "module") else model
        if (image_inputs is not None
                and isinstance(getattr(unwrapped_model, "image_generator", None), SDXLConditioningAdapter)):
            captions = inputs.get("text_texts")
            if captions is not None:
                if self._sdxl_text_encoder is None:
                    from megatransformer.utils.sdxl_text_encoder import SDXLTextTargetEncoder
                    dev = next(unwrapped_model.parameters()).device
                    self._sdxl_text_encoder = SDXLTextTargetEncoder(device=dev, dtype=torch.float16)
                image_clip_seq_labels, image_clip_pooled_labels = self._sdxl_text_encoder.encode(captions)

        # Z-Image-adapter path: compute the Qwen3-4B(caption) regression targets
        # (resampled to seq_len), lazy-loading the frozen 4-bit Qwen3 encoder once.
        # No-op unless image_generator is a ZImageConditioningAdapter.
        image_cond_labels = None
        image_cond_mask = None
        if (image_inputs is not None
                and isinstance(getattr(unwrapped_model, "image_generator", None), ZImageConditioningAdapter)):
            # Tier-0: inject per-dim Qwen3 whitening stats into the adapter ONCE, before the
            # first forward. Persists as buffers -> in the checkpoint (eval/chat de-whiten).
            if self.image_whiten_stats_path and not self._whiten_injected:
                _st = torch.load(self.image_whiten_stats_path, map_location="cpu")
                unwrapped_model.image_generator.set_whiten_stats(_st["mean"], _st["std"])
                self._whiten_injected = True
                print(f"[whiten] injected Qwen3 stats from {self.image_whiten_stats_path} "
                      f"-> Z-Image adapter (whiten={unwrapped_model.image_generator.whiten})", flush=True)
            # Tier-1: ramp the adapter's InfoNCE weight from 0 -> config max over ramp_steps
            # (phase-relative, so a --fresh_schedule warm-start ramps from the phase start).
            _gen = unwrapped_model.image_generator
            _cmax = float(getattr(_gen.config, "contrastive_weight", 0.0))
            if _cmax > 0:
                if self._image_contrastive_phase_start is None:
                    self._image_contrastive_phase_start = global_step
                _frac = min(1.0, max(0, global_step - self._image_contrastive_phase_start)
                            / max(1, self.image_contrastive_ramp_steps))
                _gen.contrastive_weight = _cmax * _frac
                if model.training and global_step % self.args.logging_steps == 0:
                    metrics.log_scalar("train/image_contrastive_weight", _gen.contrastive_weight,
                                       global_step, skip_zero=False)
            captions = inputs.get("text_texts")
            if captions is not None:
                model_device = next(unwrapped_model.parameters()).device
                if self._zimage_text_encoder is None:
                    from megatransformer.utils.zimage_text_encoder import Qwen3TextTargetEncoder
                    icfg = unwrapped_model.image_generator.config
                    # Target encoder can live on a separate GPU (--image_target_device) to
                    # free the training card's memory + compute; bf16 there if it has headroom.
                    tgt_device = self.image_target_device or str(model_device)
                    load_4bit = getattr(icfg, "target_load_in_4bit", True) and not self.image_target_bf16
                    self._zimage_text_encoder = Qwen3TextTargetEncoder(
                        model_name=getattr(icfg, "target_model", "Tongyi-MAI/Z-Image-Turbo"),
                        seq_len=getattr(icfg, "seq_len", 64),
                        device=tgt_device,
                        max_length=getattr(icfg, "target_max_length", 512),
                        load_in_4bit=load_4bit,
                    )
                    print(f"[Qwen3TextTargetEncoder] device={tgt_device} "
                          f"precision={'4bit' if load_4bit else 'bf16'}", flush=True)
                _t0 = time.perf_counter()
                # Encoder may be on a different device; targets move back to the training card.
                _gen = unwrapped_model.image_generator
                _native = (getattr(_gen, "ar_flow_head", None) is not None
                           or bool(getattr(_gen, "flow_native_length", False)))
                if _native:
                    # NATIVE length + mask (no K-slot resample), for BOTH the T4 AR head and the
                    # T5 native parallel head: each emits one slot per real Qwen3 token, so the
                    # target must keep its true length.
                    # NB: read the config off the adapter, NOT the local `icfg` -- that is only
                    # bound inside the lazy encoder-construction branch above, so it is undefined
                    # on every step after the first (UnboundLocalError).
                    _icfg = _gen.config
                    _mx = int(getattr(_icfg, "ar_max_len", 128)
                              if getattr(_gen, "ar_flow_head", None) is not None
                              else getattr(_icfg, "flow_max_len", 128))
                    image_cond_labels, image_cond_mask = self._zimage_text_encoder.encode_native(
                        captions, max_len=_mx)
                    image_cond_labels = image_cond_labels.to(model_device)
                    image_cond_mask = image_cond_mask.to(model_device)
                else:
                    image_cond_labels = self._zimage_text_encoder.encode(captions).to(model_device)
                if model.training and global_step % self.args.logging_steps == 0:
                    metrics.log_scalar("train/target_encoder_ms",
                                       (time.perf_counter() - _t0) * 1000.0, global_step, skip_zero=False)

        # Enable per-iteration stat tracking at logging steps. Training only: the stats are
        # logged under train/ and collecting them costs ~6 GPU syncs per recurrent
        # iteration (~192 per step), so paying that on eval batches bought nothing.
        should_log = model.training and global_step % self.args.logging_steps == 0
        model_for_stats = model.module if hasattr(model, 'module') else model
        if should_log and hasattr(model_for_stats, 'recurrent_block'):
            model_for_stats.recurrent_block.track_iteration_stats = True

        # NAR masking replaces scheduled sampling: both corrupt the voice input, but SS
        # simulates AR exposure bias while this IS the NAR training objective.
        nar_mask, nar_coda_units = None, None
        if self.voice_nar and voice_inputs is not None:
            voice_inputs, nar_mask, nar_coda_units = self._apply_nar_masking(
                model, voice_inputs, voice_lengths, is_synthesis, global_step)

        ss_prob = self._scheduled_sampling_prob(global_step) if model.training else 0.0
        onpolicy_unit_ids = None
        if ss_prob > 0.0 and voice_inputs is not None and is_synthesis is not None and bool(is_synthesis.any()):
            voice_inputs, onpolicy_unit_ids = self._apply_scheduled_sampling(
                model, ss_prob, text_input_ids, voice_inputs, voice_lengths, is_synthesis, global_step,
                voice_unit_ids=inputs.get("voice_unit_ids"),
            )

        # NAR→AR curriculum: voice→voice attention scale at this step (None => off).
        voice_attn_alpha = self._voice_attn_alpha(global_step)
        if (voice_attn_alpha is not None and model.training
                and global_step % self.args.logging_steps == 0):
            metrics.log_scalar("train/voice_attn_alpha", voice_attn_alpha, global_step, skip_zero=False)

        # NAR→AR prenet curriculum: per-step ramped prenet dropout (None => static config value).
        voice_prenet_dropout = self._voice_prenet_dropout(global_step)
        if (voice_prenet_dropout is not None and model.training
                and global_step % self.args.logging_steps == 0):
            metrics.log_scalar("train/voice_prenet_dropout", voice_prenet_dropout, global_step, skip_zero=False)

        outputs = model(
            text_input_ids=text_input_ids,
            audio_inputs=audio_inputs,
            audio_lengths=audio_lengths,
            voice_inputs=voice_inputs,
            voice_lengths=voice_lengths,
            voice_chunk_map=voice_chunk_map,
            image_inputs=image_inputs,
            image_latent_labels=image_latent_labels,
            image_clip_seq_labels=image_clip_seq_labels,
            image_clip_pooled_labels=image_clip_pooled_labels,
            image_cond_labels=image_cond_labels,
            image_cond_mask=image_cond_mask,
            precomputed_latents=self.precomputed_latents,
            decode_outputs=False,
            is_synthesis=is_synthesis,
            voice_attn_alpha=voice_attn_alpha,
            voice_coda_units=nar_coda_units,
            voice_prenet_dropout=voice_prenet_dropout,
            cfg_text_dropout_prob=self.voice_cfg_text_dropout_prob,
        )

        if should_log and hasattr(model_for_stats, 'recurrent_block'):
            model_for_stats.recurrent_block.track_iteration_stats = False

        # Deduped path: F0 at 50Hz, not per segment. Expand the coda's segment-rate hidden
        # state by the GT durations (teacher forcing) to the 50Hz timeline, run the F0 head
        # there, and hand the result to the ordinary F0 loss below -- which then compares it
        # to the un-pooled 50Hz contour. The hidden state carries the text (it has attended
        # to the transcript), so the contour stays text-conditioned; expanding raw centroids
        # instead would strip that and just re-derive the SMG's own text-free predictor.
        if (self.voice_dedup and outputs.get("voice_hidden") is not None
                and inputs.get("voice_durations") is not None
                and inputs.get("voice_f0_contour") is not None):
            from megatransformer.utils.codebook import frame_to_segment_index
            h = outputs["voice_hidden"]                                  # (B, M, d)
            contour = inputs["voice_f0_contour"]
            Tf = contour.shape[-1]
            seg_idx = frame_to_segment_index(
                inputs["voice_durations"], inputs["voice_dedup_lengths"], Tf,
            ).to(h.device)
            h_exp = h.gather(1, seg_idx.unsqueeze(-1).expand(-1, -1, h.shape[-1]))  # (B, Tf, d)
            unwrapped = model.module if hasattr(model, "module") else model
            outputs["voice_f0_preds"] = unwrapped.voice_generator.f0_head(h_exp).squeeze(-1)

        # for k, v in outputs.items():
        #     if isinstance(v, torch.Tensor):
        #         megatransformer_utils.print_debug_tensor(f"World model output: {k}", v)

        # ── Compute losses ──────────────────────────────────────────────

        device = text_input_ids.device if text_input_ids is not None else next(model.parameters()).device
        total_loss = torch.tensor(0.0, device=device)
        loss_components = {}

        # Infer the task type from the batch composition for per-task logging.
        # With ModalityGroupedSampler, each batch is homogeneous.
        has_voice_data = voice_inputs is not None
        has_image_data = image_inputs is not None
        has_synthesis = is_synthesis is not None and is_synthesis.any()
        has_transcription = is_synthesis is not None and (~is_synthesis).any()

        if has_voice_data and has_synthesis:
            task_type = "voice_synthesis"
        elif has_voice_data and has_transcription:
            task_type = "voice_transcription"
        elif has_image_data and has_synthesis:
            task_type = "image_synthesis"
        elif has_image_data and has_transcription:
            task_type = "image_transcription"
        else:
            task_type = "text_continuation"

        # Text: cross-entropy on logits vs shifted targets (placeholders already removed).
        # Whitened by log(vocab_size) so that the trivial uniform-prediction
        # baseline corresponds to a normalized loss of 1.0, matching the predict-
        # the-mean baseline of the whitened image/voice/audio reconstruction losses.
        # Optionally skip text loss when text is conditioning rather than target
        # (image_synthesis / voice_synthesis). Standard practice in Flamingo/GIT/BLIP.
        skip_text_loss = (
            self.mask_text_loss_in_synthesis
            and task_type in ("image_synthesis", "voice_synthesis")
        )
        # DURATION-TOKEN EXEMPTION. --mask_text_loss_in_synthesis skips the ENTIRE text loss
        # on synthesis examples, which is right for the transcript (it is conditioning, not a
        # target) but would give the DUR_* token zero gradient -- so the model would emit a
        # never-learned bucket at generation while training loss looked perfectly healthy.
        # This is the same silent-failure shape as the M-RoPE eval flag, so it is an explicit
        # re-target rather than a quiet special case: keep the loss, mask every target that is
        # NOT a duration bucket.
        duration_only = False
        _dur_keep = None
        if skip_text_loss and text_targets is not None:
            _keep = torch.zeros_like(text_targets, dtype=torch.bool)
            if self.emit_duration_token:
                _lo = self._sp.base + 9
                _hi = _lo + constants.N_DURATION_BUCKETS
                _dur_keep = (text_targets >= _lo) & (text_targets < _hi)
                _keep |= _dur_keep
            # EOS EXEMPTION. The synthesis layout ends [.. BOV][PH][EOV][eos], so after
            # placeholder-stripping and the causal shift the target AT the EOV position is
            # EOS -- "the utterance is over, stop". --mask_text_loss_in_synthesis masks it
            # with everything else, so the model receives ZERO gradient on the one position
            # generation actually samples at: right after a media block. It is not that the
            # model prefers to start another block, it is that the position was never
            # trained. Measured 2026-08-24 at step 11076: 2.75 utterances per prompt on
            # average, max 6, from a model whose EOV fires correctly on 12 of 14 blocks.
            if self.unmask_eos_in_synthesis:
                _keep |= (text_targets == self._eos_id)
            # BISTREAM TEXT EXEMPTION, same shape of silent failure as the duration one.
            # Under chunk interleaving the transcript is no longer purely conditioning: the
            # model has to know when it has said everything the text so far supports, and
            # under INNER MONOLOGUE it must generate the text chunks itself. With the whole
            # text loss masked it can never learn to, while training loss looks perfectly
            # healthy -- so this is an explicit per-ROW re-target, not a quiet special case.
            # Per row because a batch mixes unistream and bistream samples.
            if self.bistream_text_loss:
                _bi = inputs.get("voice_is_bistream")
                if _bi is not None and _bi.shape[0] == text_targets.shape[0] and bool(_bi.any()):
                    _keep |= _bi.to(text_targets.device).bool().unsqueeze(1).expand_as(_keep)
            if bool(_keep.any()):
                text_targets = text_targets.masked_fill(~_keep, -100)
                skip_text_loss = False
                # "duration_only" gates the duration ACCURACY metric, which is only
                # interpretable when duration buckets are the only kept targets. With
                # bistream text also kept it is computed on its own subset instead.
                duration_only = _dur_keep is not None
                if _dur_keep is not None:
                    _dur_keep = _dur_keep & (text_targets != -100)
        logits = outputs.get("logits")
        if logits is not None and text_targets is not None and not skip_text_loss:
            B, T, V = logits.size()
            # Align logits and targets (may differ by at most 1 due to
            # uninterleaver padding vs target padding across batch items)
            T_min = min(T, text_targets.shape[1])
            logits = logits[:, :T_min, :].contiguous()
            text_targets = text_targets[:, :T_min].contiguous()
            B, T, V = logits.size()
            text_loss_raw = self.text_loss_fn(
                logits.reshape(B * T, V),
                text_targets.reshape(B * T),
            )
            text_loss_norm = text_loss_raw / self.log_vocab_size
            total_loss = total_loss + self.text_loss_weight * text_loss_norm
            loss_components["text_loss_raw"] = text_loss_raw.detach()
            loss_components["text_loss_norm"] = text_loss_norm.detach()
            # Per-task text loss so we can see transcription vs continuation
            # independently. Same value, just logged under a task-specific key.
            loss_components[f"text_loss_norm/{task_type}"] = text_loss_norm.detach()
            if duration_only:
                with torch.no_grad():
                    # Restricted to duration positions: with the bistream exemption on, the
                    # kept targets also include ordinary text, which would silently turn
                    # "duration accuracy" into "text accuracy".
                    _m = (_dur_keep[:, :T_min].reshape(-1) if _dur_keep is not None
                          else (text_targets.reshape(-1) != -100))
                    if bool(_m.any()):
                        _pred = logits.reshape(-1, V)[_m].argmax(-1)
                        _tgt = text_targets.reshape(-1)[_m]
                        loss_components["voice_duration_acc"] = (_pred == _tgt).float().mean().detach()
                        # off-by-one counts as near-miss: buckets are 7.5% wide, so an adjacent
                        # bucket is a ~7% length error and effectively correct once EOV trims.
                        loss_components["voice_duration_acc_pm1"] = (
                            (_pred - _tgt).abs() <= 1).float().mean().detach()

        # Audio: whitened L1+MSE + variance-matching aux loss (masked by feature lengths)
        audio_latent_preds = outputs.get("audio_latent_preds")
        if audio_latent_preds is not None and audio_latent_labels is not None and audio_latent_preds.numel() > 0:
            audio_modality_loss, audio_components = self._compute_modality_recon_loss(
                "audio_latent", audio_latent_preds, audio_latent_labels,
                self.audio_var_loss_weight,
                self.audio_var_barrier_weight,
                lengths=audio_loss_lengths,
            )
            total_loss = total_loss + self.audio_latent_loss_weight * audio_modality_loss
            loss_components.update(audio_components)

        # Voice, DISCRETE path: cross-entropy over k-means units. Replaces the continuous
        # regression below rather than supplementing it -- the two are different answers to
        # the same question, and mixing them reintroduces the regression objective the
        # units exist to escape. Regression lets the model be "approximately right" by
        # extrapolating the audio history, which is solvable without the text (measured:
        # cosine-sim ~0.02 to target, text-emb grad ~7e-4). CE forces it to NAME the unit.
        voice_unit_logits = outputs.get("voice_unit_logits")
        voice_unit_ids = inputs.get("voice_unit_ids")
        used_unit_loss = False
        if voice_unit_logits is not None and voice_unit_ids is not None:
            # Discrete voice model: the continuous recon head stays OFF (set below) whether
            # or not a supervised loss is actually added this batch.
            used_unit_loss = True
            B, T, K = voice_unit_logits.shape
            tgt = voice_unit_ids[:, :T].to(voice_unit_logits.device).long()  # -100 = padding
            if nar_mask is not None and nar_mask.shape[0] == tgt.shape[0]:
                # MaskGIT: supervise ONLY masked positions. Scoring revealed ones rewards
                # copying an input the model was handed -- the trivial solution, and at low
                # mask ratios it swamps the gradient from the positions that matter.
                tgt = tgt.masked_fill(~nar_mask[:, :tgt.shape[1]].to(tgt.device), -100)
            if tgt.shape[1] < T:  # logits can outrun targets by the shifted-input frame
                tgt = torch.nn.functional.pad(tgt, (0, T - tgt.shape[1]), value=-100)
            # SYNTHESIS-ONLY supervision. In transcription the voice is INPUT and the
            # non-shifted coda predicts each frame's unit from its OWN features -- a trivial
            # near-identity task that dilutes the metric/gradient and would mis-teach the EOV
            # terminal token (a generation-completion signal, meaningless when reading voice).
            # Mask transcription rows to the CE ignore_index. No-op for an all-synthesis batch
            # (current runs) or when no direction info is present.
            if is_synthesis is not None and is_synthesis.shape[0] == B and not bool(is_synthesis.all()):
                tgt = tgt.clone()
                tgt[~is_synthesis.to(tgt.device).bool()] = -100
            if bool((tgt != -100).any()):
                if self.voice_early_text_weight_alpha > 1.0 and self.voice_early_text_weight_frames > 0:
                    # Early-text loss weighting: onset frames have the least AR history, so the
                    # crutch can't help and text must carry -- up-weight them so gradient
                    # concentrates where text is NECESSARY. Weight decays alpha->1 over the first
                    # K frames; renormalized by total weight so the loss SCALE is unchanged (no
                    # silent LR inflation) and reduces to the plain mean when alpha=1.
                    Kf = self.voice_early_text_weight_frames
                    alpha = self.voice_early_text_weight_alpha
                    ce = F.cross_entropy(
                        voice_unit_logits.reshape(B * T, K), tgt.reshape(B * T),
                        ignore_index=-100, reduction="none",
                    ).reshape(B, T)
                    valid = (tgt != -100).float()
                    pos = torch.arange(T, device=ce.device).float()
                    frame_w = 1.0 + (alpha - 1.0) * torch.clamp(1.0 - pos / Kf, min=0.0)  # (T,)
                    w = frame_w.unsqueeze(0) * valid  # (B, T); 0 on pad/transcription
                    denom = w.sum().clamp(min=1.0)
                    unit_loss_raw = (ce * w).sum() / denom
                    loss_components["voice_early_text_meanw"] = (denom / valid.sum().clamp(min=1.0)).detach()
                else:
                    unit_loss_raw = self.voice_unit_loss_fn(
                        voice_unit_logits.reshape(B * T, K), tgt.reshape(B * T),
                    )
                # Whiten by log(K) — CE at uniform predictions is log(K) — so the weight is an
                # honest emphasis multiplier and the number is comparable to the other losses:
                # 1.0 = no better than guessing, 0 = perfect.
                unit_loss_norm = unit_loss_raw / math.log(max(2, K))
                total_loss = total_loss + self.voice_latent_loss_weight * unit_loss_norm
                loss_components["voice_unit_ce_loss_raw"] = unit_loss_raw.detach()
                loss_components["voice_unit_ce_loss_norm"] = unit_loss_norm.detach()

                # --- KL distillation from the frozen CosyVoice 2 speech LM ----------------
                # Hard-label CE collapses a ONE-TO-MANY target (the teacher's predictive
                # entropy on this data is ~4.02 nats) onto a single id. The teacher's soft
                # distribution carries the "which continuations are plausible" structure that
                # CE discards -- and free-running diagnostics show the student is already
                # on-manifold and well-formed, failing ONLY at text->content binding, which is
                # exactly what this transfers.
                teacher = getattr(self, "voice_distill_teacher", None)
                if teacher is not None and self.voice_distill_weight > 0:
                    texts = inputs.get("voice_texts")
                    lens = inputs.get("voice_feature_lengths")
                    if texts is not None and lens is not None:
                        lens = lens.reshape(lens.shape[0], -1)[:, 0] if lens.dim() > 1 else lens
                        # ON-POLICY: condition the teacher on the SAME (partly self-generated)
                        # history the student was fed. Off-policy KD only ever supervises states
                        # reachable from a perfect prefix -- which is exactly where this model is
                        # already near-teacher (early_text_delta +0.050 vs +0.054). Its failure is
                        # off that manifold: once it drifts it loops, and nothing in a GT-only
                        # objective ever tells it how to come back. CE targets stay GROUND TRUTH,
                        # so the signal is "predict the true continuation from where you actually
                        # are", which is the recovery behavior free-running needs.
                        distill_ids = voice_unit_ids
                        if self.voice_onpolicy_distill and onpolicy_unit_ids is not None:
                            distill_ids = onpolicy_unit_ids
                            loss_components["voice_distill_onpolicy_frac"] = (
                                (onpolicy_unit_ids != voice_unit_ids.to(onpolicy_unit_ids.device)
                                 ).float().mean().detach())
                        t_logits, t_mask = teacher(texts, distill_ids, lens, T)
                        t_mask = t_mask & (tgt != -100)          # never supervise pad/transcription
                        if bool(t_mask.any()):
                            temp = self.voice_distill_temperature
                            s_logp = F.log_softmax(
                                voice_unit_logits[t_mask][:, :teacher.student_vocab].float() / temp, dim=-1)
                            t_prob = F.softmax(t_logits[t_mask].to(s_logp.device) / temp, dim=-1)
                            # forward KL(teacher || student): mode-covering, the standard KD
                            # direction. T^2 keeps the gradient scale temperature-independent.
                            kl = F.kl_div(s_logp, t_prob, reduction="batchmean") * (temp ** 2)
                            total_loss = total_loss + self.voice_distill_weight * kl
                            loss_components["voice_distill_kl"] = kl.detach()
                            with torch.no_grad():
                                t_acc = (t_logits[t_mask].argmax(-1) == tgt[t_mask]).float().mean()
                                agree = (t_logits[t_mask].argmax(-1)
                                         == voice_unit_logits[t_mask][:, :teacher.student_vocab].argmax(-1)
                                         ).float().mean()
                                loss_components["voice_distill_teacher_acc"] = t_acc.detach()
                                loss_components["voice_distill_agreement"] = agree.detach()
                with torch.no_grad():
                    valid = tgt != -100
                    acc = (voice_unit_logits.argmax(-1)[valid] == tgt[valid]).float().mean()
                    # THE metric: chance is 1/K. Baselines MEASURED on the training units
                    # (scripts_local/ngram_unit_baseline.py; cosine/codedim32/K=250): the
                    # "repeat previous unit" AR crutch = 0.117, the unigram floor = 0.016,
                    # and the text-free n-gram ceiling (best backed-off n-gram, held out) =
                    # ~0.21 (5-gram seen-only asymptote ~0.23). Clearing ~0.22 is the first
                    # real sign the model uses MORE than local unit history -- i.e. text /
                    # long-range context (the "snap"); below that, gains are indistinguishable
                    # from better n-gram statistics. These are CODEBOOK-SPECIFIC -- re-measure
                    # for a new codebook. (The old "~0.30" here was stale/wrong for this VQ.)
                    # Reference lines for this curve are drawn as separate TB "runs" that
                    # share this exact tag (scripts_local/write_baseline_runs.py) so they
                    # overlay as flat lines on the same chart -- no per-step logging here.
                    loss_components["voice_unit_accuracy"] = acc.detach()
            # else: no synthesis voice rows this batch -> voice adds no generation loss.

        # Duration loss (deduped path): L1 on log-frames, masked to real segments. The
        # coda predicts log(duration) per segment; targets are the run lengths the dataset
        # emitted. Position i predicts segment i's duration (same alignment as the F0/VUV
        # heads, which also read per-segment), NOT shifted like the unit CE -- duration is
        # a property OF the current segment, not a next-token prediction.
        dur_preds = outputs.get("voice_duration_preds")
        dur_tgt = inputs.get("voice_durations")
        if dur_preds is not None and dur_tgt is not None and self.voice_duration_loss_weight > 0:
            T = dur_preds.shape[-1]
            dt = dur_tgt[:, :T].to(dur_preds.device).float()
            if dt.shape[1] < T:
                dt = torch.nn.functional.pad(dt, (0, T - dt.shape[1]))
            mask = (dt >= 1).float()                              # 0-padded segments dropped
            log_tgt = torch.log(dt.clamp(min=1.0))
            dur_l1 = (torch.abs(dur_preds.float() - log_tgt) * mask).sum() / mask.sum().clamp(min=1.0)
            total_loss = total_loss + self.voice_duration_loss_weight * dur_l1
            loss_components["voice_duration_loss"] = dur_l1.detach()
            with torch.no_grad():
                # Frames off, on the real scale, so it reads in the same units as the data.
                frame_err = (torch.abs(torch.exp(dur_preds.float()) - dt) * mask).sum() / mask.sum().clamp(min=1.0)
                loss_components["voice_duration_frame_l1"] = frame_err.detach()

        # Voice F0/VUV: the prosody half of the split. Units carry content (CE above);
        # this carries the contour. Voicing-weighted L1 on log-F0 + BCE on voicing,
        # mirroring SMG.forward's own formulation exactly (smg.py:1113-1125) so the two
        # models optimize the same quantity the same way and the world's contour is a
        # drop-in for the SMG's predictor at inference.
        voice_f0_preds = outputs.get("voice_f0_preds")
        # SPEAKER-NORMALIZED contour, not absolute F0. The world model has no speaker
        # embedding by design (it is modality-general; speaker identity belongs to the
        # decoder), and absolute pitch is dominated by a speaker offset it cannot see.
        # It predicts the contour; the SMG denormalizes using ECAPA.
        f0_tgt = inputs.get("voice_f0_contour", inputs.get("voice_f0"))
        vuv_tgt = inputs.get("voice_vuv")
        if voice_f0_preds is not None and f0_tgt is not None and vuv_tgt is not None:
            T = voice_f0_preds.shape[-1]
            f0_t = f0_tgt[..., :T].to(voice_f0_preds.device).float()
            vuv_t = vuv_tgt[..., :T].to(voice_f0_preds.device).float()
            if f0_t.shape[-1] < T:  # pad if targets are shorter than the coda's frames
                pad = T - f0_t.shape[-1]
                f0_t = torch.nn.functional.pad(f0_t, (0, pad))
                vuv_t = torch.nn.functional.pad(vuv_t, (0, pad))
            # Weight by voicing: F0 is undefined on unvoiced frames, so supervising them
            # would train the head to fit noise. vuv is a soft 0-1 periodicity, not a mask.
            # SYNTHESIS-ONLY (same rationale as the unit CE): zero the voicing weight on
            # transcription rows so they contribute nothing to the F0 loss or the baseline
            # metric below. No-op for an all-synthesis batch.
            if (is_synthesis is not None and is_synthesis.shape[0] == vuv_t.shape[0]
                    and not bool(is_synthesis.all())):
                vuv_t = vuv_t * is_synthesis.to(vuv_t.device).float().unsqueeze(1)
            vsum = vuv_t.sum().clamp(min=1e-8)
            f0_loss = ((voice_f0_preds.float() - f0_t).abs() * vuv_t).sum() / vsum
            total_loss = total_loss + self.voice_f0_loss_weight * f0_loss
            loss_components["voice_f0_loss"] = f0_loss.detach()

            # vuv_t is used ONLY to weight the F0 loss above -- there is no voicing head
            # here. Voicing is phonemic and so recoverable from the units this coda emits,
            # and the SMG predicts it from content + ECAPA. This model is speaker-blind,
            # while soft periodicity is partly a speaker trait, so a head here would be
            # strictly worse-informed than the one downstream.

            with torch.no_grad():
                # Does the head recover the CONTOUR, or just the speaker's mean pitch?
                # The SMG's predictor manages only ~4% over the mean baseline from
                # (units, speaker); the coda has the text and should beat that clearly.
                mask = vuv_t > 0.5
                if mask.sum() > 8:
                    mu = (f0_t * vuv_t).sum() / vsum
                    mean_l1 = ((mu - f0_t).abs() * vuv_t).sum() / vsum
                    loss_components["voice_f0_vs_mean_baseline"] = (f0_loss / mean_l1.clamp(min=1e-8)).detach()

        # Voice: whitened L1+MSE + variance-matching aux loss (masked by feature lengths)
        voice_latent_preds = outputs.get("voice_latent_preds")
        if used_unit_loss:
            voice_latent_preds = None  # discrete path owns the voice loss
        if voice_latent_preds is not None and voice_latent_labels is not None and voice_latent_preds.numel() > 0:
            voice_modality_loss, voice_components = self._compute_modality_recon_loss(
                "voice_latent", voice_latent_preds, voice_latent_labels,
                self.voice_var_loss_weight,
                self.voice_var_barrier_weight,
                lengths=voice_loss_lengths,
                logvar=outputs.get("voice_latent_logvar"),
                beta_nll=self.voice_beta_nll,
            )
            total_loss = total_loss + self.voice_latent_loss_weight * voice_modality_loss
            loss_components.update(voice_components)

        # Stop loss for audio/voice autoregressive generation.
        #
        # Previously the target was `1 for all frames at or past the real length`
        # and the loss was BCE over ALL T positions. That lets the stop head
        # reach near-zero loss by learning the easy signal "my input is a
        # padding frame" — which never fires at inference (autoregressive
        # predictions never produce padding-shaped frames). Diagnosis in
        # feedback_stop_head_exposure_bias.md.
        #
        # Revised formulation:
        # 1. Supervise ONLY real-content positions [0, length-1] (mask out
        #    all padding positions). The model never sees padding in its
        #    supervised range, so it can't learn padding-detection as a
        #    proxy for stop.
        # 2. Target is 1 ONLY at position `length-1` (the last real frame's
        #    own position) — meaning "after predicting this frame, stop."
        #    At inference this fires the right iteration: we generate
        #    exactly `length` real frames total.
        # 3. Class imbalance is severe (1 positive per ~length negatives
        #    per sample). Use BCEWithLogits pos_weight ~= mean(length-1) to
        #    balance, otherwise the loss is dominated by the easy negative
        #    class and the stop head never learns to fire.
        #
        # Whitened by log(2) (random-guess BCE baseline) so loss_weight=1.0
        # gives it equal footing with other whitened losses.
        log2 = 0.6931471805599453  # math.log(2)
        for mod, lengths, weight in [
            ("audio", audio_loss_lengths, self.audio_stop_loss_weight),
            ("voice", voice_loss_lengths, self.voice_stop_loss_weight),
        ]:
            stop_logits = outputs.get(f"{mod}_stop_logits")
            if stop_logits is not None and lengths is not None and weight > 0:
                T = stop_logits.shape[-1]
                lengths_flat = lengths.view(-1)  # (B,)
                device = stop_logits.device
                pos = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
                # Supervised positions: [0, length-1] inclusive. Padding masked out.
                supervised_mask = pos < lengths_flat.unsqueeze(1)  # (B, T)
                # Target: 1 only at position == length-1 (last real frame).
                stop_target = (pos == (lengths_flat - 1).unsqueeze(1)).float()
                # Per-batch class balancing: pos_weight ≈ mean(length-1) gives
                # roughly equal loss contribution from positive and negative
                # classes per sample. Clamp ≥1 for degenerate short samples.
                avg_neg = (lengths_flat.float() - 1.0).clamp(min=1.0).mean()
                pos_weight = avg_neg.detach()
                stop_loss_per_pos = torch.nn.functional.binary_cross_entropy_with_logits(
                    stop_logits, stop_target, pos_weight=pos_weight, reduction="none",
                )
                mask_f = supervised_mask.float()
                stop_loss_raw = (stop_loss_per_pos * mask_f).sum() / mask_f.sum().clamp(min=1.0)
                stop_loss_norm = stop_loss_raw / log2
                total_loss = total_loss + weight * stop_loss_norm
                loss_components[f"{mod}_stop_loss_raw"] = stop_loss_raw.detach()
                loss_components[f"{mod}_stop_loss_norm"] = stop_loss_norm.detach()

        # Image losses: only compute for synthesis (text→image) samples.
        # For transcription (image→text), image tokens are input-only — the model
        # reads them to generate text, it doesn't need to predict/reconstruct them.
        has_synthesis = is_synthesis is not None and is_synthesis.any()
        # If is_synthesis is None (e.g. memorization test without collator direction),
        # fall back to computing image loss on all samples (backward compat).
        compute_image_loss = has_synthesis or is_synthesis is None

        if compute_image_loss:
            # Image loss path. Two cases:
            #   1) DiffusionBridgeImageDecoder returns `image_diffusion_loss` directly
            #      (computed inside the decoder via flow matching).
            #   2) ImageDecoder returns `image_latent_preds` and we compute the
            #      whitened L1+MSE + variance-matching aux loss from it.
            # SDXL-adapter mode: CLIP-conditioning regression loss (MSE + InfoNCE),
            # computed inside the adapter (synthesis-masked). Predicts conditioning,
            # not a latent, so the diffusion/direct branches below don't fire.
            image_clip_loss_t = outputs.get("image_clip_loss")
            if image_clip_loss_t is not None:
                total_loss = total_loss + self.image_clip_loss_weight * image_clip_loss_t
                loss_components["image_clip_loss"] = image_clip_loss_t.detach()
                if "image_clip_mse_loss" in outputs:
                    loss_components["image_clip_mse_loss"] = outputs["image_clip_mse_loss"]
                if "image_contrastive_loss" in outputs:
                    loss_components["image_contrastive_loss"] = outputs["image_contrastive_loss"]
                if "image_contrastive_negatives" in outputs:
                    # Memory-queue depth actually used this step (warms up to queue_size).
                    loss_components["image_contrastive_negatives"] = outputs["image_contrastive_negatives"]
                if "image_flow_loss" in outputs:
                    # T3: the rectified-flow objective (the actual training signal when the
                    # adapter has a flow head; image_clip_mse_loss is then only the aux head).
                    loss_components["image_flow_loss"] = outputs["image_flow_loss"]

            image_diffusion_loss_t = outputs.get("image_diffusion_loss")
            if image_diffusion_loss_t is not None:
                # Diffusion bridge mode: trust the decoder's loss directly.
                total_loss = total_loss + self.image_latent_loss_weight * image_diffusion_loss_t
                loss_components["image_diffusion_loss"] = image_diffusion_loss_t.detach()
                # Also log the unwhitened (raw MSE) for comparison.
                image_diffusion_loss_raw = outputs.get("image_diffusion_loss_raw")
                if image_diffusion_loss_raw is not None:
                    loss_components["image_diffusion_loss_raw"] = image_diffusion_loss_raw
                # Log the rough x_0 estimate's whitened L1 vs labels for monitoring
                # (NOT added to loss — the diffusion loss is the actual training signal).
                with torch.no_grad():
                    x_0_est = outputs.get("image_latent_preds")
                    if x_0_est is not None and image_latent_labels is not None:
                        label_std = image_latent_labels.std().clamp_min(self._loss_eps)
                        loss_components["image_diffusion_x0_est_l1_loss_norm"] = (
                            self.latent_l1_loss(x_0_est, image_latent_labels) / label_std
                        )
            else:
                # Direct prediction mode: whitened L1+MSE + variance-matching aux.
                image_latent_preds_t = outputs.get("image_latent_preds")
                if image_latent_preds_t is not None and image_latent_labels is not None and image_latent_preds_t.numel() > 0:
                    image_modality_loss, image_components = self._compute_modality_recon_loss(
                        "image", image_latent_preds_t, image_latent_labels,
                        self.image_var_loss_weight,
                        self.image_var_barrier_weight,
                    )
                    total_loss = total_loss + self.image_latent_loss_weight * image_modality_loss
                    loss_components.update(image_components)

        # ── TensorBoard logging ─────────────────────────────────────────

        # Non-finite loss handling: skip the batch instead of crashing. We build
        # a zero-valued loss that is graph-connected to a model parameter so
        # backward() produces zero gradients without errors. The offending
        # batch is logged (once per step) for post-hoc analysis.
        if not torch.isfinite(total_loss):
            breakdown = {}
            for k, v in loss_components.items():
                if torch.is_tensor(v):
                    try:
                        breakdown[k] = float(v.item())
                    except Exception:
                        breakdown[k] = "<tensor read failed>"
                else:
                    breakdown[k] = v

            self._nan_skip_count = getattr(self, "_nan_skip_count", 0) + 1
            try:
                loss_scalar = float(total_loss.item())
            except Exception:
                loss_scalar = float("nan")
            print(
                f"[NaN skip #{self._nan_skip_count}] Non-finite world model loss at step {global_step}, "
                f"skipping batch. total_loss={loss_scalar}\n"
                f"  components: {breakdown}\n"
                f"  is_synthesis: {is_synthesis.tolist() if is_synthesis is not None else None}",
                flush=True,
            )

            any_param = next(p for p in model.parameters() if p.requires_grad)
            zero_loss = any_param.sum() * 0.0
            if return_outputs:
                return zero_loss, outputs
            return zero_loss

        # Stash the task type and loss components so prediction_step can
        # accumulate them for per-task eval curves (no-op during training).
        self._last_task_type = task_type
        self._last_loss_components = loss_components

        # Training curve only. prediction_step routes eval through compute_loss with
        # model.eval() and a FROZEN global_step, and eval_steps is normally a multiple of
        # logging_steps — so without this guard every eval batch re-emitted these tags at
        # the exact step the train point occupies, welding eval-distribution values onto
        # the train curve. Eval metrics come from evaluate()'s per-task accumulator
        # instead, which logs one properly averaged point per eval under eval/.
        if model.training and global_step % self.args.logging_steps == 0:
            metrics.log_scalar("train/total_loss", total_loss, global_step)
            for name, value in loss_components.items():
                metrics.log_scalar(f"train/{name}", value, global_step)

            # Recurrent output stats (variance and entropy per modality)
            for key, value in outputs.items():
                if key.startswith("recurrent_out/"):
                    metrics.log_scalar(f"train/{key}", value, global_step)

            # Per-iteration activation stats from recurrent block
            iteration_stats = outputs.get("iteration_stats")
            if iteration_stats:
                from megatransformer.utils import visualization
                fig = visualization.render_iteration_stats(iteration_stats)
                metrics.log_figure("train/iteration_stats", fig, global_step)
                from matplotlib import pyplot as plt
                plt.close(fig)

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def _get_module_groups(self):
        """Build name→module mapping for gradient norm logging (cached).

        Provides both coarse (prelude/coda/recurrent) and fine-grained
        (per-layer, per-submodule) groups for diagnosing gradient flow.
        """
        if self._module_groups is not None:
            return self._module_groups

        model = self.model
        groups = {}

        # Text prelude. From-scratch mode has a wte + prelude blocks; pretrained-LLM mode has a
        # frozen body plus the trainable extension (special_embed) + input_proj + translator MLP.
        tfe = model.text_feature_extractor
        groups["text_embedding"] = tfe
        if hasattr(tfe, 'wte'):
            groups["text_embedding/wte"] = tfe.wte
        for attr, tag in [("special_embed", "text_prelude/special_embed"),
                          ("input_proj", "text_prelude/input_proj"),
                          ("translator", "text_prelude/translator")]:
            if hasattr(tfe, attr) and getattr(tfe, attr) is not None:
                groups[tag] = getattr(tfe, attr)

        # Preludes: coarse + per-layer
        for prefix, extractor in [
            ("audio_prelude", model.audio_feature_extractor),
            ("voice_prelude", model.voice_feature_extractor),
            ("image_prelude", model.image_feature_extractor),
        ]:
            if extractor is None:
                continue
            groups[prefix] = extractor
            if hasattr(extractor, 'projection'):
                groups[f"{prefix}/projection"] = extractor.projection
            if hasattr(extractor, 'prelude'):
                for i, block in enumerate(extractor.prelude):
                    groups[f"{prefix}/layer{i}"] = block
                    groups[f"{prefix}/layer{i}/attn"] = block.self_attn
                    groups[f"{prefix}/layer{i}/ffn"] = block.ffn

        # Recurrent block: coarse + per-block + projection
        groups["recurrent"] = model.recurrent_block
        if model.recurrent_block.projection is not None:
            groups["recurrent/projection"] = model.recurrent_block.projection
        for i, block in enumerate(model.recurrent_block.recurrent_blocks):
            groups[f"recurrent/block{i}"] = block
            groups[f"recurrent/block{i}/attn"] = block.self_attn
            groups[f"recurrent/block{i}/ffn"] = block.ffn

        # Codas: coarse + per-layer
        for prefix, generator in [
            ("text_coda", model.text_generator),
            ("audio_coda", model.audio_generator),
            ("voice_coda", model.voice_generator),
            ("image_generator", model.image_generator),
        ]:
            if generator is None:
                continue
            groups[prefix] = generator
            if hasattr(generator, 'coda'):
                for i, block in enumerate(generator.coda):
                    groups[f"{prefix}/layer{i}"] = block
                    # Attention-free (MLP) coda blocks have no self_attn/ffn — log only the
                    # coarse per-layer group for them.
                    if hasattr(block, 'self_attn'):
                        groups[f"{prefix}/layer{i}/attn"] = block.self_attn
                    if hasattr(block, 'ffn'):
                        groups[f"{prefix}/layer{i}/ffn"] = block.ffn
            if getattr(generator, 'lm_head', None) is not None:
                groups[f"{prefix}/lm_head"] = generator.lm_head
            # Pretrained-LLM text coda: trainable translator + control-token head.
            if hasattr(generator, 'out_translator'):
                groups[f"{prefix}/out_translator"] = generator.out_translator
            if getattr(generator, 'special_head', None) is not None:
                groups[f"{prefix}/special_head"] = generator.special_head
            if hasattr(generator, 'feature_projection'):
                groups[f"{prefix}/feature_proj"] = generator.feature_projection
            if hasattr(generator, 'unpatchify'):
                groups[f"{prefix}/unpatchify"] = generator.unpatchify
            if hasattr(generator, 'temporal_refine') and generator.temporal_refine is not None:
                groups[f"{prefix}/temporal_refine"] = generator.temporal_refine
            # Cross-attention image decoder specific layers
            if hasattr(generator, 'encoder_layers'):
                for i, block in enumerate(generator.encoder_layers):
                    groups[f"{prefix}/encoder{i}"] = block
                    groups[f"{prefix}/encoder{i}/attn"] = block.self_attn
                    groups[f"{prefix}/encoder{i}/ffn"] = block.ffn
            if hasattr(generator, 'encoder_output_norm'):
                groups[f"{prefix}/encoder_norm"] = generator.encoder_output_norm
            if hasattr(generator, 'layers') and hasattr(generator, 'spatial_queries'):
                for i, block in enumerate(generator.layers):
                    groups[f"{prefix}/decoder{i}"] = block
                    groups[f"{prefix}/decoder{i}/self_attn"] = block.self_attn
                    if hasattr(block, 'cross_attn'):
                        groups[f"{prefix}/decoder{i}/cross_attn"] = block.cross_attn
                    groups[f"{prefix}/decoder{i}/ffn"] = block.ffn

        # Image generator special parameters
        if model.image_generator is not None:
            if hasattr(model.image_generator, 'spatial_queries'):
                groups["image_generator/spatial_queries"] = model.image_generator.spatial_queries
            if hasattr(model, 'image_coda_input_norm'):
                groups["image_generator/input_norm"] = model.image_coda_input_norm
            if hasattr(model, 'image_gen_queries'):
                groups["image_generator/gen_queries"] = model.image_gen_queries
            if hasattr(model, 'image_text_conditioning'):
                groups["image_generator/text_conditioning"] = model.image_text_conditioning

        self._module_groups = groups
        return groups

    def _voice_attn_alpha(self, global_step: int) -> Optional[float]:
        """Voice→voice attention scale for the NAR→AR curriculum at ``global_step``.

        Returns None when the curriculum is off (no bias built, fast attention path).
        Otherwise: ``floor`` for the first ``mask_steps`` (NAR phase, history severed),
        then a ``floor``→``cap`` ramp over the next ``ramp_steps`` shaped by
        ``ramp_power``, then ``cap`` (AR). A function of the step only, so train and eval
        at the same step use the same alpha.

        ``ramp_power`` shapes the ramp: 1.0 = linear; >1 = EASE-IN (slow start) — alpha
        crawls through the low range and accelerates late. This matters because the
        fragility sweep showed the model tolerates the low-alpha band and only breaks
        higher up, and the linear ramp diverged (grad explosion ~alpha 0.35): spending
        far more steps re-integrating history at small alpha is the fix. E.g. power=3
        reaches alpha=0.3 only ~67% into the ramp (0.67^3), then covers 0.3→1 quickly.
        """
        if not self._voice_ar_attn_enabled:
            return None
        mask_steps = max(0, self.voice_ar_attn_mask_steps)
        ramp = max(1, self.voice_ar_attn_ramp_steps)
        floor = self.voice_ar_attn_floor
        cap = self.voice_ar_attn_cap
        power = max(1e-6, self.voice_ar_attn_ramp_power)
        if global_step < mask_steps:
            return floor
        t = global_step - mask_steps
        if t >= ramp:
            return cap
        progress = (t / ramp) ** power
        return floor + (cap - floor) * progress

    def _voice_prenet_dropout(self, global_step: int) -> Optional[float]:
        """Per-step Tacotron-2 prenet dropout for the NAR→AR prenet curriculum.

        Returns None when the ramp is off (target<=0 or ramp_steps<=0) — the forward then
        uses the prelude config's static prenet_dropout (which build set to the target), so
        the constant --voice_prenet_dropout behaves exactly as before. When the ramp is on:
        0.0 for the first start_step, then a linear 0→target ramp over ramp_steps, then
        target. Ramped UP (not held high from step 0) because early text is noise — the
        bottleneck is only useful once the text pathway has something to offer; front-loading
        it just starves both crutches at once and stalls convergence.

        A function of the step only, so train and eval at the same step match. Attacks the
        shifted-INPUT crutch (position t is handed frame t-1), the dominant one that
        _voice_attn_alpha (which only severs voice→voice attention over history) leaves intact.
        """
        if not self._voice_prenet_ramp_enabled:
            return None
        target = self.voice_prenet_dropout
        start = max(0, self.voice_prenet_dropout_start_step)
        ramp = max(1, self.voice_prenet_dropout_ramp_steps)
        if global_step < start:
            return 0.0
        t = global_step - start
        if t >= ramp:
            return target
        return target * (t / ramp)

    def _scheduled_sampling_prob(self, global_step: int) -> float:
        """Per-frame probability of feeding the model's OWN prediction instead of ground
        truth, ramped linearly from 0 over --voice_scheduled_sampling_ramp_steps, beginning
        at --voice_scheduled_sampling_start_step.

        Ramped, not constant: at step 0 the model's predictions are noise, and training on
        noise-as-history teaches nothing. The ramp lets the model first learn to predict,
        then progressively removes the ground-truth crutch it would otherwise rely on
        forever.

        The ramp is measured from start_step, NOT from global_step 0. Without that, a warm
        start via --resume_from_checkpoint enters with global_step already past the ramp
        length (e.g. resuming at 50000 with ramp 2000 gives min(1.0, 25) = 1.0), so the
        probability jumps to its full value on the first step and the ramp flag is silently
        inert. Set start_step to the resume step to get the ramp actually asked for.
        """
        if self.voice_scheduled_sampling_prob <= 0.0:
            return 0.0
        start = self.voice_scheduled_sampling_start_step
        if global_step < start:
            return 0.0
        ramp = self.voice_scheduled_sampling_ramp_steps
        if ramp <= 0:                       # 0 = no ramp: full value AT start_step, not one
            return self.voice_scheduled_sampling_prob   # step later (which max(1, ramp) gave)
        return self.voice_scheduled_sampling_prob * min(1.0, (global_step - start) / ramp)

    def _apply_nar_masking(self, model, voice_inputs, voice_lengths, is_synthesis, global_step):
        """MaskGIT-style corruption of the voice INPUT, for masked-parallel (NAR) training.

        Replaces a random subset of frames with a learned MASK feature and returns the mask so
        the caller can restrict the unit loss to those positions. The ratio is drawn from
        `r = cos(pi*u/2), u~U(0,1)` (the MaskGIT schedule), which covers the whole range the
        iterative decoder walks through: r near 1 is "generate from text alone", r near 0 is
        "repair one token given the rest", and inference visits both.

        WHY A CAUSAL TRUNK IS STILL FINE: trunk_hidden[t'] encodes every unit revealed at
        positions <= t', so a BIDIRECTIONAL coda attending over all trunk hidden states sees
        every revealed unit, including ones to its right. That is why NAR needs
        --voice_coda_bidirectional rather than a second unit-injection path into the coda.
        """
        if voice_inputs is None or is_synthesis is None or not bool(is_synthesis.any()):
            return voice_inputs, None, None
        unwrapped = model.module if hasattr(model, "module") else model
        mask_feat = getattr(unwrapped, "voice_mask_feature", None)
        if mask_feat is None:
            return voice_inputs, None, None

        b, n, C, t = voice_inputs.shape
        dev = voice_inputs.device
        u = torch.rand(b, n, 1, 1, device=dev)
        # MASK-RATIO DISTRIBUTION.
        #
        # "cosine" is MaskGIT's, and it is wrong for this task in two ways. It puts most of its
        # mass at moderate ratios, where the utterance is largely reconstructable from
        # neighbouring units -- so the objective is mostly solvable WITHOUT reading the text,
        # and the gradient pressure toward text conditioning is weak. Measured at 23k: revealing
        # 25% of frames raised accuracy 5x (0.0169 -> 0.0860) while text_delta FELL
        # (+0.0037 -> +0.0025), i.e. the model learned bidirectional inpainting instead of
        # reading. Second, inference STARTS fully masked, so r~1 is the condition that
        # determines the first commitments and anchors every later round -- and it is exactly
        # the condition cosine trains least.
        #
        # "high" samples r ~ U[nar_mask_ratio_min, 1.0] (default 0.85), which removes the
        # shortcut and aligns training with the operating point.
        # "linear" is uniform over [min, 1] with min defaulting to 0 -- the neutral control, so
        # that "high beats cosine" can be separated from "anything but cosine beats cosine".
        _sched = getattr(self, "voice_nar_mask_schedule", "cosine")
        _lo = float(getattr(self, "voice_nar_mask_ratio_min", 0.85))
        # ANNEAL. Start fully masked so text is the ONLY available signal and the text pathway
        # has to form, then admit context gradually and see whether it SUPPLEMENTS text or
        # REPLACES it. The failure this addresses is measured: with cosine masking the model
        # reached AR-level unit accuracy (0.0860 vs 0.0887 at matched step) while deriving 3%
        # of it from the transcript instead of 31% -- context was the better predictor, so text
        # was never needed. Removing the shortcut only while the pathway forms is the
        # coarse->refine pattern, applied to the mask ratio.
        #
        # NOTE the honest caveat: gradient descent has no loyalty to features it built earlier,
        # so if context is still the easier predictor once admitted, the crutch can simply
        # reassert. The readout is text_delta at r=1.0 across the anneal -- if it decays as the
        # floor drops, the curriculum did not lock in.
        _anneal = int(getattr(self, "voice_nar_mask_anneal_steps", 0))
        if _anneal > 0:
            _floor = float(getattr(self, "voice_nar_mask_ratio_floor", 0.25))
            _t = min(1.0, max(0.0, global_step / float(_anneal)))
            _lo = 1.0 + (_floor - 1.0) * _t          # 1.0 -> floor, linear in step
            _sched = "high"                           # anneal implies U[_lo, 1]
            if global_step % self.args.logging_steps == 0:
                metrics.log_scalar("train/nar_mask_ratio_min", _lo, global_step, skip_zero=False)
        if _sched == "high":
            ratio = _lo + (1.0 - _lo) * u                          # U[ratio_min, 1]
        elif _sched == "linear":
            ratio = u                                              # U[0, 1] -- ignores ratio_min
        else:
            ratio = torch.cos(u * (math.pi / 2))                   # (b, n, 1, 1) in (0, 1]
        masked = torch.rand(b, n, 1, t, device=dev) < ratio
        # Synthesis rows only: in transcription the voice IS the input being read, so masking
        # it is damage rather than signal.
        masked = masked & is_synthesis.view(b, 1, 1, 1).to(torch.bool)
        # Never mask padding -- no target there, and it would dilute the loss denominator.
        if voice_lengths is not None:
            lens = voice_lengths.reshape(b, -1)[:, 0] if voice_lengths.dim() > 1 else voice_lengths
            idx = torch.arange(t, device=dev).view(1, 1, 1, t)
            masked = masked & (idx < lens.view(b, 1, 1, 1).to(dev))
        mixed = torch.where(masked, mask_feat.view(1, 1, C, 1).to(voice_inputs.dtype), voice_inputs)

        # TRUNK-TEXT-ONLY: the trunk gets EVERY synthesis frame masked regardless of ratio, so
        # revealed units cannot reach it; they are handed to the coda separately. Without this,
        # revealed units enter through the causal prelude and the trunk itself can learn to
        # lean on local unit context instead of text -- which is where the measured crutch is
        # (trunk text-attributed fraction 0.220 at r=1.0 vs 0.029 at r=0.25).
        trunk_in = mixed
        coda_units = None
        if getattr(self, "voice_nar_trunk_text_only", False):
            _synth = is_synthesis.view(b, 1, 1, 1).to(torch.bool).expand_as(masked)
            if voice_lengths is not None:
                _synth = _synth & (torch.arange(t, device=dev).view(1, 1, 1, t)
                                   < lens.view(b, 1, 1, 1).to(dev))
            trunk_in = torch.where(_synth, mask_feat.view(1, 1, C, 1).to(voice_inputs.dtype),
                                   voice_inputs)
            coda_units = mixed

        if global_step % self.args.logging_steps == 0:
            metrics.log_scalar("train/nar_mask_frac", masked.float().mean().item(), global_step,
                               skip_zero=False)
        return trunk_in, masked.squeeze(2).reshape(b * n, t), coda_units

    @torch.no_grad()
    def _apply_scheduled_sampling(
        self, model, ss_prob, text_input_ids, voice_inputs, voice_lengths, is_synthesis, global_step,
        voice_unit_ids=None,
    ):
        """Replace a fraction of the teacher-forced frames with the model's own predictions.

        True scheduled sampling is sequential (decode frame t, feed it to t+1), which is
        unusable here — 500 serial forwards through a 16-iteration recurrence per batch.
        This is the standard transformer approximation (Mihaylova & Martins 2019): one
        no-grad teacher-forced pass to get predictions, mix them into the input, then the
        real pass. The mixed-in frames are one step "stale" (they were predicted from
        ground-truth history, not from other predictions), so it under-states real
        exposure bias — but it removes the guarantee that the history is perfect, which is
        the property the model is currently exploiting.

        Costs one extra forward: ~1.6x step time, since forward is ~2/3 of compute here.
        Only the INPUT is mixed; voice_latent_labels stays ground truth (the trainer holds
        it separately and never passes it to the model).
        """
        tf_out = model(
            text_input_ids=text_input_ids,
            voice_inputs=voice_inputs,
            voice_lengths=voice_lengths,
            precomputed_latents=self.precomputed_latents,
            decode_outputs=False,
            is_synthesis=is_synthesis,
        )
        # What to feed back must MATCH what generate() feeds back, or this simulates the
        # wrong exposure bias. On the DISCRETE path generate() feeds the sampled unit's
        # centroid (world_model.py: voice_codebook[unit_id]); the continuous
        # feature_projection head is unused there and, because used_unit_loss skips its
        # loss, untrained — feeding it would inject off-manifold noise. So: units path ->
        # argmax-unit centroid; continuous path -> the regression mean.
        unwrapped = model.module if hasattr(model, "module") else model
        codebook = getattr(unwrapped, "voice_codebook", None)
        unit_logits = tf_out.get("voice_unit_logits")            # (num_seg, T, K+1)
        if unit_logits is not None and codebook is not None:
            # SAMPLE rather than argmax when a temperature is set. Greedy decoding at 44k
            # gives 96% adjacent repeats, i.e. the argmax IS "repeat the previous unit" --
            # substituting that trains recovery from a state the sampler never visits.
            # Sampling at the deployment temperature substitutes states the model actually
            # reaches. 0.0 keeps the old argmax behavior.
            _t = float(getattr(self, "voice_onpolicy_temperature", 0.0) or 0.0)
            if _t > 0.0:
                _p = torch.softmax(unit_logits.float() / _t, dim=-1)
                ids = torch.multinomial(_p.reshape(-1, _p.shape[-1]), 1).reshape(unit_logits.shape[:-1])
            else:
                ids = unit_logits.argmax(-1)                      # (num_seg, T), in [0, K]
            # The unit head is K+1-way (EOV = id K), but the codebook has only K rows, so an
            # argmax of EOV would index out of bounds (device-side assert). EOV is a terminator
            # with no centroid; clamp it to a valid code for this SS INPUT substitution (rare,
            # and only mixed in per the random use_pred mask below, not a target).
            ids = ids.clamp(max=codebook.shape[0] - 1)
            preds = codebook.to(ids.device)[ids].permute(0, 2, 1)  # (num_seg, C, T)
        else:
            preds = tf_out.get("voice_latent_preds")             # (num_seg, C, T)

        # The coda returns (num_seg, C, T); voice_inputs is (b, n, C, t) with num_seg=b*n.
        # Reshape by element count rather than requiring identical shapes — the old
        # equality check compared 3-D vs 4-D and was therefore ALWAYS false, making
        # scheduled sampling a silent no-op on every path since it was written.
        if preds is None or preds.numel() != voice_inputs.numel():
            if not getattr(self, "_warned_ss_shape", False):
                self._warned_ss_shape = True
                shape = tuple(preds.shape) if preds is not None else None
                print(f"[scheduled_sampling] disabled: preds numel {shape} != "
                      f"voice_inputs {tuple(voice_inputs.shape)}", flush=True)
            return voice_inputs
        preds = preds.reshape(voice_inputs.shape)

        # Per-FRAME mask (broadcast over channels): a frame is either the model's or the
        # truth, never a per-channel chimera of both, which is not a state the model will
        # ever be fed at generation.
        b, n, _, t = voice_inputs.shape
        use_pred = torch.rand(b, n, 1, t, device=voice_inputs.device) < ss_prob
        # Synthesis rows only — transcription feeds real audio as INPUT, so corrupting it
        # there would just be damage.
        synth = is_synthesis.view(b, 1, 1, 1).to(torch.bool)
        use_pred = use_pred & synth
        mixed = torch.where(use_pred, preds.detach().to(voice_inputs.dtype), voice_inputs)

        # ON-POLICY DISTILLATION needs the ids the student was actually fed, not the GT ids:
        # scoring the teacher on GT while the student saw a corrupted history asks "what
        # follows the TRUE prefix" when the student is standing somewhere else. Returning
        # the mixed ids lets the teacher be scored on the SAME history -- "what would the
        # teacher do from where you are", which is the whole point of going on-policy.
        mixed_ids = None
        if voice_unit_ids is not None and unit_logits is not None and codebook is not None:
            gt = voice_unit_ids.to(ids.device).long()
            n_pos = min(gt.shape[-1], ids.shape[-1])
            if gt.reshape(-1, gt.shape[-1]).shape[0] == ids.shape[0]:
                gt_f = gt.reshape(-1, gt.shape[-1])[:, :n_pos]
                sel = use_pred.reshape(ids.shape[0], -1)[:, :n_pos]
                # `gt_f >= 0` preserves -100 padding, which both the CE target and the
                # teacher mask key off; substituting there would supervise pad positions.
                mixed_ids = torch.where(sel & (gt_f >= 0), ids[:, :n_pos], gt_f)
                mixed_ids = mixed_ids.reshape(gt.shape[0], -1) if gt.dim() == 2 else mixed_ids
            elif not getattr(self, "_warned_onpolicy_shape", False):
                self._warned_onpolicy_shape = True
                print(f"[onpolicy_distill] disabled: unit ids {tuple(gt.shape)} do not line up "
                      f"with logits {tuple(ids.shape)}; teacher stays on ground truth", flush=True)

        if global_step % self.args.logging_steps == 0:
            metrics.log_scalar("train/scheduled_sampling_prob", ss_prob, global_step, skip_zero=False)
        return mixed, mixed_ids

    def _log_precision_once(self, model):
        """Report the ACTIVE compute precision, probed from INSIDE the model's forward.

        Every obvious place to check this lies:
          - Parameter dtype reads float32 even when bf16 works: --bf16 keeps fp32 master
            weights and casts per-op.
          - torch.is_autocast_enabled() in compute_loss is always False on GPU. HF's
            autocast_smart_context_manager (trainer.py:3969) returns nullcontext for
            anything but CPU AMP.
          - Model output dtype is always fp32: Accelerate wraps forward in
            convert_outputs_to_fp32(autocast(...)) (accelerator.py:1780).

        Accelerate applies autocast around model.forward, so the only honest probe is a
        hook on a SUBMODULE, which runs inside that context. The hook prints once and
        removes itself.
        """
        if getattr(self, "_precision_probe_installed", False):
            return
        self._precision_probe_installed = True

        unwrapped = model.module if hasattr(model, "module") else model
        target = getattr(unwrapped, "recurrent_block", None) or unwrapped
        handle = {}

        def _probe(_module, _args):
            autocast_on = torch.is_autocast_enabled()
            try:
                autocast_dtype = torch.get_autocast_dtype("cuda")
            except (AttributeError, TypeError):  # torch < 2.4
                autocast_dtype = torch.get_autocast_gpu_dtype()
            mixed = getattr(getattr(self, "accelerator", None), "mixed_precision", "<none>")
            print(
                f"[precision] autocast_in_model={'ON' if autocast_on else 'OFF'} "
                f"dtype={autocast_dtype if autocast_on else 'n/a'} | "
                f"accelerate.mixed_precision={mixed} | "
                f"args.bf16={self.args.bf16} args.fp16={self.args.fp16} | "
                f"tf32_matmul={torch.backends.cuda.matmul.allow_tf32}",
                flush=True,
            )
            if handle.get("h") is not None:
                handle["h"].remove()

        handle["h"] = target.register_forward_pre_hook(_probe)

    def _log_grad_norms(self, global_step):
        """Compute and log L2 and RMS gradient norms per module group."""
        for name, entry in self._get_module_groups().items():
            total_norm_sq = 0.0
            num_params = 0
            if isinstance(entry, torch.nn.Parameter):
                if entry.grad is not None:
                    total_norm_sq = entry.grad.data.float().norm(2).item() ** 2
                    num_params = entry.numel()
            else:
                for p in entry.parameters():
                    if p.grad is not None:
                        total_norm_sq += p.grad.data.float().norm(2).item() ** 2
                        num_params += p.numel()
            metrics.log_scalar(f"train/grad_norm/{name}", total_norm_sq ** 0.5, global_step, skip_zero=False)
            if num_params > 0:
                metrics.log_scalar(f"train/grad_rms/{name}", (total_norm_sq / num_params) ** 0.5, global_step, skip_zero=False)

    def _get_eval_sampler(self, eval_dataset) -> Optional[torch.utils.data.Sampler]:
        """Group eval batches by task type so per-modality eval metrics fire.

        Without this override, HF Trainer uses SequentialSampler at eval.
        With `MultimodalShardedDataset`'s round-robin task assignment
        (idx % n_tasks), sequential iteration produces mixed-modality batches,
        which trigger the batch-size-mismatch null-out in world_model.py
        forward. All eval batches collapse to `text_continuation` task_type
        and voice/image eval metrics never appear. Using
        `ModalityGroupedSampler` at eval produces homogeneous batches like
        training does, so per-task eval curves (e.g.
        `eval/voice_synthesis/voice_latent_l1_norm`) populate correctly.
        """
        if eval_dataset is self.eval_dataset and self._eval_shard_sampler is not None:
            return self._eval_shard_sampler

        # Ad-hoc eval with a different dataset — build a matching sampler
        # on the fly if the dataset supports it.
        if eval_dataset is not None and hasattr(eval_dataset, 'get_sampler'):
            import torch.distributed as dist
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            return eval_dataset.get_sampler(
                shuffle=False,
                seed=42,
                batch_size=self.args.per_device_eval_batch_size,
                world_size=world_size,
            )

        return super()._get_eval_sampler(eval_dataset)

    def create_optimizer(self):
        """Build optimizer with optional DiT-specific LR group.

        HF Trainer's default builds two groups (decay / no-decay) at a single
        LR. When `lr_dit` is set, we subdivide into four groups so parameters
        under `model.image_generator` get their own LR — useful when the DiT
        is the destabilizing module and needs a lower LR than the rest of
        the model. When `lr_dit` is None or when `optimizers=(..)` was passed
        pre-built (e.g. Muon), this delegates to HF's default.
        """
        if self.optimizer is not None:
            return self.optimizer
        if self.lr_dit is None:
            return super().create_optimizer()

        from transformers import Trainer as _HFTrainer

        opt_model = self.model_wrapped if hasattr(self, "model_wrapped") else self.model
        decay_parameters = set(self.get_decay_parameter_names(opt_model))

        def is_dit(name: str) -> bool:
            # Match both plain and DDP/DeepSpeed-wrapped param names.
            return ".image_generator." in f".{name}" or name.startswith("image_generator.")

        def is_flow(name: str) -> bool:
            # The generative head (T3 parallel `flow_head` OR T4 autoregressive
            # `ar_flow_head`). Either is a FRESH module bolted onto an already-converged
            # adapter, so it wants a from-scratch LR (~1e-4) while the warm-started Q-Former
            # beside it must stay slow -- and both live under image_generator.*, so lr_dit
            # alone cannot separate them.
            n = f".{name}"
            return ".image_generator.flow_head." in n or ".image_generator.ar_flow_head." in n

        _lr_flow = self.lr_flow if self.lr_flow is not None else self.lr_dit
        groups = {
            "flow_decay": {"params": [], "weight_decay": self.args.weight_decay, "lr": _lr_flow},
            "flow_no_decay": {"params": [], "weight_decay": 0.0, "lr": _lr_flow},
            "dit_decay": {"params": [], "weight_decay": self.args.weight_decay, "lr": self.lr_dit},
            "dit_no_decay": {"params": [], "weight_decay": 0.0, "lr": self.lr_dit},
            "main_decay": {"params": [], "weight_decay": self.args.weight_decay, "lr": self.args.learning_rate},
            "main_no_decay": {"params": [], "weight_decay": 0.0, "lr": self.args.learning_rate},
        }

        for n, p in opt_model.named_parameters():
            if not p.requires_grad:
                continue
            in_flow = is_flow(n)
            in_dit = is_dit(n) and not in_flow          # flow head is checked first
            in_decay = n in decay_parameters
            key = (
                "flow_decay" if in_flow and in_decay else
                "flow_no_decay" if in_flow else
                "dit_decay" if in_dit and in_decay else
                "dit_no_decay" if in_dit else
                "main_decay" if in_decay else
                "main_no_decay"
            )
            groups[key]["params"].append(p)

        group_items = [(name, g) for name, g in groups.items() if g["params"]]
        optimizer_grouped_parameters = [g for _, g in group_items]
        # Parallel tags aligned to optimizer.param_groups order, so create_scheduler
        # can hand each group its own LR-schedule lambda (DiT vs trunk/main).
        self._lr_group_tags = ["dit" if name.startswith(("dit", "flow")) else "main"
                               for name, _ in group_items]

        optimizer_cls, optimizer_kwargs = _HFTrainer.get_optimizer_cls_and_kwargs(self.args, opt_model)
        # The 'lr' baked into each group takes precedence, but optimizer_kwargs
        # still needs a default 'lr' for AdamW's constructor signature.
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        # Print a summary so the user can verify routing.
        if self.args.local_rank in (-1, 0):
            flow_params = sum(p.numel() for g in ("flow_decay", "flow_no_decay") for p in groups[g]["params"])
            dit_params = sum(p.numel() for g in ("dit_decay", "dit_no_decay") for p in groups[g]["params"])
            main_params = sum(p.numel() for g in ("main_decay", "main_no_decay") for p in groups[g]["params"])
            print(
                f"[create_optimizer] DiT LR split enabled:\n"
                f"  DiT params  (image_generator.*): {dit_params:,} @ lr={self.lr_dit}\n"
                + (f"  Flow params (flow_head.*):       {flow_params:,} @ lr={_lr_flow}\n" if flow_params else "")
                +
                f"  Main params (everything else):   {main_params:,} @ lr={self.args.learning_rate}"
            )

        return self.optimizer

    def _build_lr_lambda(self, schedule: str, warmup: int, total: int):
        """Return a step->multiplier fn (relative to the group's base LR). All
        schedules share a linear 0->1 warmup; after warmup:
          constant : flat 1.0 (the diffusion/flow-matching default; use for the DiT)
          cosine   : 1.0 -> min_ratio over the remaining steps
          wsd      : flat 1.0 until the final `lr_wsd_decay_frac` of steps, then a
                     cosine decay 1.0 -> min_ratio (stable target for most of the run,
                     one short anneal at the end)
        """
        min_ratio = float(self.lr_min_ratio)
        decay_frac = min(max(float(self.lr_wsd_decay_frac), 1e-6), 1.0)

        def lam(step: int) -> float:
            if warmup > 0 and step < warmup:
                return step / max(1, warmup)
            p = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))  # post-warmup progress
            if schedule == "constant":
                return 1.0
            if schedule == "cosine":
                return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * p))
            if schedule == "wsd":
                stable = 1.0 - decay_frac
                if p <= stable:
                    return 1.0
                q = (p - stable) / decay_frac  # 0->1 across the decay tail
                return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * q))
            return 1.0

        return lam

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """Give the DiT and trunk param groups DISTINCT LR schedules.

        Default (differential_lr_schedule=False) delegates to HF, whose single
        scheduler multiplies every param group by the SAME curve — so a cosine
        --lr_scheduler_type would decay the DiT along with the trunk. When enabled
        (and lr_dit is set, so the 4-group split exists), we build one LambdaLR
        with a per-group lambda list: DiT groups follow `lr_dit_schedule`, all
        other groups follow `lr_trunk_schedule`.
        """
        if not self.differential_lr_schedule or self.lr_dit is None:
            return super().create_scheduler(num_training_steps, optimizer)
        if self.lr_scheduler is not None:
            return self.lr_scheduler

        optimizer = optimizer if optimizer is not None else self.optimizer
        tags = getattr(self, "_lr_group_tags", None)
        if tags is None or len(tags) != len(optimizer.param_groups):
            print("[create_scheduler] WARNING: group tags missing/misaligned; "
                  "falling back to HF single-curve scheduler.")
            return super().create_scheduler(num_training_steps, optimizer)
        if self.is_deepspeed_enabled:
            # DeepSpeed builds/owns its own scheduler when ds_config has a
            # "scheduler" block, which would override this. Warn loudly.
            print("[create_scheduler] WARNING: DeepSpeed is enabled — a 'scheduler' "
                  "block in the ds_config will OVERRIDE this per-group schedule. "
                  "Remove it (let HF drive the scheduler) or verify LR curves in TB.")

        from torch.optim.lr_scheduler import LambdaLR
        warmup = self.args.get_warmup_steps(num_training_steps)
        dit_lam = self._build_lr_lambda(self.lr_dit_schedule, warmup, num_training_steps)
        trunk_lam = self._build_lr_lambda(self.lr_trunk_schedule, warmup, num_training_steps)
        lambdas = [dit_lam if t == "dit" else trunk_lam for t in tags]
        self.lr_scheduler = LambdaLR(optimizer, lambdas)

        if self.args.local_rank in (-1, 0):
            n_dit = sum(1 for t in tags if t == "dit")
            print(
                f"[create_scheduler] differential LR schedule:\n"
                f"  DiT groups   ({n_dit}): schedule={self.lr_dit_schedule} @ base lr={self.lr_dit}\n"
                f"  trunk groups ({len(tags) - n_dit}): schedule={self.lr_trunk_schedule} @ base lr={self.args.learning_rate}\n"
                f"  warmup={warmup}, total_steps={num_training_steps}, "
                f"min_ratio={self.lr_min_ratio}, wsd_decay_frac={self.lr_wsd_decay_frac}"
            )
        return self.lr_scheduler

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to capture gradient norms between backward and optimizer step."""
        model_to_use = model.module if hasattr(model, 'module') else model

        loss = super().training_step(model, inputs, num_items_in_batch)

        # After training_step, gradients should still be available (optimizer
        # step hasn't happened yet). Log norms at logging frequency.
        global_step = self.state.global_step + self.step_offset
        if global_step % self.args.logging_steps == 0:
            if self.is_deepspeed_enabled:
                # Under ZeRO, per-param .grad is sharded/None — per-module norms
                # aren't meaningful. Log the engine's global grad norm instead.
                grad_norm = None
                try:
                    grad_norm = model.get_global_grad_norm()
                except Exception:
                    pass
                if grad_norm is not None:
                    if hasattr(grad_norm, "item"):
                        grad_norm = grad_norm.item()
                    metrics.log_scalar("train/grad_norm/global", grad_norm, global_step, skip_zero=False)
            else:
                has_grads = any(p.grad is not None for p in model_to_use.parameters())
                if has_grads:
                    self._log_grad_norms(global_step)

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to route eval through compute_loss (same as training).

        The default Trainer.prediction_step calls model(**inputs), passing raw
        collator keys as kwargs. Our model expects different arg names, so we
        reuse compute_loss which handles the mapping.
        """
        model.eval()
        with torch.no_grad():
            with torch.autocast(device_type=self.args.device.type, dtype=torch.bfloat16, enabled=self.args.bf16):
                loss = self.compute_loss(model, inputs)

        # Accumulate per-task eval metrics if evaluate() has set up the buffer.
        if self._eval_task_accumulator is not None and self._last_task_type is not None:
            bucket = self._eval_task_accumulator.setdefault(self._last_task_type, {})
            def _add(key, val):
                if not math.isfinite(val):
                    return
                cur_sum, cur_count = bucket.get(key, (0.0, 0))
                bucket[key] = (cur_sum + val, cur_count + 1)

            if torch.is_tensor(loss):
                try:
                    _add("total_loss", float(loss.item()))
                except Exception:
                    pass
            for k, v in (self._last_loss_components or {}).items():
                if torch.is_tensor(v) and v.numel() == 1:
                    try:
                        _add(k, float(v.item()))
                    except Exception:
                        pass
        return (loss, None, None)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Run evaluation, accumulating per-task losses, then log them.

        Saves a checkpoint BEFORE eval+visualization so that long-running eval
        or visualization errors don't lose trained weights.

        Produces `eval/{task_type}/{component}` curves alongside HF's
        default `eval_loss`. Task type is inferred from batch composition in
        compute_loss (ModalityGroupedSampler makes each batch homogeneous).
        """
        # Save checkpoint before eval/visualization to avoid losing progress
        # if eval crashes (e.g. missing dependency, shape mismatch, OOM).
        try:
            self._save_checkpoint(self.model, trial=None)
        except Exception as e:
            print(f"Warning: pre-eval checkpoint save failed: {e}")

        self._eval_task_accumulator = {}
        # EMA AT EVAL. The EMA callback maintained shadow weights and saved them to
        # ema_state.pt, but nothing ever evaluated them -- every eval metric and every
        # training render showed the RAW weights, so --use_ema was measurable only by
        # loading ema_state.pt by hand afterwards.
        #
        # The swap wraps ONLY super().evaluate(): the pre-eval checkpoint above must keep
        # writing RAW weights (otherwise pytorch_model.bin silently becomes the EMA and a
        # resume would restart from averaged weights), while the viz callback's on_evaluate
        # -- which fires INSIDE super().evaluate() -- lands inside the swap, so the audio
        # renders reflect the EMA too. restore() is in a finally so a crashed eval cannot
        # leave the live weights replaced by the average.
        _ema = getattr(self, "ema", None)
        _swapped = False
        if _ema is not None:
            _ema.apply_shadow()
            _swapped = True
        try:
            output = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            if _swapped:
                _ema.restore()
            global_step = self.state.global_step + self.step_offset
            for task_type, bucket in self._eval_task_accumulator.items():
                for component, (s, c) in bucket.items():
                    if c > 0:
                        metrics.log_scalar(
                            f"eval/{task_type}/{component}",
                            s / c,
                            global_step,
                            skip_zero=False,
                        )
            self._eval_task_accumulator = None
        return output

    @staticmethod
    def _count_params(module):
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    def start_train_print(self, args):
        model = self.model
        print(f"Model architecture:\n{model}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n{'=' * 60}")
        print(f"World model: {model.__class__.__name__}")
        print(f"Total parameters:     {total_params:>14,}")
        print(f"Trainable parameters: {trainable_params:>14,}")
        print(f"{'=' * 60}")

        # Preludes (feature extractors) — only show instantiated modules
        preludes = {"Text prelude": model.text_feature_extractor}
        if model.audio_feature_extractor is not None:
            preludes["Audio prelude"] = model.audio_feature_extractor
        if model.voice_feature_extractor is not None:
            preludes["Voice prelude"] = model.voice_feature_extractor
        if model.image_feature_extractor is not None:
            preludes["Image prelude"] = model.image_feature_extractor

        print(f"\n  Preludes (feature extractors):")
        prelude_total = 0
        for name, mod in preludes.items():
            t, tr = self._count_params(mod)
            prelude_total += t
            frozen = " (frozen)" if tr == 0 and t > 0 else ""
            print(f"    {name:<20s} {t:>12,}{frozen}")
        print(f"    {'─' * 34}")
        print(f"    {'Subtotal':<20s} {prelude_total:>12,}")

        # Recurrent block
        rec_t, rec_tr = self._count_params(model.recurrent_block)
        print(f"\n  Recurrent block:       {rec_t:>12,}")

        # Codas (generators) — only show instantiated modules
        codas = {"Text coda": model.text_generator}
        if model.audio_generator is not None:
            codas["Audio coda"] = model.audio_generator
        if model.voice_generator is not None:
            codas["Voice coda"] = model.voice_generator
        if model.image_generator is not None:
            codas["Image coda"] = model.image_generator

        print(f"\n  Codas (generators):")
        coda_total = 0
        for name, mod in codas.items():
            t, tr = self._count_params(mod)
            coda_total += t
            frozen = " (frozen)" if tr == 0 and t > 0 else ""
            print(f"    {name:<20s} {t:>12,}{frozen}")
        print(f"    {'─' * 34}")
        print(f"    {'Subtotal':<20s} {coda_total:>12,}")

        # Summary
        print(f"\n  {'─' * 40}")
        print(f"  Preludes:              {prelude_total:>12,}  ({100*prelude_total/total_params:.1f}%)")
        print(f"  Recurrent block:       {rec_t:>12,}  ({100*rec_t/total_params:.1f}%)")
        print(f"  Codas:                 {coda_total:>12,}  ({100*coda_total/total_params:.1f}%)")
        print(f"{'=' * 60}")

        print(f"\nActive modalities: text={self.include_text}, audio={self.include_audio}, "
              f"voice={self.include_voice}, image={self.include_image}")
        tied = getattr(model.config, 'tie_word_embeddings', False)
        print(f"Precomputed latents: {self.precomputed_latents}")
        if tied and hasattr(model.text_feature_extractor, 'wte'):
            tied_params = sum(p.numel() for p in model.text_feature_extractor.wte.parameters())
            print(f"Tied word embeddings: True ({tied_params:,} params shared)")
        print(f"Loss weights: text={self.text_loss_weight}, audio={self.audio_latent_loss_weight}, "
              f"voice={self.voice_latent_loss_weight}, image={self.image_latent_loss_weight}\n")


def _voice_feature_channels(args) -> Optional[int]:
    """Width of the cached voice 'features' tensors.

    The voice coda predicts features and the SMG decodes those same features to mel, so
    the world model's feature_channels and the SMG's sive_encoder_dim are one quantity,
    not two that must be kept in sync. --voice_smg_sive_encoder_dim therefore drives both
    when given; --voice_feature_channels covers runs with no SMG loaded.
    """
    explicit = getattr(args, 'voice_feature_channels', None)
    if explicit is not None:
        return explicit
    return getattr(args, 'voice_smg_sive_encoder_dim', None)


def load_model(args, device='cuda'):
    include_modes = [m.strip() for m in args.include_modes.split(",")]
    overrides = {"include_modes": include_modes}
    if getattr(args, 'tie_word_embeddings', False):
        overrides["tie_word_embeddings"] = True
    if getattr(args, 'gen_query_mode', None) is not None:
        overrides["gen_query_mode"] = args.gen_query_mode
    if getattr(args, 'n_image_gen_positions', None) is not None:
        overrides["n_image_gen_positions"] = args.n_image_gen_positions

    # Pre-construction overrides for nested configs
    needs_override = (getattr(args, 'iteration_norm', None) is not None or
                      getattr(args, 'share_block_weights', False) or
                      getattr(args, 'max_seq_len', None) is not None or
                      _voice_feature_channels(args) is not None or
                      getattr(args, 'voice_prenet_dropout', 0.0) > 0.0 or
                      getattr(args, 'voice_codebook_path', None) is not None or
                      getattr(args, 'voice_predict_f0', False) or
                      getattr(args, 'voice_cfg_text_dropout_prob', 0.0) > 0.0 or
                      getattr(args, 'mean_thinking_steps', None) is not None or
                      getattr(args, 'image_contrastive_queue_size', None) is not None or
                      getattr(args, 'image_flow_aux_mse_weight', None) is not None or
                      getattr(args, 'voice_stochastic_output', False) or
                      getattr(args, 'use_mrope', False))
    if needs_override:
        import copy
        from megatransformer.config.world.world_model import WORLD_MODEL_CONFIGS
        from megatransformer.config.common import MegaTransformerBlockConfig
        config = copy.deepcopy(WORLD_MODEL_CONFIGS[args.config])
        if getattr(args, 'iteration_norm', None) is not None:
            config.recurrent_block_config.iteration_norm = args.iteration_norm
        if getattr(args, 'share_block_weights', False):
            config.recurrent_block_config.share_block_weights = True
        if getattr(args, 'use_mrope', False):
            # Only the TRUNK: it is the one module that sees text and media together, so it
            # is the only place a text<->voice coordinate can live. Preludes/codas keep their
            # existing per-stream RoPE.
            config.recurrent_block_config.block_config.use_mrope = True
            config.mrope_voice_rate = float(getattr(args, 'mrope_voice_rate', 6.0))
            config.mrope_scale_side = str(getattr(args, 'mrope_scale_side', 'voice'))
        if getattr(args, 'voice_prenet_dropout', 0.0) and args.voice_prenet_dropout > 0.0:
            config.voice_prelude_config.prenet_dropout = args.voice_prenet_dropout
        if getattr(args, 'voice_cfg_text_dropout_prob', 0.0) > 0.0:
            config.voice_cfg_enabled = True  # create the null_text_embed param (CFG)
        if getattr(args, 'voice_nar', False):
            # Masked-parallel voice. Two coupled changes: the MASK parameter, and a
            # BIDIRECTIONAL coda -- with a causal coda, position t cannot see units revealed
            # to its right and iterative refinement degenerates to left-to-right infilling.
            config.voice_nar = True
            config.voice_coda_config.coda_config.causal = False
            if getattr(args, 'voice_nar_trunk_text_only', False):
                # Builds voice_coda_units_proj; without this the injection path silently
                # does not exist and the flag would be a no-op.
                config.voice_nar_trunk_text_only = True
        if getattr(args, 'text_encoder_model', None):
            # Single gate: swap the from-scratch text prelude/coda for a pretrained LLM body +
            # translators + the LLM's LM head. None (default) leaves the model byte-identical.
            # The 9 control tokens live at ids >= the LLM's native vocab (the special_embed/head
            # extension), so special_token_base MUST equal the LLM's vocab_size and n_special_tokens
            # is fixed at 9 (the collator always injects all 9). eos becomes the LLM's native eos.
            from transformers import AutoConfig
            _llm_cfg = AutoConfig.from_pretrained(args.text_encoder_model)
            native_vocab = int(_llm_cfg.vocab_size)
            native_eos = _llm_cfg.eos_token_id
            if native_eos is None:
                from transformers import AutoTokenizer
                native_eos = AutoTokenizer.from_pretrained(args.text_encoder_model).eos_token_id
            config.special_token_base = native_vocab
            config.eos_token_id = int(native_eos)
            config.__post_init__()  # re-derive interleaver placeholder ids for the new base
            config.text_encoder = {
                "model": args.text_encoder_model,
                "freeze": not getattr(args, 'text_encoder_unfreeze', False),
                "translator_hidden_mult": getattr(args, 'text_encoder_translator_mult', 2.0),
                # Off by default -> the coda shares the LLM's tied head (the tested path for
                # every existing voice/image run). On -> coda owns a trunk-width readout.
                "trainable_head": getattr(args, 'text_encoder_trainable_head', False),
                # Opt-in only: the duration buckets add 32 rows to special_embed/special_head,
                # which is a MODEL SHAPE. A run that does not ask for them keeps 9.
                "n_special_tokens": (constants.N_SPECIAL_TOKENS_WITH_DURATION
                                     if getattr(args, 'voice_nar_duration_token', False)
                                     else constants.N_SPECIAL_TOKENS),
            }
        if getattr(args, 'voice_gen_query_mode', None):
            # Gen-query voice synthesis: creates voice_gen_queries + voice_coda_prev_proj
            # params (world_model.py __init__). Removes the AR crutch from the shared trunk.
            config.voice_gen_query_mode = args.voice_gen_query_mode
            if getattr(args, 'voice_gen_query_no_coda_prev', False):
                # Drop the coda's previous-centroid signal -> truly crutch-free (coda sees only
                # the text-driven trunk output). Pair with --voice_coda_type mlp.
                config.voice_gen_query_coda_prev = False
        if getattr(args, 'voice_coda_type', None):
            # Voice coda body: "mlp" swaps the transformer stack for an attention-free
            # position-wise FFN (no cross-position mixing, no KV cache), so the coda can't
            # neighbor-extrapolate emitted history. Pair with --voice_gen_query_mode.
            config.voice_coda_config.coda_type = args.voice_coda_type
            if getattr(args, 'voice_coda_mlp_ratio', None) is not None:
                config.voice_coda_config.mlp_ratio = args.voice_coda_mlp_ratio
        if getattr(args, 'mean_thinking_steps', None) is not None:
            # Must be set PRE-construction, unlike --backprop_depth: besides driving the
            # Poisson sampler, it is l_eff for the depth-scaled residual init
            # (recurrent.py:161-173), so patching it onto a built model would leave the
            # weights initialized for the old depth.
            config.recurrent_block_config.mean_thinking_steps = args.mean_thinking_steps
        if getattr(args, 'image_contrastive_queue_size', None) is not None:
            # Z-Image adapter InfoNCE memory queue. Pre-construction: it allocates a
            # (queue_size, seq_dim) buffer in the adapter's __init__.
            if not isinstance(config.image_coda_config, ZImageAdapterConfig):
                raise SystemExit("--image_contrastive_queue_size requires a Z-Image adapter config "
                                 f"(got {type(config.image_coda_config).__name__}).")
            config.image_coda_config.contrastive_queue_size = int(args.image_contrastive_queue_size)
        if getattr(args, 'image_flow_aux_mse_weight', None) is not None:
            # Weight of the point-head MSE that rides alongside the flow objective. Its gradient
            # runs seq_head -> cross_dec -> self_enc -> TRUNK, so this is not a pure diagnostic
            # knob: 0.0 removes a real training signal into the trunk, not just a readout.
            # Pre-construction because the adapter copies the value into an attribute in its
            # __init__ (zimage_adapter.py:169) -- patching the config afterwards would be inert.
            # `is not None`, NOT truthiness: 0.0 is the whole point of the flag.
            if not isinstance(config.image_coda_config, ZImageAdapterConfig):
                raise SystemExit("--image_flow_aux_mse_weight requires a Z-Image adapter config "
                                 f"(got {type(config.image_coda_config).__name__}).")
            config.image_coda_config.flow_aux_mse_weight = float(args.image_flow_aux_mse_weight)
        if getattr(args, 'voice_predict_f0', False):
            config.voice_coda_config.predict_f0 = True
        if getattr(args, 'voice_dedup', False):
            # Segment-rate path: the coda predicts a per-segment duration, and the dataset
            # must emit the deduped streams. Implies units (the segments ARE units) and F0
            # (the contour rides the segments), so require both rather than silently
            # half-configuring.
            if not getattr(args, 'voice_codebook_path', None):
                raise SystemExit("--voice_dedup requires --voice_codebook_path (segments are units).")
            config.voice_coda_config.predict_duration = True
            if not getattr(args, 'voice_predict_f0', False):
                config.voice_coda_config.predict_f0 = True
                print("[world] --voice_dedup implies F0: enabling predict_f0", flush=True)
        _btc = int(getattr(args, "bistream_text_chunk", 0) or 0)
        _bvc = int(getattr(args, "bistream_voice_chunk", 0) or 0)
        if bool(_btc) != bool(_bvc):
            raise SystemExit(
                "--bistream_text_chunk and --bistream_voice_chunk must be set together "
                f"(got {_btc} and {_bvc}). One without the other silently disables bistream.")
        if _btc and not getattr(args, 'voice_codebook_path', None):
            raise SystemExit(
                "bistream requires --voice_codebook_path: fill_token is a UNIT id (K+1), so "
                "there is nothing to derive it from on the continuous path.")
        if _btc and getattr(args, 'voice_nar', False):
            raise SystemExit(
                "bistream is an AR layout and does not compose with --voice_nar. Masked-parallel "
                "decoding reveals frames in confidence order, which has no notion of 'the text so "
                "far', so the fill_token target it would learn is meaningless. See "
                "docs/plans/bistream-inner-monologue.md ('AR, not NAR').")
        codebook_path = getattr(args, 'voice_codebook_path', None)
        if codebook_path:
            from megatransformer.utils.codebook import load_codebook
            K = int(load_codebook(codebook_path).shape[0])
            # Sizes the coda's classifier output layer, so it must be set before the model
            # is built. Derived from the codebook itself rather than a flag: a mismatch
            # between K and the codebook would be silent and catastrophic. +1 for the EOV
            # terminal token (class index K): the coda classifies K content units plus one
            # end-of-voice token, which replaces the old stop head. The codebook stays K
            # entries -- EOV has no centroid; generation stops on it before any lookup.
            # +1 more for bistream's fill_token (id K+1), which terminates a CHUNK as EOV
            # terminates the UTTERANCE. Two separate tokens by design: "send me more text"
            # and "I am done speaking" are different events, and collapsing them would make
            # the end of every chunk look like the end of the utterance.
            _bi = int(getattr(args, "bistream_text_chunk", 0) or 0) > 0
            config.voice_coda_config.unit_vocab_size = K + (2 if _bi else 1)
            print(f"[world] discrete voice units ON: K={K} "
                  f"(+1 EOV{' +1 fill' if _bi else ''}) from {codebook_path}", flush=True)
        voice_feature_channels = _voice_feature_channels(args)
        if voice_feature_channels is not None:
            # The prelude projects features -> d_model and the coda predicts d_model -> features,
            # so both ends must agree with the cached tensors' channel width. The config default
            # (128) tracks neither SIVE (256) nor ContentVec (256/768); a mismatch surfaces as a
            # matmul shape error in the prelude's projection rather than a useful message.
            config.voice_prelude_config.feature_channels = voice_feature_channels
            config.voice_coda_config.feature_channels = voice_feature_channels
        if getattr(args, 'voice_stochastic_output', False):
            config.voice_coda_config.stochastic_output = True
            if getattr(args, 'voice_logvar_init', None) is not None:
                config.voice_coda_config.logvar_init = args.voice_logvar_init
            if getattr(args, 'voice_logvar_clamp_min', None) is not None:
                config.voice_coda_config.logvar_clamp_min = args.voice_logvar_clamp_min
            if getattr(args, 'voice_logvar_clamp_max', None) is not None:
                config.voice_coda_config.logvar_clamp_max = args.voice_logvar_clamp_max
        if getattr(args, 'max_seq_len', None) is not None:
            # Size causal masks for the full interleaved sequence length:
            # text max_seq_len + worst-case media tokens (voice ~210, image ~64)
            # + boundary tokens. Use a generous buffer so new modalities or
            # longer media don't immediately trip the mask-too-small error.
            mpe = args.max_seq_len + 512
            # Top-level field (text prelude also has one; keep in sync)
            config.text_prelude_config.max_position_embeddings = mpe
            # Walk every nested MegaTransformerBlockConfig and bump its buffer
            def _bump_mpe(obj):
                if isinstance(obj, MegaTransformerBlockConfig):
                    obj.max_position_embeddings = mpe
                    return
                if hasattr(obj, '__dict__'):
                    for v in vars(obj).values():
                        _bump_mpe(v)
            _bump_mpe(config)
        for k, v in overrides.items():
            setattr(config, k, v)
        WORLD_MODEL_CONFIGS[args.config + "_cli_override"] = config
        args_config = args.config + "_cli_override"
    else:
        args_config = args.config

    model = model_loading_utils.load_model(
        MegaTransformerWorldModel,
        args_config,
        checkpoint_path=args.resume_from_checkpoint,
        overrides=overrides,
        device=device,
    )

    # Apply CLI overrides to nested image configs (can't go through top-level overrides)
    if model.image_feature_extractor is not None:
        if getattr(args, 'no_image_input_norm', False):
            model.image_feature_extractor.input_norm = None
        elif getattr(args, 'image_input_norm_type', None) is not None:
            import torch.nn as nn_mod
            lat_ch = model.config.image_prelude_config.image_config.latent_channels
            if args.image_input_norm_type == "instancenorm":
                model.image_feature_extractor.input_norm = nn_mod.InstanceNorm2d(lat_ch, affine=False)
            elif args.image_input_norm_type == "layernorm":
                model.image_feature_extractor.input_norm = nn_mod.LayerNorm(lat_ch, elementwise_affine=False)
    if getattr(args, 'no_image_output_denorm', False) and model.image_generator is not None:
        model.image_generator.use_output_denorm = False
    if getattr(args, 'voice_codebook_path', None):
        from megatransformer.utils.codebook import load_codebook
        model.set_voice_codebook(load_codebook(args.voice_codebook_path))
    if getattr(args, 'backprop_depth', None) is not None:
        model.recurrent_block.backprop_depth = args.backprop_depth
    if getattr(args, 'block_init_gain', None) is not None:
        gain = args.block_init_gain
        for block in model.recurrent_block.recurrent_blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight, gain=gain)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    if getattr(args, 'projection_init_gain', None) is not None and model.recurrent_block.projection is not None:
        nn.init.xavier_uniform_(model.recurrent_block.projection.weight, gain=args.projection_init_gain)
        if model.recurrent_block.projection.bias is not None:
            nn.init.zeros_(model.recurrent_block.projection.bias)

    # Diffusion bridge decoder: optional latent_scale override.
    # Only meaningful when image_generator is a DiffusionBridgeImageDecoder
    # — silently ignored for the direct ImageDecoder.
    from megatransformer.model.image.diffusion_decoder import DiffusionBridgeImageDecoder
    if isinstance(model.image_generator, DiffusionBridgeImageDecoder):
        # min_snr_gamma runtime override. Setting to 0 disables min-SNR
        # weighting entirely (config field becomes None).
        mgs_override = getattr(args, 'min_snr_gamma', None)
        if mgs_override is not None:
            if mgs_override <= 0:
                model.image_generator.config.min_snr_gamma = None
                print("[load_model] min-SNR weighting disabled (--min_snr_gamma <= 0)")
            else:
                model.image_generator.config.min_snr_gamma = float(mgs_override)
                print(f"[load_model] min_snr_gamma overridden to {float(mgs_override)}")

        # Structural override: drop the Q-Former bridge so the DiT cross-attends the
        # trunk's image-position outputs directly. Nulling the already-built module
        # here (before create_optimizer runs) keeps its params out of the optimizer
        # and the checkpoint. Fresh-run only — the param set differs from a bridged
        # checkpoint, so this is not resume-compatible with a bridged run.
        if getattr(args, 'image_no_bridge', False):
            model.image_generator.config.use_bridge = False
            model.image_generator.bridge = None
            print("[load_model] image DiT bridge DISABLED (--image_no_bridge): "
                  "trunk image-position outputs feed the DiT cross-attention directly")

        new_scale = None
        if getattr(args, 'image_latent_channel_scales', None) is not None:
            scales = [float(s) for s in args.image_latent_channel_scales.split(',')]
            if len(scales) != model.image_generator.config.latent_channels:
                raise ValueError(
                    f"--image_latent_channel_scales has {len(scales)} values, but "
                    f"latent_channels={model.image_generator.config.latent_channels}"
                )
            new_scale = torch.tensor(scales, dtype=torch.float)
        elif getattr(args, 'image_latent_scale', None) is not None:
            c = model.image_generator.config.latent_channels
            new_scale = torch.full((c,), float(args.image_latent_scale))
        if new_scale is not None:
            # Reshape (C,) → (1, C, 1, 1) and copy into the existing buffer so
            # the device/dtype stay consistent.
            buf = model.image_generator.latent_scale
            new_scale = new_scale.view(1, -1, 1, 1).to(device=buf.device, dtype=buf.dtype)
            buf.copy_(new_scale)

    return model


def _build_distill_teacher(args):
    """Frozen CosyVoice 2 speech LM for KL distillation, or None when disabled.

    A load failure is downgraded to a warning so a bad path can't kill a training run --
    but it prints loudly, because silently training without the teacher while believing
    otherwise would invalidate the run.
    """
    model_dir = getattr(args, "voice_cosyvoice2_distill_model_dir", None)
    if not model_dir or getattr(args, "voice_distill_weight", 0.0) <= 0:
        return None
    try:
        from megatransformer.model.voice.cosyvoice2_teacher import CosyVoice2Teacher
        dev = getattr(args, "voice_distill_device", None) or "cuda"
        dtype = torch.bfloat16 if getattr(args, "voice_distill_bf16", True) else torch.float32
        t = CosyVoice2Teacher.from_pretrained(
            model_dir,
            runtime_dir=getattr(args, "voice_cosyvoice2_runtime_dir", None),
            device=dev, dtype=dtype,
        )
        n = sum(p.numel() for p in t.parameters()) / 1e6
        print(f"[distill] CosyVoice 2 teacher loaded ({n:.1f}M, {dtype}, {dev}); "
              f"weight={args.voice_distill_weight} T={getattr(args, 'voice_distill_temperature', 1.0)}",
              flush=True)
        return t
    except Exception as e:
        print(f"WARNING: failed to load the distillation teacher ({type(e).__name__}: {e}) "
              f"-- TRAINING WILL PROCEED WITHOUT DISTILLATION.", flush=True)
        return None


def create_trainer(
    args,
    model,
    optimizer,
    training_args,
    data_collator,
    train_dataset,
    eval_dataset,
):
    include_modes = [m.strip() for m in args.include_modes.split(",")]

    return WorldModelTrainer(
        model=model,
        optimizers=(optimizer, None),
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        cmdline=args.cmdline,
        git_commit_hash=args.commit_hash or "",
        step_offset=args.start_step,
        text_loss_weight=args.text_loss_weight,
        audio_latent_loss_weight=args.audio_latent_loss_weight,
        voice_latent_loss_weight=args.voice_latent_loss_weight,
        image_latent_loss_weight=getattr(args, 'image_latent_loss_weight', 1.0),
        image_clip_loss_weight=getattr(args, 'image_clip_loss_weight', 1.0),
        image_target_device=getattr(args, 'image_target_device', None),
        image_target_bf16=getattr(args, 'image_target_bf16', False),
        image_whiten_stats_path=getattr(args, 'image_whiten_stats_path', None),
        image_contrastive_ramp_steps=getattr(args, 'image_contrastive_ramp_steps', 2000),
        audio_var_loss_weight=getattr(args, 'audio_var_loss_weight', 1.0),
        voice_var_loss_weight=getattr(args, 'voice_var_loss_weight', 1.0),
        image_var_loss_weight=getattr(args, 'image_var_loss_weight', 1.0),
        audio_var_barrier_weight=getattr(args, 'audio_var_barrier_weight', 0.0),
        voice_var_barrier_weight=getattr(args, 'voice_var_barrier_weight', 0.0),
        image_var_barrier_weight=getattr(args, 'image_var_barrier_weight', 0.0),
        audio_stop_loss_weight=getattr(args, 'audio_stop_loss_weight', 1.0),
        voice_stop_loss_weight=getattr(args, 'voice_stop_loss_weight', 1.0),
        voice_beta_nll=getattr(args, 'voice_beta_nll', 0.5),
        voice_f0_loss_weight=getattr(args, 'voice_f0_loss_weight', 1.0),
        voice_dedup=getattr(args, 'voice_dedup', False),
        voice_duration_loss_weight=getattr(args, 'voice_duration_loss_weight', 1.0),
        voice_scheduled_sampling_prob=getattr(args, 'voice_scheduled_sampling_prob', 0.0),
        voice_scheduled_sampling_ramp_steps=getattr(args, 'voice_scheduled_sampling_ramp_steps', 10000),
        voice_scheduled_sampling_start_step=getattr(args, 'voice_scheduled_sampling_start_step', 0),
        voice_ar_attn_mask_steps=getattr(args, 'voice_ar_attn_mask_steps', 0),
        voice_ar_attn_ramp_steps=getattr(args, 'voice_ar_attn_ramp_steps', 0),
        voice_ar_attn_floor=getattr(args, 'voice_ar_attn_floor', 0.0),
        voice_ar_attn_cap=getattr(args, 'voice_ar_attn_cap', 1.0),
        voice_ar_attn_ramp_power=getattr(args, 'voice_ar_attn_ramp_power', 1.0),
        voice_prenet_dropout=getattr(args, 'voice_prenet_dropout', 0.0),
        voice_cfg_text_dropout_prob=getattr(args, 'voice_cfg_text_dropout_prob', 0.0),
        voice_early_text_weight_alpha=getattr(args, 'voice_early_text_weight_alpha', 1.0),
        voice_early_text_weight_frames=getattr(args, 'voice_early_text_weight_frames', 0),
        voice_prenet_dropout_ramp_steps=getattr(args, 'voice_prenet_dropout_ramp_steps', 0),
        voice_prenet_dropout_start_step=getattr(args, 'voice_prenet_dropout_start_step', 0),
        voice_distill_teacher=_build_distill_teacher(args),
        voice_distill_weight=getattr(args, 'voice_distill_weight', 0.0),
        voice_distill_temperature=getattr(args, 'voice_distill_temperature', 1.0),
        voice_onpolicy_distill=getattr(args, 'voice_onpolicy_distill', False),
        voice_onpolicy_temperature=getattr(args, 'voice_onpolicy_temperature', 0.8),
        include_text="text" in include_modes,
        include_audio="audio" in include_modes,
        include_voice="voice" in include_modes,
        include_image="image" in include_modes,
        precomputed_latents=args.precomputed_latents,
        text_label_smoothing=args.text_label_smoothing,
        lr_dit=getattr(args, 'lr_dit', None),
        lr_flow=getattr(args, 'lr_flow', None),
        differential_lr_schedule=getattr(args, 'differential_lr_schedule', False),
        lr_dit_schedule=getattr(args, 'lr_dit_schedule', 'constant'),
        lr_trunk_schedule=getattr(args, 'lr_trunk_schedule', 'cosine'),
        lr_min_ratio=getattr(args, 'lr_min_ratio', 0.1),
        lr_wsd_decay_frac=getattr(args, 'lr_wsd_decay_frac', 0.2),
        mask_text_loss_in_synthesis=getattr(args, 'mask_text_loss_in_synthesis', False),
        emit_duration_token=getattr(args, 'voice_nar_duration_token', False),
        bistream_text_loss=getattr(args, 'bistream_text_loss', False),
        unmask_eos_in_synthesis=getattr(args, 'unmask_eos_in_synthesis', False),
        voice_nar=getattr(args, 'voice_nar', False),
        voice_nar_mask_schedule=getattr(args, 'voice_nar_mask_schedule', 'cosine'),
        voice_nar_mask_ratio_min=getattr(args, 'voice_nar_mask_ratio_min', 0.85),
        voice_nar_mask_anneal_steps=getattr(args, 'voice_nar_mask_anneal_steps', 0),
        voice_nar_mask_ratio_floor=getattr(args, 'voice_nar_mask_ratio_floor', 0.25),
        voice_nar_trunk_text_only=getattr(args, 'voice_nar_trunk_text_only', False),
        shard_aware_sampler=getattr(args, 'shard_aware_sampler', True),
        bucket_by_length=getattr(args, 'bucket_by_length', False),
        bucket_mega_factor=getattr(args, 'bucket_mega_factor', 25),
        voice_curriculum_max_frames=getattr(args, 'voice_curriculum_max_frames', 0),
    )


def add_cli_args(subparsers):
    sub_parser = subparsers.add_parser(
        "world", help="Train the multimodal world model (text + audio + image)"
    )

    # Data directories. For each modality, you can either:
    #   (a) pass --<mod>_cache_dir as a base (the code appends _train / _val), or
    #   (b) pass --<mod>_train_cache_dir and --<mod>_val_cache_dir explicitly.
    # Explicit per-split dirs take precedence and avoid symlink tricks when
    # train and val shards live at unrelated paths (e.g. tmpfs for train,
    # disk for val).
    sub_parser.add_argument("--text_cache_dir", type=str, default=None,
                            help="Base dir for text shards (code appends _train/_val)")
    sub_parser.add_argument("--text_train_cache_dir", type=str, default=None,
                            help="Explicit text train shard dir (overrides --text_cache_dir)")
    sub_parser.add_argument("--text_val_cache_dir", type=str, default=None,
                            help="Explicit text val shard dir (overrides --text_cache_dir)")
    sub_parser.add_argument("--audio_cache_dir", type=str, default=None,
                            help="Base dir for audio shards (code appends _train/_val)")
    sub_parser.add_argument("--audio_train_cache_dir", type=str, default=None,
                            help="Explicit audio train shard dir (overrides --audio_cache_dir)")
    sub_parser.add_argument("--audio_val_cache_dir", type=str, default=None,
                            help="Explicit audio val shard dir (overrides --audio_cache_dir)")
    sub_parser.add_argument("--voice_cache_dir", type=str, default=None,
                            help="Base dir for voice shards (code appends _train/_val)")
    sub_parser.add_argument("--voice_train_cache_dir", type=str, default=None,
                            help="Explicit voice train shard dir (overrides --voice_cache_dir)")
    sub_parser.add_argument("--voice_val_cache_dir", type=str, default=None,
                            help="Explicit voice val shard dir (overrides --voice_cache_dir)")
    sub_parser.add_argument("--voice_synthesis_cache_dir", type=str, default=None,
                            help="Optional override corpus for the voice SYNTHESIS (text->voice) "
                                 "direction only (code appends _train/_val). Unset -> synthesis "
                                 "shares --voice_cache_dir. Point this at a clean subset while "
                                 "--voice_cache_dir holds the clean+noisy transcription superset.")
    sub_parser.add_argument("--voice_synthesis_train_cache_dir", type=str, default=None,
                            help="Explicit voice synthesis train shard dir (overrides --voice_synthesis_cache_dir)")
    sub_parser.add_argument("--voice_synthesis_val_cache_dir", type=str, default=None,
                            help="Explicit voice synthesis val shard dir (overrides --voice_synthesis_cache_dir)")
    sub_parser.add_argument("--image_cache_dir", type=str, default=None,
                            help="Base dir for image shards (code appends _train/_val)")
    sub_parser.add_argument("--image_train_cache_dir", type=str, default=None,
                            help="Explicit image train shard dir (overrides --image_cache_dir)")
    sub_parser.add_argument("--image_val_cache_dir", type=str, default=None,
                            help="Explicit image val shard dir (overrides --image_cache_dir)")
    sub_parser.add_argument("--cache_dir", type=str, default=None,
                            help="Unused for world model — use per-modality cache dirs instead")

    # Modality loss weights (applied AFTER per-modality whitening — these are
    # honest emphasis multipliers, not scale-correction hacks).
    sub_parser.add_argument("--text_loss_weight", type=float, default=1.0,
                            help="Emphasis multiplier for text loss (whitened cross-entropy)")
    sub_parser.add_argument("--audio_latent_loss_weight", type=float, default=1.0,
                            help="Emphasis multiplier for audio loss (whitened L1+MSE + var-match)")
    sub_parser.add_argument("--voice_latent_loss_weight", type=float, default=1.0,
                            help="Emphasis multiplier for voice loss (whitened L1+MSE + var-match)")
    sub_parser.add_argument("--image_latent_loss_weight", type=float, default=1.0,
                            help="Emphasis multiplier for image loss (whitened L1+MSE + var-match)")
    sub_parser.add_argument("--image_clip_loss_weight", type=float, default=1.0,
                            help="Emphasis multiplier for the SDXL/Z-Image adapter conditioning "
                                 "regression loss (used when image_coda_config is an adapter config)")
    sub_parser.add_argument("--image_target_device", type=str, default=None,
                            help="Z-Image adapter: device for the Qwen3-4B target encoder, e.g. "
                                 "'cuda:1'. Frees the training GPU's memory+compute. Launch with the "
                                 "training GPU FIRST in CUDA_VISIBLE_DEVICES (it becomes cuda:0) and "
                                 "the target GPU second (cuda:1); the Trainer is forced single-GPU so "
                                 "it won't DataParallel-wrap the model.")
    sub_parser.add_argument("--image_target_bf16", action="store_true",
                            help="Z-Image adapter: load the Qwen3-4B target encoder in bf16 (~8GB, "
                                 "faster than 4-bit) instead of 4-bit. Use with a dedicated "
                                 "--image_target_device that has headroom.")
    sub_parser.add_argument("--image_whiten_stats_path", type=str, default=None,
                            help="Z-Image adapter Tier-0 whitened MSE: path to a .pt with per-dim "
                                 "{mean,std} of the Qwen3 target (from compute_qwen_whiten_stats.py). "
                                 "Injected into the adapter's whitening buffers. Requires "
                                 "--config small_sum_zimage_whiten.")
    sub_parser.add_argument("--image_contrastive_ramp_steps", type=int, default=2000,
                            help="Z-Image adapter Tier-1: ramp the InfoNCE weight 0->config max over "
                                 "this many steps from the phase start (use small_sum_zimage_whiten_t1, "
                                 "warm-started via --resume_from_checkpoint <ckpt> --fresh_schedule).")
    sub_parser.add_argument("--lr_flow", type=float, default=None,
                            help="Override LR for the T3 flow head (image_generator.flow_head.*) "
                                 "separately from --lr_dit. The flow head is FRESH while the rest of "
                                 "the adapter is warm-started, so it wants a from-scratch rate "
                                 "(~1e-4) while the Q-Former beside it stays slow; both live under "
                                 "image_generator.*, so --lr_dit alone cannot separate them. "
                                 "Defaults to --lr_dit. Follows the DiT LR schedule.")
    sub_parser.add_argument("--image_flow_aux_mse_weight", type=float, default=None,
                            help="Z-Image adapter: override flow_aux_mse_weight (config default 0.1). "
                                 "Set 0.0 to train a PURE sampler. seq_head then gets ZERO gradient and "
                                 "stays at its init, so alpha/R^2 become a frozen random readout of "
                                 "a still-moving representation -- not comparable to the regression "
                                 "runs any more. NOTE this is a "
                                 "different ablation from flow_aux_mse_detach, which keeps the term "
                                 "but stops its gradient reaching the trunk. Ignored unless the coda "
                                 "is a Z-Image adapter config.")
    sub_parser.add_argument("--image_contrastive_queue_size", type=int, default=None,
                            help="Z-Image adapter Tier-1: size of the MoCo-style memory queue of past "
                                 "Qwen3 targets used as extra InfoNCE negatives (overrides the config; "
                                 "0 = in-batch only). In-batch InfoNCE at batch_size 8 is an 8-way task "
                                 "that saturates near 0 and stops producing gradient; a 4096 queue makes "
                                 "it 4104-way. Non-persistent buffer, so checkpoints are unaffected. "
                                 "Preset small_sum_zimage_whiten_t1q already sets 4096.")

    # Variance-matching auxiliary loss weights (per modality). The aux loss
    # penalizes (std(preds)/std(labels) - 1), preventing collapse to a constant
    # when content matching is hard. Set to 0 to disable.
    sub_parser.add_argument("--audio_var_loss_weight", type=float, default=1.0,
                            help="Weight for the audio variance-matching aux loss")
    sub_parser.add_argument("--voice_var_loss_weight", type=float, default=1.0,
                            help="Weight for the voice variance-matching aux loss")
    sub_parser.add_argument("--image_var_loss_weight", type=float, default=1.0,
                            help="Weight for the image variance-matching aux loss")

    # Variance barrier loss weights (per modality). The barrier is
    # `-log(std(preds)/std(labels))` clamped to ≥ 0 — its gradient explodes
    # as the model approaches collapse, providing strong anti-collapse pressure
    # that the bounded var_loss above cannot. Default 0 (disabled); try 0.5–1.0
    # to enable.
    sub_parser.add_argument("--audio_var_barrier_weight", type=float, default=0.0,
                            help="Weight for the audio variance barrier loss")
    sub_parser.add_argument("--voice_var_barrier_weight", type=float, default=0.0,
                            help="Weight for the voice variance barrier loss")
    sub_parser.add_argument("--image_var_barrier_weight", type=float, default=0.0,
                            help="Weight for the image variance barrier loss")

    # Stop loss weights for voice/audio autoregressive stop prediction.
    # Whitened by log(2), so 1.0 = equal weight to other whitened losses.
    sub_parser.add_argument("--audio_stop_loss_weight", type=float, default=1.0,
                            help="Weight for the audio stop prediction loss")
    sub_parser.add_argument("--voice_stop_loss_weight", type=float, default=1.0,
                            help="Weight for the voice stop prediction loss")

    # Stochastic (heteroscedastic Gaussian) voice coda. When enabled the voice
    # coda predicts (mu, log_var) per SIVE frame and trains with beta-NLL instead
    # of the whitened recon + variance-matching terms; inference samples from the
    # predicted Gaussian with a temperature (deterministic at temperature 0).
    sub_parser.add_argument("--voice_stochastic_output", action="store_true",
                            help="Voice coda predicts a per-frame diagonal Gaussian (mu, log_var) and trains with beta-NLL")
    sub_parser.add_argument("--voice_beta_nll", type=float, default=0.5,
                            help="beta for the voice beta-NLL (0=plain NLL, 1=~MSE; Seitzer default 0.5)")
    sub_parser.add_argument("--voice_logvar_init", type=float, default=0.0,
                            help="Initial (homoscedastic) log-variance the log-var head's bias inits to")
    sub_parser.add_argument("--voice_logvar_clamp_min", type=float, default=-8.0,
                            help="Lower clamp on predicted voice log-variance")
    sub_parser.add_argument("--voice_logvar_clamp_max", type=float, default=4.0,
                            help="Upper clamp on predicted voice log-variance")

    # Diffusion bridge image decoder: latent scaling (only used in diffusion mode).
    # Either pass --image_latent_scale (global SD1.x-style scalar) OR
    # --image_latent_channel_scales (per-channel SD3-style, comma-separated list
    # of length latent_channels). If both are passed, channel scales take priority.
    # Silently ignored when the image generator is the direct ImageDecoder.
    sub_parser.add_argument("--image_latent_scale", type=float, default=None,
                            help="Global scalar applied to image latents in the diffusion bridge "
                                 "decoder (multiplied at training input, divided at sampling output). "
                                 "Use ~1/std(latents). For LiteVAE, ~0.896.")
    sub_parser.add_argument("--image_latent_channel_scales", type=str, default=None,
                            help="Per-channel image latent scales for the diffusion bridge decoder, "
                                 "comma-separated, length must equal latent_channels. SD3-style "
                                 "normalization. Overrides --image_latent_scale if both are set.")

    # Text loss
    sub_parser.add_argument("--text_label_smoothing", type=float, default=0.0,
                            help="Label smoothing for text cross-entropy loss")
    sub_parser.add_argument("--mask_text_loss_in_synthesis", action="store_true", default=False,
                            help="Skip text cross-entropy loss in image_synthesis and voice_synthesis batches "
                                 "where text is conditioning rather than target. Standard practice in Flamingo/GIT/BLIP. "
                                 "Default off for backward compat with existing runs; enable for new from-scratch runs. "
                                 "Text prelude still gets gradient on synthesis batches via the recurrent block + media losses.")

    # Shard-aware sampler (default on). Pass --no_shard_aware_sampler to
    # disable — required when resuming a checkpoint from a run trained
    # with the prior uniform-shuffle sampler, since enabling it changes
    # the per-epoch index order and breaks HF Trainer's batch-skip resume.
    sub_parser.add_argument("--no_shard_aware_sampler", action="store_false",
                            dest="shard_aware_sampler", default=True,
                            help="Disable shard-aware modality sampling (reverts to legacy uniform shuffle). "
                                 "Required for resuming checkpoints from pre-shard-aware runs.")

    # Per-module LR overrides
    sub_parser.add_argument("--lr_dit", type=float, default=None,
                            help="Override LR for model.image_generator (DiT) parameters. "
                                 "If set, DiT params get this LR while everything else uses --learning_rate. "
                                 "Use to tame the DiT when it's the destabilizing module. "
                                 "Ignored when --use_muon is set (Muon has its own LR split).")

    # Differential LR *schedule* for the DiT vs the trunk (requires --lr_dit).
    # Without this, HF's single scheduler applies the SAME decay curve to every
    # group, so a cosine --lr_scheduler_type silently decays the DiT too.
    sub_parser.add_argument("--differential_lr_schedule", action="store_true", default=False,
                            help="Give the DiT param group a different LR SCHEDULE from the trunk. "
                                 "Requires --lr_dit (the 4-group split). DiT follows --lr_dit_schedule, "
                                 "everything else follows --lr_trunk_schedule. No-op under --use_muon.")
    sub_parser.add_argument("--lr_dit_schedule", type=str, default="constant",
                            choices=["constant", "cosine", "wsd"],
                            help="LR schedule for the DiT group when --differential_lr_schedule is set. "
                                 "Default 'constant' — diffusion/flow-matching heads (DiT/SD3/Flux) train "
                                 "best at a flat LR. All schedules share the --warmup_steps/--warmup_ratio warmup.")
    sub_parser.add_argument("--lr_trunk_schedule", type=str, default="cosine",
                            choices=["cosine", "wsd", "constant"],
                            help="LR schedule for the trunk/main group when --differential_lr_schedule is set. "
                                 "'wsd' (warmup-stable-decay) holds a flat LR then anneals only in the final "
                                 "--lr_wsd_decay_frac of steps — gives the DiT a stable conditioning target for "
                                 "most of the run, then one short trunk anneal.")
    sub_parser.add_argument("--lr_min_ratio", type=float, default=0.1,
                            help="LR floor as a fraction of base LR for cosine/wsd decay (default 0.1).")
    sub_parser.add_argument("--lr_wsd_decay_frac", type=float, default=0.2,
                            help="Fraction of total steps spent in the WSD decay tail (default 0.2).")

    # DiT loss-weighting override (only applies to DiffusionBridgeImageDecoder)
    sub_parser.add_argument("--min_snr_gamma", type=float, default=None,
                            help="Runtime override for DiT min-SNR gamma. Lowering from the config "
                                 "default (typically 5.0) reduces gradient amplification at the hardest "
                                 "timesteps (t near 1), trading a small amount of hard-timestep focus "
                                 "for stability. Set <= 0 to disable min-SNR weighting entirely.")

    # DiT conditioning-path ablation (only applies to DiffusionBridgeImageDecoder)
    sub_parser.add_argument("--image_no_bridge", action="store_true", default=False,
                            help="Remove the Q-Former bridge so the DiT cross-attends the recurrent "
                                 "trunk's image-position outputs directly (shorter/wider conditioning "
                                 "path). Requires the DiT d_model to equal the trunk d_model (small_sum "
                                 "matches by design). Fresh-run only — not resume-compatible with a "
                                 "bridged checkpoint (different param set).")

    # Weight tying
    sub_parser.add_argument("--tie_word_embeddings", action="store_true", default=False,
                            help="Tie LM head weights to input embedding matrix")

    # Image normalization overrides
    sub_parser.add_argument("--no_image_input_norm", action="store_true", default=False,
                            help="Disable normalization on image latents before prelude")
    sub_parser.add_argument("--image_input_norm_type", type=str, default=None, choices=["layernorm", "instancenorm"],
                            help="Override image input normalization type (default: use config)")
    sub_parser.add_argument("--no_image_output_denorm", action="store_true", default=False,
                            help="Disable learnable scale/bias after image coda")

    # Precomputed latents flag
    sub_parser.add_argument("--precomputed_latents", action="store_true", default=True,
                            help="Whether media inputs are precomputed VAE latents (default: True)")
    sub_parser.add_argument("--no_precomputed_latents", action="store_false", dest="precomputed_latents",
                            help="Media inputs are raw (mel specs / images), not VAE latents")

    # Dataset limiting (for overfitting experiments)
    sub_parser.add_argument("--data_fraction", type=float, default=1.0,
                            help="Train on a seeded RANDOM fraction of the corpus (1.0 = all). "
                                 "For data-scaling experiments: at matched steps, a smaller "
                                 "fraction sees each sample proportionally more often, so the "
                                 "comparison isolates DATA QUANTITY at fixed compute. Unlike "
                                 "--max_samples (a prefix cap, i.e. the first N in shard order "
                                 "= a biased speaker subset), this preserves the speaker "
                                 "distribution in expectation.")
    sub_parser.add_argument("--data_subset_seed", type=int, default=0,
                            help="Seed for --data_fraction. Vary it to check that a slope is not "
                                 "an artifact of one particular subset draw.")
    sub_parser.add_argument("--max_samples", type=int, default=None,
                            help="Cap dataset size to N samples (for overfitting/memorization experiments)")
    sub_parser.add_argument("--use_memorization_dataset", action="store_true", default=False,
                            help="Preload all samples into RAM (fast, no shard I/O). Requires --max_samples.")

    # Generation query mode
    sub_parser.add_argument("--gen_query_mode", type=str, default=None,
                            choices=["learned", "positional_only"],
                            help="Generation query mode: 'learned' (default) or 'positional_only' (frozen sinusoidal PE only)")

    # Image generation query count (decoupled from prelude patch count)
    sub_parser.add_argument("--n_image_gen_positions", type=int, default=None,
                            help="Number of image gen query positions for synthesis. Must be a perfect "
                                 "square (e.g. 64, 144, 256). Default: use the prelude's patch count.")

    # Recurrent block overrides
    sub_parser.add_argument("--mean_thinking_steps", type=int, default=None,
                            help="Mean recurrent iterations per forward (Poisson log-normal). "
                                 "Sets effective depth = n_recurrent_blocks x this, and is also "
                                 "the l_eff for depth-scaled residual init — so it changes "
                                 "initialization, and runs at different values are not "
                                 "checkpoint-compatible. Dominates step cost: ~2/3 of compute is "
                                 "the no-grad iterations. Default: config value (32).")
    sub_parser.add_argument("--backprop_depth", type=int, default=None,
                            help="Override truncated BPTT depth (default: use config, typically 8)")
    sub_parser.add_argument("--block_init_gain", type=float, default=None,
                            help="Override recurrent block xavier init gain (default: 0.02)")
    sub_parser.add_argument("--projection_init_gain", type=float, default=None,
                            help="Override projection xavier init gain (default: use config, typically 1.0)")
    sub_parser.add_argument("--use_mrope", action="store_true", default=False,
                            help="M-RoPE in the recurrent trunk: split the rotary dims into a "
                                 "GLOBAL axis (increasing across the whole interleaved sequence, "
                                 "so multiple media examples stay distinguishable) and a LOCAL "
                                 "axis (index within the current same-modality segment, media "
                                 "scaled by 1/--mrope_voice_rate). Puts an aligned (text token, "
                                 "voice frame) pair at relative distance ~0 instead of "
                                 "L_text + 0.83*t. Same RoPE, different position integers.")
    sub_parser.add_argument("--mrope_voice_rate", type=float, default=6.0,
                            help="Nominal voice frames per text token for M-RoPE's local clock "
                                 "(25Hz speech vs SmolLM2 tokens is ~6). Only the ratio matters.")
    sub_parser.add_argument("--mrope_scale_side", type=str, default="voice", choices=["voice", "text"],
                            help="Which stream absorbs the frame rate on M-RoPE's local axis. "
                                 "'voice' (default) = voice t/rate, which leaves the 250-position "
                                 "generated stream at FRACTIONAL sub-unit spacing — a regime RoPE "
                                 "is essentially never used in. 'text' = text j*rate, same "
                                 "alignment property with both streams integer-spaced.")
    sub_parser.add_argument("--share_block_weights", action="store_true", default=False,
                            help="Share weights across all recurrent blocks (deeper 1-block)")
    sub_parser.add_argument("--iteration_norm", type=str, default=None,
                            choices=["none", "pre_projection", "post_projection"],
                            help="Override per-iteration normalization placement")

    # Max sequence length for text collator
    sub_parser.add_argument("--max_seq_len", type=int, default=2048,
                            help="Maximum token sequence length for text")

    # Visualization callback dependencies
    sub_parser.add_argument("--viz_suppress_media_tokens", dest="viz_suppress_media_tokens",
                            action="store_true", default=True,
                            help="EVAL RENDERS ONLY (changes nothing about training): ban the "
                                 "BO* boundary tokens and the three media placeholders from the "
                                 "text sampler. Every eval prompt already ENDS with the BO* it "
                                 "needs, so a sampled one is a hallucination -- measured at "
                                 "checkpoint-11076, the model terminated 12 of 14 voice blocks "
                                 "correctly with EOV and then sampled BOV again and spoke a "
                                 "second time. Default ON. Placeholders are banned too: sampling "
                                 "one is meaningless in any regime, since the interleaver "
                                 "consumes them at training time and generation has nothing to "
                                 "replace them with. EO* is NOT banned -- generate() forces it "
                                 "after a media block, which bypasses the sampler.")
    sub_parser.add_argument("--no_viz_suppress_media_tokens", dest="viz_suppress_media_tokens",
                            action="store_false",
                            help="Let the model sample BO*/placeholders at eval. Use this to "
                                 "MEASURE how often it starts a second media block unprompted "
                                 "-- a real termination signal that the ban hides along with the "
                                 "artifact. Renders stay correctly separated per utterance "
                                 "either way; only the hallucinated block's existence changes.")
    sub_parser.add_argument("--viz_voice_ras_win", type=int, default=10,
                            help="Repetition-aware sampling window for EVAL RENDERS only (10 = the "
                                 "CosyVoice 2 value; 0 disables). Does not affect training or any "
                                 "metric — it only changes what the logged audio sounds like. "
                                 "DEFAULT FLIPPED 0 -> 10 on 2026-09-10. At 0 the renders are the "
                                 "drawl arm: measured at ck90000 (n=64) the model runs 1.63x slow "
                                 "per word (16.5 frames/word vs the GT-ceiling's 10.1) and emits "
                                 "24%% fewer words, and in the like-for-like correct-length bucket "
                                 "RAS is worth +0.20 LCS-recall (0.555 -> 0.754 against a 0.862 "
                                 "ceiling). Every render judged by ear between the 2026-08-30 "
                                 "corpus switch and this date was the degraded arm — the model "
                                 "sounded far worse than it was, and the gap was an unset flag. "
                                 "Also: adj_repeat 0.103 -> 0.007, longest_run 56 -> 7. Pass 0 "
                                 "explicitly to recover the old behaviour.")
    sub_parser.add_argument("--viz_voice_ras_tau", type=float, default=0.1,
                            help="RAS repetition threshold for eval renders: RAS fires when the "
                                 "chosen unit occurred >= ras_win*tau times in the last ras_win. "
                                 "tau=0 disables RAS, which at T=0 collapses COMPLETELY (12/12 "
                                 "budget-capped, empty transcripts). Interacts with "
                                 "--viz_voice_ras_temperature: at greedy, tau 0.1/0.2/0.3 are "
                                 "flat and 0.1 wins on cap count; with the resample decoupled at "
                                 "1.0, tau 0.3 measured best. Do not sweep one without the other.")
    sub_parser.add_argument("--viz_trunk_iters", type=int, default=8,
                            help="Recurrent iteration CAP for eval RENDERS only (not for "
                                 "eval loss, which stays at full depth so historical curves "
                                 "remain comparable). DEFAULT 8 as of 2026-09-16: measured "
                                 "n=48 x 2 seeds, voice quality saturates at ~6 iterations "
                                 "(LCS 0.9179 at 18.6% of trunk compute) and mildly declines "
                                 "past it (0.9051 at 32). 8 sits just past saturation. "
                                 "0 = model default (mean_thinking_steps).")
    sub_parser.add_argument("--viz_voice_ras_temperature", type=float, default=None,
                            help="Temperature for the RAS RESAMPLE in eval renders, decoupled "
                                 "from the pick. None (default) inherits the pick's scaling, "
                                 "which is DISCONTINUOUS at zero: greedy resamples at an "
                                 "effective 1.0 while T=0.2 resamples 5x sharper than any pick, "
                                 "making 0.2-0.6 a trough where termination breaks. Set 1.0 to "
                                 "hold the resample fixed and make --viz_voice_temperature "
                                 "behave monotonically.")
    sub_parser.add_argument("--voice_cosyvoice2_distill_model_dir", type=str, default=None,
                            help="CosyVoice2-0.5B snapshot dir for KL DISTILLATION. Runs the frozen "
                                 "Qwen2-0.5B speech LM teacher-forced in the training loop and adds "
                                 "KL(teacher || student) on the voice unit logits. Needs "
                                 "--voice_distill_weight > 0 to take effect.")
    sub_parser.add_argument("--voice_distill_weight", type=float, default=0.0,
                            help="Weight on the distillation KL term (0 = off). The CE term is "
                                 "unchanged, so this ADDS soft-target supervision on top of it.")
    sub_parser.add_argument("--voice_distill_temperature", type=float, default=1.0,
                            help="Softmax temperature for both sides of the KL. >1 flattens the "
                                 "teacher and transfers more of its low-probability structure "
                                 "(the point, given ~4 nats of predictive entropy). Loss is scaled "
                                 "by T^2 so gradient magnitude stays temperature-independent.")
    sub_parser.add_argument("--voice_distill_device", type=str, default=None,
                            help="Device for the frozen teacher (default: same as training). It is "
                                 "~0.5B; in bf16 that is ~1GB plus activations.")
    sub_parser.add_argument("--voice_distill_fp32", dest="voice_distill_bf16", action="store_false",
                            default=True, help="Run the teacher in fp32 instead of bf16 (2x memory).")
    sub_parser.add_argument("--voice_cosyvoice2_model_dir", type=str, default=None,
                            help="CosyVoice2-0.5B snapshot dir. Loads the FROZEN flow+HiFT decoder "
                                 "so eval viz renders 24kHz audio from the voice coda's unit ids. "
                                 "Use INSTEAD OF --voice_smg_checkpoint_path/--vocoder on a "
                                 "CosyVoice 2 run (the SMG decodes Mimi's codebook, not this one).")
    sub_parser.add_argument("--voice_cosyvoice2_runtime_dir", type=str, default=None,
                            help="CosyVoice checkout providing the `cosyvoice` package + cv_extra "
                                 "(default: $COSYVOICE_RUNTIME, else ~/dev/projects/cosyvoice-runtime)")
    sub_parser.add_argument("--voice_cosyvoice2_device", type=str, default="cpu",
                            help="Device for the frozen CosyVoice 2 decoder (default cpu, so the "
                                 "133M of frozen weights don't take VRAM from training)")
    sub_parser.add_argument("--vocoder_checkpoint_path", type=str, default=None,
                            help="Path to vocoder checkpoint for visualization")
    sub_parser.add_argument("--vocoder_config", type=str, default=None,
                            help="Vocoder config name (e.g. 'hifigan' for pretrained SpeechBrain HiFi-GAN, no checkpoint needed)")
    sub_parser.add_argument("--image_vae_decoder_path", type=str, default=None,
                            help="Path to image VAE decoder checkpoint (not needed for litevae)")
    sub_parser.add_argument("--image_vae_decoder_config", type=str, default=None,
                            help="Image VAE decoder config. Use 'litevae' for pretrained LiteVAE (auto-downloaded)")
    sub_parser.add_argument("--voice_smg_checkpoint_path", type=str, default=None,
                            help="Path to voice SMG decoder checkpoint for decoding SIVE latents to mel specs")
    sub_parser.add_argument("--voice_smg_config", type=str, default="small",
                            help="Voice SMG decoder config name")
    sub_parser.add_argument("--voice_smg_sive_encoder_dim", type=int, default=None,
                            help="Override sive_encoder_dim for voice SMG (must match what it was trained with)")
    sub_parser.add_argument("--voice_smg_speaker_embedding_dim", type=int, default=None,
                            help="Speaker-embedding width of the voice SMG checkpoint (192 ECAPA, 768 WavLM). "
                                 "Required to load a WavLM-conditioned SMG or its FiLM/F0 speaker weights "
                                 "load as random under strict=False.")
    sub_parser.add_argument("--viz_voice_temperature", type=float, default=0.0,
                            help="Sampling temperature for voice latents in TB eval renders. "
                                 "DEFAULT 0.0 as of 2026-09-15, changed from 0.6: the RAS "
                                 "resample inherits this scaling and is discontinuous at zero, "
                                 "so 0.2-0.6 is a TROUGH where termination breaks (budget caps "
                                 "12/12 at T=0.2, 3/12 at T=0.6, 0/12 at T=0.0 and T=0.7). At "
                                 "T=0 the pick is argmax while the resample runs at an "
                                 "effective 1.0, which measured best overall. If you want a "
                                 "mid-range temperature, pass --viz_voice_ras_temperature 1.0 "
                                 "with it. Only active if the model was trained with "
                                 "--voice_stochastic_output (else generate() ignores it).")
    sub_parser.add_argument("--viz_voice_variance_floor", type=float, default=0.0,
                            help="Clamp the per-frame std floor when sampling voice latents in TB renders (0=off).")
    sub_parser.add_argument("--static_speaker_embedding_path", type=str, default=None,
                            help="Path to a .pt file containing a speaker embedding tensor for static-speaker voice decoding")
    sub_parser.add_argument("--viz_voice_prompt_audio", type=str,
                            default="logs/speaker_male.wav",
                            help="WAV of the SAME speaker as --static_speaker_embedding_path. "
                                 "Enables CosyVoice 2 zero-shot prompt conditioning in the eval "
                                 "renders: the 192-d embedding is a global summary, while the "
                                 "prompt supplies per-frame acoustic evidence (timbre, channel, "
                                 "style) no fixed-size vector carries. Needs "
                                 "--voice_cosyvoice2_model_dir for speech_tokenizer_v2.onnx. "
                                 "A failure degrades to embedding-only rendering with a warning. "
                                 "DEFAULTS to logs/speaker_male.wav, which is gitignored -- a "
                                 "fresh clone warns and falls back rather than failing. "
                                 "REQUIRED for correct speaker identity: the 192-d campplus "
                                 "embedding alone rendered a male reference as female "
                                 "(2026-09-15, ear-confirmed).")
    sub_parser.add_argument("--num_eval_samples", type=int, default=4,
                            help="Number of samples per visualization scenario")

    # Voice collator settings
    sub_parser.add_argument("--voice_max_seconds", type=float, default=10.0,
                            help="Maximum voice clip length in seconds")
    sub_parser.add_argument("--voice_sample_rate", type=int, default=16000,
                            help="Voice sample rate")
    sub_parser.add_argument("--voice_n_fft", type=int, default=1024,
                            help="FFT size for voice mel spectrograms")
    sub_parser.add_argument("--voice_n_mels", type=int, default=80,
                            help="Number of voice mel filterbanks")
    sub_parser.add_argument("--voice_hop_length", type=int, default=256,
                            help="Voice hop length")
    # Audio (general/non-speech) collator settings
    sub_parser.add_argument("--audio_max_seconds", type=float, default=10.0,
                            help="Maximum audio clip length in seconds")
    sub_parser.add_argument("--audio_sample_rate", type=int, default=16000,
                            help="Audio sample rate")
    sub_parser.add_argument("--audio_n_fft", type=int, default=1024,
                            help="FFT size for audio mel spectrograms")
    sub_parser.add_argument("--audio_n_mels", type=int, default=80,
                            help="Number of audio mel filterbanks")
    sub_parser.add_argument("--audio_hop_length", type=int, default=256,
                            help="Audio hop length")
    sub_parser.add_argument("--sive_total_stride", type=int, default=3,
                            help="Total temporal downsampling stride of the SIVE encoder (default 3 = 3x; 4x was found to over-compress SIVE)")
    sub_parser.add_argument("--voice_predict_f0", action="store_true",
                            help="Add an F0 contour regression head to the voice coda, beside the "
                                 "unit classifier. Quantization strips prosody from the units by "
                                 "construction, so the contour must come from somewhere; the SMG's own "
                                 "F0 predictor sees only (units, speaker) and recovers ~4%% of "
                                 "within-utterance F0 variation over predicting the speaker's mean "
                                 "pitch, because neither input knows the sentence is a question. The "
                                 "coda sits on text conditioning and can. The head emits a "
                                 "SPEAKER-NORMALIZED contour (sigma units); the SMG denormalizes it "
                                 "with ECAPA. Pair with --voice_f0_loss_weight.")
    sub_parser.add_argument("--voice_f0_loss_weight", type=float, default=1.0,
                            help="Weight on the coda's voicing-weighted L1 loss over the "
                                 "speaker-normalized log-F0 contour. NOTE the contour is in sigma "
                                 "units (std ~1.1), not log Hz, so this weight is on a different "
                                 "scale than a raw-F0 loss would be.")
    sub_parser.add_argument("--voice_dedup", action="store_true",
                            help="Run the voice coda on deduped (unit, duration) SEGMENTS instead of "
                                 "50Hz frames. At 50Hz the next unit is dominated by 'where am I in "
                                 "this phoneme', which teacher-forced history predicts and text "
                                 "cannot -- the measured cause of the text-conditioning collapse. "
                                 "Collapsing consecutive-equal units removes that redundancy so the "
                                 "text gradient survives (marginal text signal +0.004 -> +0.033). The "
                                 "coda predicts unit + duration per segment; generation expands by "
                                 "duration back to 50Hz for the frozen SMG. Requires "
                                 "--voice_codebook_path; implies --voice_predict_f0.")
    sub_parser.add_argument("--voice_duration_loss_weight", type=float, default=1.0,
                            help="Weight on the per-segment duration head's L1-on-log-frames loss "
                                 "(only with --voice_dedup).")
    sub_parser.add_argument("--voice_codebook_path", type=str, default=None,
                            help="k-means codebook .pt (from scripts.data.voice.fit_codebook). Turns "
                                 "ON the DISCRETE voice path: content features are snapped to their "
                                 "nearest centroid on the fly, the voice coda gains a K-way classifier "
                                 "(K read from the codebook), and the voice loss becomes cross-entropy "
                                 "over units INSTEAD OF continuous regression. Regression is solvable "
                                 "from the audio history alone, so the text earns no gradient and the "
                                 "model converges to fluent babble; CE forces it to NAME the next unit, "
                                 "which requires the text. MUST be the same codebook the SMG was trained "
                                 "with -- a re-fit reorders centroids and silently maps every unit to "
                                 "the wrong phoneme.")
    sub_parser.add_argument("--voice_cfg_text_dropout_prob", type=float, default=0.0,
                            help="Classifier-free guidance: per-synthesis-row probability of "
                                 "replacing text with a learned null embedding during training, so "
                                 "the model learns an unconditional voice distribution. Enables "
                                 "inference guidance (uncond + w*(cond-uncond)). 0 = off; typical "
                                 "0.1-0.15. For finetuning a converged base into CFG, pair with a "
                                 "fresh low LR (see notes). Deterministic at inference (no dropout).")
    sub_parser.add_argument("--voice_gen_query_mode", type=str, default=None,
                            choices=["learned_pos"],
                            help="Gen-query voice synthesis: replace the AR crutch feeding the "
                                 "SHARED recurrent trunk with a learned per-position query (text-"
                                 "only), and route local coherence to the coda via the previous "
                                 "frame's centroid (voice_coda_prev_proj). Mirrors the image path's "
                                 "gen queries; the coda + EOV still drive length autoregressively. "
                                 "Adds params (voice_gen_queries, voice_coda_prev_proj) -- start a "
                                 "FRESH run (or --fresh_schedule). None = off (AR crutch). "
                                 "'learned_pos' = the only mode.")
    sub_parser.add_argument("--voice_curriculum_max_frames", type=int, default=0,
                            help="LENGTH CURRICULUM: train only on voice utterances with "
                                 "feature_length <= this many frames (Mimi @12.5Hz: 25=~2s, "
                                 "40=~3.2s, 60=~4.8s, 125=~10s=full). 0 = off (all lengths). A "
                                 "STATIC per-stage cap: short utts are the regime where text "
                                 "already drives content (baseline early_text_delta +0.10), so "
                                 "start small and RAISE it across manual resumes (e.g. 25 -> 40 "
                                 "-> 60 -> 0) to extend text conditioning to longer sequences. "
                                 "Voice-only; eval stays uncapped for cross-stage comparability. "
                                 "Changes sampler order -> use on a fresh stage, not a byte-exact resume.")
    sub_parser.add_argument("--voice_gen_query_no_coda_prev", action="store_true",
                            help="With --voice_gen_query_mode, DROP the coda's previous-centroid "
                                 "signal (voice_coda_prev). That signal is a 1st-order neighbor "
                                 "crutch that survives even an MLP coda (a position-wise coda can "
                                 "predict unit t from centroid t-1 with no attention and no text) "
                                 "-- exactly the 'local coherence without text' failure. With this "
                                 "flag the coda sees ONLY the text-driven trunk output, so there is "
                                 "no neighbor-extrapolation crutch anywhere. Pair with "
                                 "--voice_coda_type mlp for the truly crutch-free path.")
    sub_parser.add_argument("--text_encoder_model", type=str, default=None,
                            help="SINGLE GATE for the pretrained-LLM text path. An HF model id "
                                 "(e.g. HuggingFaceTB/SmolLM2-135M) swaps the from-scratch text "
                                 "prelude/coda for that LLM's body (encoder) + MLP translators + "
                                 "its LM head. None (default) = current from-scratch path, byte-"
                                 "identical. NOTE: the LLM owns its tokenizer, so text data must be "
                                 "re-tokenized with it (fold into the clean-data re-preprocess).")
    sub_parser.add_argument("--voice_nar", action="store_true", default=False,
                            help="Masked-parallel (NAR) voice synthesis instead of autoregressive. "
                                 "Trains a MaskGIT objective: a cosine-sampled fraction of voice "
                                 "frames is replaced by a learned MASK feature and the unit loss is "
                                 "restricted to those positions. Makes the voice coda BIDIRECTIONAL "
                                 "so the head sees revealed units on both sides. Motivated by the "
                                 "measured AR failure: greedy decoding puts 96%% adjacent repeats, "
                                 "i.e. the conditional mode is 'repeat the previous unit' -- a loop "
                                 "that is structurally impossible without an AR feedback path.")
    sub_parser.add_argument("--voice_nar_mask_schedule", type=str, default="cosine",
                            choices=["cosine", "high", "linear"],
                            help="Distribution of the NAR mask ratio. 'cosine' is MaskGIT's and "
                                 "concentrates mass at moderate ratios, where the utterance is "
                                 "largely reconstructable from neighbouring units -- measured at "
                                 "23k, that produced inpainting rather than text reading "
                                 "(revealing 25%% of frames raised accuracy 5x while text_delta "
                                 "FELL). 'high' samples U[--voice_nar_mask_ratio_min, 1], removing "
                                 "the shortcut and matching inference, which starts fully masked. "
                                 "'linear' is U[0,1], the neutral uniform control.")
    sub_parser.add_argument("--voice_nar_trunk_text_only", action="store_true", default=False,
                            help="NAR variant: the trunk sees ONLY the MASK feature at voice "
                                 "positions and revealed units go straight to the coda, so the "
                                 "trunk stays text-conditioned and any inpainting shortcut is "
                                 "confined to the head. Identical to plain --voice_nar at mask "
                                 "ratio 1.0 (nothing to route); the difference appears once the "
                                 "anneal admits context.")
    sub_parser.add_argument("--voice_nar_mask_anneal_steps", type=int, default=0,
                            help="Anneal the mask floor from 1.0 (fully masked -- text is the "
                                 "ONLY signal) down to --voice_nar_mask_ratio_floor over this "
                                 "many steps, sampling U[floor(step), 1]. 0 = off. Forms the text "
                                 "pathway while no shortcut exists, then admits context. Set it "
                                 "from where text-only training actually plateaus; watch "
                                 "text_delta at r=1.0 to see whether the pathway survives the "
                                 "anneal or the crutch reasserts.")
    sub_parser.add_argument("--voice_nar_mask_ratio_floor", type=float, default=0.25,
                            help="Final mask floor for --voice_nar_mask_anneal_steps.")
    sub_parser.add_argument("--voice_nar_mask_ratio_min", type=float, default=0.85,
                            help="Lower bound for the 'high' and 'linear' mask schedules.")
    sub_parser.add_argument("--voice_nar_choice_temperature", type=float, default=1.0,
                            help="Gumbel noise on MaskGIT confidence at eval-render time, "
                                 "annealed to 0 over the rounds. 0 = greedy reveal, which "
                                 "self-reinforces the repetition mode.")
    sub_parser.add_argument("--voice_nar_rounds", type=int, default=16,
                            help="MaskGIT refinement rounds at generation. Decoding cost is K "
                                 "forward passes REGARDLESS of length, unlike AR's one-per-frame. "
                                 "K=1 samples independent per-position marginals and should sound "
                                 "obviously worse -- a useful check that the head is modelling the "
                                 "joint rather than the marginals.")
    sub_parser.add_argument("--voice_nar_duration_token", action="store_true", default=False,
                            help="NAR voice: emit a DUR_* bucket token between BOV and the voice "
                                 "placeholder, predicted from the BOV hidden state (the last "
                                 "position with full causal multimodal context, and the last one "
                                 "before gen queries must be allocated). 32 log-spaced buckets "
                                 "over [25,250] frames, measured from the training cache. Rides "
                                 "the existing trainable special_embed/special_head extension, so "
                                 "the frozen LLM is untouched. Automatically exempts the token "
                                 "from --mask_text_loss_in_synthesis, which would otherwise give "
                                 "it zero gradient while training loss looked healthy.")
    sub_parser.add_argument("--unmask_eos_in_synthesis", action="store_true", default=False,
                            help="Keep the text loss on the EOS target that follows a synthesis "
                                 "media block, even under --mask_text_loss_in_synthesis. The "
                                 "layout is [text][BOV][PH][EOV][eos], so the target at the EOV "
                                 "position is EOS -- and masking it means the model gets ZERO "
                                 "gradient on the one position generation samples at, right "
                                 "after a media block. It then has no learned reason to stop: "
                                 "measured at step 11076, 2.75 utterances per prompt (max 6) "
                                 "from a model whose EOV fires correctly on 12 of 14 blocks. "
                                 "Opt-in so in-flight runs are unaffected; recommended for any "
                                 "new synthesis run, voice or image.")
    # ------------------------------------------------------------------ bistream
    sub_parser.add_argument("--bistream_text_chunk", type=int, default=0,
                            help="BISTREAM chunk interleaving (CosyVoice 2 style): text tokens "
                                 "per chunk, k. 0 = off (all text then all speech). Turns the "
                                 "layout into [text 0:k][BOV][PH][EOV][text k:2k][BOV][PH][EOV]... "
                                 "so the text a frame realizes is ADJACENT instead of hundreds of "
                                 "positions back, and alignment is roughly monotonic. Needs "
                                 "--bistream_voice_chunk. Widens the unit head by one for the "
                                 "chunk terminator (fill_token = K+1), so a bistream checkpoint "
                                 "is NOT loadable into a unistream model.")
    sub_parser.add_argument("--bistream_voice_chunk", type=int, default=0,
                            help="Speech frames per chunk, s. Use 30, NOT CosyVoice 2's 15: theirs "
                                 "assumes 3 frames/token and our corpus measures 5.9 (n=6102, "
                                 "median 5.906), so 5:15 would exhaust the text about halfway "
                                 "through the utterance and leave the back half unaligned.")
    sub_parser.add_argument("--bistream_prob", type=float, default=0.5,
                            help="Fraction of ELIGIBLE synthesis samples that go bistream; the rest "
                                 "stay unistream. 0.5 copies CosyVoice 2, and keeps the unistream "
                                 "half directly comparable to what is trained today. A sample is "
                                 "eligible when the chunks before the last one fit "
                                 "(n_frames > (n_text_chunks-1)*s) -- the exact condition, not "
                                 "their speech/text ratio gate, which would reject ~half our data.")
    sub_parser.add_argument("--bistream_text_loss", action="store_true", default=False,
                            help="Keep the text loss alive on chunk-interleaved rows even under "
                                 "--mask_text_loss_in_synthesis. REQUIRED for inner monologue (the "
                                 "model generating its own transcript); optional for bistream TTS, "
                                 "where the transcript is still fed at inference. Without it the "
                                 "interleaved text gets zero gradient and the model can never speak "
                                 "at will -- while training loss looks perfectly healthy.")
    sub_parser.add_argument("--voice_onpolicy_distill", action="store_true", default=False,
                            help="Score the distillation teacher on the model's OWN "
                                 "(scheduled-sampling-substituted) unit history instead of the "
                                 "ground truth. Off-policy KD only supervises states reachable "
                                 "from a perfect prefix -- where this model is already near the "
                                 "teacher. Its failure is off that manifold. Requires "
                                 "--voice_scheduled_sampling_prob > 0 and --voice_distill_weight > 0.")
    sub_parser.add_argument("--voice_onpolicy_temperature", type=float, default=0.8,
                            help="Temperature for sampling the substituted units (0 = argmax, the "
                                 "old behavior). Argmax substitutes the MODE, which greedy decoding "
                                 "shows is 'repeat the previous unit' -- a state the sampler never "
                                 "visits. 0.8 matches the measured best decode point.")
    sub_parser.add_argument("--text_encoder_unfreeze", action="store_true",
                            help="Fine-tune the pretrained LLM (default: frozen). Use a low LR; "
                                 "differential-LR wiring is a follow-up -- frozen is the tested path.")
    sub_parser.add_argument("--text_encoder_translator_mult", type=float, default=2.0,
                            help="Hidden width multiple for the prelude/coda MLP translators "
                                 "(hidden = trunk_d_model * mult).")
    sub_parser.add_argument("--text_encoder_trainable_head", action="store_true",
                            help="Keep the frozen LLM on the INPUT side only: give the text coda its "
                                 "OWN trainable readout at TRUNK width (vocab + control tokens in one "
                                 "matrix) instead of sharing the LLM's tied lm_head. Default (off) "
                                 "shares the head, which pins every output distribution to the LLM's "
                                 "frozen token geometry and caps logit-matrix rank at llm_d+1. "
                                 "Costs ~+38M params at trunk_d 768 / vocab 49152.")
    sub_parser.add_argument("--text_encoder_n_special_tokens", type=int, default=0,
                            help="Size of the TRAINABLE control-token extension (tied special_embed/"
                                 "special_head) for BOV/EOV/placeholders etc., appended at ids >= the "
                                 "LLM's native vocab. Frozen-LLM-safe (soft-prompt style). 0 = none.")
    sub_parser.add_argument("--voice_coda_type", type=str, default=None,
                            choices=["transformer", "mlp"],
                            help="Voice coda body: 'transformer' (causal self-attention stack, "
                                 "default) or 'mlp' (attention-free position-wise FFN stack, no KV "
                                 "cache). The MLP coda decodes each trunk position to a mimi unit "
                                 "INDEPENDENTLY, so it cannot use local coherence as a crutch that "
                                 "bypasses text -- pair with --voice_gen_query_mode for a fully "
                                 "crutch-free voice path. None = leave the config default (transformer).")
    sub_parser.add_argument("--voice_coda_mlp_ratio", type=float, default=None,
                            help="FFN expansion ratio for --voice_coda_type mlp (hidden = "
                                 "round(d_model * ratio)). Default (None) uses the config's 4.0.")
    sub_parser.add_argument("--voice_early_text_weight_alpha", type=float, default=1.0,
                            help="Early-text loss weighting: peak per-frame CE weight at the "
                                 "utterance ONSET, decaying linearly to 1 by frame "
                                 "--voice_early_text_weight_frames. Onset frames have the least AR "
                                 "history, so the crutch can't help and text must carry -- this "
                                 "concentrates gradient where text is necessary. Renormalized so "
                                 "loss scale is unchanged. 1.0 = off (plain mean); typical 2-4.")
    sub_parser.add_argument("--voice_early_text_weight_frames", type=int, default=0,
                            help="Onset region (frames) over which the early-text weight decays "
                                 "alpha->1. 0 = off. Typical 8-16 (~0.6-1.3s at 12.5Hz), matching "
                                 "where early_text_delta shows the clean text signal.")
    sub_parser.add_argument("--voice_prenet_dropout", type=float, default=0.0,
                            help="Tacotron-2 prenet dropout on the AUTOREGRESSIVE voice path "
                                 "(shifted teacher forcing in training, own-output feedback at "
                                 "generation). Under teacher forcing the true previous frames "
                                 "predict frame t so well that the text earns no gradient and the "
                                 "model converges to unconditional babble; this bottleneck forces "
                                 "the text to become the reliable signal. Stays ON at inference, "
                                 "by design. 0 = off, 0.5 = the Tacotron 2 value. This is the RAMP "
                                 "TARGET when --voice_prenet_dropout_ramp_steps > 0, else a constant.")
    sub_parser.add_argument("--voice_prenet_dropout_ramp_steps", type=int, default=0,
                            help="Linear ramp length (steps) over which prenet dropout goes 0 → "
                                 "--voice_prenet_dropout. 0 (default) = constant (no ramp), the "
                                 "prior behavior. Ramped UP (not held high from step 0) because "
                                 "early text is noise: the bottleneck only helps once the text "
                                 "pathway has something to offer. Attacks the shifted-INPUT crutch "
                                 "that --voice_ar_attn_* leaves intact; the two compose. Suggested: "
                                 "20000, aligned with the attention ramp.")
    sub_parser.add_argument("--voice_prenet_dropout_start_step", type=int, default=0,
                            help="Hold prenet dropout at 0 for this many steps before the ramp "
                                 "begins. Set to --voice_ar_attn_mask_steps to start starving the "
                                 "input crutch exactly when history-attention is handed back.")
    sub_parser.add_argument("--voice_scheduled_sampling_prob", type=float, default=0.0,
                            help="Per-frame probability of feeding the model's OWN prediction "
                                 "instead of ground truth on the AR path, ramped from 0 over "
                                 "--voice_scheduled_sampling_ramp_steps. Attacks the same "
                                 "teacher-forcing crutch as --voice_prenet_dropout but by removing "
                                 "the guarantee that history is perfect. COSTS ~1.6x step time (one "
                                 "extra no-grad forward per step). 0 = off.")
    sub_parser.add_argument("--voice_scheduled_sampling_ramp_steps", type=int, default=10000,
                            help="Linear ramp length for --voice_scheduled_sampling_prob. 0 = no "
                                 "ramp (full value from --voice_scheduled_sampling_start_step "
                                 "onward), which is the right choice on a warm start from a "
                                 "competent checkpoint, where the ramp\'s premise — that early "
                                 "predictions are noise — no longer holds. Ramped "
                                 "because early predictions are noise, and training on "
                                 "noise-as-history teaches nothing.")
    sub_parser.add_argument("--voice_scheduled_sampling_start_step", type=int, default=0,
                            help="Global step at which the --voice_scheduled_sampling_prob ramp "
                                 "BEGINS; the probability is 0 before it. Default 0 = ramp from "
                                 "the start of training, correct for a fresh run. On a warm start "
                                 "set it to the resume step (e.g. 50000 when resuming from "
                                 "checkpoint-50000), otherwise global_step is already past the "
                                 "ramp length on the first step and the probability jumps "
                                 "straight to its full value — the ramp flag does nothing.")
    sub_parser.add_argument("--voice_ar_attn_mask_steps", type=int, default=0,
                            help="NAR→AR curriculum (Variant B): number of steps to HOLD "
                                 "voice→voice attention at --voice_ar_attn_floor (the NAR phase, "
                                 "voice history severed). The voice AR leans on the voice-history "
                                 "crutch so the text earns no gradient; down-scaling voice→voice "
                                 "attention in the prelude + recurrent trunk + coda forces content "
                                 "onto the text pathway. Input stays shifted-TF (a voice position "
                                 "still sees its own frame t-1 — a weak 1-frame residual). REPLACES "
                                 "and is mutually exclusive with --voice_scheduled_sampling_prob. "
                                 "0 (default) = curriculum off. Suggested: 10000.")
    sub_parser.add_argument("--voice_ar_attn_ramp_steps", type=int, default=0,
                            help="Linear ramp length (steps) over which voice→voice attention scale "
                                 "goes --voice_ar_attn_floor → --voice_ar_attn_cap after the "
                                 "--voice_ar_attn_mask_steps NAR phase, handing history back once the "
                                 "text pathway is established. Suggested: 20000.")
    sub_parser.add_argument("--voice_ar_attn_floor", type=float, default=0.0,
                            help="Minimum voice→voice attention scale alpha during the NAR phase "
                                 "(0 = fully severed: a voice position attends text + its own "
                                 "shifted-TF frame only). Adding log(alpha) to a voice→voice QK "
                                 "score scales that softmax weight by alpha.")
    sub_parser.add_argument("--voice_ar_attn_cap", type=float, default=1.0,
                            help="Maximum voice→voice attention scale alpha the ramp reaches "
                                 "(1.0 = full attention restored; <1.0 keeps some suppression "
                                 "permanent, which then also applies at generation).")
    sub_parser.add_argument("--voice_ar_attn_ramp_power", type=float, default=1.0,
                            help="Ramp shape: 1.0 = linear; >1 = EASE-IN (slow start) so alpha "
                                 "crawls through the low band and accelerates late. Use >1 (e.g. 3) "
                                 "when the low-alpha history-reintroduction is the fragile part — the "
                                 "linear ramp diverged (grad explosion ~alpha 0.35); an ease-in "
                                 "spends most ramp steps re-integrating history at small alpha.")
    sub_parser.add_argument("--voice_token_budget", type=int, default=None,
                            help="Max content frames generate() may emit before force-closing with "
                                 "EOV, for the TensorBoard TTS renders. Default: derived from "
                                 "--voice_max_seconds/--voice_sample_rate/--voice_hop_length/"
                                 "--sive_total_stride, i.e. the same number as the collator's trim "
                                 "ceiling (500 for 10s ContentVec @hop320/stride1). The stop head "
                                 "normally ends generation before this; it is a hard cap.")
    sub_parser.add_argument("--audio_token_budget", type=int, default=None,
                            help="As --voice_token_budget, for the non-speech audio modality.")
    sub_parser.add_argument("--voice_feature_channels", type=int, default=None,
                            help="Channel width of the cached voice 'features' tensors — the voice prelude "
                                 "projects from this and the voice coda predicts into it. Must match the "
                                 "content encoder the cache was built with (SIVE: 256; ContentVec: 256 for "
                                 "final_proj, 768 for last_hidden_state). Defaults to "
                                 "--voice_smg_sive_encoder_dim (same quantity: the coda's output is the "
                                 "SMG's input), then to the config value (128). Only needed when no SMG "
                                 "is loaded.")

    return sub_parser
