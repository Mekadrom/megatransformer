import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatransformer.model.world.world_model import MegaTransformerWorldModel
from megatransformer.scripts.train.trainer import CommonTrainer
from megatransformer.utils import model_loading_utils, megatransformer_utils, metrics


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
        # If True, skip text loss in image_synthesis / voice_synthesis batches
        # where text is conditioning rather than target. Default False for
        # backward compatibility with existing runs. Standard practice in
        # multimodal LMs (Flamingo, GIT, BLIP) is True — applying text loss
        # during synthesis competes with the generation objective.
        mask_text_loss_in_synthesis: bool = False,
        # Group within-modality samples by shard when shuffling. Default
        # True. Set False to reproduce the legacy uniform shuffle order
        # when resuming a checkpoint from a pre-shard-aware run.
        shard_aware_sampler: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.lr_dit = lr_dit
        self.mask_text_loss_in_synthesis = mask_text_loss_in_synthesis
        self.shard_aware_sampler = shard_aware_sampler

        self.cmdline = cmdline
        self.git_commit_hash = git_commit_hash
        self.step_offset = step_offset or 0

        self.text_loss_weight = text_loss_weight
        self.audio_latent_loss_weight = audio_latent_loss_weight
        self.voice_latent_loss_weight = voice_latent_loss_weight
        self.image_latent_loss_weight = image_latent_loss_weight

        self.audio_var_loss_weight = audio_var_loss_weight
        self.voice_var_loss_weight = voice_var_loss_weight
        self.image_var_loss_weight = image_var_loss_weight

        self.audio_var_barrier_weight = audio_var_barrier_weight
        self.voice_var_barrier_weight = voice_var_barrier_weight
        self.image_var_barrier_weight = image_var_barrier_weight

        self.audio_stop_loss_weight = audio_stop_loss_weight
        self.voice_stop_loss_weight = voice_stop_loss_weight

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
        try:
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
            )

        # Eval sampler — mirrors the train sampler structure to ensure eval
        # batches are homogeneous by task type. Without this, HF Trainer's
        # default SequentialSampler produces mixed-modality eval batches,
        # which trigger the batch-size-mismatch null-out in world_model.py
        # forward and collapse all eval task_type classification to
        # text_continuation (because voice/image get nulled for non-text-heavy
        # batches). Result: only `world_eval/text_continuation/*` metrics
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
        self.text_loss_fn = nn.CrossEntropyLoss(label_smoothing=text_label_smoothing)
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

        if not self.has_logged_cli and torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
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
        if text_input_ids is not None and self.include_text:
            from megatransformer.utils.constants import (
                AUDIO_PLACEHOLDER_TOKEN_ID,
                VOICE_PLACEHOLDER_TOKEN_ID,
                IMAGE_PLACEHOLDER_TOKEN_ID,
            )
            placeholder_ids = {AUDIO_PLACEHOLDER_TOKEN_ID, VOICE_PLACEHOLDER_TOKEN_ID, IMAGE_PLACEHOLDER_TOKEN_ID}

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

            # Model input: keep placeholders for the interleaver
            text_input_ids = full_ids[:, :-1].contiguous()

            # Count non-PH positions in model input (= number of logits per item)
            non_ph_input = torch.ones_like(text_input_ids, dtype=torch.bool)
            for pid in placeholder_ids:
                non_ph_input &= (text_input_ids != pid)

            target_list = []
            for b in range(full_ids.shape[0]):
                clean = full_ids[b][non_ph_mask_full[b]]  # non-PH tokens in order
                shifted = clean[1:]  # causal shift: predict next non-PH token
                K = non_ph_input[b].sum().item()  # number of logits this item will produce
                # Truncate or pad targets to exactly K
                if shifted.shape[0] >= K:
                    target_list.append(shifted[:K])
                else:
                    target_list.append(torch.cat([shifted, shifted.new_zeros(K - shifted.shape[0])]))

            # Pad and stack targets across batch
            max_len = max(t.shape[0] for t in target_list)
            padded_targets = []
            for t in target_list:
                if t.shape[0] < max_len:
                    padded_targets.append(torch.cat([t, t.new_zeros(max_len - t.shape[0])]))
                else:
                    padded_targets.append(t)
            text_targets = torch.stack(padded_targets)  # [B, T_text]

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

        # Voice: same shape contract as audio (SIVE features). Separate placeholder token.
        voice_inputs = None
        voice_lengths = None
        voice_latent_labels = None
        voice_loss_lengths = None  # (B,) raw feature lengths for masked loss
        if self.include_voice:
            voice_data = inputs.get("voice_features")  # [B, C, T]
            if voice_data is not None:
                voice_inputs = voice_data.unsqueeze(1)  # [B, 1, C, T]
                voice_latent_labels = voice_data.clone()

                voice_feat_lengths = inputs.get("voice_feature_lengths")
                if voice_feat_lengths is not None:
                    voice_lengths = voice_feat_lengths.unsqueeze(1)
                    voice_loss_lengths = voice_feat_lengths  # (B,) for masked loss

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

        # Enable per-iteration stat tracking at logging steps
        should_log = global_step % self.args.logging_steps == 0
        model_for_stats = model.module if hasattr(model, 'module') else model
        if should_log and hasattr(model_for_stats, 'recurrent_block'):
            model_for_stats.recurrent_block.track_iteration_stats = True

        outputs = model(
            text_input_ids=text_input_ids,
            audio_inputs=audio_inputs,
            audio_lengths=audio_lengths,
            voice_inputs=voice_inputs,
            voice_lengths=voice_lengths,
            image_inputs=image_inputs,
            image_latent_labels=image_latent_labels,
            precomputed_latents=self.precomputed_latents,
            decode_outputs=False,
            is_synthesis=is_synthesis,
        )

        if should_log and hasattr(model_for_stats, 'recurrent_block'):
            model_for_stats.recurrent_block.track_iteration_stats = False

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

        # Voice: whitened L1+MSE + variance-matching aux loss (masked by feature lengths)
        voice_latent_preds = outputs.get("voice_latent_preds")
        if voice_latent_preds is not None and voice_latent_labels is not None and voice_latent_preds.numel() > 0:
            voice_modality_loss, voice_components = self._compute_modality_recon_loss(
                "voice_latent", voice_latent_preds, voice_latent_labels,
                self.voice_var_loss_weight,
                self.voice_var_barrier_weight,
                lengths=voice_loss_lengths,
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

        if global_step % self.args.logging_steps == 0:
            metrics.log_scalar("world/total_loss", total_loss, global_step)
            for name, value in loss_components.items():
                metrics.log_scalar(f"world/{name}", value, global_step)

            # Recurrent output stats (variance and entropy per modality)
            for key, value in outputs.items():
                if key.startswith("recurrent_out/"):
                    metrics.log_scalar(f"world/{key}", value, global_step)

            # Per-iteration activation stats from recurrent block
            iteration_stats = outputs.get("iteration_stats")
            if iteration_stats:
                from megatransformer.utils import visualization
                fig = visualization.render_iteration_stats(iteration_stats)
                metrics.log_figure("world/iteration_stats", fig, global_step)
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

        # Text embedding
        groups["text_embedding"] = model.text_feature_extractor
        groups["text_embedding/wte"] = model.text_feature_extractor.wte

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
                    groups[f"{prefix}/layer{i}/attn"] = block.self_attn
                    groups[f"{prefix}/layer{i}/ffn"] = block.ffn
            if hasattr(generator, 'lm_head'):
                groups[f"{prefix}/lm_head"] = generator.lm_head
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
            metrics.log_scalar(f"world/grad_norm/{name}", total_norm_sq ** 0.5, global_step, skip_zero=False)
            if num_params > 0:
                metrics.log_scalar(f"world/grad_rms/{name}", (total_norm_sq / num_params) ** 0.5, global_step, skip_zero=False)

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
        `world_eval/voice_synthesis/voice_latent_l1_norm`) populate correctly.
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

        groups = {
            "dit_decay": {"params": [], "weight_decay": self.args.weight_decay, "lr": self.lr_dit},
            "dit_no_decay": {"params": [], "weight_decay": 0.0, "lr": self.lr_dit},
            "main_decay": {"params": [], "weight_decay": self.args.weight_decay, "lr": self.args.learning_rate},
            "main_no_decay": {"params": [], "weight_decay": 0.0, "lr": self.args.learning_rate},
        }

        for n, p in opt_model.named_parameters():
            if not p.requires_grad:
                continue
            in_dit = is_dit(n)
            in_decay = n in decay_parameters
            key = (
                "dit_decay" if in_dit and in_decay else
                "dit_no_decay" if in_dit else
                "main_decay" if in_decay else
                "main_no_decay"
            )
            groups[key]["params"].append(p)

        optimizer_grouped_parameters = [g for g in groups.values() if g["params"]]

        optimizer_cls, optimizer_kwargs = _HFTrainer.get_optimizer_cls_and_kwargs(self.args, opt_model)
        # The 'lr' baked into each group takes precedence, but optimizer_kwargs
        # still needs a default 'lr' for AdamW's constructor signature.
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        # Print a summary so the user can verify routing.
        if self.args.local_rank in (-1, 0):
            dit_params = sum(p.numel() for g in ("dit_decay", "dit_no_decay") for p in groups[g]["params"])
            main_params = sum(p.numel() for g in ("main_decay", "main_no_decay") for p in groups[g]["params"])
            print(
                f"[create_optimizer] DiT LR split enabled:\n"
                f"  DiT params  (image_generator.*): {dit_params:,} @ lr={self.lr_dit}\n"
                f"  Main params (everything else):   {main_params:,} @ lr={self.args.learning_rate}"
            )

        return self.optimizer

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
                    metrics.log_scalar("world/grad_norm/global", grad_norm, global_step, skip_zero=False)
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

        Produces `world_eval/{task_type}/{component}` curves alongside HF's
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
        try:
            output = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            global_step = self.state.global_step + self.step_offset
            for task_type, bucket in self._eval_task_accumulator.items():
                for component, (s, c) in bucket.items():
                    if c > 0:
                        metrics.log_scalar(
                            f"world_eval/{task_type}/{component}",
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
        if tied:
            tied_params = sum(p.numel() for p in model.text_feature_extractor.wte.parameters())
            print(f"Tied word embeddings: True ({tied_params:,} params shared)")
        print(f"Loss weights: text={self.text_loss_weight}, audio={self.audio_latent_loss_weight}, "
              f"voice={self.voice_latent_loss_weight}, image={self.image_latent_loss_weight}\n")


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
                      getattr(args, 'max_seq_len', None) is not None)
    if needs_override:
        import copy
        from megatransformer.config.world.world_model import WORLD_MODEL_CONFIGS
        from megatransformer.config.common import MegaTransformerBlockConfig
        config = copy.deepcopy(WORLD_MODEL_CONFIGS[args.config])
        if getattr(args, 'iteration_norm', None) is not None:
            config.recurrent_block_config.iteration_norm = args.iteration_norm
        if getattr(args, 'share_block_weights', False):
            config.recurrent_block_config.share_block_weights = True
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
        audio_var_loss_weight=getattr(args, 'audio_var_loss_weight', 1.0),
        voice_var_loss_weight=getattr(args, 'voice_var_loss_weight', 1.0),
        image_var_loss_weight=getattr(args, 'image_var_loss_weight', 1.0),
        audio_var_barrier_weight=getattr(args, 'audio_var_barrier_weight', 0.0),
        voice_var_barrier_weight=getattr(args, 'voice_var_barrier_weight', 0.0),
        image_var_barrier_weight=getattr(args, 'image_var_barrier_weight', 0.0),
        audio_stop_loss_weight=getattr(args, 'audio_stop_loss_weight', 1.0),
        voice_stop_loss_weight=getattr(args, 'voice_stop_loss_weight', 1.0),
        include_text="text" in include_modes,
        include_audio="audio" in include_modes,
        include_voice="voice" in include_modes,
        include_image="image" in include_modes,
        precomputed_latents=args.precomputed_latents,
        text_label_smoothing=args.text_label_smoothing,
        lr_dit=getattr(args, 'lr_dit', None),
        mask_text_loss_in_synthesis=getattr(args, 'mask_text_loss_in_synthesis', False),
        shard_aware_sampler=getattr(args, 'shard_aware_sampler', True),
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

    # DiT loss-weighting override (only applies to DiffusionBridgeImageDecoder)
    sub_parser.add_argument("--min_snr_gamma", type=float, default=None,
                            help="Runtime override for DiT min-SNR gamma. Lowering from the config "
                                 "default (typically 5.0) reduces gradient amplification at the hardest "
                                 "timesteps (t near 1), trading a small amount of hard-timestep focus "
                                 "for stability. Set <= 0 to disable min-SNR weighting entirely.")

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
    sub_parser.add_argument("--backprop_depth", type=int, default=None,
                            help="Override truncated BPTT depth (default: use config, typically 8)")
    sub_parser.add_argument("--block_init_gain", type=float, default=None,
                            help="Override recurrent block xavier init gain (default: 0.02)")
    sub_parser.add_argument("--projection_init_gain", type=float, default=None,
                            help="Override projection xavier init gain (default: use config, typically 1.0)")
    sub_parser.add_argument("--share_block_weights", action="store_true", default=False,
                            help="Share weights across all recurrent blocks (deeper 1-block)")
    sub_parser.add_argument("--iteration_norm", type=str, default=None,
                            choices=["none", "pre_projection", "post_projection"],
                            help="Override per-iteration normalization placement")

    # Max sequence length for text collator
    sub_parser.add_argument("--max_seq_len", type=int, default=2048,
                            help="Maximum token sequence length for text")

    # Visualization callback dependencies
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
    sub_parser.add_argument("--voice_smg_latent_channels", type=int, default=None,
                            help="Override latent_channels for voice SMG (must match what it was trained with)")
    sub_parser.add_argument("--static_speaker_embedding_path", type=str, default=None,
                            help="Path to a .pt file containing a speaker embedding tensor for static-speaker voice decoding")
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
    sub_parser.add_argument("--sive_total_stride", type=int, default=4,
                            help="Total temporal downsampling stride of the SIVE encoder (e.g. 4 for 4x, 3 for 3x)")

    return sub_parser
