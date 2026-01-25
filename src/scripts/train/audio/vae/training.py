import os

import torch


from typing import Any, Optional

from torch.amp import autocast
from torch.nn import functional as F

from model.audio.criteria import MultiResolutionSTFTLoss, MultiScaleMelLoss
from model.audio.vae.criteria import ArcFaceLoss, AudioPerceptualLoss, MultiScaleMelSpectrogramLoss
from model.audio.vae.discriminator import (
    MelDomainCombinedDiscriminator,
    MelDomainMultiPeriodDiscriminator,
    MelDomainMultiScaleDiscriminator,
    MelInstanceNoiseScheduler,
    add_mel_instance_noise,
    compute_mel_discriminator_loss,
    compute_mel_generator_gan_loss,
    r1_mel_gradient_penalty
)
from model.audio.vae.vae import AudioCVAEDecoderOnly, AudioVAE
from model.discriminator import compute_adaptive_weight
from scripts.train.trainer import CommonTrainer
from utils import model_loading_utils


class LearnedSpeakerClassifier(torch.nn.Module):
    """
    Classifier to predict speaker ID from learned speaker embeddings.

    This classifier operates on the learned speaker embeddings.
    This explicitly trains the speaker head to produce speaker-discriminative features.

    Architecture: Simple MLP since input is already a pooled vector [B, embedding_dim]
    """
    def __init__(self, embedding_dim: int, num_speakers: int, hidden_dim: int = 256):
        super().__init__()

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(hidden_dim, num_speakers),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.classifier:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, speaker_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - direct classification (no gradient reversal).

        Args:
            speaker_embedding: [B, embedding_dim] learned speaker embedding from encoder

        Returns:
            [B, num_speakers] logits for speaker classification
        """
        # Handle 3D input [B, 1, D] -> [B, D]
        if speaker_embedding.dim() == 3:
            speaker_embedding = speaker_embedding.squeeze(1)

        return self.classifier(speaker_embedding)


class AudioCVAEGANTrainer(CommonTrainer):
    """
    Custom trainer for VAE with optional GAN training.
    Handles alternating generator/discriminator updates.

    Supports discriminator regularization:
    - Instance noise: adds Gaussian noise to both real and fake mel spectrograms
    - R1 gradient penalty: penalizes gradient norm on real mel spectrograms

    For sharded datasets, uses ShardAwareSampler to minimize shard loading overhead.

    Supports audio perceptual losses:
    - Multi-scale mel loss (works on mel spectrograms directly)
    """

    def __init__(
        self,
        *args,
        cmdline,
        git_commit_hash,
        step_offset: int = 0,
        discriminator: Optional[torch.nn.Module] = None,
        discriminator_optimizer: Optional[torch.optim.Optimizer] = None,
        gan_loss_weight: float = 0.5,
        feature_matching_weight: float = 0.0,
        discriminator_update_frequency: int = 1,
        gan_start_condition_key: Optional[str] = None,
        gan_start_condition_value: Optional[Any] = None,
        # Discriminator regularization
        instance_noise_std: float = 0.0,  # Initial std for instance noise (0 = disabled)
        instance_noise_decay_steps: int = 50000,  # Steps to decay noise to 0
        r1_penalty_weight: float = 0.0,  # Weight for R1 gradient penalty (0 = disabled)
        r1_penalty_interval: int = 16,  # Apply R1 penalty every N steps (expensive)
        # GAN warmup (ramps GAN loss contribution from 0 to full weight)
        gan_warmup_steps: int = 0,  # Steps to ramp up GAN loss (0 = no warmup)
        # Adaptive discriminator weighting (VQGAN-style)
        use_adaptive_weight: bool = False,  # Automatically balance GAN vs reconstruction gradients
        # KL annealing (ramps KL weight from 0 to full over training)
        kl_annealing_steps: int = 0,  # Steps to ramp KL weight from 0 to 1 (0 = disabled)
        # Audio perceptual loss (mel domain and waveform domain if applicable)
        audio_perceptual_loss: Optional[AudioPerceptualLoss] = None,
        audio_perceptual_loss_weight: float = 0.0,
        audio_perceptual_loss_start_step: int = 0,  # Step to start applying perceptual loss
        vocoder: Optional[torch.nn.Module] = None,  # For waveform-based losses
        # FiLM statistics tracking
        log_film_stats: bool = False,  # Whether to log FiLM scale/shift statistics
        # FiLM contrastive loss - encourages different speaker embeddings to produce different FiLM outputs
        film_contrastive_loss_weight: float = 0.0,  # Weight for FiLM contrastive loss (0 = disabled)
        film_contrastive_loss_start_step: int = 0,  # Step to start FiLM contrastive loss
        film_contrastive_margin_max: float = 0.1,  # Max margin value for hinge loss
        film_contrastive_margin_rampup_steps: int = 5000,  # Steps to ramp margin from 0 to max
        # Mu-only reconstruction loss (for diffusion compatibility)
        mu_only_recon_weight: float = 0.0,  # Weight for mu-only reconstruction loss (0 = disabled)
        learned_speaker_classifier: Optional[torch.nn.Module] = None,
        learned_speaker_classifier_optimizer: Optional[torch.optim.Optimizer] = None,
        speaker_id_loss_weight: float = 0.0,  # Weight for speaker ID loss (0 = disabled)
        speaker_id_loss_type: str = "arcface",  # "classifier" or "arcface"
        speaker_id_loss_start_step: int = 0,  # Step to start speaker ID loss (0 = from beginning)
        speaker_id_loss_rampup_steps: int = 0,  # Steps to ramp weight from 0 to max (0 = no rampup)
        # F0 predictor pretraining settings
        f0_predictor_freeze_steps: int = 0,  # Steps to keep F0 predictor frozen (0 = no freezing)
        f0_warmup_use_gt_steps: int = 0,  # Steps to use GT F0 instead of predicted (0 = disabled)
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Store shard-aware sampler if available
        self._shard_sampler = None
        if hasattr(self.train_dataset, 'get_sampler'):
            self._shard_sampler = self.train_dataset.get_sampler(shuffle=True, seed=42)
            print("Using ShardAwareSampler for efficient shard loading")
        self.writer = None

        self.step_offset = step_offset if step_offset is not None else 0
        self.cmdline = cmdline
        self.git_commit_hash = git_commit_hash

        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.gan_loss_weight = gan_loss_weight
        self.feature_matching_weight = feature_matching_weight
        self.discriminator_update_frequency = discriminator_update_frequency
        self.gan_start_condition_key = gan_start_condition_key
        self.gan_start_condition_value = gan_start_condition_value
        self.gan_already_started = False
        self.gan_start_step = None  # Track when GAN training started (for warmup)
        self.gan_warmup_steps = gan_warmup_steps

        # Adaptive discriminator weighting (VQGAN-style)
        self.use_adaptive_weight = use_adaptive_weight

        # KL annealing settings
        self.kl_annealing_steps = kl_annealing_steps

        # Discriminator regularization settings
        self.r1_penalty_weight = r1_penalty_weight
        self.r1_penalty_interval = r1_penalty_interval

        # Audio perceptual loss settings
        self.audio_perceptual_loss = audio_perceptual_loss
        self.audio_perceptual_loss_weight = audio_perceptual_loss_weight
        self.audio_perceptual_loss_start_step = audio_perceptual_loss_start_step
        self.vocoder = vocoder

        # Instance noise scheduler (decays over training)
        self.noise_scheduler = None
        if instance_noise_std > 0:
            self.noise_scheduler = MelInstanceNoiseScheduler(
                initial_std=instance_noise_std,
                final_std=0.0,
                decay_steps=instance_noise_decay_steps,
                decay_type="linear",
            )

        # GradScaler for discriminator when using mixed precision
        # The Trainer has its own scaler for the main model, but discriminator needs separate one
        self.discriminator_scaler = None
        if discriminator is not None:
            self.discriminator_scaler = torch.amp.GradScaler(enabled=False)  # Will be enabled in compute_loss

        # FiLM statistics tracking
        self.log_film_stats = log_film_stats

        # FiLM contrastive loss settings
        self.film_contrastive_loss_weight = film_contrastive_loss_weight
        self.film_contrastive_loss_start_step = film_contrastive_loss_start_step
        self.film_contrastive_margin_max = film_contrastive_margin_max
        self.film_contrastive_margin_rampup_steps = film_contrastive_margin_rampup_steps

        # Mu-only reconstruction loss (for diffusion compatibility)
        self.mu_only_recon_weight = mu_only_recon_weight

        # Learned speaker embedding classification
        self.learned_speaker_classifier = learned_speaker_classifier
        self.learned_speaker_classifier_optimizer = learned_speaker_classifier_optimizer
        self.speaker_id_loss_weight = speaker_id_loss_weight
        self.speaker_id_loss_type = speaker_id_loss_type  # "classifier" or "arcface"
        self.speaker_id_loss_start_step = speaker_id_loss_start_step
        self.speaker_id_loss_rampup_steps = speaker_id_loss_rampup_steps
        self.speaker_id_training_started = False  # Track when training has started (for checkpointing)

        # F0 predictor pretraining settings
        self.f0_predictor_freeze_steps = f0_predictor_freeze_steps
        self.f0_warmup_use_gt_steps = f0_warmup_use_gt_steps

        self.has_logged_cli = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        global_step = self.state.global_step + self.step_offset

        self._ensure_tensorboard_writer()

        # gets reset any time training is resumed; it can be assumed that the cli changed, so log at the step value it was resumed from
        if not self.has_logged_cli and self.writer is not None:
            self.writer.add_text("training/command_line", self.cmdline, global_step)
            self.writer.add_text("training/git_commit_hash", self.git_commit_hash, global_step)
            self.has_logged_cli = True

        # Unfreeze F0 predictor after freeze_steps
        # Handle wrapped models (DeepSpeed, DDP)
        unwrapped_model = model.module if hasattr(model, 'module') else model
        if (self.f0_predictor_freeze_steps > 0 and
            global_step >= self.f0_predictor_freeze_steps and
            hasattr(unwrapped_model, 'f0_predictor') and unwrapped_model.f0_predictor is not None):
            # Check if still frozen (first param's requires_grad)
            first_param = next(unwrapped_model.f0_predictor.parameters(), None)
            if first_param is not None and not first_param.requires_grad:
                print(f"Step {global_step}: Unfreezing F0 predictor")
                for param in unwrapped_model.f0_predictor.parameters():
                    param.requires_grad = True

        features = inputs.get("features", None)
        mel_spec = inputs["mel_specs"]
        mel_spec_masks = inputs.get("mel_spec_masks", None)
        mel_spec_lengths = inputs.get("mel_lengths", None)
        speaker_embedding = inputs.get("speaker_embedding", None)
        target_f0 = inputs.get("target_f0", None)
        target_voiced = inputs.get("target_voiced", None)

        # print(f"Mask length: {torch.cumprod(mel_spec_masks, dim=-1).sum(dim=-1)}")

        # Compute KL weight multiplier for KL annealing (ramps from 0 to 1)
        kl_weight_multiplier = 1.0
        if self.kl_annealing_steps > 0:
            kl_weight_multiplier = min(1.0, global_step / self.kl_annealing_steps)

        # Determine if we should use GT F0 for warmup
        # This lets the F0 embedding learn with clean signal before relying on predictor
        use_gt_f0 = (self.f0_warmup_use_gt_steps > 0 and
                     global_step < self.f0_warmup_use_gt_steps and
                     target_f0 is not None and target_voiced is not None)

        is_vae = hasattr(model, "encoder") and model.encoder is not None

        if is_vae:
            # Forward pass through VAE model (with optional mask for reconstruction loss and lengths for attention)
            # Request FiLM stats if logging is enabled
            recon, mu, logvar, losses = model(
                features=features,
                target=mel_spec,
                mask=mel_spec_masks,
                speaker_embedding=speaker_embedding,
                length=mel_spec_lengths,
                kl_weight_multiplier=kl_weight_multiplier,
                return_film_stats=self.log_film_stats,
                target_f0=target_f0,
                target_voiced=target_voiced,
                use_gt_f0=use_gt_f0,
            )
        else:
            recon, losses = model(
                features=features,
                target=mel_spec,
                mask=mel_spec_masks,
                speaker_embedding=speaker_embedding,
                return_film_stats=self.log_film_stats,
                target_f0=target_f0,
                target_voiced=target_voiced,
                use_gt_f0=use_gt_f0,
            )
            mu = None
            logvar = None

        # Extract FiLM stats if available
        film_stats = losses.pop("film_stats", None) if self.log_film_stats else None

        # Determine which speaker embedding to use for decode operations:
        # - If model learns speaker embedding, use the learned embedding from encoder output
        # - Otherwise, use the pretrained speaker embedding from dataset
        learned_speaker_embedding = losses.get("learned_speaker_embedding", None)
        decode_speaker_embedding = learned_speaker_embedding if learned_speaker_embedding is not None else speaker_embedding

        # Get VAE reconstruction loss
        vae_loss = losses["total_loss"]

        # Mu-only reconstruction loss (trains decoder to produce good outputs from mu directly)
        # This ensures diffusion-generated latents decode well without needing reparameterization noise
        mu_only_recon_loss = torch.tensor(0.0, device=mel_spec.device)
        if mu is not None and self.mu_only_recon_weight > 0:
            # Decode mu directly (no reparameterization noise)
            recon_mu_only = model.decode(mu.detach(), speaker_embedding=decode_speaker_embedding, features=features)[..., :mel_spec.shape[-1]]

            # Compute masked loss if mask is available (prevents optimizing padded regions)
            if mel_spec_masks is not None:
                # Expand mask to match mel_spec shape: [B, T] -> [B, 1, 1, T]
                mask_expanded = mel_spec_masks.unsqueeze(1).unsqueeze(2)
                # Count valid elements per sample: sum(mask) * C * n_mels
                valid_elements = mask_expanded.sum() * mel_spec.shape[1] * mel_spec.shape[2]

                # Masked MSE loss
                mel_length = recon_mu_only.size(-1)
                if hasattr(model, 'mse_loss_weight') and model.mse_loss_weight > 0:
                    masked_squared_error = ((recon_mu_only - mel_spec[..., :mel_length]) ** 2) * mask_expanded[..., :mel_length]
                    masked_mse = masked_squared_error.sum() / (valid_elements + 1e-5)
                    mu_only_recon_loss = mu_only_recon_loss + model.mse_loss_weight * masked_mse

                # Masked L1 loss
                if hasattr(model, 'l1_loss_weight') and model.l1_loss_weight > 0:
                    masked_abs_error = torch.abs(recon_mu_only - mel_spec[..., :mel_length]) * mask_expanded[..., :mel_length]
                    masked_l1 = masked_abs_error.sum() / (valid_elements + 1e-5)
                    mu_only_recon_loss = mu_only_recon_loss + model.l1_loss_weight * masked_l1

                # Fallback to masked MSE if no weights defined
                if mu_only_recon_loss.item() == 0.0:
                    masked_squared_error = ((recon_mu_only - mel_spec[..., :mel_length]) ** 2) * mask_expanded[..., :mel_length]
                    mu_only_recon_loss = masked_squared_error.sum() / (valid_elements + 1e-5)
            else:
                # No mask available - use standard unmasked losses
                if hasattr(model, 'mse_loss_weight') and model.mse_loss_weight > 0:
                    mu_only_recon_loss = mu_only_recon_loss + model.mse_loss_weight * F.mse_loss(recon_mu_only, mel_spec)
                if hasattr(model, 'l1_loss_weight') and model.l1_loss_weight > 0:
                    mu_only_recon_loss = mu_only_recon_loss + model.l1_loss_weight * F.l1_loss(recon_mu_only, mel_spec)
                # Fallback to MSE if no weights defined
                if mu_only_recon_loss.item() == 0.0:
                    mu_only_recon_loss = F.mse_loss(recon_mu_only, mel_spec)

            vae_loss = vae_loss + self.mu_only_recon_weight * mu_only_recon_loss
            losses["mu_only_recon_loss"] = mu_only_recon_loss

        # GAN losses (only after condition is met)
        g_gan_loss = torch.tensor(0.0, device=mel_spec.device)
        d_loss = torch.tensor(0.0, device=mel_spec.device)

        if self.is_gan_enabled(global_step, vae_loss):
            if not self.gan_already_started:
                # First step of GAN training - record start step for warmup
                self.gan_start_step = global_step
                print(f"GAN training starting at step {global_step}")
            self.gan_already_started = True

            # Compute GAN warmup factor (ramps from 0 to 1 over gan_warmup_steps)
            gan_warmup_factor = 1.0
            if self.gan_warmup_steps > 0 and self.gan_start_step is not None:
                steps_since_gan_start = global_step - self.gan_start_step
                gan_warmup_factor = min(1.0, steps_since_gan_start / self.gan_warmup_steps)

            # Get current instance noise std (decays over training)
            noise_std = 0.0
            if self.noise_scheduler is not None:
                noise_std = self.noise_scheduler.get_std(global_step)

            # Discriminator Update
            if global_step % self.discriminator_update_frequency == 0:
                self.discriminator.train()

                # Ensure discriminator is on the same device as inputs
                if next(self.discriminator.parameters()).device != mel_spec.device:
                    self.discriminator = self.discriminator.to(mel_spec.device)

                # Apply instance noise to both real and fake mel spectrograms
                real_for_disc = add_mel_instance_noise(mel_spec, noise_std) if noise_std > 0 else mel_spec
                fake_for_disc = add_mel_instance_noise(recon.detach(), noise_std) if noise_std > 0 else recon.detach()

                # Compute discriminator loss in fp32 to avoid gradient underflow
                # Mixed precision can cause discriminator gradients to vanish
                with autocast(mel_spec.device.type, dtype=torch.float32, enabled=False):
                    # Cast inputs to fp32 for discriminator
                    # Add channel dimension: [B, n_mels, T] -> [B, 1, n_mels, T]
                    # Discriminator expects [B, 1, n_mels, T] format
                    real_fp32 = real_for_disc.float()
                    fake_fp32 = fake_for_disc.float()
                    if real_fp32.dim() == 3:
                        real_fp32 = real_fp32.unsqueeze(1)
                    if fake_fp32.dim() == 3:
                        fake_fp32 = fake_fp32.unsqueeze(1)

                    d_loss, d_loss_dict = compute_mel_discriminator_loss(
                        self.discriminator,
                        real_mels=real_fp32,
                        fake_mels=fake_fp32,
                    )

                # R1 gradient penalty (on clean real mels, not noisy)
                r1_loss = torch.tensor(0.0, device=mel_spec.device)
                if self.r1_penalty_weight > 0 and global_step % self.r1_penalty_interval == 0:
                    # Add channel dimension for discriminator: [B, n_mels, T] -> [B, 1, n_mels, T]
                    mel_spec_4d = mel_spec.float().unsqueeze(1) if mel_spec.dim() == 3 else mel_spec.float()
                    r1_loss = r1_mel_gradient_penalty(mel_spec_4d, self.discriminator)
                    d_loss = d_loss + self.r1_penalty_weight * r1_loss

                # Log discriminator diagnostics
                if global_step % self.args.logging_steps == 0 and self.writer is not None:
                    for key, val in d_loss_dict.items():
                        self._log_scalar(f"train/{key}", val, global_step)
                    if self.r1_penalty_weight > 0:
                        self._log_scalar("train/r1_penalty", r1_loss, global_step)
                    if noise_std > 0:
                        self._log_scalar("train/instance_noise_std", noise_std, global_step)

                    # Log how different real and fake mel spectrograms are
                    with torch.no_grad():
                        real_fake_mse = torch.nn.functional.mse_loss(mel_spec, recon).item()
                        real_fake_l1 = torch.nn.functional.l1_loss(mel_spec, recon).item()
                        self._log_scalar("train/real_fake_mse", real_fake_mse, global_step)
                        self._log_scalar("train/real_fake_l1", real_fake_l1, global_step)

                # Update discriminator (only during training when gradients are enabled)
                if self.discriminator_optimizer is not None and self.discriminator.training and torch.is_grad_enabled():
                    self.discriminator_optimizer.zero_grad()
                    d_loss.backward()

                    # Log gradient statistics to diagnose training issues
                    if global_step % self.args.logging_steps == 0 and self.writer is not None:
                        total_grad_norm = 0.0
                        for p in self.discriminator.parameters():
                            if p.grad is not None:
                                total_grad_norm += p.grad.norm().item() ** 2
                        total_grad_norm = total_grad_norm ** 0.5
                        self._log_scalar("train/d_grad_norm", total_grad_norm, global_step)

                    self.discriminator_optimizer.step()

            # Generator GAN Loss
            device_type = mel_spec.device.type
            dtype = torch.bfloat16 if self.args.bf16 else torch.float16 if self.args.fp16 else torch.float32
            with autocast(device_type, dtype=dtype, enabled=self.args.fp16 or self.args.bf16):
                # Add channel dimension for discriminator: [B, n_mels, T] -> [B, 1, n_mels, T]
                mel_spec_for_gen = mel_spec.unsqueeze(1) if mel_spec.dim() == 3 else mel_spec
                recon_for_gen = recon.unsqueeze(1) if recon.dim() == 3 else recon
                g_gan_loss, g_loss_dict = compute_mel_generator_gan_loss(
                    self.discriminator,
                    real_mels=mel_spec_for_gen,
                    fake_mels=recon_for_gen,
                    feature_matching_weight=self.feature_matching_weight,
                )

            if global_step % self.args.logging_steps == 0 and self.writer is not None:
                for key, val in g_loss_dict.items():
                    self._log_scalar(f"train/{key}", val, global_step)
                # Log warmup factor
                self._log_scalar("train/gan_warmup_factor", gan_warmup_factor, global_step)

            # Apply warmup factor to GAN loss
            g_gan_loss = gan_warmup_factor * g_gan_loss

        # Total generator loss with optional adaptive weighting
        adaptive_weight = torch.tensor(self.gan_loss_weight, device=mel_spec.device)
        if self.use_adaptive_weight and self.gan_already_started and g_gan_loss.requires_grad:
            # VQGAN-style adaptive weighting: balance GAN and reconstruction gradients
            # Uses the last decoder layer as proxy for decoder output gradients
            try:
                last_layer = model.decoder.final_conv.weight
                adaptive_weight = compute_adaptive_weight(
                    vae_loss, g_gan_loss, last_layer, self.gan_loss_weight
                )
            except (AttributeError, RuntimeError) as e:
                # Fallback to fixed weight if adaptive weight fails
                # (e.g., model doesn't have expected structure, or grad computation fails)
                if global_step % self.args.logging_steps == 0:
                    print(f"Warning: adaptive weight computation failed ({e}), using fixed weight")
                adaptive_weight = torch.tensor(self.gan_loss_weight, device=mel_spec.device)

        total_loss = vae_loss + adaptive_weight * g_gan_loss

        # Audio perceptual loss (requires vocoder for waveform-based losses)
        # Only apply after audio_perceptual_loss_start_step to let L1/MSE settle first
        audio_perceptual_loss_value = torch.tensor(0.0, device=mel_spec.device)
        audio_perceptual_losses = {}
        perceptual_loss_enabled = (
            self.audio_perceptual_loss is not None
            and self.audio_perceptual_loss_weight > 0
            and global_step >= self.audio_perceptual_loss_start_step
        )
        if perceptual_loss_enabled:
            # Get waveforms if vocoder is available
            pred_waveform = None
            target_waveform = None
            if self.vocoder is not None:
                with torch.no_grad():
                    # Vocoder expects [B, n_mels, T] - recon is already in that format
                    vocoder_outputs = self.vocoder(recon.float())
                    if isinstance(vocoder_outputs, dict):
                        pred_waveform = vocoder_outputs["pred_waveform"]
                    else:
                        pred_waveform = vocoder_outputs

                    target_vocoder_outputs = self.vocoder(mel_spec.float())
                    if isinstance(target_vocoder_outputs, dict):
                        target_waveform = target_vocoder_outputs["pred_waveform"]
                    else:
                        target_waveform = target_vocoder_outputs

            # Compute audio perceptual losses
            # Mel spec is [B, 1, n_mels, T], squeeze channel dim for multi-scale mel loss
            audio_perceptual_losses = self.audio_perceptual_loss(
                pred_mel=recon.squeeze(1),  # [B, n_mels, T]
                target_mel=mel_spec.squeeze(1),  # [B, n_mels, T]
                target_speaker_embedding=speaker_embedding,
                pred_waveform=pred_waveform,
                target_waveform=target_waveform,
                mask=mel_spec_masks,  # [B, T] mask to exclude padded regions
            )
            audio_perceptual_loss_value = audio_perceptual_losses.get("total_perceptual_loss", torch.tensor(0.0, device=mel_spec.device))
            total_loss = total_loss + self.audio_perceptual_loss_weight * audio_perceptual_loss_value

        # Learned speaker embedding classification loss
        # Uses direct gradients (no reversal) to train the speaker head to be discriminative
        speaker_id_loss = torch.tensor(0.0, device=mel_spec.device)
        speaker_id_acc = 0.0
        speaker_id_enabled = (
            self.learned_speaker_classifier is not None
            and self.speaker_id_loss_weight > 0
            and learned_speaker_embedding is not None  # Only works with learned speaker embeddings
        )
        if speaker_id_enabled:
            # Get speaker IDs from batch
            speaker_ids = inputs.get("speaker_ids", None)
            if speaker_ids is not None and not isinstance(speaker_ids, torch.Tensor):
                speaker_ids = torch.tensor(speaker_ids, device=mel_spec.device, dtype=torch.long)

            if speaker_ids is not None:
                # Mark training as started (for checkpoint saving)
                if not self.speaker_id_training_started:
                    # Get num_speakers depending on loss type
                    if self.speaker_id_loss_type == "arcface":
                        num_speakers = self.learned_speaker_classifier.num_speakers
                    else:
                        num_speakers = self.learned_speaker_classifier.classifier[-1].out_features
                    print(f"Learned speaker ID classification starting at step {global_step}")
                    print(f"  loss_type: {self.speaker_id_loss_type}")
                    print(f"  speaker_ids: min={speaker_ids.min().item()}, max={speaker_ids.max().item()}, "
                          f"unique={len(torch.unique(speaker_ids))}, batch_size={len(speaker_ids)}")
                    print(f"  classifier num_speakers: {num_speakers}")
                    print(f"  learned_speaker_embedding shape: {learned_speaker_embedding.shape}")

                    # Validate speaker_ids are in valid range
                    if speaker_ids.min() < 0:
                        raise ValueError(f"speaker_ids contains negative values (min={speaker_ids.min().item()})")
                    if speaker_ids.max() >= num_speakers:
                        raise ValueError(
                            f"speaker_ids max ({speaker_ids.max().item()}) >= num_speakers ({num_speakers}). "
                            f"Either increase --num_speakers or check if speaker_ids are 1-indexed "
                            f"(should be 0-indexed for cross_entropy)."
                        )
                    self.speaker_id_training_started = True

                # Ensure classifier is on same device
                if next(self.learned_speaker_classifier.parameters()).device != mel_spec.device:
                    self.learned_speaker_classifier.to(mel_spec.device)

                if self.speaker_id_loss_type == "arcface":
                    # ArcFace loss: returns (loss, logits, accuracy) tuple
                    # Update ArcFace weights FIRST with detached embeddings
                    if self.learned_speaker_classifier_optimizer is not None and torch.is_grad_enabled():
                        self.learned_speaker_classifier_optimizer.zero_grad()
                        classifier_loss, _, _ = self.learned_speaker_classifier(
                            learned_speaker_embedding.detach(), speaker_ids
                        )
                        classifier_loss.backward()
                        self.learned_speaker_classifier_optimizer.step()

                    # Forward pass with gradients flowing to encoder
                    speaker_id_loss, speaker_id_logits, speaker_id_acc = self.learned_speaker_classifier(
                        learned_speaker_embedding, speaker_ids
                    )
                else:
                    # Standard classifier: returns logits only
                    # Update classifier FIRST with detached embeddings (train classifier to recognize speakers)
                    if self.learned_speaker_classifier_optimizer is not None and torch.is_grad_enabled():
                        self.learned_speaker_classifier_optimizer.zero_grad()
                        classifier_logits = self.learned_speaker_classifier(learned_speaker_embedding.detach())
                        classifier_loss = F.cross_entropy(classifier_logits, speaker_ids)
                        classifier_loss.backward()
                        self.learned_speaker_classifier_optimizer.step()

                    # Forward pass with gradients flowing to encoder (train speaker head to produce discriminative embeddings)
                    speaker_id_logits = self.learned_speaker_classifier(learned_speaker_embedding)
                    speaker_id_loss = F.cross_entropy(speaker_id_logits, speaker_ids)

                    # Compute accuracy for logging
                    with torch.no_grad():
                        speaker_id_preds = speaker_id_logits.argmax(dim=-1)
                        speaker_id_acc = (speaker_id_preds == speaker_ids).float().mean().item()

                # Compute effective weight with ramping
                ramp_progress = 1.0  # Default: fully ramped
                if self.speaker_id_loss_start_step > 0 and global_step < self.speaker_id_loss_start_step:
                    # Haven't reached start step yet
                    effective_speaker_id_weight = 0.0
                    ramp_progress = 0.0
                elif self.speaker_id_loss_rampup_steps > 0:
                    # Ramp from 0 to max over rampup_steps
                    steps_since_start = max(0, global_step - self.speaker_id_loss_start_step)
                    ramp_progress = min(1.0, steps_since_start / self.speaker_id_loss_rampup_steps)
                    effective_speaker_id_weight = self.speaker_id_loss_weight * ramp_progress
                else:
                    # No ramping
                    effective_speaker_id_weight = self.speaker_id_loss_weight

                # Add to total loss (direct gradients to encoder's speaker head)
                total_loss = total_loss + effective_speaker_id_weight * speaker_id_loss

                # Log speaker ID metrics
                if global_step % self.args.logging_steps == 0 and self.writer is not None:
                    prefix = "train/" if model.training else "eval/"
                    self._log_scalar(f"{prefix}speaker_id/loss", speaker_id_loss, global_step)
                    self._log_scalar(f"{prefix}speaker_id/accuracy", speaker_id_acc, global_step, skip_zero=False)
                    self._log_scalar(f"{prefix}speaker_id/weighted_loss", effective_speaker_id_weight * speaker_id_loss, global_step)
                    self._log_scalar(f"{prefix}speaker_id/effective_weight", effective_speaker_id_weight, global_step)
                    self._log_scalar(f"{prefix}speaker_id/ramp_progress", ramp_progress, global_step, skip_zero=False)

        # FiLM contrastive loss - encourages different speaker embeddings to produce different outputs
        # This penalizes the decoder for ignoring speaker embeddings
        # Key insight: weight loss by embedding similarity so we don't penalize same/similar speakers
        film_contrastive_loss = torch.tensor(0.0, device=mel_spec.device)

        # Use decode_speaker_embedding (learned or pretrained, extracted earlier)
        film_contrastive_enabled = (
            self.film_contrastive_loss_weight > 0
            and global_step >= self.film_contrastive_loss_start_step
            and decode_speaker_embedding is not None
            and decode_speaker_embedding.shape[0] > 1  # Need at least 2 samples for shuffling
        )

        if film_contrastive_enabled:
            # Compute margin alpha (ramps from 0 to film_contrastive_margin_max over rampup_steps)
            steps_since_start = global_step - self.film_contrastive_loss_start_step
            film_contrastive_margin_alpha = min(1.0, steps_since_start / max(1, self.film_contrastive_margin_rampup_steps))
            margin = self.film_contrastive_margin_max * film_contrastive_margin_alpha

            # Shuffle speaker embeddings to create mismatched (audio, wrong_speaker) pairs
            batch_size = decode_speaker_embedding.shape[0]
            perm = torch.randperm(batch_size, device=decode_speaker_embedding.device)
            # Ensure no sample maps to itself (for valid pairs)
            same_indices = (perm == torch.arange(batch_size, device=perm.device))
            if same_indices.any():
                # Shift indices that map to themselves
                perm[same_indices] = (perm[same_indices] + 1) % batch_size
            shuffled_speaker_embedding = decode_speaker_embedding[perm]

            # Compute embedding similarity to weight the loss
            # Flatten to [B, D] for cosine similarity
            emb_flat = decode_speaker_embedding.squeeze(1) if decode_speaker_embedding.dim() == 3 else decode_speaker_embedding
            shuffled_emb_flat = shuffled_speaker_embedding.squeeze(1) if shuffled_speaker_embedding.dim() == 3 else shuffled_speaker_embedding
            emb_similarity = F.cosine_similarity(emb_flat, shuffled_emb_flat, dim=-1)  # [B]

            # Weight loss by how different the embeddings are:
            # - Same speaker (sim ≈ 1) → weight ≈ 0 → no penalty (correct behavior)
            # - Similar speakers (sim ≈ 0.8) → weight ≈ 0.2 → small penalty
            # - Very different speakers (sim ≈ 0.3) → weight ≈ 0.7 → stronger penalty
            emb_diff_weight = (1.0 - emb_similarity).clamp(0, 1)  # [B]

            # Decode with shuffled speaker embeddings (detach mu to only train decoder's FiLM)
            # Pass features for F0 prediction if enabled
            if is_vae:
                recon_shuffled = model.decode(mu.detach(), speaker_embedding=shuffled_speaker_embedding, features=features)
            else:
                recon_shuffled = model.decode(features, speaker_embedding=shuffled_speaker_embedding, features=features)

            # Handle 2D decoder output: [B, 1, 80, T] -> [B, 80, T]
            if recon_shuffled.dim() == 4 and recon_shuffled.shape[1] == 1:
                recon_shuffled = recon_shuffled.squeeze(1)

            # Compute per-sample output difference
            # recon shape: [B, 1, n_mels, T] or [B, n_mels, T]
            # Truncate to min time dim (decoder conv stack can produce slightly different lengths)
            min_time = min(recon.shape[-1], recon_shuffled.shape[-1])
            recon_truncated = recon[..., :min_time]
            recon_shuffled_truncated = recon_shuffled[..., :min_time]
            output_diff_per_sample = (recon_truncated - recon_shuffled_truncated).pow(2).mean(dim=list(range(1, recon.dim())))  # [B]

            # Hinge loss: want output_diff > margin for different speakers
            # Weighted by embedding difference so same/similar speakers aren't penalized
            per_sample_loss = emb_diff_weight * F.relu(margin - output_diff_per_sample)
            film_contrastive_loss = per_sample_loss.mean()

            total_loss = total_loss + self.film_contrastive_loss_weight * film_contrastive_loss

            # Log FiLM contrastive metrics
            if global_step % self.args.logging_steps == 0 and self.writer is not None:
                prefix = "train/" if model.training else "eval/"
                # Use skip_zero=False for metrics that start at 0 due to margin rampup
                self._log_scalar(f"{prefix}film_contrastive/loss", film_contrastive_loss, global_step, skip_zero=False)
                self._log_scalar(f"{prefix}film_contrastive/output_diff_mean", output_diff_per_sample.mean(), global_step)
                self._log_scalar(f"{prefix}film_contrastive/emb_similarity_mean", emb_similarity.mean(), global_step)
                self._log_scalar(f"{prefix}film_contrastive/emb_diff_weight_mean", emb_diff_weight.mean(), global_step)
                self._log_scalar(f"{prefix}film_contrastive/margin", margin, global_step, skip_zero=False)
                self._log_scalar(f"{prefix}film_contrastive/margin_alpha", film_contrastive_margin_alpha, global_step, skip_zero=False)
                self._log_scalar(f"{prefix}film_contrastive/weighted_loss",
                               self.film_contrastive_loss_weight * film_contrastive_loss, global_step, skip_zero=False)

        # Log losses (skip non-loss values like learned_speaker_embedding)
        if global_step % self.args.logging_steps == 0 and self.writer is not None:
            prefix = "train/" if model.training else "eval/"
            for loss_name, loss in losses.items():
                # Skip non-scalar tensors (e.g., learned_speaker_embedding)
                if isinstance(loss, torch.Tensor) and loss.numel() > 1 and not loss_name.endswith("_loss"):
                    continue
                self._log_scalar(f"{prefix}vae_{loss_name}", loss.mean() if isinstance(loss, torch.Tensor) else loss, global_step)
            # Log mu and logvar stats
            if mu is not None and logvar is not None:
                self._log_scalar(f"{prefix}vae_mu_mean", mu.mean(), global_step)
                self._log_scalar(f"{prefix}vae_mu_std", mu.std(), global_step)
                self._log_scalar(f"{prefix}vae_logvar_mean", logvar.mean(), global_step)
                # Mean variance (what diffusion will see) - useful for setting latent_std
                self._log_scalar(f"{prefix}vae_mean_variance", logvar.exp().mean(), global_step)
                self._log_scalar(f"{prefix}vae_mean_std", logvar.exp().mean().sqrt(), global_step)

            self._log_scalar(f"{prefix}g_gan_loss", g_gan_loss, global_step)
            self._log_scalar(f"{prefix}total_loss", total_loss.mean(), global_step)
            # Log adaptive weight when using adaptive weighting
            if self.use_adaptive_weight and self.gan_already_started:
                self._log_scalar(f"{prefix}adaptive_gan_weight", adaptive_weight, global_step)
            # Log KL weight multiplier when annealing is enabled
            if self.kl_annealing_steps > 0 and mu is not None and logvar is not None:
                self._log_scalar(f"{prefix}kl_weight_multiplier", kl_weight_multiplier, global_step)

            # Per-channel latent statistics (for detecting channel collapse)
            # mu shape: [B, C, T] - compute stats per channel (average over batch, mel, time)
            if mu is not None and logvar is not None:
                per_channel_mu_mean = mu.mean(dim=(0, 2))  # [C]
                per_channel_mu_std = mu.std(dim=(0, 2))  # [C]
                per_channel_var = logvar.exp().mean(dim=(0, 2))  # [C]
                # Per-channel KL: 0.5 * (mu^2 + var - log(var) - 1), averaged over batch and spatial
                per_channel_kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).mean(dim=(0, 2))  # [C]

                for c in range(mu.shape[1]):
                    self._log_scalar(f"{prefix}channel_{c}/mu_mean", per_channel_mu_mean[c], global_step)
                    self._log_scalar(f"{prefix}channel_{c}/mu_std", per_channel_mu_std[c], global_step)
                    self._log_scalar(f"{prefix}channel_{c}/variance", per_channel_var[c], global_step)
                    self._log_scalar(f"{prefix}channel_{c}/kl", per_channel_kl[c], global_step)

            # Log audio perceptual losses
            if audio_perceptual_losses:
                for loss_name, loss_val in audio_perceptual_losses.items():
                    self._log_scalar(f"{prefix}audio_perceptual/{loss_name}", loss_val, global_step)
                self._log_scalar(f"{prefix}audio_perceptual_weighted", self.audio_perceptual_loss_weight * audio_perceptual_loss_value, global_step)

            # Log waveform-domain losses
            if waveform_loss_enabled:
                self._log_scalar(f"{prefix}waveform/stft_loss", waveform_stft_loss_value, global_step)
                self._log_scalar(f"{prefix}waveform/stft_loss_weighted", self.waveform_stft_loss_weight * waveform_stft_loss_value, global_step)
                self._log_scalar(f"{prefix}waveform/mel_loss", waveform_mel_loss_value, global_step)
                self._log_scalar(f"{prefix}waveform/mel_loss_weighted", self.waveform_mel_loss_weight * waveform_mel_loss_value, global_step)

            # Log speaker embedding statistics (learned or pretrained, whichever is used for decoding)
            if decode_speaker_embedding is not None:
                # Flatten to [B, D] if needed
                spk_emb = decode_speaker_embedding.squeeze(1) if decode_speaker_embedding.dim() == 3 else decode_speaker_embedding
                self._log_scalar(f"{prefix}speaker_emb/mean", spk_emb.mean(), global_step)
                self._log_scalar(f"{prefix}speaker_emb/std", spk_emb.std(), global_step)
                # L2 norm per sample, then average
                l2_norms = torch.norm(spk_emb, p=2, dim=-1)
                self._log_scalar(f"{prefix}speaker_emb/l2_norm_mean", l2_norms.mean(), global_step)
                self._log_scalar(f"{prefix}speaker_emb/l2_norm_min", l2_norms.min(), global_step)
                self._log_scalar(f"{prefix}speaker_emb/l2_norm_max", l2_norms.max(), global_step)
                # Log whether using learned embedding
                if learned_speaker_embedding is not None:
                    self._log_scalar(f"{prefix}speaker_emb/is_learned", 1.0, global_step)

                    # Log within-speaker vs between-speaker similarity (measures embedding separability)
                    speaker_ids = inputs.get("speaker_ids", None)
                    if speaker_ids is not None:
                        if not isinstance(speaker_ids, torch.Tensor):
                            speaker_ids = torch.tensor(speaker_ids, device=mel_spec.device, dtype=torch.long)

                        # Normalize embeddings for cosine similarity
                        emb_flat = spk_emb  # Already [B, D] from above
                        emb_norm = F.normalize(emb_flat, dim=-1)

                        # Compute similarity matrix [B, B]
                        sim_matrix = emb_norm @ emb_norm.T

                        # Create masks for same-speaker and different-speaker pairs
                        batch_size = speaker_ids.shape[0]
                        same_speaker_mask = speaker_ids.unsqueeze(0) == speaker_ids.unsqueeze(1)  # [B, B]
                        eye_mask = torch.eye(batch_size, dtype=torch.bool, device=same_speaker_mask.device)

                        # Within-speaker: same speaker, excluding self-similarity (diagonal)
                        within_mask = same_speaker_mask & ~eye_mask
                        # Between-speaker: different speakers
                        between_mask = ~same_speaker_mask

                        if within_mask.any():
                            within_sim = sim_matrix[within_mask].mean()
                            self._log_scalar(f"{prefix}speaker_emb/within_speaker_sim", within_sim, global_step)

                        if between_mask.any():
                            between_sim = sim_matrix[between_mask].mean()
                            self._log_scalar(f"{prefix}speaker_emb/between_speaker_sim", between_sim, global_step)

                        if within_mask.any() and between_mask.any():
                            sim_margin = within_sim - between_sim
                            self._log_scalar(f"{prefix}speaker_emb/sim_margin", sim_margin, global_step)

                        # === Debug metrics for diagnosing embedding collapse ===

                        # 1. All-pairs similarity statistics (excluding self-similarity)
                        off_diag_mask = ~eye_mask
                        all_pairs_sim = sim_matrix[off_diag_mask]
                        self._log_scalar(f"{prefix}speaker_emb/all_pairs_sim_min", all_pairs_sim.min(), global_step)
                        self._log_scalar(f"{prefix}speaker_emb/all_pairs_sim_max", all_pairs_sim.max(), global_step)
                        self._log_scalar(f"{prefix}speaker_emb/all_pairs_sim_median", all_pairs_sim.median(), global_step)
                        self._log_scalar(f"{prefix}speaker_emb/all_pairs_sim_std", all_pairs_sim.std(), global_step)

                        # 2. Per-dimension statistics (detect dimension collapse)
                        # emb_flat shape: [B, D]
                        per_dim_mean = emb_flat.mean(dim=0)  # [D]
                        per_dim_std = emb_flat.std(dim=0)    # [D]
                        # Count how many dimensions have very low variance (< 0.01)
                        collapsed_dims = (per_dim_std < 0.01).sum().item()
                        active_dims = (per_dim_std >= 0.01).sum().item()
                        self._log_scalar(f"{prefix}speaker_emb/collapsed_dims", collapsed_dims, global_step, skip_zero=False)
                        self._log_scalar(f"{prefix}speaker_emb/active_dims", active_dims, global_step, skip_zero=False)
                        self._log_scalar(f"{prefix}speaker_emb/per_dim_std_mean", per_dim_std.mean(), global_step)
                        self._log_scalar(f"{prefix}speaker_emb/per_dim_std_min", per_dim_std.min(), global_step)
                        self._log_scalar(f"{prefix}speaker_emb/per_dim_std_max", per_dim_std.max(), global_step)

                        # 3. Similarity to batch centroid (are all embeddings collapsing to same point?)
                        centroid = emb_flat.mean(dim=0, keepdim=True)  # [1, D]
                        centroid_norm = F.normalize(centroid, dim=-1)
                        sim_to_centroid = (emb_norm * centroid_norm).sum(dim=-1)  # [B]
                        self._log_scalar(f"{prefix}speaker_emb/sim_to_centroid_mean", sim_to_centroid.mean(), global_step)
                        self._log_scalar(f"{prefix}speaker_emb/sim_to_centroid_min", sim_to_centroid.min(), global_step)
                        # High mean + low variance = collapse

                        # 4. L2 norm variance (should have some variation across samples)
                        self._log_scalar(f"{prefix}speaker_emb/l2_norm_std", l2_norms.std(), global_step)

                        # 5. Effective dimensionality via PCA-like measure
                        # Compute variance explained by top-k principal components
                        # This is expensive, so only compute occasionally
                        if global_step % (self.args.logging_steps * 10) == 0:
                            try:
                                # Center the data
                                centered = emb_flat - emb_flat.mean(dim=0, keepdim=True)
                                # SVD to get singular values (proportional to sqrt of eigenvalues)
                                _, s, _ = torch.svd(centered)
                                # Variance explained by each component
                                var_explained = s ** 2 / (s ** 2).sum()
                                # Cumulative variance
                                cumvar = var_explained.cumsum(dim=0)
                                # Effective dimensionality: how many dims to explain 95% variance
                                eff_dims_95 = (cumvar < 0.95).sum().item() + 1
                                eff_dims_90 = (cumvar < 0.90).sum().item() + 1
                                self._log_scalar(f"{prefix}speaker_emb/eff_dims_95pct", eff_dims_95, global_step, skip_zero=False)
                                self._log_scalar(f"{prefix}speaker_emb/eff_dims_90pct", eff_dims_90, global_step, skip_zero=False)
                                # Top singular value ratio (if top is dominant, embeddings are 1D)
                                top1_ratio = var_explained[0].item() if len(var_explained) > 0 else 0.0
                                self._log_scalar(f"{prefix}speaker_emb/top1_var_ratio", top1_ratio, global_step)
                            except Exception:
                                pass  # SVD can fail on degenerate matrices

            # Log FiLM statistics (for diagnosing speaker conditioning health)
            if film_stats is not None:
                for stat_name, stat_value in film_stats.items():
                    self._log_scalar(f"{prefix}film/{stat_name}", stat_value, global_step)

        outputs = {
            "loss": total_loss,
            "rec": recon,
        }

        return (total_loss, outputs) if return_outputs else total_loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """
        Override prediction_step to handle VAE inputs correctly during evaluation.
        The default Trainer calls model(**inputs) which doesn't work with VAE.forward().
        """
        model.eval()

        with torch.no_grad():
            # Unpack inputs the same way as compute_loss
            features = inputs.get("features", None)
            mel_spec = inputs["mel_specs"]
            mel_spec_lengths = inputs.get("mel_lengths", None)
            mel_spec_masks = inputs.get("mel_spec_masks", None)
            speaker_embedding = inputs.get("speaker_embedding", None)

            # Move to device
            mel_spec = mel_spec.to(self.args.device)

            if mel_spec_lengths is not None:
                mel_spec_lengths = mel_spec_lengths.to(self.args.device)
            if mel_spec_masks is not None:
                mel_spec_masks = mel_spec_masks.to(self.args.device)
            if speaker_embedding is not None:
                speaker_embedding = speaker_embedding.to(self.args.device)

            # Use autocast for mixed precision (same as training)
            dtype = torch.bfloat16 if self.args.bf16 else torch.float16 if self.args.fp16 else torch.float32
            with autocast(self.args.device.type, dtype=dtype, enabled=self.args.bf16 or self.args.fp16):
                # Forward pass
                _, _, _, losses = model(
                    features=features,
                    target=mel_spec,
                    mask=mel_spec_masks,
                    speaker_embedding=speaker_embedding,
                    length=mel_spec_lengths,
                )

                loss = losses["total_loss"]

        # Return (loss, logits, labels) - for VAE we don't have traditional logits/labels
        return (loss, None, None)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save both VAE and discriminator."""
        super().save_model(output_dir, _internal_call)

        if output_dir is None:
            output_dir = self.args.output_dir

        # Save discriminator if GAN training has started
        if hasattr(self, "discriminator") and self.discriminator is not None and self.gan_already_started:
            os.makedirs(output_dir, exist_ok=True)
            discriminator_path = os.path.join(output_dir, "discriminator.pt")
            torch.save({
                "discriminator_state_dict": self.discriminator.state_dict(),
                "discriminator_optimizer_state_dict": (
                    self.discriminator_optimizer.state_dict()
                    if self.discriminator_optimizer is not None else None
                ),
            }, discriminator_path)
            print(f"Discriminator saved to {discriminator_path}")

        # Save learned speaker classifier (speaker ID on embeddings) if enabled and training has started
        if hasattr(self, "learned_speaker_classifier") and self.learned_speaker_classifier is not None and self.speaker_id_training_started:
            os.makedirs(output_dir, exist_ok=True)
            learned_speaker_classifier_path = os.path.join(output_dir, "learned_speaker_classifier.pt")
            torch.save({
                "learned_speaker_classifier_state_dict": self.learned_speaker_classifier.state_dict(),
                "learned_speaker_classifier_optimizer_state_dict": (
                    self.learned_speaker_classifier_optimizer.state_dict()
                    if self.learned_speaker_classifier_optimizer is not None else None
                ),
            }, learned_speaker_classifier_path)
            print(f"Learned speaker classifier saved to {learned_speaker_classifier_path}")

    def start_train_print(self, args):
        model = self.model
        discriminator = self.discriminator if hasattr(self, 'discriminator') else None
        vocoder = self.vocoder if hasattr(self, 'vocoder') else None
        print(f"Model structure: {model}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        if hasattr(model, 'encoder') and model.encoder is not None:
            print(f"  VAE Encoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")
        print(f"  VAE Decoder parameters: {sum(p.numel() for p in model.decoder.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"Audio settings:")
        print(f"  Sample rate: {args.audio_sample_rate}")
        print(f"  N mels: {args.audio_n_mels}")
        print(f"  N FFT: {args.audio_n_fft}")
        print(f"  Hop length: {args.audio_hop_length}")
        print(f"  Max frames: {args.audio_max_frames}")
        print(f"  Latent channels: {args.latent_channels}")
        print(f"  Speaker encoder type: {args.speaker_encoder_type}")
        print(f"  Speaker embedding dim: {args.speaker_embedding_dim}")
        print(f"  Speaker embedding proj dim: {args.speaker_embedding_proj_dim} (0=no projection)")
        print(f"  Normalize speaker embedding: {args.normalize_speaker_embedding}")
        print(f"  FiLM scale bound: {args.film_scale_bound} (0=unbounded)")
        print(f"  FiLM shift bound: {args.film_shift_bound} (0=unbounded)")
        if args.learn_speaker_embedding:
            print(f"  Learned speaker embedding: ENABLED (dim={args.learned_speaker_dim})")
        else:
            print(f"  Learned speaker embedding: DISABLED (using pretrained from dataset)")
        if args.use_gan and discriminator is not None:
            print(f"GAN training: enabled")
            print(f"Discriminator structure: {discriminator}")
            print(f"  Discriminator config: {args.discriminator_config}")
            print(f"  Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
            multi_scale_discs = [d for d in discriminator.discriminators if isinstance(d, MelDomainMultiScaleDiscriminator)]
            multi_period_discs = [d for d in discriminator.discriminators if isinstance(d, MelDomainMultiPeriodDiscriminator)]
            if multi_scale_discs:
                print(f"    Multi-scale discriminators parameters: {sum(p.numel() for d in multi_scale_discs for p in d.parameters()):,}")
            if multi_period_discs:
                print(f"    Multi-period discriminators parameters: {sum(p.numel() for d in multi_period_discs for p in d.parameters()):,}")
            print(f"  GAN loss weight: {args.gan_loss_weight}")
            print(f"  Feature matching weight: {args.feature_matching_weight}")
            print(f"  Discriminator LR: {args.discriminator_lr}")
            print(f"  GAN start condition: {args.gan_start_condition_key}={args.gan_start_condition_value}")
            if args.instance_noise_std > 0:
                print(f"  Instance noise: initial_std={args.instance_noise_std}, decay_steps={args.instance_noise_decay_steps}")
            if args.r1_penalty_weight > 0:
                print(f"  R1 penalty: weight={args.r1_penalty_weight}, interval={args.r1_penalty_interval}")
            if args.gan_warmup_steps > 0:
                print(f"  GAN warmup: {args.gan_warmup_steps} steps (ramps loss from 0 to full)")
            if args.use_adaptive_weight:
                print(f"  Adaptive GAN weight: enabled (VQGAN-style gradient balancing)")
        if hasattr(self, 'audio_perceptual_loss') and self.audio_perceptual_loss is not None:
            print(f"Audio perceptual loss: enabled (total_weight={args.audio_perceptual_loss_weight})")
            if args.waveform_stft_loss_weight > 0 or args.waveform_mel_loss_weight > 0:
                print(f"Waveform-domain losses: enabled (gradients flow through frozen vocoder)")
                print(f"  STFT loss weight: {args.waveform_stft_loss_weight if vocoder is not None else 0.0}")
                print(f"  Mel loss weight: {args.waveform_mel_loss_weight if vocoder is not None else 0.0}")
            if args.audio_perceptual_loss_start_step > 0:
                print(f"  Start step: {args.audio_perceptual_loss_start_step} (delayed to let L1/MSE settle)")
            print(f"  Multi-scale mel weight: {args.multi_scale_mel_loss_weight}")
            if vocoder is not None:
                print(f"  Using vocoder for waveform conversion: {args.vocoder_config}")
            else:
                print(f"  No vocoder loaded - only multi-scale mel loss active")
        if args.kl_annealing_steps > 0:
            print(f"KL annealing: {args.kl_annealing_steps} steps (ramps KL weight from 0 to 1)")
        if args.free_bits > 0:
            print(f"Free bits: {args.free_bits} nats per channel (prevents posterior collapse)")
        if args.speaker_embedding_dropout > 0:
            print(f"Speaker embedding dropout: {args.speaker_embedding_dropout} (encourages disentanglement)")
        if args.instance_norm_latents:
            print(f"Instance norm on latents: enabled (removes speaker statistics from z)")
        if args.mu_only_recon_weight > 0:
            print(f"Mu-only reconstruction loss: weight={args.mu_only_recon_weight} (trains decoder for diffusion compatibility)")


def load_model(args):
    if args.config.endswith("_decoder_only"):
        return model_loading_utils.load_model(AudioCVAEDecoderOnly, args.config,  checkpoint_path=args.resume_from_checkpoint, overrides={
            "latent_channels": args.latent_channels,
        })
    else:
        return model_loading_utils.load_model(AudioVAE, args.config,  checkpoint_path=args.resume_from_checkpoint, overrides={
            "latent_channels": args.latent_channels,
        })


def create_trainer(
    args,
    model,
    training_args,
    data_collator,
    train_dataset,
    eval_dataset,
    shared_window_buffer,
    vocoder,
    device,
):
    # Create waveform-domain loss criteria if enabled
    waveform_stft_loss = None
    waveform_mel_loss = None
    multi_scale_mel_loss = None
    if vocoder is not None:
        if args.waveform_stft_loss_weight > 0:
            waveform_stft_loss = MultiResolutionSTFTLoss(
                shared_window_buffer=shared_window_buffer,
                fft_sizes=[256, 512, 1024, 2048],
                hop_sizes=[64, 128, 256, 512],
            )
            waveform_stft_loss.to(device)
            print(f"Created MultiResolutionSTFTLoss (weight={args.waveform_stft_loss_weight})")

        if args.waveform_mel_loss_weight > 0:
            waveform_mel_loss = MultiScaleMelLoss(
                shared_window_buffer=shared_window_buffer,
                sample_rate=args.audio_sample_rate,
                n_mels=args.n_mels,
            )
            waveform_mel_loss.to(device)
            print(f"Created MultiScaleMelLoss (weight={args.waveform_mel_loss_weight})")

    # doesn't need vocoder to be enabled
    if args.multi_scale_mel_loss_weight > 0:
        multi_scale_mel_loss = MultiScaleMelSpectrogramLoss()
        multi_scale_mel_loss.to(device)
        print(f"Created MultiScaleMelSpectrogramLoss (weight={args.multi_scale_mel_loss_weight})")

    # Create audio perceptual loss if enabled
    audio_perceptual_loss = None
    if args.audio_perceptual_loss_weight > 0:
        # Create audio perceptual loss
        audio_perceptual_loss = AudioPerceptualLoss(
            vocoder,
            waveform_stft_loss,
            waveform_mel_loss,
            multi_scale_mel_loss,
            waveform_stft_weight=args.waveform_stft_loss_weight,
            waveform_mel_weight=args.waveform_mel_loss_weight,
            multi_scale_mel_weight=args.multi_scale_mel_loss_weight,
        )
        audio_perceptual_loss.to(device)
        # Freeze all perceptual loss weights
        for param in audio_perceptual_loss.parameters():
            param.requires_grad = False

        # Warn if waveform-based perceptual losses requested but no vocoder
        if vocoder is None and (args.waveform_stft_loss_weight > 0 or args.waveform_mel_loss_weight > 0):
            print("WARNING: Audio perceptual loss with waveform-based components requested but no vocoder loaded. Waveform losses will be disabled.")

    # Create discriminator if GAN training is enabled
    discriminator = None
    discriminator_optimizer = None
    if args.use_gan:
        # keep on cpu, transfer to device when activated by provided criteria
        discriminator = MelDomainCombinedDiscriminator.from_config(args.discriminator_config)

        discriminator_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=args.discriminator_lr,
            betas=(0.0, 0.99),
            weight_decay=0.0,
        )

        # Try to load existing discriminator checkpoint
        discriminator, discriminator_optimizer, disc_loaded = model_loading_utils.load_discriminator(
            args.resume_from_checkpoint, discriminator, discriminator_optimizer, device
        )

        discriminator = discriminator.cpu()

        if disc_loaded:
            print("Loaded discriminator from checkpoint")

    # Create learned speaker classifier for speaker ID loss on learned embeddings
    learned_speaker_classifier = None
    learned_speaker_classifier_optimizer = None
    if args.speaker_id_loss_weight > 0 and args.learn_speaker_embedding:
        # Auto-detect num_speakers from dataset if not already set
        if num_speakers <= 0:
            num_speakers = train_dataset.num_speakers
            print(f"Auto-detected {num_speakers} speakers from dataset")

        # Check if dataset has speaker_ids available
        if not train_dataset.include_speaker_ids:
            print("WARNING: Speaker ID loss enabled but dataset may not have speaker_ids. "
                  "Re-run preprocessing with --include_speaker_id and re-merge shards.")

        # Create classifier based on loss type
        if args.speaker_id_loss_type == "arcface":
            # ArcFace: angular margin loss for tighter embedding clustering (like ECAPA-TDNN)
            learned_speaker_classifier = ArcFaceLoss(
                embedding_dim=args.learned_speaker_dim,
                num_speakers=num_speakers,
                scale=args.arcface_scale,
                margin=args.arcface_margin,
            ).to(device)
        else:
            # Simple MLP classifier with cross-entropy
            learned_speaker_classifier = LearnedSpeakerClassifier(
                embedding_dim=args.learned_speaker_dim,
                num_speakers=num_speakers,
                hidden_dim=256,
            ).to(device)

        learned_speaker_classifier_optimizer = torch.optim.AdamW(
            learned_speaker_classifier.parameters(),
            lr=args.speaker_id_classifier_lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        # Try to load existing learned speaker classifier checkpoint
        learned_speaker_classifier, learned_speaker_classifier_optimizer, lsc_loaded = model_loading_utils.load_learned_speaker_classifier(
            args.resume_from_checkpoint, learned_speaker_classifier, learned_speaker_classifier_optimizer, device
        )
        if lsc_loaded:
            print("Loaded learned speaker classifier from checkpoint")

        if args.local_rank == 0 or not args.use_deepspeed:
            print(f"Learned speaker ID classification: enabled")
            print(f"  Loss type: {args.speaker_id_loss_type}")
            print(f"  Embedding dim (input): {args.learned_speaker_dim}")
            print(f"  Num speakers: {num_speakers}")
            print(f"  Speaker ID loss weight: {args.speaker_id_loss_weight}")
            print(f"  Speaker ID classifier LR: {args.speaker_id_classifier_lr}")
            if args.speaker_id_loss_type == "arcface":
                print(f"  ArcFace scale: {args.arcface_scale}")
                print(f"  ArcFace margin: {args.arcface_margin} radians")
            print(f"  Learned speaker classifier parameters: {sum(p.numel() for p in learned_speaker_classifier.parameters()):,}")
    elif args.speaker_id_loss_weight > 0 and not args.learn_speaker_embedding:
        print("WARNING: speaker_id_loss_weight > 0 but learn_speaker_embedding is False. "
              "Speaker ID loss requires learned speaker embeddings. Disabling speaker ID loss.")

    return AudioCVAEGANTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        cmdline=args.cmdline,
        git_commit_hash=args.commit_hash,
        step_offset=args.start_step,
        discriminator=discriminator,
        discriminator_optimizer=discriminator_optimizer,
        gan_loss_weight=args.gan_loss_weight,
        feature_matching_weight=args.feature_matching_weight,
        discriminator_update_frequency=args.discriminator_update_frequency,
        gan_start_condition_key=args.gan_start_condition_key,
        gan_start_condition_value=args.gan_start_condition_value,
        instance_noise_std=args.instance_noise_std,
        instance_noise_decay_steps=args.instance_noise_decay_steps,
        r1_penalty_weight=args.r1_penalty_weight,
        r1_penalty_interval=args.r1_penalty_interval,
        gan_warmup_steps=args.gan_warmup_steps,
        use_adaptive_weight=args.use_adaptive_weight,
        kl_annealing_steps=args.kl_annealing_steps,
        audio_perceptual_loss=audio_perceptual_loss,
        audio_perceptual_loss_weight=args.audio_perceptual_loss_weight,
        audio_perceptual_loss_start_step=args.audio_perceptual_loss_start_step,
        vocoder=vocoder,
        log_film_stats=args.log_film_stats,
        film_contrastive_loss_weight=args.film_contrastive_loss_weight,
        film_contrastive_loss_start_step=args.film_contrastive_loss_start_step,
        film_contrastive_margin_max=args.film_contrastive_margin_max,
        film_contrastive_margin_rampup_steps=args.film_contrastive_margin_rampup_steps,
        mu_only_recon_weight=args.mu_only_recon_weight,
        learned_speaker_classifier=learned_speaker_classifier,
        learned_speaker_classifier_optimizer=learned_speaker_classifier_optimizer,
        speaker_id_loss_weight=args.speaker_id_loss_weight,
        speaker_id_loss_type=args.speaker_id_loss_type,
        speaker_id_loss_start_step=args.speaker_id_loss_start_step,
        speaker_id_loss_rampup_steps=args.speaker_id_loss_rampup_steps,
        f0_predictor_freeze_steps=args.f0_predictor_freeze_steps,
        f0_warmup_use_gt_steps=args.f0_warmup_use_gt_steps,
    )


def add_cli_args(subparsers):
    sub_parser = subparsers.add_parser("audio-cvae", help="Preprocess audio dataset through SIVE for VAE training")

    sub_parser.add_argument("--audio_max_frames", type=int, default=1875,
                           help="Maximum number of frames for audio input (overrides config file)")
    sub_parser.add_argument("--audio_n_mels", type=int, default=80,
                            help="Number of mel frequency bins (overrides config file)")
    sub_parser.add_argument("--audio_sample_rate", type=int, default=16000,
                            help="Audio sample rate (overrides config file)")
    sub_parser.add_argument("--audio_n_fft", type=int, default=1024,
                            help="FFT size for audio mel spectrograms (overrides config file)")
    sub_parser.add_argument("--audio_hop_length", type=int, default=256,
                            help="Hop length for audio mel spectrograms (overrides config file)")
    
    # VAE settings
    sub_parser.add_argument("--latent_channels", type=int, default=32,
                           help="Number of latent channels in VAE bottleneck")
    # Speaker encoder type determines embedding dimension
    sub_parser.add_argument("--speaker_encoder_type", type=str, default="ecapa_tdnn",
                           help="Type of pretrained speaker encoder to use (ecapa_tdnn or wavlm)")
    sub_parser.add_argument("--speaker_embedding_dim", type=int, default=192,
                           help="Dimension of speaker embeddings from the speaker encoder")
    # Optional projection to reduce speaker embedding dim before FiLM (0 = no projection)
    # Useful for reducing params when using large embeddings like WavLM (768-dim)
    sub_parser.add_argument("--speaker_embedding_proj_dim", type=int, default=0,
                           help="Dimension to project speaker embeddings to before FiLM (0 = no projection)")
    sub_parser.add_argument("--normalize_speaker_embedding", type=str, default="true",
                           help="Whether to L2-normalize speaker embeddings before FiLM (true/false)")
    # FiLM bounding - prevents extreme scale/shift values that can cause artifacts
    # Scale bound of 0.5 means (1 + scale) ranges from 0.5 to 1.5 (never zeroes out)
    # Set to 0 to disable bounding (unbounded FiLM)
    sub_parser.add_argument("--film_scale_bound", type=float, default=0.5,
                           help="Bound for FiLM scale modulation (0 = unbounded)")
    sub_parser.add_argument("--film_shift_bound", type=float, default=0.5,
                            help="Bound for FiLM shift modulation (0 = unbounded)")
    # Zero-init FiLM output weights - forces model to learn FiLM from scratch instead of relying on bias
    sub_parser.add_argument("--zero_init_film_bias", type=str, default="false",
                           help="Whether to zero-initialize FiLM bias terms (true/false)")
    # Remove bias from FiLM projections entirely - zero embedding = zero modulation (structurally enforced)
    sub_parser.add_argument("--film_no_bias", type=str, default="false",
                           help="Whether to remove bias terms from FiLM projections (true/false)")

    # Learned speaker embedding: if True, encoder outputs a learned speaker embedding instead of using pretrained
    # The speaker head uses global pooling to remove temporal structure, outputting a single speaker vector
    sub_parser.add_argument("--learn_speaker_embedding", type=str, default="false",
                           help="Whether to use a learned speaker embedding (true/false)")
    sub_parser.add_argument("--learned_speaker_dim", type=int, default=256,
                           help="Dimension of learned speaker embedding")

    # FiLM contrastive loss - encourages different speaker embeddings to produce different FiLM outputs
    sub_parser.add_argument("--film_contrastive_loss_weight", type=float, default=0.0,
                           help="Weight for FiLM contrastive loss (0 = disabled)")
    sub_parser.add_argument("--film_contrastive_loss_start_step", type=int, default=0,
                           help="Step to start applying FiLM contrastive loss (0 = from start)")
    # FiLM contrastive margin scheduling (ramps from 0 to max, measure of contribution from none to all)
    sub_parser.add_argument("--film_contrastive_margin_max", type=float, default=0.1,
                           help="Maximum margin for FiLM contrastive loss")
    sub_parser.add_argument("--film_contrastive_margin_rampup_steps", type=int, default=5000,
                           help="Number of steps to ramp up FiLM contrastive margin")

    # VAE loss weights
    sub_parser.add_argument("--recon_loss_weight", type=float, default=1.0,
                           help="Weight for reconstruction loss")
    sub_parser.add_argument("--mse_loss_weight", type=float, default=1.0,
                           help="Weight for MSE loss")
    sub_parser.add_argument("--l1_loss_weight", type=float, default=1.0,
                           help="Weight for L1 loss")
    sub_parser.add_argument("--kl_divergence_loss_weight", type=float, default=1e-6,
                           help="Weight for KL divergence loss")
    
    # Audio perceptual loss settings (speech-focused)
    # Total weight for all audio perceptual losses (0 = disabled)
    sub_parser.add_argument("--audio_perceptual_loss_weight", type=float, default=0.0,
                           help="Total weight for audio perceptual loss (0 = disabled)")
    # Waveform-domain losses (require vocoder, gradients flow through frozen vocoder)
    # These losses operate on waveforms generated by the vocoder, providing direct audio supervision
    sub_parser.add_argument("--waveform_stft_loss_weight", type=float, default=0.0,
                           help="Weight for MultiResolutionSTFTLoss on waveforms")
    sub_parser.add_argument("--waveform_mel_loss_weight", type=float, default=0.0,
                           help="Weight for MultiScaleMelLoss on waveforms")
    # Individual component weights (relative to total audio perceptual loss weight)
    sub_parser.add_argument("--multi_scale_mel_loss_weight", type=float, default=1.0,
                           help="Weight for multi-scale mel spectrogram loss component")
    # Step to start applying perceptual loss (0 = from start, >0 = delay to let L1/MSE settle)
    sub_parser.add_argument("--audio_perceptual_loss_start_step", type=int, default=0,
                           help="Step to start applying audio perceptual loss (0 = from start)")
    
    # Vocoder settings (optional - for audio generation during visualization AND waveform losses)
    sub_parser.add_argument("--vocoder_checkpoint_path", type=str, default=None,
                           help="Path to pretrained vocoder checkpoint (for waveform generation/losses)")
    sub_parser.add_argument("--vocoder_config", type=str, default="tiny_attention_freq_domain_vocoder",
                           help="Vocoder config name (from vocoder_configs.py)")

    # GAN training settings
    sub_parser.add_argument("--use_gan", type=str, default="false",
                            help="Whether to use GAN training (true/false)")
    sub_parser.add_argument("--gan_start_condition_key", type=str, default="step",
                            help="Condition to start GAN training: 'step' or 'loss'")
    sub_parser.add_argument("--gan_start_condition_value", type=str, default="0",
                            help="Value for GAN start condition (int for 'step', float for 'loss')")
    sub_parser.add_argument("--discriminator_lr", type=float, default=2e-4,
                            help="Learning rate for discriminator optimizer")
    sub_parser.add_argument("--gan_loss_weight", type=float, default=0.5,
                            help="Weight for GAN loss component")
    sub_parser.add_argument("--feature_matching_weight", type=float, default=0.0,
                            help="Weight for feature matching loss component")
    sub_parser.add_argument("--discriminator_update_frequency", type=int, default=1,
                            help="Number of discriminator updates per generator update")
    sub_parser.add_argument("--discriminator_config", type=str, default="mini_multi_scale",
                            help="Discriminator configuration name")

    # Discriminator regularization settings
    sub_parser.add_argument("--instance_noise_std", type=float, default=0.0,
                            help="Initial standard deviation for instance noise (0 = disabled)")
    sub_parser.add_argument("--instance_noise_decay_steps", type=int, default=50000,
                            help="Number of steps to decay instance noise to zero")
    sub_parser.add_argument("--r1_penalty_weight", type=float, default=0.0,
                            help="Weight for R1 gradient penalty (0 = disabled)")
    sub_parser.add_argument("--r1_penalty_interval", type=int, default=16,
                            help="Interval (in steps) to apply R1 gradient penalty")
    sub_parser.add_argument("--gan_warmup_steps", type=int, default=0,
                            help="Number of steps to warm up GAN loss (ramps from 0 to full)")
    # Adaptive discriminator weighting (VQGAN-style): automatically balances GAN vs reconstruction gradients
    # This prevents the discriminator from dominating and causing artifacts
    sub_parser.add_argument("--use_adaptive_weight", type=str, default="false",
                            help="Whether to use adaptive discriminator weighting (true/false)")

    # KL annealing: ramps KL weight from 0 to full over N steps (0 = disabled, no annealing)
    sub_parser.add_argument("--kl_annealing_steps", type=int, default=0,
                            help="Number of steps for KL annealing (0 = disabled)")

    # Free bits: minimum KL per channel to prevent posterior collapse (0 = disabled)
    sub_parser.add_argument("--free_bits", type=float, default=0.0,
                            help="Free bits threshold in nats per channel (0 = disabled)")

    # Speaker embedding dropout: probability of zeroing speaker embedding during training
    # Encourages disentanglement by forcing decoder to learn to use embedding when available
    sub_parser.add_argument("--speaker_embedding_dropout", type=float, default=0.0,
                            help="Probability of dropping speaker embedding during training")

    # Instance normalization on latents for speaker disentanglement
    # Removes per-instance statistics (mean/variance) which often encode speaker characteristics
    # Speaker info is then re-injected via FiLM conditioning only
    sub_parser.add_argument("--instance_norm_latents", type=str, default="false",
                            help="Whether to apply instance normalization on latents (true/false)")

    # Instance normalization on input mel spectrogram for speaker-invariant features
    # Normalizes each mel bin across time (like CMVN), stripping per-utterance speaker statistics
    sub_parser.add_argument("--use_input_instance_norm", type=str, default="false",
                            help="Whether to apply instance normalization on input mel spectrograms (true/false)")

    # F0 predictor pretraining settings
    # Load pretrained F0 predictor checkpoint for better initialization
    sub_parser.add_argument("--f0_predictor_checkpoint", type=str, default=None,
                            help="Path to pretrained F0 predictor checkpoint (for initialization)")
    # Number of steps to keep F0 predictor frozen (0 = no freezing)
    sub_parser.add_argument("--f0_predictor_freeze_steps", type=int, default=0,
                            help="Number of steps to keep F0 predictor frozen (0 = no freezing)")
    # Use GT F0 for first N steps to let embedding learn with clean signal (0 = disabled)
    sub_parser.add_argument("--f0_warmup_use_gt_steps", type=int, default=0,
                            help="Number of steps to use GT F0 for warmup (0 = disabled)")

    sub_parser.add_argument("--num_speakers", type=int, default=0,
                           help="Number of speaker classes for speaker ID loss (0 = auto-detect from dataset)")

    # Speaker ID classification on learned speaker embeddings
    # Only works when learn_speaker_embedding=True
    sub_parser.add_argument("--speaker_id_loss_weight", type=float, default=0.0,
                           help="Weight for speaker ID classification loss (0 = disabled)")
    sub_parser.add_argument("--speaker_id_classifier_lr", type=float, default=1e-4,
                            help="Learning rate for speaker ID classifier")
    # Loss type: "classifier" (MLP + cross-entropy) or "arcface" (angular margin for tighter clustering)
    sub_parser.add_argument("--speaker_id_loss_type", type=str, default="arcface",
                           help="Speaker ID loss type: 'classifier' or 'arcface'")
    # ArcFace hyperparameters (only used when speaker_id_loss_type="arcface")
    sub_parser.add_argument("--arcface_scale", type=float, default=30.0,
                            help="ArcFace scale parameter")
    sub_parser.add_argument("--arcface_margin", type=float, default=0.2,
                            help="ArcFace margin parameter")
    # Speaker ID loss scheduling (ramps weight from 0 to max over rampup_steps starting at start_step)
    sub_parser.add_argument("--speaker_id_loss_start_step", type=int, default=0,
                           help="Step to start ramping up speaker ID loss weight (0 = from beginning)")
    sub_parser.add_argument("--speaker_id_loss_rampup_steps", type=int, default=0,
                           help="Number of steps to ramp up speaker ID loss weight (0 = no rampup)")

    # FiLM statistics logging - track scale/shift statistics for diagnosing conditioning health
    sub_parser.add_argument("--log_film_stats", type=str, default="false",
                           help="Whether to log FiLM statistics during training (true/false)")

    # Mu-only reconstruction loss: trains decoder to produce good outputs from mu directly
    # This ensures diffusion-generated latents decode well without needing reparameterization noise
    sub_parser.add_argument("--mu_only_recon_weight", type=float, default=0.0,
                           help="Weight for mu-only reconstruction loss (0 = disabled)")

    sub_parser.add_argument("--f0_loss_weight", type=float, default=0.1,
                           help="Weight for F0 prediction loss")
    sub_parser.add_argument("--vuv_loss_weight", type=float, default=0.1,
                           help="Weight for V/UV prediction loss")

    sub_parser.add_argument("--cache_dir", type=str, default="../cached_datasets/sive_cvae_f0",
                           help="Directory for cached datasets")

    return sub_parser
