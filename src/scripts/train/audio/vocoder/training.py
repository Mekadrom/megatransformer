import os

import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import Any, Optional

from transformers.integrations.integration_utils import TensorBoardCallback

from config.audio.vocoder.vocoder import VocoderConfig
from model.audio.criteria import HighFreqSTFTLoss, MultiResolutionSTFTLoss, PhaseLoss, StableMelSpectrogramLoss, Wav2Vec2PerceptualLoss
from model.audio.vocoder.discriminator import WaveformDomainDiscriminator, compute_discriminator_losses, compute_generator_losses
from model.audio.vocoder.vocoder import Vocoder
from model.discriminator import compute_adaptive_weight
from scripts.train.trainer import CommonTrainer
from utils import model_loading_utils
from utils.audio_utils import SharedWindowBuffer


class VocoderGANTrainer(CommonTrainer):
    """
    Custom trainer for vocoder with GAN training.
    Handles alternating generator/discriminator updates.

    Supports dynamic GAN start conditions:
    - "step": Start GAN at a specific training step
    - "reconstruction_criteria_met": Start when reconstruction loss drops below threshold

    For sharded datasets, uses ShardAwareSampler to minimize shard loading overhead.
    """

    def __init__(
        self,
        *args,
        shared_window_buffer: SharedWindowBuffer,
        config: VocoderConfig,
        step_offset,
        n_fft,
        hop_length,
        cmdline,
        git_commit_hash,
        sc_loss_weight: float = 1.0,
        mag_loss_weight: float = 3.0,
        waveform_l1_loss_weight: float = 0.1,
        phase_loss_weight: float = 1.0,
        phase_ip_loss_weight: float = 1.0,
        phase_iaf_loss_weight: float = 1.0,
        phase_gd_loss_weight: float = 1.0,
        high_freq_stft_loss_weight: float = 0.0,
        high_freq_stft_cutoff_bin: int = 256,
        high_freq_mag_penalty_weight: float = 0.1,
        direct_mag_loss_weight: float = 0.0,
        mel_recon_loss_weight: float = 1.0,
        wav2vec2_loss_weight: float = 0.0,
        recon_annealing_start_step: int = 5000,
        recon_annealing_steps: int = 5000,
        perceptual_inclusion_start_step: int = 5000,
        perceptual_inclusion_annealing_steps: int = 5000,
        perceptual_annealing_start_step: int = 20000,
        perceptual_annealing_steps: int = 10000,
        discriminator: Optional[WaveformDomainDiscriminator] = None,
        discriminator_optimizer: Optional[torch.optim.Optimizer] = None,
        gan_adv_loss_weight: float = 1.0,
        gan_feature_matching_loss_weight: float = 2.0,
        discriminator_update_frequency: int = 1,
        gan_start_condition_key: Optional[str] = None,
        gan_start_condition_value: Optional[Any] = None,
        mpd_loss_weight: float = 1.0,
        msd_loss_weight: float = 1.0,
        mrsd_loss_weight: float = 1.0,
        mpd_adv_loss_weight: float = 1.0,
        msd_adv_loss_weight: float = 1.0,
        mrsd_adv_loss_weight: float = 1.0,
        mpd_fm_loss_weight: float = 1.0,
        msd_fm_loss_weight: float = 1.0,
        mrsd_fm_loss_weight: float = 1.0,
        use_adaptive_weight: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.shared_window_buffer = shared_window_buffer
        self.config = config

        # Store shard-aware sampler if available
        self._shard_sampler = None
        if hasattr(self.train_dataset, 'get_sampler'):
            self._shard_sampler = self.train_dataset.get_sampler(shuffle=True, seed=42)
            print("Using ShardAwareSampler for efficient shard loading")

        self.step_offset = step_offset if step_offset is not None else 0
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.cmdline = cmdline
        self.git_commit_hash = git_commit_hash

        self.sc_loss_weight = sc_loss_weight
        self.mag_loss_weight = mag_loss_weight
        self.waveform_l1_loss_weight = waveform_l1_loss_weight
        self.phase_loss_weight = phase_loss_weight
        self.phase_ip_loss_weight = phase_ip_loss_weight
        self.phase_iaf_loss_weight = phase_iaf_loss_weight
        self.phase_gd_loss_weight = phase_gd_loss_weight
        self.high_freq_stft_loss_weight = high_freq_stft_loss_weight
        self.high_freq_stft_cutoff_bin = high_freq_stft_cutoff_bin
        self.direct_mag_loss_weight = direct_mag_loss_weight
        self.high_freq_mag_penalty_weight = high_freq_mag_penalty_weight
        self.mel_recon_loss_weight = mel_recon_loss_weight
        self.wav2vec2_loss_weight = wav2vec2_loss_weight
        self.recon_annealing_start_step = recon_annealing_start_step
        self.recon_annealing_steps = recon_annealing_steps
        self.perceptual_inclusion_start_step = perceptual_inclusion_start_step
        self.perceptual_inclusion_annealing_steps = perceptual_inclusion_annealing_steps
        self.perceptual_annealing_start_step = perceptual_annealing_start_step
        self.perceptual_annealing_steps = perceptual_annealing_steps

        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.gan_adv_loss_weight = gan_adv_loss_weight
        self.gan_feature_matching_loss_weight = gan_feature_matching_loss_weight
        self.discriminator_update_frequency = discriminator_update_frequency
        self.gan_start_condition_key = gan_start_condition_key
        self.gan_start_condition_value = gan_start_condition_value
        self.gan_already_started = False
        self.mpd_loss_weight = mpd_loss_weight
        self.msd_loss_weight = msd_loss_weight
        self.mrsd_loss_weight = mrsd_loss_weight
        self.mpd_adv_loss_weight = mpd_adv_loss_weight
        self.msd_adv_loss_weight = msd_adv_loss_weight
        self.mrsd_adv_loss_weight = mrsd_adv_loss_weight
        self.mpd_fm_loss_weight = mpd_fm_loss_weight
        self.msd_fm_loss_weight = msd_fm_loss_weight
        self.mrsd_fm_loss_weight = mrsd_fm_loss_weight
        self.use_adaptive_weight = use_adaptive_weight
        self.writer = None

        self.has_logged_cli = False

        # Loss functions
        self.stft_loss = MultiResolutionSTFTLoss(shared_window_buffer=shared_window_buffer)
        self.mel_recon_loss = StableMelSpectrogramLoss(
            shared_window_buffer=shared_window_buffer,
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
        )

        self.phase_loss = None
        if phase_loss_weight > 0.0:
            self.phase_loss = PhaseLoss(shared_window_buffer=shared_window_buffer, n_fft=config.n_fft, hop_length=config.hop_length)

        self.high_freq_stft_loss = None
        if high_freq_stft_loss_weight > 0.0:
            self.high_freq_stft_loss = HighFreqSTFTLoss(
                shared_window_buffer=shared_window_buffer,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                cutoff_bin=high_freq_stft_cutoff_bin
            )

        self.wav2vec2_loss = None
        if wav2vec2_loss_weight > 0.0:
            self.wav2vec2_loss = Wav2Vec2PerceptualLoss(
                model_name=config.wav2vec2_model,
                sample_rate=config.sample_rate,
            )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        global_step = self.state.global_step + self.step_offset

        self._ensure_tensorboard_writer()

        if not self.has_logged_cli:
            self.writer.add_text("training/command_line", self.cmdline, global_step)
            self.writer.add_text("training/git_commit_hash", self.git_commit_hash, global_step)
            self.has_logged_cli = True

        # Forward pass through generator (vocoder)
        mel_specs = inputs["mel_specs"]
        mel_spec_masks = inputs["mel_spec_masks"]
        waveform_labels = inputs["waveforms"]
        waveform_masks = inputs["waveform_masks"]

        outputs = model(mel_specs)

        # Get reconstruction losses from model
        pred_waveform = outputs["pred_waveform"]
        pred_stft = outputs["pred_stft"]
        pred_magnitude = outputs["pred_magnitude"]

        # Ensure waveform_labels has batch dimension to match pred_waveform
        if waveform_labels.dim() == 1:
            waveform_labels = waveform_labels.unsqueeze(0)

        # Align waveform lengths
        min_len = min(pred_waveform.shape[-1], waveform_labels.shape[-1])
        pred_waveform_aligned = pred_waveform[..., :min_len]
        waveform_labels_aligned = waveform_labels[..., :min_len]
        waveform_masks_aligned = waveform_masks[..., :min_len]

        # Compute losses (masked if waveform_masks provided)
        waveform_l1 = (torch.abs(pred_waveform_aligned - waveform_labels_aligned) * waveform_masks_aligned).sum() / waveform_masks_aligned.sum()

        # STFT loss expects [B, 1, T] shape
        sc_loss, mag_loss = self.stft_loss(
            pred_waveform_aligned.unsqueeze(1) if pred_waveform_aligned.dim() == 2 else pred_waveform_aligned,
            waveform_labels_aligned.unsqueeze(1) if waveform_labels_aligned.dim() == 2 else waveform_labels_aligned,
        )

        target_complex_stfts = torch.stft(
            waveform_labels_aligned.to(torch.float32), self.config.n_fft, self.config.hop_length,
            window=self.shared_window_buffer.get_window(self.config.n_fft, waveform_labels_aligned.device), return_complex=True
        )[..., :mel_spec_masks.shape[-1]]

        direct_mag_loss = 0.0
        if pred_stft is not None and target_complex_stfts is not None:
            pred_mag = pred_stft.abs()
            target_mag = target_complex_stfts.abs()
            # Use 1e-5 minimum for bf16 numerical stability
            direct_mag_loss = F.l1_loss(
                torch.log(pred_mag.clamp(min=1e-5)),
                torch.log(target_mag.clamp(min=1e-5))
            )

        mel_recon_loss_value = self.mel_recon_loss(pred_waveform_aligned, mel_specs[..., :mel_spec_masks.shape[-1]])

        ip_loss = iaf_loss = gd_loss = phase_loss_value = 0.0
        if self.phase_loss is not None:
            ip_loss, iaf_loss, gd_loss = self.phase_loss(
                pred_waveform_aligned,
                target_complex_stfts=target_complex_stfts,
                precomputed_stft=pred_stft,
            )
            phase_loss_value = (self.phase_ip_loss_weight * ip_loss +
                                self.phase_iaf_loss_weight * iaf_loss +
                                self.phase_gd_loss_weight * gd_loss)

        high_freq_stft_loss_value = 0.0
        if self.high_freq_stft_loss is not None:
            high_freq_stft_loss_value = self.high_freq_stft_loss(
                pred_waveform_aligned,
                waveform_labels_aligned,
                target_complex_stfts=target_complex_stfts,
                precomputed_stft=pred_stft
            )

        wav2vec2_loss_value = 0.0
        if self.wav2vec2_loss is not None:
            wav2vec2_loss_value = self.wav2vec2_loss(
                pred_waveform_aligned,
                waveform_labels_aligned,
            )

        high_freq_penalty = pred_magnitude[..., -20:, :].pow(2).mean()

        sc_loss_weight = self.get_loss_weight(
            self.sc_loss_weight,
            global_step,
            start_step=0,
            rampup_steps=0,
            anneal_start_step=self.recon_annealing_start_step,
            anneal_steps=self.recon_annealing_steps,
            min_ratio=0.5*self.sc_loss_weight
        )

        mag_loss_weight = self.get_loss_weight(
            self.mag_loss_weight,
            global_step,
            start_step=0,
            rampup_steps=0,
            anneal_start_step=self.recon_annealing_start_step,
            anneal_steps=self.recon_annealing_steps,
            min_ratio=0.15*self.mag_loss_weight
        )

        waveform_l1_loss_weight = self.get_loss_weight(
            self.waveform_l1_loss_weight,
            global_step,
            start_step=0,
            rampup_steps=0,
            anneal_start_step=self.recon_annealing_start_step,
            anneal_steps=self.recon_annealing_steps,
            min_ratio=0.15*self.waveform_l1_loss_weight
        )

        phase_loss_weight = self.get_loss_weight(
            self.phase_loss_weight,
            global_step,
            start_step=0,
            rampup_steps=0,
            anneal_start_step=self.recon_annealing_start_step,
            anneal_steps=self.recon_annealing_steps,
            min_ratio=0.675*self.phase_loss_weight
        )

        high_freq_stft_loss_weight = self.get_loss_weight(
            self.high_freq_stft_loss_weight,
            global_step,
            start_step=0,
            rampup_steps=0,
            anneal_start_step=self.recon_annealing_start_step,
            anneal_steps=self.recon_annealing_steps,
            min_ratio=0.325*self.high_freq_stft_loss_weight
        )

        high_freq_mag_penalty_weight = self.get_loss_weight(
            self.high_freq_mag_penalty_weight,
            global_step,
            start_step=0,
            rampup_steps=0,
            anneal_start_step=self.recon_annealing_start_step,
            anneal_steps=self.recon_annealing_steps,
            min_ratio=0.15*self.high_freq_mag_penalty_weight
        )

        direct_mag_loss_weight = self.get_loss_weight(
            self.direct_mag_loss_weight,
            global_step,
            start_step=0,
            rampup_steps=0,
            anneal_start_step=self.recon_annealing_start_step,
            anneal_steps=self.recon_annealing_steps,
            min_ratio=0.15*self.direct_mag_loss_weight
        )

        mel_recon_loss_weight = self.get_loss_weight(
            self.mel_recon_loss_weight,
            global_step,
            start_step=self.perceptual_inclusion_start_step,
            rampup_steps=self.perceptual_inclusion_annealing_steps,
            anneal_start_step=self.perceptual_annealing_start_step,
            anneal_steps=self.perceptual_annealing_steps,
            min_ratio=0.675*self.mel_recon_loss_weight
        )

        wav2vec2_loss_weight = self.get_loss_weight(
            self.wav2vec2_loss_weight,
            global_step,
            start_step=self.perceptual_inclusion_start_step,
            rampup_steps=self.perceptual_inclusion_annealing_steps,
            anneal_start_step=self.perceptual_annealing_start_step,
            anneal_steps=self.perceptual_annealing_steps,
            min_ratio=0.675*self.wav2vec2_loss_weight
        )

        recon_loss = (sc_loss_weight * sc_loss +
                        mag_loss_weight * mag_loss +
                        waveform_l1_loss_weight * waveform_l1 +
                        phase_loss_weight * phase_loss_value +
                        high_freq_stft_loss_weight * high_freq_stft_loss_value +
                        high_freq_mag_penalty_weight * high_freq_penalty +
                        direct_mag_loss_weight * direct_mag_loss +
                        mel_recon_loss_weight * mel_recon_loss_value +
                        wav2vec2_loss_weight * wav2vec2_loss_value)

        # log weights
        if global_step % self.args.logging_steps == 0 and self.writer is not None:
            self._log_scalar("train/loss_weights/sc_loss_weight", sc_loss_weight)
            self._log_scalar("train/loss_weights/mag_loss_weight", mag_loss_weight)
            self._log_scalar("train/loss_weights/waveform_l1_loss_weight", waveform_l1_loss_weight)
            self._log_scalar("train/loss_weights/phase_loss_weight", phase_loss_weight)
            self._log_scalar("train/loss_weights/high_freq_stft_loss_weight", high_freq_stft_loss_weight)
            self._log_scalar("train/loss_weights/high_freq_mag_penalty_weight", high_freq_mag_penalty_weight)
            self._log_scalar("train/loss_weights/direct_mag_loss_weight", direct_mag_loss_weight)
            self._log_scalar("train/loss_weights/mel_recon_loss_weight", mel_recon_loss_weight)
            self._log_scalar("train/loss_weights/wav2vec2_loss_weight", wav2vec2_loss_weight)

        outputs.update({
            "loss": recon_loss,
            "waveform_l1": waveform_l1,
            "sc_loss": sc_loss,
            "mag_loss": mag_loss,
            "phase_loss": phase_loss_value,
            "phase_ip_loss": ip_loss,
            "phase_iaf_loss": iaf_loss,
            "phase_gd_loss": gd_loss,
            "high_freq_stft_loss": high_freq_stft_loss_value,
            "direct_mag_loss": direct_mag_loss,
            "mel_recon_loss": mel_recon_loss_value,
            "wav2vec2_loss": wav2vec2_loss_value,
            "high_freq_mag_penalty": high_freq_penalty,
        })

        # Ensure waveform_labels has correct shape
        if waveform_labels.dim() == 1:
            waveform_labels = waveform_labels.unsqueeze(0)

        # Align lengths
        min_len = min(pred_waveform.shape[-1], waveform_labels.shape[-1])
        pred_waveform_aligned = pred_waveform[..., :min_len]
        waveform_labels_aligned = waveform_labels[..., :min_len]

        # Check if GAN training should be enabled
        gan_enabled = self.is_gan_enabled(global_step, recon_loss)

        # Set the flag once GAN starts (stays enabled for rest of training)
        if gan_enabled and not self.gan_already_started:
            self.gan_already_started = True
            print(f"GAN training started at step {global_step}")

        g_loss_gan = torch.tensor(0.0, device=pred_waveform.device)
        g_loss_fm = torch.tensor(0.0, device=pred_waveform.device)
        d_loss = torch.tensor(0.0, device=pred_waveform.device)

        if gan_enabled:
            # Discriminator Update
            dtype = torch.bfloat16 if self.args.bf16 else torch.float16 if self.args.fp16 else torch.float32
            if global_step % self.discriminator_update_frequency == 0:
                self.discriminator.train()
                self.discriminator.to(pred_waveform.device)
                self.discriminator.to(dtype)

                # Detach generator outputs for discriminator update
                with torch.no_grad():
                    pred_waveform_detached = pred_waveform_aligned.detach()

                # Get discriminator outputs for real and fake
                # autocast needs to be used here
                device_type = pred_waveform.device.type
                with torch.autocast(device_type, dtype=dtype, enabled=self.args.fp16 or self.args.bf16):
                    disc_real = self.discriminator(waveform_labels_aligned)
                    disc_fake = self.discriminator(pred_waveform_detached)

                d_losses = compute_discriminator_losses(disc_real, disc_fake)

                # Compute discriminator losses
                d_loss_mpd = d_losses["d_loss_mpd"]
                d_loss_msd = d_losses["d_loss_msd"]
                d_loss_mrsd = d_losses["d_loss_mrsd"]
                
                d_loss = self.mpd_loss_weight * d_loss_mpd + self.msd_loss_weight * d_loss_msd + self.mrsd_loss_weight * d_loss_mrsd

                if global_step % self.args.logging_steps == 0 and self.writer is not None:
                    for disc_crit in disc_real.keys():
                        for o, output in enumerate(disc_real[disc_crit][0]):
                            self._log_scalar(f"train/disc_real_{disc_crit}/{o}/avg", output.mean())

                    for disc_crit in disc_fake.keys():
                        for o, output in enumerate(disc_fake[disc_crit][0]):
                            self._log_scalar(f"train/disc_fake_{disc_crit}/{o}/avg", output.mean())
                        
                    self._log_scalar("train/d_loss_mpd", d_loss_mpd)
                    self._log_scalar("train/d_loss_msd", d_loss_msd)
                    self._log_scalar("train/d_loss_mrsd", d_loss_mrsd)
                    self._log_scalar("train/d_loss_total", d_loss)

                # Update discriminator (only during training when gradients are enabled)
                if self.discriminator_optimizer is not None and self.discriminator.training and torch.is_grad_enabled():
                    self.discriminator_optimizer.zero_grad()
                    d_loss.backward()
                    self.discriminator_optimizer.step()

            # Generator GAN Loss
            # Get discriminator outputs for fake (for generator update)
            device_type = pred_waveform.device.type
            dtype = torch.bfloat16 if self.args.bf16 else torch.float16 if self.args.fp16 else torch.float32
            with torch.autocast(device_type, dtype=dtype, enabled=self.args.fp16 or self.args.bf16):
                disc_real = self.discriminator(waveform_labels_aligned.detach())
                disc_fake = self.discriminator(pred_waveform_aligned)

            g_losses = compute_generator_losses(disc_fake, disc_real)

            # Generator adversarial loss
            g_adv_mpd = g_losses["g_adv_mpd"]
            g_fm_mpd = g_losses["g_fm_mpd"]
            g_adv_msd = g_losses["g_adv_msd"]
            g_fm_msd = g_losses["g_fm_msd"]
            g_adv_mrsd = g_losses["g_adv_mrsd"]
            g_fm_mrsd = g_losses["g_fm_mrsd"]

            g_loss_adv = self.mpd_adv_loss_weight * g_adv_mpd + self.msd_adv_loss_weight * g_adv_msd + self.mrsd_adv_loss_weight * g_adv_mrsd
            g_loss_fm = self.mpd_fm_loss_weight * g_fm_mpd + self.msd_fm_loss_weight * g_fm_msd + self.mrsd_fm_loss_weight * g_fm_mrsd
            g_loss_gan = self.gan_adv_loss_weight * g_loss_adv + self.gan_feature_matching_loss_weight * g_loss_fm

            if global_step % self.args.logging_steps == 0 and self.writer is not None:
                self._log_scalar("train/g_adv_mpd", g_adv_mpd)
                self._log_scalar("train/g_fm_mpd", g_fm_mpd)
                self._log_scalar("train/g_adv_msd", g_adv_msd)
                self._log_scalar("train/g_fm_msd", g_fm_msd)
                self._log_scalar("train/g_adv_mrsd", g_adv_mrsd)
                self._log_scalar("train/g_fm_mrsd", g_fm_mrsd)
                self._log_scalar("train/g_adv_total", g_loss_adv)
                self._log_scalar("train/g_fm_total", g_loss_fm)
                self._log_scalar("train/g_loss_total", g_loss_gan)

        # Total generator loss with optional adaptive weighting
        adaptive_weight = torch.tensor(self.gan_adv_loss_weight, device=mel_specs.device)
        if self.use_adaptive_weight and self.gan_already_started and g_loss_gan.requires_grad:
            # VQGAN-style adaptive weighting: balance GAN and reconstruction gradients
            # Uses the last decoder layer as proxy for decoder output gradients
            try:
                m: Vocoder = model
                last_parameters = [
                    *m.mag_head_low.parameters(), *m.mag_head_high.parameters(),
                    *m.phase_head_low.parameters(), *m.phase_head_high.parameters()
                ]
                adaptive_weight = compute_adaptive_weight(
                    recon_loss, g_loss_gan, last_parameters, self.gan_adv_loss_weight
                )
            except (AttributeError, RuntimeError) as e:
                # Fallback to fixed weight if adaptive weight fails
                # (e.g., model doesn't have expected structure, or grad computation fails)
                if global_step % self.args.logging_steps == 0:
                    print(f"Warning: adaptive weight computation failed ({e}), using fixed weight")
                adaptive_weight = torch.tensor(self.gan_adv_loss_weight, device=mel_specs.device)

        # Total generator loss
        total_loss = recon_loss + adaptive_weight * g_loss_gan

        # Log individual losses
        if global_step % self.args.logging_steps == 0 and self.writer is not None:
            prefix = "train/" if model.training else "eval/"
            self._log_scalar(f"{prefix}waveform_l1", outputs.get("waveform_l1", 0))
            self._log_scalar(f"{prefix}sc_loss", outputs.get("sc_loss", 0))
            self._log_scalar(f"{prefix}mag_loss", outputs.get("mag_loss", 0))
            self._log_scalar(f"{prefix}mel_recon_loss", outputs.get("mel_recon_loss", 0))
            self._log_scalar(f"{prefix}phase_ip_loss", outputs.get("phase_ip_loss", 0))
            self._log_scalar(f"{prefix}phase_iaf_loss", outputs.get("phase_iaf_loss", 0))
            self._log_scalar(f"{prefix}phase_gd_loss", outputs.get("phase_gd_loss", 0))
            self._log_scalar(f"{prefix}phase_loss", outputs.get("phase_loss", 0))
            self._log_scalar(f"{prefix}high_freq_stft_loss", outputs.get("high_freq_stft_loss", 0))
            self._log_scalar(f"{prefix}wav2vec2_loss", outputs.get("wav2vec2_loss", 0))
            self._log_scalar(f"{prefix}recon_loss", recon_loss)
            self._log_scalar(f"{prefix}total_loss", total_loss)

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
            mel_specs = inputs["mel_specs"]

            # Move to device
            mel_specs = mel_specs.to(self.args.device)

            # Use autocast for mixed precision (same as training)
            dtype = torch.bfloat16 if self.args.bf16 else torch.float16 if self.args.fp16 else torch.float32
            with torch.autocast(self.args.device.type, dtype=dtype, enabled=self.args.bf16 or self.args.fp16):
                loss = self.compute_loss(model, inputs, return_outputs=True)[0]

        # Return (loss, logits, labels) - for VAE we don't have traditional logits/labels
        return (loss, None, None)

    def _log_scalar(self, tag, value):
        global_step = self.state.global_step + self.step_offset
        if self.writer is not None:
            if isinstance(value, torch.Tensor):
                value = value.item()
            if value != 0.0:
                self.writer.add_scalar(tag, value, global_step)

    def _ensure_tensorboard_writer(self):
        if hasattr(self, "writer") and self.writer is not None:
            return

        for callback in self.callback_handler.callbacks:
            if isinstance(callback, TensorBoardCallback):
                self.writer = callback.tb_writer
                return

        self.writer = None

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save both generator and discriminator."""
        super().save_model(output_dir, _internal_call)

        if output_dir is None:
            output_dir = self.args.output_dir

        # Save discriminator if GAN training has started
        if self.discriminator is not None and self.gan_already_started:
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

    def start_train_print(self, args):
        print(f"Model structure: {self.model}")
        if args.use_gan and self.discriminator is not None:
            print(f"Discriminator structure: {self.discriminator}")
            print(f"GAN training: {'enabled' if args.use_gan else 'disabled'}")
            print(f"  GAN loss weight: {args.gan_adv_loss_weight}")
            print(f"  Feature matching loss weight: {args.gan_feature_matching_loss_weight}")
            print(f"  MPD adversarial loss weight: {args.mpd_adv_loss_weight}")
            print(f"  MSD adversarial loss weight: {args.msd_adv_loss_weight}")
            print(f"  MRSD adversarial loss weight: {args.mrsd_adv_loss_weight}")
            print(f"  MPD feature matching loss weight: {args.mpd_fm_loss_weight}")
            print(f"  MSD feature matching loss weight: {args.msd_fm_loss_weight}")
            print(f"  MRSD feature matching loss weight: {args.mrsd_fm_loss_weight}")
            print(f"  Discriminator LR: {args.discriminator_lr}")
            print(f"  GAN start condition: {args.gan_start_condition_key}={args.gan_start_condition_value}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"  Vocoder parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        for name, module in self.model.named_children():
            print(f"    {name} parameters: {sum(p.numel() for p in module.parameters()):,}")
        if args.use_gan and self.discriminator is not None:
            print(f"  Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
            print(f"    MPD parameters: {sum(p.numel() for p in self.discriminator.mpd.parameters()):,}")
            print(f"    MSD parameters: {sum(p.numel() for p in self.discriminator.msd.parameters()):,}")
            print(f"    MRSD parameters: {sum(p.numel() for p in self.discriminator.mrsd.parameters()):,}")


def load_model(args, shared_window_buffer: SharedWindowBuffer):
    # return model_loading_utils.load_model(Vocoder, args.config,  checkpoint_path=args.resume_from_checkpoint, overrides={
    #     'shared_window_buffer': shared_window_buffer,
    #     'sc_loss_weight': args.sc_loss_weight,
    #     'mag_loss_weight': args.mag_loss_weight,
    #     'waveform_l1_loss_weight': args.waveform_l1_loss_weight,
    #     'phase_loss_weight': args.phase_loss_weight,
    #     'phase_ip_loss_weight': args.phase_ip_loss_weight,
    #     'phase_iaf_loss_weight': args.phase_iaf_loss_weight,
    #     'phase_gd_loss_weight': args.phase_gd_loss_weight,
    #     'high_freq_stft_loss_weight': args.high_freq_stft_loss_weight,
    #     'high_freq_stft_cutoff_bin': args.high_freq_stft_cutoff_bin,
    #     'high_freq_mag_penalty_weight': args.high_freq_mag_penalty_weight,
    #     'direct_mag_loss_weight': args.direct_mag_loss_weight,
    #     'mel_recon_loss_weight': args.mel_recon_loss_weight,
    #     'wav2vec2_loss_weight': args.wav2vec2_loss_weight,
    #     'recon_annealing_start_step': args.recon_annealing_start_step,
    #     'recon_annealing_steps': args.recon_annealing_steps,
    #     'perceptual_inclusion_start_step': args.perceptual_inclusion_start_step,
    #     'perceptual_inclusion_annealing_steps': args.perceptual_inclusion_annealing_steps,
    #     'perceptual_annealing_start_step': args.perceptual_annealing_start_step,
    #     'perceptual_annealing_steps': args.perceptual_annealing_steps,
    # })
    return model_loading_utils.load_model(Vocoder, args.config,  checkpoint_path=args.resume_from_checkpoint, overrides={
        'shared_window_buffer': shared_window_buffer,
    })


def create_trainer(
    args,
    model,
    optimizer,
    training_args,
    data_collator,
    train_dataset,
    eval_dataset,
    shared_window_buffer: SharedWindowBuffer,
):
    # Create discriminator if GAN training is enabled
    discriminator = None
    discriminator_optimizer = None

    if args.use_gan:
        # keep on cpu, transfer to device when activated by provided criteria
        discriminator = WaveformDomainDiscriminator.from_config(shared_window_buffer, args.discriminator_config)

        discriminator_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=args.discriminator_lr,
            betas=(0.0, 0.99),
            weight_decay=0.0,
        )

        # Try to load existing discriminator checkpoint
        discriminator, discriminator_optimizer, disc_loaded = model_loading_utils.load_discriminator(
            args.resume_from_checkpoint, discriminator, discriminator_optimizer, "cpu"
        )

        # move to cpu until gan training starts
        discriminator = discriminator.cpu()

        if disc_loaded:
            print("Loaded discriminator from checkpoint")

    return VocoderGANTrainer(
        model=model,
        optimizers=(optimizer, None),
        args=training_args,
        shared_window_buffer=shared_window_buffer,
        config=model.config,
        step_offset=args.start_step,
        n_fft=args.audio_n_fft,
        hop_length=args.audio_hop_length,
        cmdline=args.cmdline,
        git_commit_hash=args.commit_hash,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        discriminator=discriminator if args.use_gan else None,
        discriminator_optimizer=discriminator_optimizer if args.use_gan else None,
        gan_adv_loss_weight=args.gan_adv_loss_weight,
        gan_feature_matching_loss_weight=args.gan_feature_matching_loss_weight,
        mpd_loss_weight=args.mpd_loss_weight,
        msd_loss_weight=args.msd_loss_weight,
        mrsd_loss_weight=args.mrsd_loss_weight,
        gan_start_condition_key=args.gan_start_condition_key,
        gan_start_condition_value=args.gan_start_condition_value,
        mpd_adv_loss_weight=args.mpd_adv_loss_weight,
        msd_adv_loss_weight=args.msd_adv_loss_weight,
        mrsd_adv_loss_weight=args.mrsd_adv_loss_weight,
        mpd_fm_loss_weight=args.mpd_fm_loss_weight,
        msd_fm_loss_weight=args.msd_fm_loss_weight,
        mrsd_fm_loss_weight=args.mrsd_fm_loss_weight,
        direct_mag_loss_weight=args.direct_mag_loss_weight,
        use_adaptive_weight=args.use_adaptive_weight,
    )


def add_cli_args(subparsers):
    sub_parser = subparsers.add_parser("vocoder", help="Preprocess audio dataset through SIVE for VAE training")

    sub_parser.add_argument('--audio_max_seconds', type=float, default=10.0,
                            help="Maximum audio length in seconds (overrides config file)")
    sub_parser.add_argument("--audio_n_mels", type=int, default=80,
                            help="Number of mel frequency bins (overrides config file)")
    sub_parser.add_argument("--audio_sample_rate", type=int, default=16000,
                            help="Audio sample rate (overrides config file)")
    sub_parser.add_argument("--audio_n_fft", type=int, default=1024,
                            help="FFT size for audio mel spectrograms (overrides config file)")
    sub_parser.add_argument("--audio_hop_length", type=int, default=256,
                            help="Hop length for audio mel spectrograms (overrides config file)")
    
    # GAN training settings
    sub_parser.add_argument("--use_gan", action="store_true",
                            help="Whether to use GAN training (true/false)")
    sub_parser.add_argument("--gan_start_condition_key", type=str, default="step",
                            help="Condition to start GAN training: 'step' or 'loss'")
    sub_parser.add_argument("--gan_start_condition_value", type=str, default="0",
                            help="Value for GAN start condition (int for 'step', float for 'loss')")
    sub_parser.add_argument("--discriminator_lr", type=float, default=2e-4,
                            help="Learning rate for discriminator optimizer")
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

    sub_parser.add_argument("--sc_loss_weight", type=float, default=1.0,
                            help="Spectral convergence loss weight")
    sub_parser.add_argument("--mag_loss_weight", type=float, default=3.0,
                            help="Log magnitude loss weight")
    sub_parser.add_argument("--waveform_l1_loss_weight", type=float, default=0.1,
                            help="Waveform L1 loss weight")
    sub_parser.add_argument("--phase_loss_weight", type=float, default=0.0,
                            help="Phase loss weight")
    sub_parser.add_argument("--phase_ip_loss_weight", type=float, default=0.0,
                            help="Instantaneous phase loss weight")
    sub_parser.add_argument("--phase_iaf_loss_weight", type=float, default=0.0,
                            help="Instantaneous angular frequency loss weight")
    sub_parser.add_argument("--phase_gd_loss_weight", type=float, default=0.0,
                            help="Group delay loss weight")
    sub_parser.add_argument("--high_freq_stft_loss_weight", type=float, default=0.0,
                            help="High-frequency STFT loss weight")
    sub_parser.add_argument("--high_freq_stft_cutoff_bin", type=int, default=256,
                            help="Cutoff bin for high-frequency STFT loss")
    sub_parser.add_argument("--direct_mag_loss_weight", type=float, default=0.0,
                            help="Direct magnitude loss weight")
    sub_parser.add_argument("--high_freq_mag_penalty_weight", type=float, default=0.1,
                            help="High-frequency magnitude penalty weight")
    sub_parser.add_argument("--mel_recon_loss_weight", type=float, default=1.0,
                            help="Mel spectrogram reconstruction loss weight")
    sub_parser.add_argument("--wav2vec2_loss_weight", type=float, default=0.0,
                            help="Wav2Vec2 perceptual loss weight")
    sub_parser.add_argument("--recon_annealing_start_step", type=int, default=5000,
                            help="Reconstruction loss annealing start step")
    sub_parser.add_argument("--recon_annealing_steps", type=int, default=5000,
                            help="Number of steps over which to anneal reconstruction loss")
    sub_parser.add_argument("--perceptual_inclusion_start_step", type=int, default=5000,
                            help="Perceptual inclusion start step")
    sub_parser.add_argument("--perceptual_inclusion_annealing_steps", type=int, default=5000,
                            help="Number of steps over which to anneal perceptual inclusion")
    sub_parser.add_argument("--perceptual_annealing_start_step", type=int, default=20000,
                            help="Perceptual annealing start step")
    sub_parser.add_argument("--perceptual_annealing_steps", type=int, default=10000,
                            help="Number of steps over which to anneal perceptual loss")
    
    # Discriminator loss weights
    sub_parser.add_argument("--gan_adv_loss_weight", type=float, default=1.0,
                            help="GAN adversarial loss weight")
    sub_parser.add_argument("--gan_feature_matching_loss_weight", type=float, default=2.0,
                            help="GAN feature matching loss weight")
    sub_parser.add_argument("--mpd_loss_weight", type=float, default=0.8,
                            help="Multi-period discriminator loss weight")
    sub_parser.add_argument("--msd_loss_weight", type=float, default=0.5,
                            help="Multi-scale discriminator loss weight")
    sub_parser.add_argument("--mrsd_loss_weight", type=float, default=0.8,
                            help="Multi-resolution discriminator loss weight")
    sub_parser.add_argument("--mpd_adv_loss_weight", type=float, default=1.2,
                            help="Multi-period discriminator adversarial loss weight")
    sub_parser.add_argument("--msd_adv_loss_weight", type=float, default=0.5,
                            help="Multi-scale discriminator adversarial loss weight")
    sub_parser.add_argument("--mrsd_adv_loss_weight", type=float, default=0.7,
                            help="Multi-resolution discriminator adversarial loss weight")
    sub_parser.add_argument("--mpd_fm_loss_weight", type=float, default=1.0,
                            help="Multi-period discriminator feature matching loss weight")
    sub_parser.add_argument("--msd_fm_loss_weight", type=float, default=2.0,
                            help="Multi-scale discriminator feature matching loss weight")
    sub_parser.add_argument("--mrsd_fm_loss_weight", type=float, default=2.0,
                            help="Multi-resolution discriminator feature matching loss weight")

    sub_parser.add_argument("--cache_dir", type=str, default="../cached_datasets/audio",
                           help="Directory for cached datasets")

    return sub_parser
