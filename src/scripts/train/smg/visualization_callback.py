import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from transformers import Trainer
from scripts.train.visualization_callback import VisualizationCallback
from utils import audio_utils, metrics, visualization
from utils.audio_utils import SharedWindowBuffer
from torch.amp import autocast


class SMGVisualizationCallback(VisualizationCallback):
    """
    Callback for logging SMG mel spectrogram reconstruction during training.
    Periodically reconstructs test mel specs and logs to TensorBoard.
    Optionally converts mel spectrograms to audio using a vocoder.
    """

    def __init__(
        self,
        shared_window_buffer: SharedWindowBuffer,
        step_offset: int = 0,
        voice_max_seconds: int = 10,
        voice_sample_rate: int = 16000,
        voice_n_mels: int = 80,
        voice_n_fft: int = 1024,
        voice_hop_length: int = 256,
        vocoder_checkpoint_path: Optional[str] = None,
        vocoder_config: str = "experimental",
        vocoder: Optional[torch.nn.Module] = None,  # Pre-loaded vocoder (shared with trainer)
        num_eval_samples: int = 8,
        speaker_encoder_type: str = "ecapa_tdnn",
        free_bits: float = 0.0,
    ):
        self.shared_window_buffer = shared_window_buffer

        self.trainer: Optional[Trainer] = None
        self.step_offset = step_offset if step_offset is not None else 0
        self.voice_sample_rate = voice_sample_rate
        self.voice_n_mels = voice_n_mels
        self.voice_n_fft = voice_n_fft
        self.voice_hop_length = voice_hop_length
        self.num_eval_samples = num_eval_samples
        self.speaker_encoder_type = speaker_encoder_type
        self.free_bits = free_bits

        # Vocoder settings - use pre-loaded vocoder if provided, otherwise lazy load
        self.vocoder_checkpoint_path = vocoder_checkpoint_path
        self.vocoder_config = vocoder_config
        self.vocoder = vocoder  # May be pre-loaded (shared with trainer for waveform losses)
        self._vocoder_load_attempted = vocoder is not None  # Skip lazy loading if already provided

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Generate and log reconstructions during evaluation from eval dataset."""
        global_step = state.global_step + self.step_offset

        if not state.is_world_process_zero:
            return

        logger = metrics.get_logger()
        if logger is None:
            print("No metrics logger found, skipping eval visualization...")
            return

        # Get eval dataset from trainer
        eval_dataset = self.trainer.eval_dataset
        if eval_dataset is None or len(eval_dataset) == 0:
            print("No eval dataset available, skipping eval visualization...")
            return
        collator = self.trainer.data_collator

        print(f"Generating eval mel reconstructions at step {global_step}...")

        device = self._get_device()
        model.eval()

        # Sample random indices from eval dataset
        num_samples = min(self.num_eval_samples, len(eval_dataset))
        indices = torch.randperm(len(eval_dataset))[:num_samples].tolist()

        # Collect aggregate statistics
        all_losses = {}
        all_mu_means = []
        all_mu_stds = []
        all_logvar_means = []

        # Collect sample data for cross-speaker reconstruction
        eval_samples_data = []  # List of (mel, speaker_embedding, mu, mel_length)

        # SMG has no encoder — all is_vae branches below are dead-code paths kept only
        # to minimize churn; the flag is always False.
        is_vae = False

        with torch.no_grad():
            dtype = torch.bfloat16 if bool(args.bf16) else torch.float16 if args.fp16 else torch.float32

            with autocast(device.type, dtype=dtype, enabled=args.bf16 or args.fp16):
                for i, idx in enumerate(indices):
                    sample = eval_dataset[idx]
                    sample = collator([sample])  # Collate single sample batch
                    features = sample["features"].to(device)  # [1, C, H, W]
                    mel_specs = sample["mel_specs"]
                    mel_lengths = sample.get("mel_lengths", None)
                    mel_spec_masks = sample.get("mel_spec_masks", None)
                    speaker_embeddings = sample.get("speaker_embeddings", None)
                    sample_f0 = sample.get("f0", None)
                    sample_voiced = sample.get("voiced", None)

                    # Ensure correct shape [1, n_mels, T]
                    if mel_specs.dim() == 2:
                        mel_specs = mel_specs.unsqueeze(0).to(device)

                    # Trim to actual length before inference (avoid wasted compute on padding)
                    mel = mel_specs[..., :mel_lengths].to(device)
                    mel_spec_masks = mel_spec_masks[..., :mel_lengths].to(device) if mel_spec_masks is not None else None

                    # Prepare F0 data if available
                    target_f0 = None
                    target_voiced = None
                    if sample_f0 is not None and sample_voiced is not None:
                        target_f0 = sample_f0[:mel_lengths].unsqueeze(0).to(device)  # [1, T]
                        target_voiced = sample_voiced[:mel_lengths].unsqueeze(0).to(device)  # [1, T]

                    spk_emb = speaker_embeddings.to(device)

                    # Use reconstruct_with_attention to get attention weights
                    if is_vae:
                        recon, mu, logvar, losses = model(
                            features=features,
                            target=mel,
                            speaker_embedding=spk_emb,
                            target_f0=target_f0,
                            target_voiced=target_voiced,
                            mask=mel_spec_masks
                        )
                    else:
                        recon,losses = model(
                            features=features,
                            target=mel,
                            speaker_embedding=spk_emb,
                            target_f0=target_f0,
                            target_voiced=target_voiced,
                            mask=mel_spec_masks
                        )
                        mu = None
                        logvar = None

                    # Determine which speaker embedding to use for decode:
                    # If model learns speaker embedding, encode to get learned embedding
                    # Otherwise use the pretrained speaker embedding from dataset
                    encoder_learns_speaker = is_vae and hasattr(model.encoder, 'learn_speaker_embedding') and model.encoder.learn_speaker_embedding
                    if encoder_learns_speaker:
                        # Get learned speaker embedding from encoder
                        enc_result = model.encode(mel_specs)
                        # encode returns (mu, logvar, learned_speaker_emb) when learn_speaker_embedding=True
                        learned_spk_emb = enc_result[2] if len(enc_result) > 2 else None
                        decode_spk_emb = learned_spk_emb if learned_spk_emb is not None else spk_emb
                    else:
                        decode_spk_emb = spk_emb

                    # Store sample data for cross-speaker reconstruction later
                    eval_samples_data.append({
                        "features": features,  # [1, C, H, W]
                        "mel": mel,  # [1, n_mels, T]
                        "speaker_embedding": speaker_embeddings,  # [192] or None (pretrained)
                        "mu": mu.cpu() if mu is not None else None,  # [1, C, H, W]
                        "mel_length": mel_lengths,
                    })

                    if is_vae:
                        # Generate mu-only reconstruction (no sampling, z = mu)
                        # This is what diffusion will see during inference
                        recon_mu_only = model.decode(mu, speaker_embedding=decode_spk_emb, features=features)

                    if is_vae:
                        kl_per_element: torch.Tensor = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                        if self.free_bits > 0:
                            # Free bits: apply minimum KL per channel to prevent posterior collapse
                            # Sum over spatial dims, mean over batch -> per-channel KL: [C]
                            spatial_dims = list(range(2, mu.dim()))  # [2, 3] for 4D, [2] for 3D
                            kl_per_channel = kl_per_element.sum(dim=spatial_dims).mean(dim=0)  # [C]

                            # Clamp each channel's KL to at least free_bits
                            kl_per_channel = torch.clamp(kl_per_channel, min=self.free_bits)

                            # Sum over channels for total KL
                            kl_divergence = kl_per_channel.sum()
                        else:
                            # Original behavior: sum over all latent dims, mean over batch
                            latent_dims = list(range(1, mu.dim()))  # [1, 2, 3] for 4D, [1, 2] for 3D
                            kl_divergence = kl_per_element.sum(dim=latent_dims).mean()
                        losses["kl_divergence"] = kl_divergence

                    # Add manually computed losses for logging (model already returns losses dict)
                    # Overwrite mse_loss with properly computed one (trimmed mel)
                    losses["mse_loss"] = F.mse_loss(recon, mel)

                    if is_vae:
                        # Collect statistics
                        all_mu_means.append(mu.mean().item())
                        all_mu_stds.append(mu.std().item())
                        all_logvar_means.append(logvar.mean().item())

                    for loss_name, loss_val in losses.items():
                        # Skip non-scalar values like learned_speaker_embedding
                        if isinstance(loss_val, torch.Tensor):
                            if loss_val.numel() != 1:
                                continue
                            loss_val = loss_val.item()
                        elif not isinstance(loss_val, (int, float)):
                            continue
                        if loss_name not in all_losses:
                            all_losses[loss_name] = []
                        all_losses[loss_name].append(loss_val)

                    # Get tensors for visualization
                    mel_cpu = mel.squeeze(0).cpu().numpy()
                    recon_cpu = recon[0].squeeze(0).float().cpu().numpy()
                    if is_vae:
                        recon_mu_only_cpu = recon_mu_only[0].squeeze(0).float().cpu().numpy()

                    # Trimmed versions (without padding)
                    mel_trimmed = mel_cpu[..., :mel_lengths]
                    recon_trimmed = recon_cpu[..., :mel_lengths]
                    if is_vae:
                        recon_mu_only_trimmed = recon_mu_only_cpu[..., :mel_lengths]

                    # Log individual mel spectrograms
                    fig = visualization.render_mel_spectrogram(mel_trimmed, hop_length=self.voice_hop_length, sample_rate=self.voice_sample_rate, n_fft=self.voice_n_fft)
                    metrics.log_figure(f"eval_smg/original/{i}", fig, global_step)
                    plt.close(fig)

                    fig = visualization.render_mel_spectrogram(recon_trimmed, hop_length=self.voice_hop_length, sample_rate=self.voice_sample_rate, n_fft=self.voice_n_fft)
                    metrics.log_figure(f"eval_smg/reconstruction/{i}", fig, global_step)
                    plt.close(fig)

                    if is_vae:
                        fig = visualization.render_mel_spectrogram(recon_mu_only_trimmed, hop_length=self.voice_hop_length, sample_rate=self.voice_sample_rate, n_fft=self.voice_n_fft)
                        metrics.log_figure(f"eval_smg/reconstruction_mu_only/{i}", fig, global_step)
                        plt.close(fig)

                    # Log trimmed comparison
                    fig = visualization.render_mel_comparison(recon_trimmed, mel_trimmed)
                    metrics.log_figure(f"eval_smg/comparison/{i}", fig, global_step)
                    plt.close(fig)

                    if is_vae:
                        fig = visualization.render_mel_comparison(recon_mu_only_trimmed, mel_trimmed)
                        metrics.log_figure(f"eval_smg/comparison_mu_only/{i}", fig, global_step)
                        plt.close(fig)

                    # Log per-example losses (skip non-scalar values like learned_speaker_embedding)
                    for loss_name, loss_val in losses.items():
                        if isinstance(loss_val, torch.Tensor):
                            if loss_val.numel() != 1:
                                continue
                            loss_val = loss_val.item()
                        elif not isinstance(loss_val, (int, float)):
                            continue
                        metrics.log_scalar(f"eval_smg/example_{i}/{loss_name}", loss_val, global_step)

                    # Log latent channel visualizations for first few samples
                    if is_vae and i < 4:
                        mu_sample = mu[0].float().cpu()  # [latent_channels, H, W]
                        mu_min, mu_max = mu_sample.min(), mu_sample.max()
                        mu_norm = (mu_sample - mu_min) / (mu_max - mu_min + 1e-5)
                        for c in range(min(mu_norm.shape[0], 8)):  # Limit to first 8 channels
                            metrics.log_image(
                                f"eval_smg/example_{i}/mu_channel_{c}",
                                mu_norm[c:c+1, None, :],
                                global_step
                            )

                    # Log attention weights for first few samples
                    if i < 4:
                        # Compute downsampled T for trimming padding from attention visualizations
                        # Time strides depend on config, but for "small" it's 5*5*1 = 25x
                        # We can infer this from the attention shape
                        if is_vae and hasattr(model.encoder, 'time_strides'):
                            # Compute downsampled length
                            T_down = mel_lengths
                            for stride in model.encoder.time_strides:
                                T_down = (T_down + stride - 1) // stride

                    # Convert mel spectrograms to audio using vocoder
                    if self.vocoder is not None:
                        mel_tensor = mel.squeeze(0).float().cpu()
                        recon_mel_tensor = recon[0].squeeze(0).float().cpu()

                        # Log ground truth audio
                        try:
                            waveform = visualization.render_vocoder_audio(self.vocoder, mel_tensor[..., :mel_lengths])
                            metrics.log_audio(f"eval_smg/original_audio/{i}", waveform, global_step, self.voice_sample_rate)
                        except Exception as e:
                            print(f"Vocoder failed for original audio {i}: {e}")
                        # Log reconstruction audio
                        try:
                            waveform = visualization.render_vocoder_audio(self.vocoder, recon_mel_tensor[..., :mel_lengths])
                            metrics.log_audio(f"eval_smg/recon_audio/{i}", waveform, global_step, self.voice_sample_rate)
                        except Exception as e:
                            print(f"Vocoder failed for recon audio {i}: {e}")
                        # Log mu-only reconstruction audio (what diffusion will produce)
                        if is_vae:
                            recon_mu_only_mel_tensor = recon_mu_only[0].squeeze(0).float().cpu()
                            try:
                                waveform = visualization.render_vocoder_audio(self.vocoder, recon_mu_only_mel_tensor[..., :mel_lengths])
                                metrics.log_audio(f"eval_smg/recon_mu_only_audio/{i}", waveform, global_step, self.voice_sample_rate)
                            except Exception as e:
                                print(f"Vocoder failed for mu-only recon audio {i}: {e}")

        # Log aggregate statistics
        for loss_name, loss_vals in all_losses.items():
            metrics.log_scalar(f"eval_smg/mean_{loss_name}", np.mean(loss_vals), global_step)
            metrics.log_scalar(f"eval_smg/std_{loss_name}", np.std(loss_vals), global_step)

        if is_vae:
            metrics.log_scalar("eval_smg/mean_mu_mean", np.mean(all_mu_means), global_step)
            metrics.log_scalar("eval_smg/mean_mu_std", np.mean(all_mu_stds), global_step)
            metrics.log_scalar("eval_smg/mean_logvar_mean", np.mean(all_logvar_means), global_step)

        # Cross-speaker reconstruction on eval samples
        # Select samples that have speaker embeddings
        samples_with_speakers = [
            (i, s) for i, s in enumerate(eval_samples_data)
            if s["speaker_embedding"] is not None
        ]

        # Check if model uses learned speaker embeddings
        encoder_learns_speaker = is_vae and hasattr(model.encoder, 'learn_speaker_embedding') and model.encoder.learn_speaker_embedding

        if len(samples_with_speakers) >= 2:
            print("Generating cross-speaker reconstructions on eval samples...")

            with torch.no_grad():
                with autocast(device.type, dtype=dtype, enabled=args.bf16 or args.fp16):
                    # Create 4 random pairs (or fewer if not enough samples)
                    num_pairs = min(4, len(samples_with_speakers) // 2)
                    pair_indices = torch.randperm(len(samples_with_speakers))[:num_pairs * 2].tolist()

                    for pair_idx in range(num_pairs):
                        idx_a = pair_indices[pair_idx * 2]
                        idx_b = pair_indices[pair_idx * 2 + 1]

                        sample_a_idx, sample_a = samples_with_speakers[idx_a]
                        sample_b_idx, sample_b = samples_with_speakers[idx_b]

                        # Reconstruct A's content with B's speaker embedding
                        if is_vae:
                            z_a = sample_a["mu"].to(device)
                        else:
                            z_a = sample_a["features"].to(device)  # In non-VAE case, "mu" is actually z

                        # Get speaker embedding for B (encode to get learned, or use pretrained)
                        if encoder_learns_speaker:
                            # Encode B's mel to get learned speaker embedding
                            mel_b_input = sample_b["mel"].to(device)
                            enc_result_b = model.encode(mel_b_input)
                            spk_emb_b = enc_result_b[2]  # learned_speaker_emb
                        else:
                            spk_emb_b = sample_b["speaker_embedding"].to(device)

                        # For cross-speaker, use source features with target speaker for F0 prediction
                        features_a = sample_a["features"].to(device)
                        cross_recon_a_with_b = model.decode(z_a, speaker_embedding=spk_emb_b, features=features_a)

                        # Reconstruct B's content with A's speaker embedding
                        if is_vae:
                            z_b = sample_b["mu"].to(device)
                        else:
                            z_b = sample_b["features"].to(device)  # In non-VAE case, "mu" is actually z

                        # Get speaker embedding for A (encode to get learned, or use pretrained)
                        if encoder_learns_speaker:
                            # Encode A's mel to get learned speaker embedding
                            mel_a_input = sample_a["mel"].to(device)
                            enc_result_a = model.encode(mel_a_input)
                            spk_emb_a = enc_result_a[2]  # learned_speaker_emb
                        else:
                            spk_emb_a = sample_a["speaker_embedding"].to(device)

                        features_b = sample_b["features"].to(device)
                        cross_recon_b_with_a = model.decode(z_b, speaker_embedding=spk_emb_a, features=features_b)

                        # Log A with B's speaker
                        mel_a_trimmed = sample_a["mel"].squeeze(0).cpu().numpy()[..., :sample_a["mel_length"]]
                        cross_a_trimmed = cross_recon_a_with_b[0].squeeze(0).float().cpu().numpy()[..., :sample_a["mel_length"]]

                        metrics.log_text(
                            f"eval_smg/cross_speaker/pair{pair_idx}",
                            f"content{sample_a_idx}_spk{sample_b_idx}",
                            global_step
                        )
                        fig = visualization.render_mel_spectrogram(mel_a_trimmed, hop_length=self.voice_hop_length, sample_rate=self.voice_sample_rate, n_fft=self.voice_n_fft)
                        metrics.log_figure(f"eval_smg/cross_speaker/pair{pair_idx}/original", fig, global_step)
                        plt.close(fig)

                        fig = visualization.render_mel_spectrogram(cross_a_trimmed, hop_length=self.voice_hop_length, sample_rate=self.voice_sample_rate, n_fft=self.voice_n_fft)
                        metrics.log_figure(f"eval_smg/cross_speaker/pair{pair_idx}/reconstruction", fig, global_step)
                        plt.close(fig)

                        fig = visualization.render_mel_comparison(cross_a_trimmed, mel_a_trimmed)
                        metrics.log_figure(f"eval_smg/cross_speaker/pair{pair_idx}/comparison", fig, global_step)
                        plt.close(fig)

                        # Log B with A's speaker
                        mel_b_trimmed = sample_b["mel"].squeeze(0).cpu().numpy()[..., :sample_b["mel_length"]]
                        cross_b_trimmed = cross_recon_b_with_a[0].squeeze(0).float().cpu().numpy()[..., :sample_b["mel_length"]]

                        metrics.log_text(
                            f"eval_smg/cross_speaker/pair{pair_idx}",
                            f"content{sample_b_idx}_spk{sample_a_idx}",
                            global_step
                        )
                        fig = visualization.render_mel_spectrogram(mel_b_trimmed, hop_length=self.voice_hop_length, sample_rate=self.voice_sample_rate, n_fft=self.voice_n_fft)
                        metrics.log_figure(f"eval_smg/cross_speaker/pair{pair_idx}/original", fig, global_step)
                        plt.close(fig)

                        fig = visualization.render_mel_spectrogram(cross_b_trimmed, hop_length=self.voice_hop_length, sample_rate=self.voice_sample_rate, n_fft=self.voice_n_fft)
                        metrics.log_figure(f"eval_smg/cross_speaker/pair{pair_idx}/reconstruction", fig, global_step)
                        plt.close(fig)

                        fig = visualization.render_mel_comparison(cross_b_trimmed, mel_b_trimmed)
                        metrics.log_figure(f"eval_smg/cross_speaker/pair{pair_idx}/comparison", fig, global_step)
                        plt.close(fig)

                        # Log audio if vocoder available
                        if self.vocoder is not None:
                            try:
                                waveform = visualization.render_vocoder_audio(self.vocoder, cross_recon_a_with_b[0].squeeze(0).float().cpu()[..., :sample_a["mel_length"]])
                                metrics.log_audio(f"eval_smg/cross_speaker/pair{pair_idx}_ab/audio", waveform, global_step, self.voice_sample_rate)

                                waveform_mels = audio_utils.extract_mels(
                                    self.shared_window_buffer,
                                    torch.from_numpy(waveform)
                                ).cpu().numpy()

                                fig = visualization.render_mel_spectrogram(waveform_mels, hop_length=self.voice_hop_length, sample_rate=self.voice_sample_rate, n_fft=self.voice_n_fft)
                                metrics.log_figure(f"eval_smg/cross_speaker/pair{pair_idx}_ab/waveform_mel", fig, global_step)
                                plt.close(fig)
                            except Exception as e:
                                print(f"Vocoder failed for cross-speaker pair {pair_idx} AB: {e}")

                            try:
                                waveform = visualization.render_vocoder_audio(self.vocoder, cross_recon_b_with_a[0].squeeze(0).float().cpu()[..., :sample_b["mel_length"]])
                                metrics.log_audio(f"eval_smg/cross_speaker/pair{pair_idx}_ba/audio", waveform, global_step, self.voice_sample_rate)

                                waveform_mels = audio_utils.extract_mels(
                                    self.shared_window_buffer,
                                    torch.from_numpy(waveform)
                                ).cpu().numpy()

                                fig = visualization.render_mel_spectrogram(waveform_mels, hop_length=self.voice_hop_length, sample_rate=self.voice_sample_rate, n_fft=self.voice_n_fft)
                                metrics.log_figure(f"eval_smg/cross_speaker/pair{pair_idx}_ba/waveform_mel", fig, global_step)
                                plt.close(fig)
                            except Exception as e:
                                print(f"Vocoder failed for cross-speaker pair {pair_idx} BA: {e}")

            print(f"Cross-speaker reconstruction complete: {num_pairs} pairs logged")

        print(f"Eval visualization complete: {num_samples} samples logged")
        metrics.flush()
