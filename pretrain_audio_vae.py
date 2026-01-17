import os

os.environ["DEEPSPEED_UNIT_TEST"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ['NCCL_TIMEOUT'] = '1200000'

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from typing import Any, Dict, List, Mapping, Optional, Union

from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers.integrations import TensorBoardCallback

from dataset_loading import audio_loading
from dataset_loading.audio_diffusion_dataset import CachedAudioDiffusionDataset
from shard_utils import AudioVAEShardedDataset, AudioVAEDataCollator, ShardAwareSampler
from model.audio.discriminators import (
    GradientReversalFunction,
    MelMultiPeriodDiscriminator,
    MelMultiScaleDiscriminator,
    mel_discriminator_config_lookup,
    compute_mel_discriminator_loss,
    compute_mel_generator_gan_loss,
    add_mel_instance_noise,
    r1_mel_gradient_penalty,
    MelInstanceNoiseScheduler,
)
from model.image.discriminators import compute_adaptive_weight
from model.audio.criteria import AudioPerceptualLoss
from model.audio.vae import model_config_lookup
from model.audio.vocoders.vocoders import model_config_lookup as vocoder_config_lookup
from utils import megatransformer_utils
from utils.audio_utils import SharedWindowBuffer
from utils.model_loading_utils import load_model
from utils.training_utils import EarlyStoppingCallback, setup_int8_training


class SpeakerClassifier(torch.nn.Module):
    """
    Enhanced classifier to predict speaker ID from latent representations.

    Uses convolutional processing to extract spatial patterns + multi-statistic
    pooling (mean/std/max) to capture channel dynamics. Much more capable than
    simple global average pooling.

    Used with GRL: classifier tries to predict speaker, but reversed gradients
    train the encoder to remove speaker information from latents.

    Architecture:
        Conv blocks (expand channels, extract patterns) ->
        Multi-statistic pooling (mean/std/max per channel) ->
        MLP classifier with residual connection
    """
    def __init__(self, latent_channels: int, num_speakers: int, hidden_dim: int = 512):
        super().__init__()

        # Convolutional feature extraction - process spatial patterns
        # Input: [B, latent_channels, H, W] e.g. [B, 8, 10, 157]
        self.conv_blocks = torch.nn.Sequential(
            # Block 1: expand channels, extract local patterns
            torch.nn.Conv2d(latent_channels, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.GELU(),

            # Block 2: deeper features
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.GELU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce spatial dims

            # Block 3: high-level speaker patterns
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.GELU(),
        )

        # Channel attention - weight channels by importance for speaker ID
        self.channel_attention = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 256),
            torch.nn.Sigmoid(),
        )

        # Multi-statistic pooling gives us 256 * 3 = 768 features
        pooled_features = 256 * 3

        # MLP classifier with residual
        self.fc1 = torch.nn.Linear(pooled_features, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, num_speakers)

        self.dropout = torch.nn.Dropout(0.3)
        self.layer_norm1 = torch.nn.LayerNorm(hidden_dim)
        self.layer_norm2 = torch.nn.LayerNorm(hidden_dim)

    def forward(self, latent: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Forward pass with gradient reversal.

        Args:
            latent: [B, C, H, W] latent representation from encoder
            alpha: Gradient reversal strength (0 = no reversal, 1 = full reversal)

        Returns:
            [B, num_speakers] logits for speaker classification
        """
        # Apply gradient reversal to encoder gradients
        reversed_latent = GradientReversalFunction.apply(latent, alpha)

        # Convolutional feature extraction
        features = self.conv_blocks(reversed_latent)  # [B, 256, H', W']

        # Channel attention - reweight channels by speaker relevance
        attn_weights = self.channel_attention(features)  # [B, 256]
        features = features * attn_weights.unsqueeze(-1).unsqueeze(-1)  # [B, 256, H', W']

        # Multi-statistic pooling - capture mean, variance, and peaks
        feat_mean = features.mean(dim=(2, 3))  # [B, 256]
        feat_std = features.std(dim=(2, 3))    # [B, 256]
        feat_max = features.amax(dim=(2, 3))   # [B, 256]

        pooled = torch.cat([feat_mean, feat_std, feat_max], dim=1)  # [B, 768]

        # MLP with residual connection
        x = self.fc1(pooled)
        x = torch.nn.functional.gelu(x)
        x = self.layer_norm1(x)
        x = self.dropout(x)

        residual = x
        x = self.fc2(x)
        x = torch.nn.functional.gelu(x)
        x = self.layer_norm2(x + residual)  # Residual connection
        x = self.dropout(x)

        return self.fc_out(x)


class LearnedSpeakerClassifier(torch.nn.Module):
    """
    Classifier to predict speaker ID from learned speaker embeddings.

    Unlike the GRL SpeakerClassifier (which operates on latents with gradient reversal),
    this classifier operates on the learned speaker embeddings with DIRECT gradients.
    This explicitly trains the speaker head to produce speaker-discriminative features.

    Used together with GRL:
    - GRL on latents: "don't encode speaker info here" (adversarial)
    - This classifier on speaker embeddings: "DO encode speaker info here" (direct supervision)

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


class ArcFaceLoss(torch.nn.Module):
    """
    Additive Angular Margin Loss (ArcFace) for learning speaker embeddings.

    Unlike simple classification which only learns a decision boundary, ArcFace
    explicitly shapes the embedding geometry by:
    - Normalizing embeddings to unit hypersphere
    - Adding angular margin to target class
    - Scaling logits to sharpen the softmax

    This forces same-speaker embeddings to cluster tightly together with angular
    separation between different speakers - the same property that makes ECAPA-TDNN
    embeddings effective for speaker verification.

    Reference: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"

    Args:
        embedding_dim: Dimension of input speaker embeddings
        num_speakers: Number of speaker classes
        scale: Logit scale factor (higher = sharper softmax, typical: 30-64)
        margin: Angular margin in radians (typical: 0.2-0.5)
    """
    def __init__(
        self,
        embedding_dim: int,
        num_speakers: int,
        scale: float = 30.0,
        margin: float = 0.2,
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.num_speakers = num_speakers
        self.embedding_dim = embedding_dim

        # Learnable class center weights (one per speaker)
        self.weight = torch.nn.Parameter(torch.FloatTensor(num_speakers, embedding_dim))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        Compute ArcFace loss.

        Args:
            embeddings: [B, D] speaker embeddings (will be L2 normalized)
            labels: [B] speaker class labels

        Returns:
            Tuple of (loss, logits, accuracy) for logging
        """
        # Handle 3D input [B, 1, D] -> [B, D]
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(1)

        # L2 normalize embeddings and weights to project onto unit hypersphere
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weights_norm = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity (dot product of normalized vectors)
        cos_theta = F.linear(embeddings_norm, weights_norm)  # [B, num_speakers]

        # Clamp for numerical stability before acos
        cos_theta_clamped = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)

        # Convert to angle
        theta = torch.acos(cos_theta_clamped)

        # Add angular margin only to the target class
        # This pushes the target class embedding further from the decision boundary
        one_hot = F.one_hot(labels, num_classes=self.num_speakers).float()
        theta_m = theta + one_hot * self.margin

        # Convert back to cosine (with margin applied to target)
        cos_theta_m = torch.cos(theta_m)

        # Scale logits (higher scale = sharper probability distribution)
        logits = self.scale * cos_theta_m

        # Standard cross-entropy on scaled logits
        loss = F.cross_entropy(logits, labels)

        # Compute accuracy for logging (use original cos_theta for predictions)
        with torch.no_grad():
            preds = cos_theta.argmax(dim=-1)
            accuracy = (preds == labels).float().mean().item()

        return loss, logits, accuracy


def get_writer(trainer: Trainer) -> Optional[SummaryWriter]:
    """Get TensorBoard writer from trainer callbacks."""
    if hasattr(trainer, 'callback_handler'):
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, TensorBoardCallback):
                if callback.tb_writer is not None:
                    return callback.tb_writer
    return None


class AudioVAEReconstructionCallback(TrainerCallback):
    """
    Callback for logging VAE mel spectrogram reconstruction during training.
    Periodically reconstructs test mel specs and logs to TensorBoard.
    Optionally converts mel spectrograms to audio using a vocoder.
    """

    def __init__(
        self,
        shared_window_buffer: SharedWindowBuffer,
        step_offset: int = 0,
        generation_steps: int = 1000,
        audio_sample_rate: int = 16000,
        audio_n_mels: int = 80,
        audio_n_fft: int = 1024,
        audio_hop_length: int = 256,
        audio_max_frames: int = 1875,
        vocoder_checkpoint_path: Optional[str] = None,
        vocoder_config: str = "experimental",
        num_eval_samples: int = 8,
        speaker_encoder_type: str = "ecapa_tdnn",
    ):
        self.shared_window_buffer = shared_window_buffer

        self.trainer: Optional[Trainer] = None
        self.step_offset = step_offset if step_offset is not None else 0
        self.generation_steps = generation_steps
        self.audio_sample_rate = audio_sample_rate
        self.audio_n_mels = audio_n_mels
        self.audio_n_fft = audio_n_fft
        self.audio_hop_length = audio_hop_length
        self.audio_max_frames = audio_max_frames
        self.num_eval_samples = num_eval_samples
        self.speaker_encoder_type = speaker_encoder_type

        # Vocoder settings
        self.vocoder_checkpoint_path = vocoder_checkpoint_path
        self.vocoder_config = vocoder_config
        self.vocoder = None
        self._vocoder_load_attempted = False

        self.example_speaker_embeddings = []  # Store speaker embeddings for example audio

        # Load example audio files and compute mel specs
        self.example_paths = [
            "inference/examples/test_alm_1.mp3",
            "inference/examples/test_alm_2.mp3",
        ]
        # Pre-extracted speaker embedding paths - suffix based on encoder type
        # ecapa_tdnn (default): test_alm_speaker_embedding_1.pt
        # wavlm: test_alm_speaker_embedding_1_wavlm.pt
        emb_suffix = "" if speaker_encoder_type == "ecapa_tdnn" else f"_{speaker_encoder_type}"
        self.example_speaker_embedding_paths = [
            f"inference/examples/test_alm_speaker_embedding_1{emb_suffix}.pt",
            f"inference/examples/test_alm_speaker_embedding_2{emb_suffix}.pt",
        ]
        self.example_mels = []
        self.example_original_lengths = []  # Store original mel lengths before padding

        for path in self.example_paths:
            if os.path.exists(path):
                try:
                    waveform, orig_sr = torchaudio.load(path)
                    # Resample if needed
                    if orig_sr != audio_sample_rate:
                        waveform = torchaudio.transforms.Resample(
                            orig_freq=orig_sr, new_freq=audio_sample_rate
                        )(waveform)
                    # Convert to mono if stereo
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)

                    # Compute mel spectrogram
                    mel = audio_loading.extract_mels(
                        self.shared_window_buffer,
                        waveform,
                        self.audio_sample_rate,
                        self.audio_n_mels,
                        self.audio_n_fft,
                        self.audio_hop_length
                    )

                    # Store original length before padding
                    original_length = mel.shape[-1]

                    # Pad or truncate to max frames
                    if mel.shape[-1] < audio_max_frames:
                        mel = F.pad(mel, (0, audio_max_frames - mel.shape[-1]), value=0)
                    elif mel.shape[-1] > audio_max_frames:
                        mel = mel[..., :audio_max_frames]
                        original_length = audio_max_frames  # Truncated, so original is the max

                    # Add channel dimension: [n_mels, T] -> [1, n_mels, T]
                    if mel.dim() == 2:
                        mel = mel.unsqueeze(0)

                    self.example_mels.append(mel)
                    self.example_original_lengths.append(original_length)
                    print(f"Loaded example mel from {path}: shape {mel.shape}, original length {original_length}")
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
            else:
                print(f"Example audio not found: {path}")

    def _load_vocoder(self):
        """Lazily load vocoder on first use."""
        if self._vocoder_load_attempted:
            return
        self._vocoder_load_attempted = True

        if self.vocoder_checkpoint_path is None:
            return

        if not os.path.exists(self.vocoder_checkpoint_path):
            print(f"Vocoder checkpoint not found at {self.vocoder_checkpoint_path}")
            return

        try:
            vocoder = vocoder_config_lookup[self.vocoder_config](
                shared_window_buffer=self.shared_window_buffer,
            )

            load_model(False, vocoder, self.vocoder_checkpoint_path)

            # Remove weight normalization for inference optimization
            if hasattr(vocoder.vocoder, 'remove_weight_norm'):
                vocoder.vocoder.remove_weight_norm()

            vocoder.eval()
            self.vocoder = vocoder
            print(f"Loaded vocoder from {self.vocoder_checkpoint_path}")
            print(f"Vocoder parameters: {sum(p.numel() for p in vocoder.parameters()):,}")
        except Exception as e:
            print(f"Failed to load vocoder: {e}")
            self.vocoder = None

    def _load_speaker_embeddings(self, device: torch.device):
        """Load speaker embeddings from pre-extracted .pt files (no runtime extraction)."""
        if len(self.example_mels) == 0:
            return

        # Load pre-extracted speaker embeddings - these MUST exist
        for i, emb_path in enumerate(self.example_speaker_embedding_paths):
            if os.path.exists(emb_path):
                try:
                    speaker_emb = torch.load(emb_path, weights_only=True)
                    self.example_speaker_embeddings.append(speaker_emb)
                    print(f"Loaded speaker embedding from {emb_path}: shape {speaker_emb.shape}, L2 norm {speaker_emb.norm():.4f}")
                except Exception as e:
                    raise RuntimeError(f"Failed to load speaker embedding from {emb_path}: {e}")
            else:
                raise FileNotFoundError(
                    f"Pre-extracted speaker embedding not found: {emb_path}\n"
                    f"Please extract it using: python scripts/extract_speaker_embedding.py "
                    f"--audio_path <audio_file> --output_path {emb_path}"
                )

        print(f"Loaded {len(self.example_speaker_embeddings)} pre-extracted speaker embeddings")

    def _log_vocoder_audio(self, writer: SummaryWriter, mel_spec: torch.Tensor, global_step: int, tag: str):
        """Convert mel spectrogram to audio using vocoder and log to TensorBoard."""
        try:
            # Ensure mel is [B, n_mels, T] for vocoder
            if mel_spec.dim() == 2:
                mel_spec = mel_spec.unsqueeze(0)  # [1, n_mels, T]

            # Run vocoder on CPU (float32 for stability)
            mel_spec = mel_spec.float()

            with torch.no_grad():
                outputs = self.vocoder(mel_spec)
                if isinstance(outputs, dict):
                    waveform = outputs["pred_waveform"]
                else:
                    waveform = outputs

            # Ensure 1D waveform
            if waveform.dim() > 1:
                waveform = waveform.squeeze()

            # Normalize to [-1, 1] range
            waveform = waveform / (waveform.abs().max() + 1e-8)

            # Log audio to TensorBoard
            writer.add_audio(
                tag,
                waveform.numpy(),
                global_step,
                sample_rate=self.audio_sample_rate
            )
        except Exception as e:
            print(f"Failed to generate audio with vocoder: {e}")

    def _get_device(self):
        """Determine the device to use for inference."""
        if torch.distributed.is_initialized():
            return torch.device(f"cuda:{torch.distributed.get_rank()}")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Generate and log reconstructions during evaluation from eval dataset."""
        global_step = state.global_step + self.step_offset

        if not state.is_world_process_zero:
            return

        writer = get_writer(self.trainer)
        if writer is None:
            print("No TensorBoard writer found, skipping eval visualization...")
            return

        # Get eval dataset from trainer
        eval_dataset = self.trainer.eval_dataset
        if eval_dataset is None or len(eval_dataset) == 0:
            print("No eval dataset available, skipping eval visualization...")
            return

        print(f"Generating eval mel reconstructions at step {global_step}...")

        # Lazily load vocoder
        self._load_vocoder()

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

        with torch.no_grad():
            dtype = torch.bfloat16 if bool(args.bf16) else torch.float16 if args.fp16 else torch.float32

            with autocast(device.type, dtype=dtype, enabled=args.bf16 or args.fp16):
                for i, idx in enumerate(indices):
                    sample = eval_dataset[idx]
                    mel = sample["mel_spec"]
                    speaker_embedding = sample.get("speaker_embedding", None)

                    # Ensure correct shape [1, n_mels, T]
                    if mel.dim() == 2:
                        mel = mel.unsqueeze(0)

                    mel_length = sample.get("mel_spec_length", mel.shape[-1])

                    # Trim to actual length before inference (avoid wasted compute on padding)
                    mel = mel[..., :mel_length]

                    # Prepare speaker embedding if available
                    spk_emb = None
                    if speaker_embedding is not None:
                        spk_emb = speaker_embedding.unsqueeze(0).to(device)
                        if spk_emb.dim() == 2:
                            spk_emb = spk_emb.unsqueeze(1)  # [B, 1, D]

                    # Use reconstruct_with_attention to get attention weights
                    mel_input = mel.unsqueeze(0).to(device)
                    recon, mu, logvar, enc_attn, dec_attn = model.reconstruct_with_attention(
                        mel_input,
                        speaker_embedding=spk_emb,
                    )

                    # Determine which speaker embedding to use for decode:
                    # If model learns speaker embedding, encode to get learned embedding
                    # Otherwise use the pretrained speaker embedding from dataset
                    encoder_learns_speaker = hasattr(model.encoder, 'learn_speaker_embedding') and model.encoder.learn_speaker_embedding
                    if encoder_learns_speaker:
                        # Get learned speaker embedding from encoder
                        enc_result = model.encode(mel_input)
                        # encode returns (mu, logvar, learned_speaker_emb) when learn_speaker_embedding=True
                        learned_spk_emb = enc_result[2] if len(enc_result) > 2 else None
                        decode_spk_emb = learned_spk_emb if learned_spk_emb is not None else spk_emb
                    else:
                        decode_spk_emb = spk_emb

                    # Store sample data for cross-speaker reconstruction later
                    eval_samples_data.append({
                        "mel": mel,  # [1, n_mels, T]
                        "speaker_embedding": speaker_embedding,  # [192] or None (pretrained)
                        "mu": mu.cpu(),  # [1, C, H, W]
                        "mel_length": mel_length,
                    })

                    # Generate mu-only reconstruction (no sampling, z = mu)
                    # This is what diffusion will see during inference
                    recon_mu_only = model.decode(mu, speaker_embedding=decode_spk_emb)

                    # Debug: print stats for first sample to diagnose mu_only vs reparameterized difference
                    if i == 0:
                        z = model.reparameterize(mu, logvar)
                        std = torch.exp(0.5 * logvar)
                        print(f"[DEBUG mu_only] decoder.speaker_embedding_dim: {model.decoder.speaker_embedding_dim}")
                        if spk_emb is not None:
                            print(f"[DEBUG mu_only] spk_emb: shape={spk_emb.shape}, "
                                  f"mean={spk_emb.mean().item():.4f}, std={spk_emb.std().item():.4f}")
                        else:
                            print("[DEBUG mu_only] spk_emb: None")
                        print(f"[DEBUG mu_only] mu: mean={mu.mean().item():.4f}, std={mu.std().item():.4f}, "
                              f"min={mu.min().item():.4f}, max={mu.max().item():.4f}")
                        print(f"[DEBUG mu_only] logvar: mean={logvar.mean().item():.4f}, "
                              f"std_from_logvar: mean={std.mean().item():.4f}, min={std.min().item():.6f}, max={std.max().item():.4f}")
                        print(f"[DEBUG mu_only] z: mean={z.mean().item():.4f}, std={z.std().item():.4f}")
                        print(f"[DEBUG mu_only] recon (from reconstruct_with_attention): "
                              f"mean={recon.mean().item():.4f}, std={recon.std().item():.4f}, "
                              f"min={recon.min().item():.4f}, max={recon.max().item():.4f}")
                        print(f"[DEBUG mu_only] recon_mu_only: "
                              f"mean={recon_mu_only.mean().item():.4f}, std={recon_mu_only.std().item():.4f}, "
                              f"min={recon_mu_only.min().item():.4f}, max={recon_mu_only.max().item():.4f}")
                        # Also check if recon from decode(z) matches reconstruct_with_attention
                        recon_z_direct = model.decode(z, speaker_embedding=decode_spk_emb)
                        print(f"[DEBUG mu_only] recon from decode(z): "
                              f"mean={recon_z_direct.mean().item():.4f}, std={recon_z_direct.std().item():.4f}")

                    # Compute losses manually for logging
                    losses = {}
                    losses["mse_loss"] = F.mse_loss(recon, mel_input)
                    losses["kl_divergence"] = -0.5 * torch.mean(
                        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3])
                    )

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
                    recon_mu_only_cpu = recon_mu_only[0].squeeze(0).float().cpu().numpy()

                    # Trimmed versions (without padding)
                    mel_trimmed = mel_cpu[..., :mel_length]
                    recon_trimmed = recon_cpu[..., :mel_length]
                    recon_mu_only_trimmed = recon_mu_only_cpu[..., :mel_length]

                    # Log individual mel spectrograms
                    writer.add_image(
                        f"eval_vae/original/{i}",
                        self._visualize_mel_spec(mel_trimmed),
                        global_step
                    )
                    writer.add_image(
                        f"eval_vae/reconstruction/{i}",
                        self._visualize_mel_spec(recon_trimmed),
                        global_step
                    )
                    writer.add_image(
                        f"eval_vae/reconstruction_mu_only/{i}",
                        self._visualize_mel_spec(recon_mu_only_trimmed),
                        global_step
                    )

                    # Log trimmed comparison
                    self._log_mel_comparison(
                        writer, recon_trimmed, mel_trimmed, global_step,
                        tag=f"eval_vae/comparison/{i}"
                    )
                    self._log_mel_comparison(
                        writer, recon_mu_only_trimmed, mel_trimmed, global_step,
                        tag=f"eval_vae/comparison_mu_only/{i}"
                    )

                    # Log per-example losses (skip non-scalar values like learned_speaker_embedding)
                    for loss_name, loss_val in losses.items():
                        if isinstance(loss_val, torch.Tensor):
                            if loss_val.numel() != 1:
                                continue
                            loss_val = loss_val.item()
                        elif not isinstance(loss_val, (int, float)):
                            continue
                        writer.add_scalar(f"eval_vae/example_{i}/{loss_name}", loss_val, global_step)

                    # Log latent channel visualizations for first few samples
                    if i < 4:
                        mu_sample = mu[0].float().cpu()  # [latent_channels, H, W]
                        mu_min, mu_max = mu_sample.min(), mu_sample.max()
                        mu_norm = (mu_sample - mu_min) / (mu_max - mu_min + 1e-5)
                        for c in range(min(mu_norm.shape[0], 8)):  # Limit to first 8 channels
                            writer.add_image(
                                f"eval_vae/example_{i}/mu_channel_{c}",
                                mu_norm[c:c+1, :, :],
                                global_step
                            )

                    # Log attention weights for first few samples
                    if i < 4:
                        # Compute downsampled T for trimming padding from attention visualizations
                        # Time strides depend on config, but for "small" it's 5*5*1 = 25x
                        # We can infer this from the attention shape
                        enc_T_actual = None
                        if hasattr(model, 'encoder') and hasattr(model.encoder, 'time_strides'):
                            # Compute downsampled length
                            T_down = mel_length
                            for stride in model.encoder.time_strides:
                                T_down = (T_down + stride - 1) // stride
                            enc_T_actual = T_down

                        self._log_attention_weights(
                            writer, enc_attn, global_step,
                            tag_prefix=f"eval_vae/example_{i}/encoder_attention",
                            T_actual=enc_T_actual,
                        )
                        self._log_attention_weights(
                            writer, dec_attn, global_step,
                            tag_prefix=f"eval_vae/example_{i}/decoder_attention",
                            T_actual=enc_T_actual,  # Decoder bottleneck has same resolution
                        )

                    # Convert mel spectrograms to audio using vocoder
                    if self.vocoder is not None:
                        mel_tensor = mel.squeeze(0).float().cpu()
                        recon_mel_tensor = recon[0].squeeze(0).float().cpu()
                        recon_mu_only_mel_tensor = recon_mu_only[0].squeeze(0).float().cpu()

                        # Log ground truth audio
                        self._log_vocoder_audio(
                            writer, mel_tensor[..., :mel_length], global_step,
                            tag=f"eval_vae/original_audio/{i}"
                        )
                        # Log reconstruction audio
                        self._log_vocoder_audio(
                            writer, recon_mel_tensor[..., :mel_length], global_step,
                            tag=f"eval_vae/recon_audio/{i}"
                        )
                        # Log mu-only reconstruction audio (what diffusion will produce)
                        self._log_vocoder_audio(
                            writer, recon_mu_only_mel_tensor[..., :mel_length], global_step,
                            tag=f"eval_vae/recon_mu_only_audio/{i}"
                        )

        # Log aggregate statistics
        for loss_name, loss_vals in all_losses.items():
            writer.add_scalar(f"eval_vae/mean_{loss_name}", np.mean(loss_vals), global_step)
            writer.add_scalar(f"eval_vae/std_{loss_name}", np.std(loss_vals), global_step)

        writer.add_scalar("eval_vae/mean_mu_mean", np.mean(all_mu_means), global_step)
        writer.add_scalar("eval_vae/mean_mu_std", np.mean(all_mu_stds), global_step)
        writer.add_scalar("eval_vae/mean_logvar_mean", np.mean(all_logvar_means), global_step)

        # Cross-speaker reconstruction on eval samples
        # Select samples that have speaker embeddings
        samples_with_speakers = [
            (i, s) for i, s in enumerate(eval_samples_data)
            if s["speaker_embedding"] is not None
        ]

        # Check if model uses learned speaker embeddings
        encoder_learns_speaker = hasattr(model.encoder, 'learn_speaker_embedding') and model.encoder.learn_speaker_embedding

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
                        mu_a = sample_a["mu"].to(device)

                        # Get speaker embedding for B (encode to get learned, or use pretrained)
                        if encoder_learns_speaker:
                            # Encode B's mel to get learned speaker embedding
                            mel_b_input = sample_b["mel"].unsqueeze(0).to(device)
                            enc_result_b = model.encode(mel_b_input)
                            spk_emb_b = enc_result_b[2]  # learned_speaker_emb
                        else:
                            spk_emb_b = sample_b["speaker_embedding"].unsqueeze(0).to(device)
                            if spk_emb_b.dim() == 2:
                                spk_emb_b = spk_emb_b.unsqueeze(1)  # [1, 1, 192]

                        cross_recon_a_with_b = model.decode(mu_a, speaker_embedding=spk_emb_b)

                        # Reconstruct B's content with A's speaker embedding
                        mu_b = sample_b["mu"].to(device)

                        # Get speaker embedding for A (encode to get learned, or use pretrained)
                        if encoder_learns_speaker:
                            # Encode A's mel to get learned speaker embedding
                            mel_a_input = sample_a["mel"].unsqueeze(0).to(device)
                            enc_result_a = model.encode(mel_a_input)
                            spk_emb_a = enc_result_a[2]  # learned_speaker_emb
                        else:
                            spk_emb_a = sample_a["speaker_embedding"].unsqueeze(0).to(device)
                            if spk_emb_a.dim() == 2:
                                spk_emb_a = spk_emb_a.unsqueeze(1)  # [1, 1, 192]

                        cross_recon_b_with_a = model.decode(mu_b, speaker_embedding=spk_emb_a)

                        # Log A with B's speaker
                        mel_a_trimmed = sample_a["mel"].squeeze(0).cpu().numpy()[..., :sample_a["mel_length"]]
                        cross_a_trimmed = cross_recon_a_with_b[0].squeeze(0).float().cpu().numpy()[..., :sample_a["mel_length"]]

                        writer.add_image(
                            f"eval_vae/cross_speaker/pair{pair_idx}_content{sample_a_idx}_spk{sample_b_idx}/original",
                            self._visualize_mel_spec(mel_a_trimmed),
                            global_step
                        )
                        writer.add_image(
                            f"eval_vae/cross_speaker/pair{pair_idx}_content{sample_a_idx}_spk{sample_b_idx}/reconstruction",
                            self._visualize_mel_spec(cross_a_trimmed),
                            global_step
                        )
                        self._log_mel_comparison(
                            writer, cross_a_trimmed, mel_a_trimmed, global_step,
                            tag=f"eval_vae/cross_speaker/pair{pair_idx}_content{sample_a_idx}_spk{sample_b_idx}/comparison"
                        )

                        # Log B with A's speaker
                        mel_b_trimmed = sample_b["mel"].squeeze(0).cpu().numpy()[..., :sample_b["mel_length"]]
                        cross_b_trimmed = cross_recon_b_with_a[0].squeeze(0).float().cpu().numpy()[..., :sample_b["mel_length"]]

                        writer.add_image(
                            f"eval_vae/cross_speaker/pair{pair_idx}_content{sample_b_idx}_spk{sample_a_idx}/original",
                            self._visualize_mel_spec(mel_b_trimmed),
                            global_step
                        )
                        writer.add_image(
                            f"eval_vae/cross_speaker/pair{pair_idx}_content{sample_b_idx}_spk{sample_a_idx}/reconstruction",
                            self._visualize_mel_spec(cross_b_trimmed),
                            global_step
                        )
                        self._log_mel_comparison(
                            writer, cross_b_trimmed, mel_b_trimmed, global_step,
                            tag=f"eval_vae/cross_speaker/pair{pair_idx}_content{sample_b_idx}_spk{sample_a_idx}/comparison"
                        )

                        # Log audio if vocoder available
                        if self.vocoder is not None:
                            self._log_vocoder_audio(
                                writer,
                                cross_recon_a_with_b[0].squeeze(0).float().cpu()[..., :sample_a["mel_length"]],
                                global_step,
                                tag=f"eval_vae/cross_speaker/pair{pair_idx}_content{sample_a_idx}_spk{sample_b_idx}/audio"
                            )
                            self._log_vocoder_audio(
                                writer,
                                cross_recon_b_with_a[0].squeeze(0).float().cpu()[..., :sample_b["mel_length"]],
                                global_step,
                                tag=f"eval_vae/cross_speaker/pair{pair_idx}_content{sample_b_idx}_spk{sample_a_idx}/audio"
                            )

            print(f"Cross-speaker reconstruction complete: {num_pairs} pairs logged")

        print(f"Eval visualization complete: {num_samples} samples logged")
        writer.flush()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        global_step = state.global_step + self.step_offset

        if ((global_step == 1) or (global_step % self.generation_steps == 0)) and state.is_world_process_zero:
            writer = get_writer(self.trainer)
            if writer is None:
                print("No TensorBoard writer found, skipping visualization...")
                return

            if len(self.example_mels) == 0:
                print("No example mels loaded, skipping visualization...")
                return

            print(f"Generating mel reconstructions at step {global_step}...")

            # Lazily load vocoder
            self._load_vocoder()

            # Determine device
            if torch.distributed.is_initialized():
                device = torch.device(f"cuda:{torch.distributed.get_rank()}")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            # Load pre-extracted speaker embeddings
            self._load_speaker_embeddings(device)

            with torch.no_grad():
                dtype = torch.bfloat16 if bool(args.bf16) else torch.float16 if args.fp16 else torch.float32

                with autocast(device.type, dtype=dtype):
                    for i, mel in enumerate(self.example_mels):
                        # Get original length for this example
                        original_length = self.example_original_lengths[i] if i < len(self.example_original_lengths) else mel.shape[-1]

                        # Trim to actual length before inference (avoid wasted compute on padding)
                        mel_trimmed = mel[..., :original_length]

                        # Get speaker embedding for this example (if available)
                        spk_emb = None
                        if i < len(self.example_speaker_embeddings):
                            spk_emb = self.example_speaker_embeddings[i].unsqueeze(0).to(device)  # [1, 192]
                            if spk_emb.dim() == 2:
                                spk_emb = spk_emb.unsqueeze(1)  # [1, 1, 192]

                        recon, mu, logvar, losses = model(
                            mel_trimmed.unsqueeze(0).to(device),
                            speaker_embedding=spk_emb,
                        )

                        # Determine which speaker embedding to use for decode:
                        # If model learns speaker embedding, use the learned one from encoder
                        # Otherwise use the pretrained speaker embedding from dataset
                        learned_spk_emb = losses.get("learned_speaker_embedding", None)
                        decode_spk_emb = learned_spk_emb if learned_spk_emb is not None else spk_emb

                        # Generate mu-only reconstruction (no sampling, z = mu)
                        # This is what diffusion will see during inference
                        recon_mu_only = model.decode(mu, speaker_embedding=decode_spk_emb)

                        # Get original length for this example
                        original_length = self.example_original_lengths[i] if i < len(self.example_original_lengths) else mel.shape[-1]

                        # Get tensors for visualization
                        mel_cpu = mel.squeeze(0).cpu().numpy()  # [n_mels, T]
                        recon_cpu = recon[0].squeeze(0).float().cpu().numpy()  # [n_mels, T]
                        recon_mu_only_cpu = recon_mu_only[0].squeeze(0).float().cpu().numpy()  # [n_mels, T]

                        # Trimmed versions (without padding)
                        mel_trimmed = mel_cpu[..., :original_length]
                        recon_trimmed = recon_cpu[..., :original_length]
                        recon_mu_only_trimmed = recon_mu_only_cpu[..., :original_length]

                        # Log trimmed (content only) mel spectrograms
                        writer.add_image(
                            f"audio_vae/original_trimmed/{i}",
                            self._visualize_mel_spec(mel_trimmed),
                            global_step
                        )
                        writer.add_image(
                            f"audio_vae/recon_trimmed/{i}",
                            self._visualize_mel_spec(recon_trimmed),
                            global_step
                        )
                        writer.add_image(
                            f"audio_vae/recon_mu_only_trimmed/{i}",
                            self._visualize_mel_spec(recon_mu_only_trimmed),
                            global_step
                        )

                        # Log trimmed comparison
                        self._log_mel_comparison(
                            writer, recon_trimmed, mel_trimmed, global_step,
                            tag=f"audio_vae/comparison_trimmed/{i}"
                        )
                        self._log_mel_comparison(
                            writer, recon_mu_only_trimmed, mel_trimmed, global_step,
                            tag=f"audio_vae/comparison_mu_only_trimmed/{i}"
                        )

                        # Log padded (full) mel spectrograms for diagnosing silence reconstruction
                        writer.add_image(
                            f"audio_vae/original_padded/{i}",
                            self._visualize_mel_spec(mel_cpu),
                            global_step
                        )
                        writer.add_image(
                            f"audio_vae/recon_padded/{i}",
                            self._visualize_mel_spec(recon_cpu),
                            global_step
                        )

                        # Log padded comparison
                        self._log_mel_comparison(
                            writer, recon_cpu, mel_cpu, global_step,
                            tag=f"audio_vae/comparison_padded/{i}"
                        )

                        # Log per-example losses (skip non-scalar values like learned_speaker_embedding)
                        for loss_name, loss_val in losses.items():
                            if isinstance(loss_val, torch.Tensor):
                                if loss_val.numel() != 1:
                                    continue  # Skip non-scalar tensors
                                loss_val = loss_val.item()
                            elif not isinstance(loss_val, (int, float)):
                                continue  # Skip non-numeric values
                            writer.add_scalar(f"audio_vae/example_{i}/{loss_name}", loss_val, global_step)

                        # Log latent channel visualizations
                        mu_sample = mu[0].float().cpu()  # [latent_channels, H, W]
                        mu_min, mu_max = mu_sample.min(), mu_sample.max()
                        mu_norm = (mu_sample - mu_min) / (mu_max - mu_min + 1e-5)
                        for c in range(mu_norm.shape[0]):
                            writer.add_image(
                                f"audio_vae/example_{i}/mu_channel_{c}",
                                mu_norm[c:c+1, :, :],
                                global_step
                            )

                        # Convert reconstructed mel to audio using vocoder (trimmed version)
                        if self.vocoder is not None:
                            recon_mel_tensor = recon[0].squeeze(0).float().cpu()  # [n_mels, T]
                            recon_mu_only_mel_tensor = recon_mu_only[0].squeeze(0).float().cpu()  # [n_mels, T]
                            # Log trimmed audio
                            self._log_vocoder_audio(
                                writer, recon_mel_tensor[..., :original_length], global_step,
                                tag=f"audio_vae/recon_audio_trimmed/{i}"
                            )
                            # Log mu-only trimmed audio (what diffusion will produce)
                            self._log_vocoder_audio(
                                writer, recon_mu_only_mel_tensor[..., :original_length], global_step,
                                tag=f"audio_vae/recon_mu_only_audio_trimmed/{i}"
                            )
                            # Log full padded audio for comparison
                            self._log_vocoder_audio(
                                writer, recon_mel_tensor, global_step,
                                tag=f"audio_vae/recon_audio_padded/{i}"
                            )

                    # Cross-speaker reconstruction: reconstruct each mel with the OTHER speaker's embedding
                    # This tests speaker disentanglement - content should be preserved, voice should change
                    encoder_learns_speaker = hasattr(model.encoder, 'learn_speaker_embedding') and model.encoder.learn_speaker_embedding

                    if len(self.example_mels) >= 2 and len(self.example_speaker_embeddings) >= 2:
                        print("Generating cross-speaker reconstructions...")
                        cross_pairs = [
                            (0, 1),  # mel 0 with speaker embedding 1
                            (1, 0),  # mel 1 with speaker embedding 0
                        ]

                        for mel_idx, spk_idx in cross_pairs:
                            mel = self.example_mels[mel_idx]
                            mel_input = mel.unsqueeze(0).to(device)

                            # Encode the mel (get latent representation)
                            enc_result = model.encode(mel_input)
                            if encoder_learns_speaker:
                                mu, logvar, _ = enc_result  # Ignore current mel's learned speaker emb
                            else:
                                mu, logvar = enc_result

                            # Get cross speaker embedding (from the OTHER sample)
                            if encoder_learns_speaker:
                                # Encode the OTHER mel to get its learned speaker embedding
                                other_mel_input = self.example_mels[spk_idx].unsqueeze(0).to(device)
                                other_enc_result = model.encode(other_mel_input)
                                cross_spk_emb = other_enc_result[2]  # learned_speaker_emb
                            else:
                                cross_spk_emb = self.example_speaker_embeddings[spk_idx].unsqueeze(0).to(device)  # [1, 192]
                                if cross_spk_emb.dim() == 2:
                                    cross_spk_emb = cross_spk_emb.unsqueeze(1)  # [1, 1, 192]

                            # Decode with the OTHER speaker's embedding
                            cross_recon = model.decode(mu, speaker_embedding=cross_spk_emb)

                            # Get original length for this mel
                            original_length = self.example_original_lengths[mel_idx] if mel_idx < len(self.example_original_lengths) else mel.shape[-1]

                            # Get tensors for visualization
                            mel_cpu = mel.squeeze(0).cpu().numpy()  # [n_mels, T]
                            cross_recon_cpu = cross_recon[0].squeeze(0).float().cpu().numpy()  # [n_mels, T]

                            # Trimmed versions (without padding)
                            mel_trimmed = mel_cpu[..., :original_length]
                            cross_recon_trimmed = cross_recon_cpu[..., :original_length]

                            # Log cross-speaker reconstruction
                            tag_suffix = f"mel{mel_idx}_spk{spk_idx}"
                            writer.add_image(
                                f"audio_vae/cross_speaker/{tag_suffix}/original",
                                self._visualize_mel_spec(mel_trimmed),
                                global_step
                            )
                            writer.add_image(
                                f"audio_vae/cross_speaker/{tag_suffix}/reconstruction",
                                self._visualize_mel_spec(cross_recon_trimmed),
                                global_step
                            )
                            self._log_mel_comparison(
                                writer, cross_recon_trimmed, mel_trimmed, global_step,
                                tag=f"audio_vae/cross_speaker/{tag_suffix}/comparison"
                            )

                            # Convert to audio using vocoder
                            if self.vocoder is not None:
                                cross_recon_mel_tensor = cross_recon[0].squeeze(0).float().cpu()
                                self._log_vocoder_audio(
                                    writer, cross_recon_mel_tensor[..., :original_length], global_step,
                                    tag=f"audio_vae/cross_speaker/{tag_suffix}/audio"
                                )

                        print("Cross-speaker reconstructions complete")

    def _visualize_mel_spec(self, mel_spec: np.ndarray) -> np.ndarray:
        """Generate mel spectrogram visualization for TensorBoard."""
        if mel_spec.ndim == 3:
            mel_spec = mel_spec.squeeze(0)

        # Normalize to [0, 1] range for visualization
        mel_spec_norm = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(
            mel_spec_norm,
            hop_length=self.audio_hop_length,
            x_axis='time',
            y_axis='mel',
            sr=self.audio_sample_rate,
            n_fft=self.audio_n_fft,
            fmin=0,
            fmax=8000,
            ax=ax,
        )
        plt.colorbar(img, format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.tight_layout()

        # Convert figure to numpy array
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape((height, width, 4))[:, :, :3]  # Drop alpha channel

        plt.close(fig)

        # TensorBoard expects (C, H, W) for add_image
        data = data.transpose(2, 0, 1)

        return data

    def _log_attention_weights(
        self,
        writer: SummaryWriter,
        attn_dict: Dict[str, Optional[torch.Tensor]],
        global_step: int,
        tag_prefix: str,
        M: int = 10,  # Expected mel bins in bottleneck (default for small config)
        T_actual: Optional[int] = None,  # Actual T (without padding) for trimming
    ):
        """
        Log 2D attention weight visualizations to TensorBoard.

        Args:
            writer: TensorBoard writer
            attn_dict: Dictionary with "weights" key containing attention tensor
                      Shape: [B, n_heads, M*T, M*T] for full 2D attention with RoPE
            global_step: Current training step
            tag_prefix: Tag prefix for TensorBoard (e.g., "eval_vae/example_0/encoder_attention")
            M: Number of mel bins in bottleneck (used to infer T from sequence length)
            T_actual: Actual valid timesteps (before padding). If provided, visualizations
                      are trimmed to show only valid positions.
        """
        weights = attn_dict.get("weights", None)
        if weights is None:
            return

        # Move to CPU and convert to numpy
        # Shape: [n_heads, seq_len, seq_len] where seq_len = M * T
        weights = weights[0].float().cpu().numpy()
        n_heads, seq_len, _ = weights.shape

        # Infer T from seq_len = M * T
        T_full = seq_len // M

        # Use T_actual if provided, otherwise use full T
        T = T_actual if T_actual is not None else T_full

        # Build mask for valid positions if we have padding
        # Positions are ordered as (m=0, t=0), (m=0, t=1), ..., (m=M-1, t=T-1)
        # Valid positions are those where t < T_actual
        if T_actual is not None and T_actual < T_full:
            # Create indices for valid positions
            valid_indices = []
            for m in range(M):
                for t in range(T_actual):
                    valid_indices.append(m * T_full + t)
            valid_indices = np.array(valid_indices)

            # Slice out valid positions from attention weights
            # weights shape: [n_heads, M*T_full, M*T_full] -> [n_heads, M*T_actual, M*T_actual]
            weights = weights[:, valid_indices, :][:, :, valid_indices]

        # 1. Global average attention map (avg across heads)
        global_avg_weights = weights.mean(axis=0)  # [M*T, M*T]

        # Log full 2D attention map
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(global_avg_weights, aspect='auto', origin='lower', cmap='viridis')
        title_suffix = " (trimmed)" if T_actual is not None and T_actual < T_full else ""
        ax.set_title(f'2D Attention (avg {n_heads} heads, M={M}, T={T}){title_suffix}')
        ax.set_xlabel('Key position (flattened M×T)')
        ax.set_ylabel('Query position (flattened M×T)')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        writer.add_figure(f"{tag_prefix}/global_2d", fig, global_step)
        plt.close(fig)

        # 2. Cross-frequency attention analysis
        # For each time position, look at how different mel bins attend to each other
        # Sum attention from (m, t) to (m', t) for same t to see freq-freq patterns
        freq_freq_attn = np.zeros((M, M))
        for t in range(T):
            for m_q in range(M):
                for m_k in range(M):
                    q_idx = m_q * T + t
                    k_idx = m_k * T + t
                    freq_freq_attn[m_q, m_k] += global_avg_weights[q_idx, k_idx]
        freq_freq_attn /= T  # Average over time positions

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(freq_freq_attn, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f'Cross-Frequency Attention (same timestep){title_suffix}')
        ax.set_xlabel('Key mel bin')
        ax.set_ylabel('Query mel bin')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        writer.add_figure(f"{tag_prefix}/cross_freq", fig, global_step)
        plt.close(fig)

        # 3. Cross-time attention analysis
        # For each mel bin, look at how different time positions attend to each other
        # Sum attention from (m, t) to (m, t') for same m to see time-time patterns
        time_time_attn = np.zeros((T, T))
        for m in range(M):
            for t_q in range(T):
                for t_k in range(T):
                    q_idx = m * T + t_q
                    k_idx = m * T + t_k
                    time_time_attn[t_q, t_k] += global_avg_weights[q_idx, k_idx]
        time_time_attn /= M  # Average over mel bins

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(time_time_attn, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f'Cross-Time Attention (same mel bin){title_suffix}')
        ax.set_xlabel('Key timestep')
        ax.set_ylabel('Query timestep')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        writer.add_figure(f"{tag_prefix}/cross_time", fig, global_step)
        plt.close(fig)

        # 4. Per-head attention maps (first 4 heads)
        num_heads_to_log = min(n_heads, 4)
        fig, axes = plt.subplots(1, num_heads_to_log, figsize=(4 * num_heads_to_log, 4))
        if num_heads_to_log == 1:
            axes = [axes]
        for h in range(num_heads_to_log):
            im = axes[h].imshow(weights[h], aspect='auto', origin='lower', cmap='viridis')
            axes[h].set_title(f'Head {h}')
            axes[h].set_xlabel('Key')
            axes[h].set_ylabel('Query')
        plt.tight_layout()
        writer.add_figure(f"{tag_prefix}/per_head", fig, global_step)
        plt.close(fig)

        # 5. Log attention statistics (on valid positions only)
        writer.add_scalar(f"{tag_prefix}/mean", float(global_avg_weights.mean()), global_step)
        writer.add_scalar(f"{tag_prefix}/max", float(global_avg_weights.max()), global_step)
        writer.add_scalar(f"{tag_prefix}/entropy", float(self._attention_entropy(global_avg_weights)), global_step)

        # Log diagonal strength (how much attention stays on same position)
        diag_strength = np.diag(global_avg_weights).mean()
        writer.add_scalar(f"{tag_prefix}/diag_strength", float(diag_strength), global_step)

        # Log cross-freq vs cross-time attention balance
        cross_freq_strength = freq_freq_attn.sum() / (M * M)
        cross_time_strength = time_time_attn.sum() / (T * T)
        writer.add_scalar(f"{tag_prefix}/cross_freq_strength", float(cross_freq_strength), global_step)
        writer.add_scalar(f"{tag_prefix}/cross_time_strength", float(cross_time_strength), global_step)

    def _attention_entropy(self, attn_weights: np.ndarray) -> float:
        """
        Compute average entropy of attention distributions.
        Higher entropy = more uniform attention, lower = more peaked.
        """
        # Clip to avoid log(0)
        attn_clipped = np.clip(attn_weights, 1e-10, 1.0)
        # Normalize rows to sum to 1 (they should already, but just in case)
        row_sums = attn_clipped.sum(axis=-1, keepdims=True)
        attn_norm = attn_clipped / (row_sums + 1e-10)
        # Compute entropy per row, then average
        entropy_per_row = -np.sum(attn_norm * np.log(attn_norm + 1e-10), axis=-1)
        return float(entropy_per_row.mean())

    def _log_mel_comparison(self, writer: SummaryWriter, pred_mel: np.ndarray, target_mel: np.ndarray, global_step: int, tag: str):
        """Log side-by-side comparison of predicted and target mel spectrograms."""
        if pred_mel.ndim == 3:
            pred_mel = pred_mel.squeeze(0)
        if target_mel.ndim == 3:
            target_mel = target_mel.squeeze(0)

        # Align lengths
        min_len = min(pred_mel.shape[-1], target_mel.shape[-1])
        pred_mel = pred_mel[..., :min_len]
        target_mel = target_mel[..., :min_len]

        # Create comparison figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Normalize for visualization
        vmin = min(pred_mel.min(), target_mel.min())
        vmax = max(pred_mel.max(), target_mel.max())

        im0 = axes[0].imshow(target_mel, aspect='auto', origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
        axes[0].set_title('Target Mel')
        axes[0].set_ylabel('Mel bin')
        axes[0].set_xlabel('Time frame')
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(pred_mel, aspect='auto', origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
        axes[1].set_title('Reconstructed Mel')
        axes[1].set_ylabel('Mel bin')
        axes[1].set_xlabel('Time frame')
        plt.colorbar(im1, ax=axes[1])

        # Error map
        error = np.abs(pred_mel - target_mel)
        im2 = axes[2].imshow(error, aspect='auto', origin='lower', cmap='viridis')
        axes[2].set_title(f'Absolute Error (mean={error.mean():.4f})')
        axes[2].set_ylabel('Mel bin')
        axes[2].set_xlabel('Time frame')
        plt.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        writer.add_figure(tag, fig, global_step)
        plt.close(fig)


class AudioVAEGANTrainer(Trainer):
    """
    Custom trainer for VAE with optional GAN training.
    Handles alternating generator/discriminator updates.

    Supports discriminator regularization:
    - Instance noise: adds Gaussian noise to both real and fake mel spectrograms
    - R1 gradient penalty: penalizes gradient norm on real mel spectrograms

    For sharded datasets, uses ShardAwareSampler to minimize shard loading overhead.

    Supports audio perceptual losses:
    - Multi-scale mel loss (works on mel spectrograms directly)
    - Wav2Vec2 perceptual loss (requires vocoder to convert to waveform)
    - PANNs perceptual loss (requires vocoder to convert to waveform)
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
        # Audio perceptual loss
        audio_perceptual_loss: Optional[torch.nn.Module] = None,
        audio_perceptual_loss_weight: float = 0.0,
        audio_perceptual_loss_start_step: int = 0,  # Step to start applying perceptual loss
        vocoder: Optional[torch.nn.Module] = None,  # For waveform-based losses
        # GRL speaker disentanglement
        speaker_classifier: Optional[torch.nn.Module] = None,
        speaker_classifier_optimizer: Optional[torch.optim.Optimizer] = None,
        grl_weight: float = 0.0,  # Weight for GRL loss (0 = disabled)
        grl_start_step: int = 0,  # Step to start GRL (let VAE learn first)
        grl_alpha_max: float = 1.0,  # Max gradient reversal strength
        grl_rampup_steps: int = 5000,  # Steps to ramp alpha from 0 to max
        # FiLM statistics tracking
        log_film_stats: bool = False,  # Whether to log FiLM scale/shift statistics
        # FiLM contrastive loss - encourages different speaker embeddings to produce different FiLM outputs
        film_contrastive_loss_weight: float = 0.0,  # Weight for FiLM contrastive loss (0 = disabled)
        film_contrastive_loss_start_step: int = 0,  # Step to start FiLM contrastive loss
        film_contrastive_margin_max: float = 0.1,  # Max margin value for hinge loss
        film_contrastive_margin_rampup_steps: int = 5000,  # Steps to ramp margin from 0 to max
        # Mu-only reconstruction loss (for diffusion compatibility)
        mu_only_recon_weight: float = 0.0,  # Weight for mu-only reconstruction loss (0 = disabled)
        # Learned speaker embedding classification (complementary to GRL)
        # GRL pushes speaker OUT of latents; this pulls speaker INTO learned embeddings
        learned_speaker_classifier: Optional[torch.nn.Module] = None,
        learned_speaker_classifier_optimizer: Optional[torch.optim.Optimizer] = None,
        speaker_id_loss_weight: float = 0.0,  # Weight for speaker ID loss (0 = disabled)
        speaker_id_loss_type: str = "arcface",  # "classifier" or "arcface"
        speaker_id_loss_start_step: int = 0,  # Step to start speaker ID loss (0 = from beginning)
        speaker_id_loss_rampup_steps: int = 0,  # Steps to ramp weight from 0 to max (0 = no rampup)
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

        # GRL speaker disentanglement settings
        self.speaker_classifier = speaker_classifier
        self.speaker_classifier_optimizer = speaker_classifier_optimizer
        self.grl_weight = grl_weight
        self.grl_start_step = grl_start_step
        self.grl_alpha_max = grl_alpha_max
        self.grl_rampup_steps = grl_rampup_steps
        self.grl_already_started = False  # Track when GRL training has actually started

        # FiLM statistics tracking
        self.log_film_stats = log_film_stats

        # FiLM contrastive loss settings
        self.film_contrastive_loss_weight = film_contrastive_loss_weight
        self.film_contrastive_loss_start_step = film_contrastive_loss_start_step
        self.film_contrastive_margin_max = film_contrastive_margin_max
        self.film_contrastive_margin_rampup_steps = film_contrastive_margin_rampup_steps

        # Mu-only reconstruction loss (for diffusion compatibility)
        self.mu_only_recon_weight = mu_only_recon_weight

        # Learned speaker embedding classification (complementary to GRL)
        self.learned_speaker_classifier = learned_speaker_classifier
        self.learned_speaker_classifier_optimizer = learned_speaker_classifier_optimizer
        self.speaker_id_loss_weight = speaker_id_loss_weight
        self.speaker_id_loss_type = speaker_id_loss_type  # "classifier" or "arcface"
        self.speaker_id_loss_start_step = speaker_id_loss_start_step
        self.speaker_id_loss_rampup_steps = speaker_id_loss_rampup_steps
        self.speaker_id_training_started = False  # Track when training has started (for checkpointing)

        self.has_logged_cli = False

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """
        Override to use shard-aware sampler for sharded datasets.

        This ensures samples are grouped by shard, minimizing disk I/O
        by loading each shard only once per epoch.
        """
        if self._shard_sampler is not None:
            # Update epoch for proper shuffling reproducibility
            epoch = 0
            if self.state is not None and self.state.epoch is not None:
                epoch = int(self.state.epoch)
            self._shard_sampler.set_epoch(epoch)
            return self._shard_sampler

        # Fall back to default sampler for non-sharded datasets
        return super()._get_train_sampler()

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.args.device}
            if self.is_deepspeed_enabled and (torch.is_floating_point(data)):
                kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        global_step = self.state.global_step + self.step_offset

        self._ensure_tensorboard_writer()

        # gets reset any time training is resumed; it can be assumed that the cli changed, so log at the step value it was resumed from
        if not self.has_logged_cli:
            self.writer.add_text("training/command_line", self.cmdline, global_step)
            self.writer.add_text("training/git_commit_hash", self.git_commit_hash, global_step)
            self.has_logged_cli = True

        mel_spec = inputs["mel_spec"]
        mel_spec_mask = inputs.get("mel_spec_mask", None)
        mel_spec_lengths = inputs.get("mel_spec_lengths", None)
        speaker_embedding = inputs.get("speaker_embedding", None)

        # Compute KL weight multiplier for KL annealing (ramps from 0 to 1)
        kl_weight_multiplier = 1.0
        if self.kl_annealing_steps > 0:
            kl_weight_multiplier = min(1.0, global_step / self.kl_annealing_steps)

        # Forward pass through VAE model (with optional mask for reconstruction loss and lengths for attention)
        # Request FiLM stats if logging is enabled
        recon, mu, logvar, losses = model(
            mel_spec,
            mask=mel_spec_mask,
            speaker_embedding=speaker_embedding,
            lengths=mel_spec_lengths,
            kl_weight_multiplier=kl_weight_multiplier,
            return_film_stats=self.log_film_stats,
        )

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
        if self.mu_only_recon_weight > 0:
            # Decode mu directly (no reparameterization noise)
            recon_mu_only = model.decode(mu.detach(), speaker_embedding=decode_speaker_embedding)[..., :mel_spec.shape[-1]]
            # Use same loss as main reconstruction (L1 or MSE depending on config)
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
                    real_fp32 = real_for_disc.float()
                    fake_fp32 = fake_for_disc.float()

                    d_loss, d_loss_dict = compute_mel_discriminator_loss(
                        self.discriminator,
                        real_mels=real_fp32,
                        fake_mels=fake_fp32,
                    )

                # R1 gradient penalty (on clean real mels, not noisy)
                r1_loss = torch.tensor(0.0, device=mel_spec.device)
                if self.r1_penalty_weight > 0 and global_step % self.r1_penalty_interval == 0:
                    r1_loss = r1_mel_gradient_penalty(mel_spec.float(), self.discriminator)
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
                g_gan_loss, g_loss_dict = compute_mel_generator_gan_loss(
                    self.discriminator,
                    real_mels=mel_spec,
                    fake_mels=recon,
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
            # Get waveforms if vocoder is available (needed for Wav2Vec2 and PANNs)
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
            )
            audio_perceptual_loss_value = audio_perceptual_losses.get("total_perceptual_loss", torch.tensor(0.0, device=mel_spec.device))
            total_loss = total_loss + self.audio_perceptual_loss_weight * audio_perceptual_loss_value

        # GRL speaker disentanglement loss
        # Uses gradient reversal to train encoder to remove speaker info from latents
        grl_loss = torch.tensor(0.0, device=mel_spec.device)
        speaker_classifier_acc = 0.0
        grl_enabled = (
            self.speaker_classifier is not None
            and self.grl_weight > 0
            and global_step >= self.grl_start_step
        )
        if grl_enabled:
            # Get speaker IDs from batch (if available)
            # Note: collator provides "speaker_ids" (plural) as a list
            speaker_ids = inputs.get("speaker_ids", None)
            if speaker_ids is not None and not isinstance(speaker_ids, torch.Tensor):
                speaker_ids = torch.tensor(speaker_ids, device=mel_spec.device, dtype=torch.long)
            if speaker_ids is not None:
                # Mark GRL as started (for checkpoint saving)
                if not self.grl_already_started:
                    # Get num_speakers from classifier output layer
                    num_speakers = self.speaker_classifier.fc_out.out_features
                    print(f"GRL training starting at step {global_step}")
                    print(f"  speaker_ids: min={speaker_ids.min().item()}, max={speaker_ids.max().item()}, "
                          f"unique={len(torch.unique(speaker_ids))}, batch_size={len(speaker_ids)}")
                    print(f"  classifier num_speakers: {num_speakers}")

                    # Validate speaker_ids are in valid range
                    if speaker_ids.min() < 0:
                        raise ValueError(f"speaker_ids contains negative values (min={speaker_ids.min().item()})")
                    if speaker_ids.max() >= num_speakers:
                        raise ValueError(
                            f"speaker_ids max ({speaker_ids.max().item()}) >= num_speakers ({num_speakers}). "
                            f"Either increase --num_speakers or check if speaker_ids are 1-indexed "
                            f"(should be 0-indexed for cross_entropy)."
                        )
                    self.grl_already_started = True

                # Ensure speaker classifier is on same device
                if next(self.speaker_classifier.parameters()).device != mel_spec.device:
                    self.speaker_classifier.to(mel_spec.device)

                # Compute GRL alpha (ramps from 0 to grl_alpha_max over grl_rampup_steps)
                steps_since_grl_start = global_step - self.grl_start_step
                grl_alpha = min(self.grl_alpha_max, self.grl_alpha_max * steps_since_grl_start / max(1, self.grl_rampup_steps))

                # Update speaker classifier FIRST (separate optimizer, before GRL forward)
                # This must happen before the GRL forward pass to avoid in-place modification errors
                # The classifier tries to MAXIMIZE accuracy, encoder tries to MINIMIZE it
                if self.speaker_classifier_optimizer is not None and torch.is_grad_enabled():
                    self.speaker_classifier_optimizer.zero_grad()
                    # Classifier loss without GRL (alpha=0 means no gradient reversal, mu detached)
                    classifier_logits = self.speaker_classifier(mu.detach(), alpha=0.0)
                    classifier_loss = F.cross_entropy(classifier_logits, speaker_ids)
                    classifier_loss.backward()
                    self.speaker_classifier_optimizer.step()

                # Forward pass through speaker classifier (with GRL)
                # mu shape: [B, C, M, T] - classifier expects this
                # This uses the UPDATED classifier weights after the step above
                speaker_logits = self.speaker_classifier(mu, alpha=grl_alpha)

                # Compute cross-entropy loss for speaker classification
                grl_loss = F.cross_entropy(speaker_logits, speaker_ids)

                # Compute accuracy for logging
                with torch.no_grad():
                    speaker_preds = speaker_logits.argmax(dim=-1)
                    speaker_classifier_acc = (speaker_preds == speaker_ids).float().mean().item()

                # Add to total loss (reversed gradients flow to encoder)
                total_loss = total_loss + self.grl_weight * grl_loss

                # Log GRL metrics
                if global_step % self.args.logging_steps == 0 and self.writer is not None:
                    prefix = "train/" if model.training else "eval/"
                    self._log_scalar(f"{prefix}grl/loss", grl_loss, global_step)
                    self._log_scalar(f"{prefix}grl/alpha", grl_alpha, global_step)
                    self._log_scalar(f"{prefix}grl/speaker_classifier_acc", speaker_classifier_acc, global_step, skip_zero=False)
                    self._log_scalar(f"{prefix}grl/weighted_loss", self.grl_weight * grl_loss, global_step)

        # Learned speaker embedding classification loss
        # Complementary to GRL: GRL pushes speaker info OUT of latents, this pulls it INTO the speaker head
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
            recon_shuffled = model.decode(mu.detach(), speaker_embedding=shuffled_speaker_embedding)

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
            if self.kl_annealing_steps > 0:
                self._log_scalar(f"{prefix}kl_weight_multiplier", kl_weight_multiplier, global_step)

            # Per-channel latent statistics (for detecting channel collapse)
            # mu shape: [B, C, M, T] - compute stats per channel (average over batch, mel, time)
            per_channel_mu_mean = mu.mean(dim=(0, 2, 3))  # [C]
            per_channel_mu_std = mu.std(dim=(0, 2, 3))  # [C]
            per_channel_var = logvar.exp().mean(dim=(0, 2, 3))  # [C]
            # Per-channel KL: 0.5 * (mu^2 + var - log(var) - 1), averaged over batch and spatial
            per_channel_kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).mean(dim=(0, 2, 3))  # [C]

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

    def is_gan_enabled(self, global_step: int, vae_loss: torch.Tensor) -> bool:
        """
        Check if GAN training should be enabled based on the configured conditions.

        Supports two modes:
        - "step": Start GAN training after a specific step
        - "reconstruction_criteria_met": Start when VAE loss drops below threshold

        Once GAN training starts, it stays enabled (via gan_already_started flag).
        """
        if self.discriminator is None:
            return False

        if self.gan_already_started:
            return True

        if self.gan_start_condition_key is None:
            # Legacy mode: always enabled if discriminator exists
            return True

        if self.gan_start_condition_key == "step":
            return global_step >= int(self.gan_start_condition_value)

        if self.gan_start_condition_key == "reconstruction_criteria_met":
            # Start GAN when VAE loss drops below threshold
            threshold = float(self.gan_start_condition_value)
            return vae_loss.item() < threshold

        return False

    def _log_scalar(self, tag, value, global_step, skip_zero=True):
        if self.writer is not None:
            if isinstance(value, torch.Tensor):
                value = value.item()
            # Skip zero values by default (for unused losses), but allow explicit logging of zeros
            if not skip_zero or value != 0.0:
                self.writer.add_scalar(tag, value, global_step)

    def _ensure_tensorboard_writer(self):
        if hasattr(self, "writer") and self.writer is not None:
            return

        for callback in self.callback_handler.callbacks:
            if isinstance(callback, TensorBoardCallback):
                self.writer = callback.tb_writer
                return

        self.writer = None

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
            mel_spec = inputs["mel_spec"]
            mel_spec_mask = inputs.get("mel_spec_mask", None)
            mel_spec_lengths = inputs.get("mel_spec_lengths", None)
            speaker_embedding = inputs.get("speaker_embedding", None)

            # Move to device
            mel_spec = mel_spec.to(self.args.device)
            if mel_spec_mask is not None:
                mel_spec_mask = mel_spec_mask.to(self.args.device)
            if mel_spec_lengths is not None:
                mel_spec_lengths = mel_spec_lengths.to(self.args.device)
            if speaker_embedding is not None:
                speaker_embedding = speaker_embedding.to(self.args.device)

            # Use autocast for mixed precision (same as training)
            dtype = torch.bfloat16 if self.args.bf16 else torch.float16 if self.args.fp16 else torch.float32
            with autocast(self.args.device.type, dtype=dtype, enabled=self.args.bf16 or self.args.fp16):
                # Forward pass
                _, _, _, losses = model(
                    mel_spec,
                    mask=mel_spec_mask,
                    speaker_embedding=speaker_embedding,
                    lengths=mel_spec_lengths,
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

        # Save speaker classifier (GRL) if enabled and training has started
        if self.speaker_classifier is not None and self.grl_already_started:
            os.makedirs(output_dir, exist_ok=True)
            speaker_classifier_path = os.path.join(output_dir, "speaker_classifier.pt")
            torch.save({
                "speaker_classifier_state_dict": self.speaker_classifier.state_dict(),
                "speaker_classifier_optimizer_state_dict": (
                    self.speaker_classifier_optimizer.state_dict()
                    if self.speaker_classifier_optimizer is not None else None
                ),
            }, speaker_classifier_path)
            print(f"Speaker classifier (GRL) saved to {speaker_classifier_path}")

        # Save learned speaker classifier (speaker ID on embeddings) if enabled and training has started
        if self.learned_speaker_classifier is not None and self.speaker_id_training_started:
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


def load_discriminator(
    resume_from_checkpoint: str,
    discriminator: torch.nn.Module,
    discriminator_optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.nn.Module, Optional[torch.optim.Optimizer], bool]:
    """
    Load discriminator from checkpoint if it exists.

    Handles errors gracefully - if loading fails, returns the fresh discriminator
    and continues training from scratch.
    """
    if resume_from_checkpoint is None:
        print("No checkpoint path provided, training discriminator from scratch")
        return discriminator, discriminator_optimizer, False

    discriminator_path = os.path.join(resume_from_checkpoint, "discriminator.pt")
    if os.path.exists(discriminator_path):
        print(f"Loading discriminator from {discriminator_path}")
        try:
            checkpoint = torch.load(discriminator_path, map_location=device, weights_only=True)
            discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

            if discriminator_optimizer is not None and checkpoint.get("discriminator_optimizer_state_dict"):
                try:
                    discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer_state_dict"])
                except Exception as e:
                    print(f"Warning: Failed to load discriminator optimizer state: {e}")
                    print("Continuing with fresh optimizer state...")

            return discriminator, discriminator_optimizer, True
        except Exception as e:
            print(f"Warning: Failed to load discriminator checkpoint: {e}")
            print("Continuing with fresh discriminator...")
            return discriminator, discriminator_optimizer, False

    print("No existing discriminator checkpoint found, training from scratch")
    return discriminator, discriminator_optimizer, False


def load_speaker_classifier(
    resume_from_checkpoint: str,
    speaker_classifier: torch.nn.Module,
    speaker_classifier_optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.nn.Module, Optional[torch.optim.Optimizer], bool]:
    """
    Load speaker classifier (GRL) from checkpoint if it exists.

    Handles errors gracefully - if loading fails, returns the fresh classifier
    and continues training from scratch.
    """
    if resume_from_checkpoint is None:
        print("No checkpoint path provided, training speaker classifier from scratch")
        return speaker_classifier, speaker_classifier_optimizer, False

    speaker_classifier_path = os.path.join(resume_from_checkpoint, "speaker_classifier.pt")
    if os.path.exists(speaker_classifier_path):
        print(f"Loading speaker classifier from {speaker_classifier_path}")
        try:
            checkpoint = torch.load(speaker_classifier_path, map_location=device, weights_only=True)
            speaker_classifier.load_state_dict(checkpoint["speaker_classifier_state_dict"])

            if speaker_classifier_optimizer is not None and checkpoint.get("speaker_classifier_optimizer_state_dict"):
                try:
                    speaker_classifier_optimizer.load_state_dict(checkpoint["speaker_classifier_optimizer_state_dict"])
                except Exception as e:
                    print(f"Warning: Failed to load speaker classifier optimizer state: {e}")
                    print("Continuing with fresh optimizer state...")

            return speaker_classifier, speaker_classifier_optimizer, True
        except Exception as e:
            print(f"Warning: Failed to load speaker classifier checkpoint: {e}")
            print("Continuing with fresh speaker classifier...")
            return speaker_classifier, speaker_classifier_optimizer, False

    print("No existing speaker classifier checkpoint found, training from scratch")
    return speaker_classifier, speaker_classifier_optimizer, False


def load_learned_speaker_classifier(
    resume_from_checkpoint: str,
    learned_speaker_classifier: torch.nn.Module,
    learned_speaker_classifier_optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.nn.Module, Optional[torch.optim.Optimizer], bool]:
    """
    Load learned speaker classifier (speaker ID on embeddings) from checkpoint if it exists.

    Handles errors gracefully - if loading fails, returns the fresh classifier
    and continues training from scratch.
    """
    if resume_from_checkpoint is None:
        print("No checkpoint path provided, training learned speaker classifier from scratch")
        return learned_speaker_classifier, learned_speaker_classifier_optimizer, False

    learned_speaker_classifier_path = os.path.join(resume_from_checkpoint, "learned_speaker_classifier.pt")
    if os.path.exists(learned_speaker_classifier_path):
        print(f"Loading learned speaker classifier from {learned_speaker_classifier_path}")
        try:
            checkpoint = torch.load(learned_speaker_classifier_path, map_location=device, weights_only=True)
            learned_speaker_classifier.load_state_dict(checkpoint["learned_speaker_classifier_state_dict"])

            if learned_speaker_classifier_optimizer is not None and checkpoint.get("learned_speaker_classifier_optimizer_state_dict"):
                try:
                    learned_speaker_classifier_optimizer.load_state_dict(checkpoint["learned_speaker_classifier_optimizer_state_dict"])
                except Exception as e:
                    print(f"Warning: Failed to load learned speaker classifier optimizer state: {e}")
                    print("Continuing with fresh optimizer state...")

            return learned_speaker_classifier, learned_speaker_classifier_optimizer, True
        except Exception as e:
            print(f"Warning: Failed to load learned speaker classifier checkpoint: {e}")
            print("Continuing with fresh learned speaker classifier...")
            return learned_speaker_classifier, learned_speaker_classifier_optimizer, False

    print("No existing learned speaker classifier checkpoint found, training from scratch")
    return learned_speaker_classifier, learned_speaker_classifier_optimizer, False


def main():
    args, unk = megatransformer_utils.parse_args()
    run_dir = os.path.join(args.logging_base_dir, args.run_name)

    # Select model configuration
    if args.config not in model_config_lookup:
        raise ValueError(f"Unknown audio VAE config: {args.config}. Available: {list(model_config_lookup.keys())}")

    # Parse extra arguments
    unk_dict = {}
    for i in range(0, len(unk), 2):
        unk_dict[unk[i].lstrip('-')] = unk[i+1]

    # Dataset settings
    use_sharded_dataset = unk_dict.get("use_sharded_dataset", "true").lower() == "true"
    train_cache_dir = unk_dict.get("train_cache_dir", "./cached_datasets/audio_vae_speaker_train")
    val_cache_dir = unk_dict.get("val_cache_dir", "./cached_datasets/audio_vae_speaker_val")

    # Audio settings (CLI args override unk_dict defaults)
    audio_max_frames = int(unk_dict.get("audio_max_frames", 1875))
    n_mels = int(unk_dict.get("n_mels", 80))
    audio_sample_rate = args.audio_sample_rate if args.audio_sample_rate is not None else int(unk_dict.get("audio_sample_rate", 16000))
    audio_n_fft = args.audio_n_fft if args.audio_n_fft is not None else int(unk_dict.get("audio_n_fft", 1024))
    audio_hop_length = args.audio_hop_length if args.audio_hop_length is not None else int(unk_dict.get("audio_hop_length", 256))

    # VAE settings
    latent_channels = int(unk_dict.get("latent_channels", 4))
    # Speaker encoder type determines embedding dimension
    speaker_encoder_type = unk_dict.get("speaker_encoder_type", "ecapa_tdnn")
    speaker_embedding_dim_default = 768 if speaker_encoder_type == "wavlm" else 192
    speaker_embedding_dim = int(unk_dict.get("speaker_embedding_dim", speaker_embedding_dim_default))
    # Optional projection to reduce speaker embedding dim before FiLM (0 = no projection)
    # Useful for reducing params when using large embeddings like WavLM (768-dim)
    speaker_embedding_proj_dim = int(unk_dict.get("speaker_embedding_proj_dim", 0))
    normalize_speaker_embedding = unk_dict.get("normalize_speaker_embedding", "true").lower() == "true"
    # FiLM bounding - prevents extreme scale/shift values that can cause artifacts
    # Scale bound of 0.5 means (1 + scale) ranges from 0.5 to 1.5 (never zeroes out)
    # Set to 0 to disable bounding (unbounded FiLM)
    film_scale_bound = float(unk_dict.get("film_scale_bound", 0.5))
    film_shift_bound = float(unk_dict.get("film_shift_bound", 0.5))
    # Zero-init FiLM output weights - forces model to learn FiLM from scratch instead of relying on bias
    zero_init_film_bias = unk_dict.get("zero_init_film_bias", "false").lower() == "true"
    # Remove bias from FiLM projections entirely - zero embedding = zero modulation (structurally enforced)
    film_no_bias = unk_dict.get("film_no_bias", "false").lower() == "true"

    # Learned speaker embedding: if True, encoder outputs a learned speaker embedding instead of using pretrained
    # The speaker head uses global pooling to remove temporal structure, outputting a single speaker vector
    # With GRL pushing speaker info out of latents, the speaker head learns to capture speaker characteristics
    learn_speaker_embedding = unk_dict.get("learn_speaker_embedding", "false").lower() == "true"
    learned_speaker_dim = int(unk_dict.get("learned_speaker_dim", 256))

    # FiLM contrastive loss - encourages different speaker embeddings to produce different FiLM outputs
    film_contrastive_loss_weight = float(unk_dict.get("film_contrastive_loss_weight", 0.0))
    film_contrastive_loss_start_step = int(unk_dict.get("film_contrastive_loss_start_step", 0))
    # FiLM contrastive margin scheduling (ramps from 0 to max, similar to GRL alpha)
    film_contrastive_margin_max = float(unk_dict.get("film_contrastive_margin_max", 0.1))
    film_contrastive_margin_rampup_steps = int(unk_dict.get("film_contrastive_margin_rampup_steps", 5000))

    # VAE loss weights
    recon_loss_weight = float(unk_dict.get("recon_loss_weight", 1.0))
    mse_loss_weight = float(unk_dict.get("mse_loss_weight", 1.0))
    l1_loss_weight = float(unk_dict.get("l1_loss_weight", 0.0))
    kl_divergence_loss_weight = float(unk_dict.get("kl_divergence_loss_weight", 1e-4))

    # Audio perceptual loss settings (speech-focused)
    # Total weight for all audio perceptual losses (0 = disabled)
    audio_perceptual_loss_weight = float(unk_dict.get("audio_perceptual_loss_weight", 0.0))
    # Step to start applying perceptual loss (0 = from start, >0 = delay to let L1/MSE settle)
    audio_perceptual_loss_start_step = int(unk_dict.get("audio_perceptual_loss_start_step", 0))
    # Individual component weights (relative to total audio perceptual loss weight)
    multi_scale_mel_weight = float(unk_dict.get("multi_scale_mel_weight", 1.0))
    wav2vec2_weight = float(unk_dict.get("wav2vec2_weight", 0.0))  # Requires vocoder for waveform
    panns_weight = float(unk_dict.get("panns_weight", 0.0))  # Requires panns-inference, vocoder
    speaker_embedding_weight = float(unk_dict.get("speaker_embedding_weight", 0.0))  # Requires speaker embedding model, vocoder
    # Wav2Vec2 model selection: 'facebook/wav2vec2-base' (~95M) or 'facebook/wav2vec2-large' (~317M)
    wav2vec2_model = unk_dict.get("wav2vec2_model", "facebook/wav2vec2-base")

    # Vocoder settings (optional - for audio generation during visualization AND perceptual loss)
    vocoder_checkpoint_path = unk_dict.get("vocoder_checkpoint_path", None)
    vocoder_config = unk_dict.get("vocoder_config", "tiny_attention_freq_domain_vocoder")

    # GAN training settings
    use_gan = unk_dict.get("use_gan", "false").lower() == "true"
    gan_start_condition_key = unk_dict.get("gan_start_condition_key", "step")  # "step" or "reconstruction_criteria_met"
    gan_start_condition_value = unk_dict.get("gan_start_condition_value", "0")  # step number or loss threshold
    discriminator_lr = float(unk_dict.get("discriminator_lr", 2e-4))
    gan_loss_weight = float(unk_dict.get("gan_loss_weight", 0.5))
    feature_matching_weight = float(unk_dict.get("feature_matching_weight", 0.0))
    discriminator_update_frequency = int(unk_dict.get("discriminator_update_frequency", 1))
    discriminator_config = unk_dict.get("discriminator_config", "mini_multi_scale")

    # Discriminator regularization settings
    instance_noise_std = float(unk_dict.get("instance_noise_std", 0.0))  # Initial std (0 = disabled)
    instance_noise_decay_steps = int(unk_dict.get("instance_noise_decay_steps", 50000))
    r1_penalty_weight = float(unk_dict.get("r1_penalty_weight", 0.0))  # Weight (0 = disabled)
    r1_penalty_interval = int(unk_dict.get("r1_penalty_interval", 16))  # Apply every N steps
    gan_warmup_steps = int(unk_dict.get("gan_warmup_steps", 0))  # Steps to ramp GAN loss from 0 to full (0 = no warmup)
    # Adaptive discriminator weighting (VQGAN-style): automatically balances GAN vs reconstruction gradients
    # This prevents the discriminator from dominating and causing artifacts
    use_adaptive_weight = unk_dict.get("use_adaptive_weight", "false").lower() == "true"

    # KL annealing: ramps KL weight from 0 to full over N steps (0 = disabled, no annealing)
    kl_annealing_steps = int(unk_dict.get("kl_annealing_steps", 0))

    # Free bits: minimum KL per channel to prevent posterior collapse (0 = disabled)
    free_bits = float(unk_dict.get("free_bits", 0.0))

    # Speaker embedding dropout: probability of zeroing speaker embedding during training
    # Encourages disentanglement by forcing decoder to learn to use embedding when available
    speaker_embedding_dropout = float(unk_dict.get("speaker_embedding_dropout", 0.0))

    # Instance normalization on latents for speaker disentanglement
    # Removes per-instance statistics (mean/variance) which often encode speaker characteristics
    # Speaker info is then re-injected via FiLM conditioning only
    instance_norm_latents = unk_dict.get("instance_norm_latents", "false").lower() == "true"

    # Instance normalization on input mel spectrogram for speaker-invariant features
    # Normalizes each mel bin across time (like CMVN), stripping per-utterance speaker statistics
    use_input_instance_norm = unk_dict.get("use_input_instance_norm", "false").lower() == "true"

    # GRL (Gradient Reversal Layer) speaker disentanglement settings
    # Trains a speaker classifier on latents with reversed gradients, encouraging encoder
    # to produce latents that are speaker-agnostic
    grl_weight = float(unk_dict.get("grl_weight", 0.0))  # Weight for GRL loss (0 = disabled)
    grl_start_step = int(unk_dict.get("grl_start_step", 5000))  # Step to start GRL (let VAE learn first)
    grl_alpha_max = float(unk_dict.get("grl_alpha_max", 1.0))  # Max gradient reversal strength
    grl_rampup_steps = int(unk_dict.get("grl_rampup_steps", 10000))  # Steps to ramp alpha from 0 to max
    grl_classifier_lr = float(unk_dict.get("grl_classifier_lr", 1e-4))  # Learning rate for speaker classifier
    num_speakers = int(unk_dict.get("num_speakers", 0))  # Number of speaker classes (0 = auto-detect from dataset)

    # Speaker ID classification on learned speaker embeddings (complementary to GRL)
    # GRL pushes speaker info OUT of latents, this pulls it INTO the speaker head
    # Only works when learn_speaker_embedding=True
    speaker_id_loss_weight = float(unk_dict.get("speaker_id_loss_weight", 0.0))  # Weight (0 = disabled)
    speaker_id_classifier_lr = float(unk_dict.get("speaker_id_classifier_lr", 1e-4))  # Learning rate
    # Loss type: "classifier" (MLP + cross-entropy) or "arcface" (angular margin for tighter clustering)
    speaker_id_loss_type = unk_dict.get("speaker_id_loss_type", "arcface")  # "classifier" or "arcface"
    # ArcFace hyperparameters (only used when speaker_id_loss_type="arcface")
    arcface_scale = float(unk_dict.get("arcface_scale", 30.0))  # Logit scale (higher = sharper softmax)
    arcface_margin = float(unk_dict.get("arcface_margin", 0.2))  # Angular margin in radians
    # Speaker ID loss scheduling (ramps weight from 0 to max over rampup_steps starting at start_step)
    speaker_id_loss_start_step = int(unk_dict.get("speaker_id_loss_start_step", 0))  # 0 = from beginning
    speaker_id_loss_rampup_steps = int(unk_dict.get("speaker_id_loss_rampup_steps", 0))  # 0 = no rampup

    # FiLM statistics logging - track scale/shift statistics for diagnosing conditioning health
    log_film_stats = unk_dict.get("log_film_stats", "false").lower() == "true"

    # Mu-only reconstruction loss: trains decoder to produce good outputs from mu directly
    # This ensures diffusion-generated latents decode well without needing reparameterization noise
    mu_only_recon_weight = float(unk_dict.get("mu_only_recon_weight", 0.0))

    # Create shared window buffer for audio processing
    shared_window_buffer = SharedWindowBuffer()

    model = model_config_lookup[args.config](
        latent_channels=latent_channels,
        speaker_embedding_dim=speaker_embedding_dim,
        speaker_embedding_proj_dim=speaker_embedding_proj_dim,
        normalize_speaker_embedding=normalize_speaker_embedding,
        film_scale_bound=film_scale_bound,
        film_shift_bound=film_shift_bound,
        zero_init_film_bias=zero_init_film_bias,
        film_no_bias=film_no_bias,
        learn_speaker_embedding=learn_speaker_embedding,
        learned_speaker_dim=learned_speaker_dim,
        recon_loss_weight=recon_loss_weight,
        mse_loss_weight=mse_loss_weight,
        l1_loss_weight=l1_loss_weight,
        kl_divergence_loss_weight=kl_divergence_loss_weight,
        free_bits=free_bits,
        speaker_embedding_dropout=speaker_embedding_dropout,
        instance_norm_latents=instance_norm_latents,
        use_input_instance_norm=use_input_instance_norm,
    )

    # Try to load existing checkpoint
    model, model_loaded = load_model(False, model, run_dir)

    # Determine device for discriminator
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.local_rank}" if args.local_rank >= 0 else "cuda")
    else:
        device = torch.device("cpu")

    # Create discriminator if GAN training is enabled
    discriminator = None
    discriminator_optimizer = None
    if use_gan:
        if discriminator_config not in mel_discriminator_config_lookup:
            raise ValueError(f"Unknown discriminator config: {discriminator_config}. Available: {list(mel_discriminator_config_lookup.keys())}")

        # keep on cpu, transfer to device when activated by provided criteria
        discriminator = mel_discriminator_config_lookup[discriminator_config]()

        discriminator_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=discriminator_lr,
            betas=(0.0, 0.99),
            weight_decay=0.0,
        )

        # Try to load existing discriminator checkpoint
        discriminator, discriminator_optimizer, disc_loaded = load_discriminator(
            args.resume_from_checkpoint, discriminator, discriminator_optimizer, device
        )

        discriminator = discriminator.cpu()

        if disc_loaded:
            print("Loaded discriminator from checkpoint")

    # Create audio perceptual loss if enabled
    audio_perceptual_loss = None
    perceptual_loss_vocoder = None
    if audio_perceptual_loss_weight > 0:
        # Check if waveform-based losses need a vocoder
        needs_vocoder = (wav2vec2_weight > 0 or panns_weight > 0)
        if needs_vocoder:
            if vocoder_checkpoint_path is None:
                print("Warning: Wav2Vec2/PANNs losses require a vocoder but no vocoder_checkpoint_path provided.")
                print("  Only multi-scale mel loss will be used. Set wav2vec2_weight=0 and panns_weight=0 to suppress this warning.")
            elif os.path.exists(vocoder_checkpoint_path):
                # Load vocoder for perceptual loss (separate from visualization vocoder)
                print(f"Loading vocoder for perceptual loss from {vocoder_checkpoint_path}")
                perceptual_loss_vocoder = vocoder_config_lookup[vocoder_config](
                    shared_window_buffer=shared_window_buffer,
                )
                perceptual_loss_vocoder, _ = load_model(False, perceptual_loss_vocoder, vocoder_checkpoint_path)
                # Remove weight normalization for inference optimization
                if hasattr(perceptual_loss_vocoder.vocoder, 'remove_weight_norm'):
                    perceptual_loss_vocoder.vocoder.remove_weight_norm()
                perceptual_loss_vocoder.eval()
                perceptual_loss_vocoder.to(device)
                # Freeze vocoder weights
                for param in perceptual_loss_vocoder.parameters():
                    param.requires_grad = False
                print(f"Loaded vocoder for perceptual loss: {sum(p.numel() for p in perceptual_loss_vocoder.parameters()):,} parameters")
            else:
                print(f"Warning: Vocoder checkpoint not found at {vocoder_checkpoint_path}")
                print("  Wav2Vec2/PANNs losses will be disabled.")

        # Create audio perceptual loss
        audio_perceptual_loss = AudioPerceptualLoss(
            sample_rate=audio_sample_rate,
            multi_scale_mel_weight=multi_scale_mel_weight,
            wav2vec2_weight=wav2vec2_weight if perceptual_loss_vocoder is not None else 0.0,
            panns_weight=panns_weight if perceptual_loss_vocoder is not None else 0.0,
            wav2vec2_model=wav2vec2_model,
            speaker_embedding_weight=speaker_embedding_weight,
        )
        audio_perceptual_loss.to(device)
        # Freeze all perceptual loss weights
        for param in audio_perceptual_loss.parameters():
            param.requires_grad = False

    if args.local_rank == 0 or not args.use_deepspeed:
        print(f"Model structure: {model}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  VAE Encoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")
        print(f"  VAE Decoder parameters: {sum(p.numel() for p in model.decoder.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"Audio settings:")
        print(f"  Sample rate: {audio_sample_rate}")
        print(f"  N mels: {n_mels}")
        print(f"  N FFT: {audio_n_fft}")
        print(f"  Hop length: {audio_hop_length}")
        print(f"  Max frames: {audio_max_frames}")
        print(f"  Latent channels: {latent_channels}")
        print(f"  Speaker encoder type: {speaker_encoder_type}")
        print(f"  Speaker embedding dim: {speaker_embedding_dim}")
        print(f"  Speaker embedding proj dim: {speaker_embedding_proj_dim} (0=no projection)")
        print(f"  Normalize speaker embedding: {normalize_speaker_embedding}")
        print(f"  FiLM scale bound: {film_scale_bound} (0=unbounded)")
        print(f"  FiLM shift bound: {film_shift_bound} (0=unbounded)")
        if learn_speaker_embedding:
            print(f"  Learned speaker embedding: ENABLED (dim={learned_speaker_dim})")
        else:
            print(f"  Learned speaker embedding: DISABLED (using pretrained from dataset)")
        if use_gan and discriminator is not None:
            print(f"GAN training: enabled")
            print(f"Discriminator structure: {discriminator}")
            print(f"  Discriminator config: {discriminator_config}")
            print(f"  Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
            multi_scale_discs = [d for d in discriminator.discriminators if isinstance(d, MelMultiScaleDiscriminator)]
            multi_period_discs = [d for d in discriminator.discriminators if isinstance(d, MelMultiPeriodDiscriminator)]
            if multi_scale_discs:
                print(f"    Multi-scale discriminators parameters: {sum(p.numel() for d in multi_scale_discs for p in d.parameters()):,}")
            if multi_period_discs:
                print(f"    Multi-period discriminators parameters: {sum(p.numel() for d in multi_period_discs for p in d.parameters()):,}")
            print(f"  GAN loss weight: {gan_loss_weight}")
            print(f"  Feature matching weight: {feature_matching_weight}")
            print(f"  Discriminator LR: {discriminator_lr}")
            print(f"  GAN start condition: {gan_start_condition_key}={gan_start_condition_value}")
            if instance_noise_std > 0:
                print(f"  Instance noise: initial_std={instance_noise_std}, decay_steps={instance_noise_decay_steps}")
            if r1_penalty_weight > 0:
                print(f"  R1 penalty: weight={r1_penalty_weight}, interval={r1_penalty_interval}")
            if gan_warmup_steps > 0:
                print(f"  GAN warmup: {gan_warmup_steps} steps (ramps loss from 0 to full)")
            if use_adaptive_weight:
                print(f"  Adaptive GAN weight: enabled (VQGAN-style gradient balancing)")
        if audio_perceptual_loss is not None:
            print(f"Audio perceptual loss: enabled (total_weight={audio_perceptual_loss_weight})")
            if audio_perceptual_loss_start_step > 0:
                print(f"  Start step: {audio_perceptual_loss_start_step} (delayed to let L1/MSE settle)")
            print(f"  Multi-scale mel weight: {multi_scale_mel_weight}")
            print(f"  Wav2Vec2 weight: {wav2vec2_weight if perceptual_loss_vocoder is not None else 0.0} (model: {wav2vec2_model})")
            print(f"  PANNs weight: {panns_weight if perceptual_loss_vocoder is not None else 0.0}")
            if perceptual_loss_vocoder is not None:
                print(f"  Using vocoder for waveform conversion: {vocoder_config}")
            else:
                print(f"  No vocoder loaded - only multi-scale mel loss active")
        if kl_annealing_steps > 0:
            print(f"KL annealing: {kl_annealing_steps} steps (ramps KL weight from 0 to 1)")
        if free_bits > 0:
            print(f"Free bits: {free_bits} nats per channel (prevents posterior collapse)")
        if speaker_embedding_dropout > 0:
            print(f"Speaker embedding dropout: {speaker_embedding_dropout} (encourages disentanglement)")
        if instance_norm_latents:
            print(f"Instance norm on latents: enabled (removes speaker statistics from z)")
        if mu_only_recon_weight > 0:
            print(f"Mu-only reconstruction loss: weight={mu_only_recon_weight} (trains decoder for diffusion compatibility)")

    model = setup_int8_training(args, model)

    if not os.path.exists(run_dir):
        os.makedirs(run_dir, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        tpu_num_cores=8 if args.use_xla else None,
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
        torch_compile=args.compile_model and not args.use_deepspeed and not args.use_xla,
        deepspeed=args.deepspeed_config if args.use_deepspeed and not args.use_xla else None,
        use_cpu=args.cpu,
        log_level=args.log_level,
        logging_first_step=True,
        local_rank=args.local_rank,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        ignore_data_skip=False,
        remove_unused_columns=False,
        eval_strategy="steps" if args.eval_steps > 0 else "no",
        eval_steps=args.eval_steps,
    )

    # Load datasets
    if use_sharded_dataset:
        print(f"Using sharded dataset format")
        print(f"  Train: {train_cache_dir}")
        print(f"  Val: {val_cache_dir}")
        train_dataset = AudioVAEShardedDataset(
            shard_dir=train_cache_dir,
            cache_size=32,
            audio_max_frames=audio_max_frames,
        )
        eval_dataset = AudioVAEShardedDataset(
            shard_dir=val_cache_dir,
            cache_size=32,
            audio_max_frames=audio_max_frames,
        )
    else:
        print(f"Using legacy diffusion dataset format")
        print(f"  Train: {train_cache_dir}")
        print(f"  Val: {val_cache_dir}")
        train_dataset = CachedAudioDiffusionDataset(
            cache_dir=train_cache_dir,
            audio_max_frames=audio_max_frames,
        )
        eval_dataset = CachedAudioDiffusionDataset(
            cache_dir=val_cache_dir,
            audio_max_frames=audio_max_frames,
        )

    # Create data collator
    data_collator = AudioVAEDataCollator(
        audio_max_frames=audio_max_frames,
        n_mels=n_mels,
        speaker_embedding_dim=speaker_embedding_dim,
    )

    # Create speaker classifier for GRL speaker disentanglement
    speaker_classifier = None
    speaker_classifier_optimizer = None
    if grl_weight > 0:
        # Auto-detect num_speakers from dataset if not specified
        if num_speakers <= 0:
            if use_sharded_dataset and hasattr(train_dataset, 'num_speakers') and train_dataset.num_speakers > 0:
                num_speakers = train_dataset.num_speakers
                print(f"Auto-detected {num_speakers} speakers from dataset")
            else:
                raise ValueError(
                    "GRL speaker disentanglement requires num_speakers > 0. "
                    "Either set --num_speakers explicitly, or use sharded dataset format with speaker IDs "
                    "(enable --include_speaker_id during preprocessing and re-merge shards)."
                )

        # Check if dataset has speaker_ids available
        if use_sharded_dataset and not train_dataset.include_speaker_ids:
            print("WARNING: GRL enabled but dataset may not have speaker_ids. "
                  "Re-run preprocessing with --include_speaker_id and re-merge shards.")

        speaker_classifier = SpeakerClassifier(
            latent_channels=latent_channels,
            num_speakers=num_speakers,
            hidden_dim=256,
        ).to(device)
        speaker_classifier_optimizer = torch.optim.AdamW(
            speaker_classifier.parameters(),
            lr=grl_classifier_lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        # Try to load existing speaker classifier checkpoint
        speaker_classifier, speaker_classifier_optimizer, sc_loaded = load_speaker_classifier(
            args.resume_from_checkpoint, speaker_classifier, speaker_classifier_optimizer, device
        )
        if sc_loaded:
            print("Loaded speaker classifier from checkpoint")

        if args.local_rank == 0 or not args.use_deepspeed:
            print(f"GRL speaker disentanglement: enabled")
            print(f"Speaker classifier structure: {speaker_classifier}")
            print(f"  Num speakers: {num_speakers}")
            print(f"  GRL weight: {grl_weight}")
            print(f"  GRL start step: {grl_start_step}")
            print(f"  GRL alpha max: {grl_alpha_max}")
            print(f"  GRL rampup steps: {grl_rampup_steps}")
            print(f"  Speaker classifier LR: {grl_classifier_lr}")
            print(f"  Speaker classifier parameters: {sum(p.numel() for p in speaker_classifier.parameters()):,}")

    # Create learned speaker classifier for speaker ID loss on learned embeddings
    # Complementary to GRL: GRL pushes speaker OUT of latents, this pulls it INTO the speaker head
    learned_speaker_classifier = None
    learned_speaker_classifier_optimizer = None
    if speaker_id_loss_weight > 0 and learn_speaker_embedding:
        # Auto-detect num_speakers from dataset if not already set by GRL
        if num_speakers <= 0:
            if use_sharded_dataset and hasattr(train_dataset, 'num_speakers') and train_dataset.num_speakers > 0:
                num_speakers = train_dataset.num_speakers
                print(f"Auto-detected {num_speakers} speakers from dataset")
            else:
                raise ValueError(
                    "Speaker ID classification requires num_speakers > 0. "
                    "Either set --num_speakers explicitly, or use sharded dataset format with speaker IDs "
                    "(enable --include_speaker_id during preprocessing and re-merge shards)."
                )

        # Check if dataset has speaker_ids available
        if use_sharded_dataset and not train_dataset.include_speaker_ids:
            print("WARNING: Speaker ID loss enabled but dataset may not have speaker_ids. "
                  "Re-run preprocessing with --include_speaker_id and re-merge shards.")

        # Create classifier based on loss type
        if speaker_id_loss_type == "arcface":
            # ArcFace: angular margin loss for tighter embedding clustering (like ECAPA-TDNN)
            learned_speaker_classifier = ArcFaceLoss(
                embedding_dim=learned_speaker_dim,
                num_speakers=num_speakers,
                scale=arcface_scale,
                margin=arcface_margin,
            ).to(device)
        else:
            # Simple MLP classifier with cross-entropy
            learned_speaker_classifier = LearnedSpeakerClassifier(
                embedding_dim=learned_speaker_dim,
                num_speakers=num_speakers,
                hidden_dim=256,
            ).to(device)

        learned_speaker_classifier_optimizer = torch.optim.AdamW(
            learned_speaker_classifier.parameters(),
            lr=speaker_id_classifier_lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        # Try to load existing learned speaker classifier checkpoint
        learned_speaker_classifier, learned_speaker_classifier_optimizer, lsc_loaded = load_learned_speaker_classifier(
            args.resume_from_checkpoint, learned_speaker_classifier, learned_speaker_classifier_optimizer, device
        )
        if lsc_loaded:
            print("Loaded learned speaker classifier from checkpoint")

        if args.local_rank == 0 or not args.use_deepspeed:
            print(f"Learned speaker ID classification: enabled")
            print(f"  Loss type: {speaker_id_loss_type}")
            print(f"  Embedding dim (input): {learned_speaker_dim}")
            print(f"  Num speakers: {num_speakers}")
            print(f"  Speaker ID loss weight: {speaker_id_loss_weight}")
            print(f"  Speaker ID classifier LR: {speaker_id_classifier_lr}")
            if speaker_id_loss_type == "arcface":
                print(f"  ArcFace scale: {arcface_scale}")
                print(f"  ArcFace margin: {arcface_margin} radians")
            print(f"  Learned speaker classifier parameters: {sum(p.numel() for p in learned_speaker_classifier.parameters()):,}")
    elif speaker_id_loss_weight > 0 and not learn_speaker_embedding:
        print("WARNING: speaker_id_loss_weight > 0 but learn_speaker_embedding is False. "
              "Speaker ID loss requires learned speaker embeddings. Disabling speaker ID loss.")

    # Create trainer
    trainer = AudioVAEGANTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        cmdline=args.cmdline,
        git_commit_hash=args.commit_hash,
        step_offset=args.start_step,
        discriminator=discriminator if use_gan else None,
        discriminator_optimizer=discriminator_optimizer if use_gan else None,
        gan_loss_weight=gan_loss_weight,
        feature_matching_weight=feature_matching_weight,
        discriminator_update_frequency=discriminator_update_frequency,
        gan_start_condition_key=gan_start_condition_key if use_gan else None,
        gan_start_condition_value=gan_start_condition_value if use_gan else None,
        instance_noise_std=instance_noise_std,
        instance_noise_decay_steps=instance_noise_decay_steps,
        r1_penalty_weight=r1_penalty_weight,
        r1_penalty_interval=r1_penalty_interval,
        gan_warmup_steps=gan_warmup_steps,
        use_adaptive_weight=use_adaptive_weight,
        audio_perceptual_loss=audio_perceptual_loss,
        audio_perceptual_loss_weight=audio_perceptual_loss_weight,
        audio_perceptual_loss_start_step=audio_perceptual_loss_start_step,
        vocoder=perceptual_loss_vocoder,
        kl_annealing_steps=kl_annealing_steps,
        # GRL speaker disentanglement
        speaker_classifier=speaker_classifier,
        speaker_classifier_optimizer=speaker_classifier_optimizer,
        grl_weight=grl_weight,
        grl_start_step=grl_start_step,
        grl_alpha_max=grl_alpha_max,
        grl_rampup_steps=grl_rampup_steps,
        # FiLM statistics logging
        log_film_stats=log_film_stats,
        # FiLM contrastive loss
        film_contrastive_loss_weight=film_contrastive_loss_weight,
        film_contrastive_loss_start_step=film_contrastive_loss_start_step,
        film_contrastive_margin_max=film_contrastive_margin_max,
        film_contrastive_margin_rampup_steps=film_contrastive_margin_rampup_steps,
        # Mu-only reconstruction loss (for diffusion compatibility)
        mu_only_recon_weight=mu_only_recon_weight,
        # Learned speaker embedding classification (complementary to GRL)
        learned_speaker_classifier=learned_speaker_classifier,
        learned_speaker_classifier_optimizer=learned_speaker_classifier_optimizer,
        speaker_id_loss_weight=speaker_id_loss_weight,
        speaker_id_loss_type=speaker_id_loss_type,
        speaker_id_loss_start_step=speaker_id_loss_start_step,
        speaker_id_loss_rampup_steps=speaker_id_loss_rampup_steps,
    )

    # Add visualization callback
    visualization_callback = AudioVAEReconstructionCallback(
        shared_window_buffer=shared_window_buffer,
        step_offset=args.start_step,
        generation_steps=args.generation_steps,
        audio_sample_rate=audio_sample_rate,
        audio_n_mels=n_mels,
        audio_n_fft=audio_n_fft,
        audio_hop_length=audio_hop_length,
        audio_max_frames=audio_max_frames,
        vocoder_checkpoint_path=vocoder_checkpoint_path,
        vocoder_config=vocoder_config,
        speaker_encoder_type=speaker_encoder_type,
    )
    trainer.add_callback(visualization_callback)

    if args.stop_step > 0:
        early_stopping_callback = EarlyStoppingCallback(stop_step=args.stop_step)
        trainer.add_callback(early_stopping_callback)

    visualization_callback.trainer = trainer

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

    print(f"Starting audio VAE training with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
