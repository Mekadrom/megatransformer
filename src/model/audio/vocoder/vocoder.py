from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.audio.vocoder.vocoder import VOCODER_CONFIGS, VocoderConfig
from model import activations
from model.audio.criteria import HighFreqSTFTLoss, MultiResolutionSTFTLoss, PhaseLoss, StableMelSpectrogramLoss, Wav2Vec2PerceptualLoss
from model.audio.vocoder.convnext import ConvNeXtBlock
from model.audio.vocoder.frequency_attention_block import FrequencyAttentionBlock
from utils import megatransformer_utils
from utils.audio_utils import SharedWindowBuffer


class Vocoder(nn.Module):
    """
    Frequency domain vocoder with attention for improved phase coherence.

    Interleaves ConvNeXt blocks with attention blocks to capture:
    - Local patterns via convolution
    - Global phase relationships via attention

    Attention helps harmonics (which are related across time) maintain coherence.
    """
    def __init__(self, shared_window_buffer: SharedWindowBuffer, config: VocoderConfig):
        super().__init__()

        self.shared_window_buffer = shared_window_buffer
        self.config = config

        self.freq_bins = config.n_fft // 2 + 1  # 513 for n_fft=1024
        
        # input projection: mel bins -> hidden
        self.input_proj = nn.Conv1d(config.n_mels, config.hidden_dim, kernel_size=7, padding=3)

        # iSTFT window
        self.register_buffer('window', shared_window_buffer.get_window(config.n_fft, torch.device('cpu')))

        self.cutoff_bin = config.cutoff_bin
        self.n_low_bins = config.cutoff_bin
        self.n_high_bins = self.freq_bins - config.cutoff_bin

        # Build backbone: ConvNeXt blocks with attention interspersed
        # Pattern: [Conv, Conv, ..., Attn, Conv, Conv, ..., Attn, ...]
        backbone_layers = []
        conv_per_attn = config.num_conv_layers // (config.num_attn_layers + 1)

        for i in range(config.num_attn_layers + 1):
            # Add ConvNeXt blocks
            for _ in range(conv_per_attn):
                backbone_layers.append(ConvNeXtBlock(config.hidden_dim, expansion=config.convnext_mult))

            # Add attention after each group (except the last)
            if i < config.num_attn_layers:
                backbone_layers.append(
                    FrequencyAttentionBlock(
                        dim=config.hidden_dim,
                        num_heads=config.attn_heads,
                        dropout=config.attn_dropout,
                    )
                )

        # Final ConvNeXt to reduce channels
        backbone_layers.append(
            ConvNeXtBlock(config.hidden_dim, ovr_out_dim=config.hidden_dim // 2, expansion=config.convnext_mult)
        )

        self.backbone = nn.ModuleList(backbone_layers)

        head_input_dim = config.hidden_dim // 2

        # Split-band magnitude heads
        self.mag_head_low = nn.Conv1d(
            head_input_dim, self.n_low_bins,
            kernel_size=config.low_freq_kernel, padding=config.low_freq_kernel // 2
        )
        self.mag_head_high = nn.Conv1d(
            head_input_dim, self.n_high_bins,
            kernel_size=config.high_freq_kernel, padding=config.high_freq_kernel // 2
        )

        # Phase heads with attention-informed features
        self.phase_head_low = nn.Sequential(
            nn.Conv1d(head_input_dim, head_input_dim, kernel_size=config.low_freq_kernel, padding=config.low_freq_kernel // 2),
            activations.Snake(head_input_dim),
            nn.Conv1d(head_input_dim, self.n_low_bins, kernel_size=config.low_freq_kernel, padding=config.low_freq_kernel // 2),
        )

        self.phase_head_high = nn.Sequential(
            activations.Snake(head_input_dim),
            nn.Conv1d(head_input_dim, self.n_high_bins, kernel_size=config.high_freq_kernel, padding=config.high_freq_kernel // 2)
        )

        self._init_weights()

        # Loss functions
        self.stft_loss = MultiResolutionSTFTLoss(shared_window_buffer=shared_window_buffer)
        self.mel_recon_loss = StableMelSpectrogramLoss(
            shared_window_buffer=shared_window_buffer,
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            mel_recon_loss_weight_linspace_max=config.mel_recon_loss_weight_linspace_max
        )

        self.phase_loss = None
        if config.phase_loss_weight > 0.0:
            self.phase_loss = PhaseLoss(shared_window_buffer=shared_window_buffer, n_fft=config.n_fft, hop_length=config.hop_length)

        self.high_freq_stft_loss = None
        if config.high_freq_stft_loss_weight > 0.0:
            self.high_freq_stft_loss = HighFreqSTFTLoss(
                shared_window_buffer=shared_window_buffer,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                cutoff_bin=config.high_freq_stft_cutoff_bin
            )

        self.wav2vec2_loss = None
        if config.wav2vec2_loss_weight > 0.0:
            self.wav2vec2_loss = Wav2Vec2PerceptualLoss(
                model_name=config.wav2vec2_model,
                sample_rate=config.sample_rate,
            )

        self.gradient_checkpointing = False

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)

        for head in [self.mag_head_low, self.mag_head_high]:
            nn.init.xavier_uniform_(head.weight, gain=0.1)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

        for m in self.phase_head_low:
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.phase_head_high:
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @classmethod
    def from_config(cls, config_name: str, shared_window_buffer: Optional[SharedWindowBuffer], **overrides) -> "Vocoder":
        """
        Create model from predefined config with optional overrides.

        Args:
            config_name: One of predefined configs
            **overrides: Override any config parameter

        Example:
            model = Vocoder.from_config("tiny", hidden_dim=256)
        """
        if config_name not in VOCODER_CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(VOCODER_CONFIGS.keys())}")

        config = VOCODER_CONFIGS[config_name]
        # Apply overrides
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(overrides)
        config = VocoderConfig(**config_dict)

        if shared_window_buffer is None:
            shared_window_buffer = SharedWindowBuffer()

        return cls(shared_window_buffer, config)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def get_phase(self, stft: torch.Tensor) -> torch.Tensor:
        """Extract phase angle from predicted STFT for loss computation."""
        return torch.angle(stft)
    
    def get_magnitude(self, stft: torch.Tensor) -> torch.Tensor:
        """Extract magnitude from predicted STFT for loss computation."""
        return stft.abs()

    def forward(
        self,
        mel_specs: torch.Tensor,
        mel_spec_masks: torch.Tensor,
        waveforms: Optional[torch.Tensor] = None,
        waveform_masks: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            mel: [B, n_mels, T] log mel spectrogram
            OR [B, 1, n_mels, T] with singleton channel dim (gets dropped)

        Returns:
            waveform: [B, T * hop_length]
            stft: [B, freq_bins, T] complex - for loss computation
        """

        if mel_specs.dim() == 4 and mel_specs.size(1) == 1:
            mel_specs = mel_specs.squeeze(1)  # Remove singleton channel dim if present

        x = self.input_proj(mel_specs)

        for block in self.backbone:
            x = block(x)

        # Magnitude prediction
        mag_pre_low = self.mag_head_low(x)
        mag_pre_high = self.mag_head_high(x)
        mag_pre = torch.cat([mag_pre_low, mag_pre_high], dim=1)
        mag = F.elu(mag_pre, alpha=1.0) + 1.0

        # Phase prediction
        phase_low = self.phase_head_low(x)
        phase_high = self.phase_head_high(x)
        phase_angle = torch.cat([phase_low, phase_high], dim=1)

        phase_real = torch.cos(phase_angle)
        phase_imag = torch.sin(phase_angle)

        # Construct complex STFT
        stft_real = mag * phase_real
        stft_imag = mag * phase_imag
        pred_stft = torch.complex(stft_real.to(torch.float32), stft_imag.to(torch.float32))[..., :mel_spec_masks.shape[-1]]

        # iSTFT to waveform
        pred_waveform = torch.istft(
            pred_stft,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            window=self.window.to(pred_stft.device),
            # length=mel_specs.size(-1) * self.config.hop_length,
            return_complex=False,
        )

        outputs = {
            "pred_waveform": pred_waveform,
            "pred_stft": pred_stft,
        }

        if waveforms is not None:
            # Ensure waveform_labels has batch dimension to match pred_waveform
            if waveforms.dim() == 1:
                waveforms = waveforms.unsqueeze(0)

            # Align waveform lengths
            min_len = min(pred_waveform.shape[-1], waveforms.shape[-1])
            pred_waveform_aligned = pred_waveform[..., :min_len]
            waveform_labels_aligned = waveforms[..., :min_len]
            waveform_masks_aligned = waveform_masks[..., :min_len]

            # Compute losses (masked if waveform_masks provided)
            waveform_l1 = (torch.abs(pred_waveform_aligned - waveform_labels_aligned) * waveform_masks_aligned).sum() / waveform_masks_aligned.sum()
            # STFT loss expects [B, 1, T] shape
            sc_loss, mag_loss, complex_stft_loss = self.stft_loss(
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
                phase_loss_value = (self.config.phase_ip_loss_weight * ip_loss +
                                    self.config.phase_iaf_loss_weight * iaf_loss +
                                    self.config.phase_gd_loss_weight * gd_loss)

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

            total_loss = (self.config.sc_loss_weight * sc_loss +
                          self.config.mag_loss_weight * mag_loss +
                          self.config.complex_stft_loss_weight * complex_stft_loss +
                          self.config.waveform_l1_loss_weight * waveform_l1 +
                          self.config.mel_recon_loss_weight * mel_recon_loss_value +
                          self.config.phase_loss_weight * phase_loss_value +
                          self.config.high_freq_stft_loss_weight * high_freq_stft_loss_value +
                          self.config.direct_mag_loss_weight * direct_mag_loss +
                          self.config.wav2vec2_loss_weight * wav2vec2_loss_value)

            outputs.update({
                "loss": total_loss,
                "waveform_l1": waveform_l1,
                "sc_loss": sc_loss,
                "mag_loss": mag_loss,
                "mel_recon_loss": mel_recon_loss_value,
                "complex_stft_loss": complex_stft_loss,
                "phase_loss": phase_loss_value,
                "phase_ip_loss": ip_loss,
                "phase_iaf_loss": iaf_loss,
                "phase_gd_loss": gd_loss,
                "high_freq_stft_loss": high_freq_stft_loss_value,
                "direct_mag_loss": direct_mag_loss,
                "wav2vec2_loss": wav2vec2_loss_value,
            })

        return outputs
