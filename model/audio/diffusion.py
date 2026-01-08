import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from einops import reduce

from model import diffusion
from model.audio.attention import AudioLinearSelfAttentionBlock, AudioDiffusionCrossAttentionBlock
from utils import configuration


class AudioConditionalGaussianDiffusion(diffusion.GaussianDiffusion):
    def __init__(self, *args, **kwargs):
        # Remove shared_window_buffer from kwargs before passing to parent
        kwargs.pop('shared_window_buffer', None)
        super().__init__(*args, **kwargs)

        self.mel_min = -11.5  # log(1e-5)
        self.mel_max = 2.0    # typical max for speech (empirically ~1.6, add small margin)
    
    def normalize(self, x_0):
        return 2 * (x_0 - self.mel_min) / (self.mel_max - self.mel_min) - 1
    
    def unnormalize(self, x):
        return (x + 1) / 2 * (self.mel_max - self.mel_min) + self.mel_min

    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, noise=None, condition=None, speaker_embedding=None, return_diagnostics=False):

        if noise is None:
            noise = torch.randn_like(x_start)

        # Offset noise: add constant noise across spatial dimensions
        if self.offset_noise_strength > 0 and self.training:
            offset_noise = torch.randn(x_start.shape[0], x_start.shape[1], 1, 1, device=x_start.device)
            noise = noise + self.offset_noise_strength * offset_noise

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # CFG: randomly drop conditioning during training
        if condition is not None and self.cfg_dropout_prob > 0 and self.training:
            batch_size = x_start.shape[0]
            dropout_mask = torch.rand(batch_size, device=x_start.device) < self.cfg_dropout_prob
            condition = condition.clone()
            condition[dropout_mask] = 0.0

        model_out = self.unet(x_noisy, t, condition, speaker_embedding=speaker_embedding)

        if self.prediction_type == "v":
            # Model predicts v, target is v
            target = self.get_v_target(x_start, noise, t)
        else:
            # Model predicts noise (epsilon)
            target = noise

        loss = F.mse_loss(model_out, target, reduction='none')
        loss_per_sample = reduce(loss, 'b ... -> b', 'mean')

        loss_weighted = loss_per_sample * self._extract(self.loss_weight, t, loss_per_sample.shape)

        if return_diagnostics:
            diagnostics = {
                "timesteps": t,
                "loss_per_sample": loss_per_sample.detach(),
                "loss_weighted_per_sample": loss_weighted.detach(),
            }
            return model_out, loss_weighted.mean(), diagnostics

        return model_out, loss_weighted.mean()

    def forward(self, x_0: torch.Tensor, condition: Optional[torch.Tensor]=None, speaker_embedding: Optional[torch.Tensor]=None, return_diagnostics: bool=False):
        """
        Forward pass for audio diffusion training.

        Args:
            x_0: [B, n_mels, T] or [B, 1, n_mels, T] mel spectrogram (log scale)
            condition: Optional [B, N, C] conditioning embeddings (e.g., T5 embeddings)
            speaker_embedding: Optional [B, D] or [B, 1, D] speaker embedding for voice conditioning
            return_diagnostics: If True, return additional diagnostic info for debugging

        Returns:
            predicted_noise: The noise predicted by the model
            loss: MSE loss between predicted and actual noise
            diagnostics (optional): Dict with per-timestep loss info and mel statistics
        """
        if x_0.dim() == 3:
            x_0 = x_0.unsqueeze(1)

        if len(x_0.shape) == 5:
            # squish batch and example dimensions if necessary
            b, e, c, h, w = x_0.shape
            x_0 = x_0.view(-1, c, h, w)
        else:
            b, c, h, w = x_0.shape
            e = None

        # Capture mel statistics before normalization for diagnostics
        mel_stats = None
        if return_diagnostics:
            mel_stats = {
                "mel_min": x_0.min().item(),
                "mel_max": x_0.max().item(),
                "mel_mean": x_0.mean().item(),
                "mel_std": x_0.std().item(),
            }

        if condition is not None:
            if len(condition.shape) == 5:
                # squish batch and example dimensions if necessary
                *_, c, h, w = condition.shape
                condition = condition.view(-1, c, h, w)

        t = self._sample_timesteps(x_0.shape[0], x_0.device)

        if self.is_normalize:
            x_0 = self.normalize(x_0)
            if return_diagnostics:
                mel_stats["mel_normalized_min"] = x_0.min().item()
                mel_stats["mel_normalized_max"] = x_0.max().item()
                mel_stats["mel_normalized_mean"] = x_0.mean().item()
                mel_stats["mel_normalized_std"] = x_0.std().item()

        if return_diagnostics:
            predicted_noise, loss, diagnostics = self.p_losses(x_0, t, condition=condition, speaker_embedding=speaker_embedding, return_diagnostics=True)
            diagnostics["mel_stats"] = mel_stats
        else:
            predicted_noise, loss = self.p_losses(x_0, t, condition=condition, speaker_embedding=speaker_embedding)

        if e is not None:
            # restore batch and example dimensions
            predicted_noise = predicted_noise.view(b, e, c, h, w)

        if return_diagnostics:
            return predicted_noise, loss, diagnostics
        return predicted_noise, loss

    @torch.no_grad()
    def sample(
        self,
        device,
        batch_size: int,
        condition: Optional[torch.Tensor]=None,
        speaker_embedding: Optional[torch.Tensor]=None,
        return_intermediate: bool=False,
        override_sampling_steps: Optional[int]=None,
        generator=None,
        guidance_scale: float=1.0,
        sampler: str="dpm_solver_pp",
        dpm_solver_order: int=2,
        **_
    ) -> torch.Tensor:
        x = torch.randn(batch_size, 1, self.config.audio_n_mels, self.config.audio_max_frames, device=device, generator=generator)

        sampling_steps = override_sampling_steps if override_sampling_steps is not None else self.sampling_timesteps

        if sampler == "dpm_solver_pp":
            x = self.dpm_solver_pp_sample_loop(
                x, condition=condition, speaker_embedding=speaker_embedding, return_intermediate=return_intermediate,
                override_sampling_steps=sampling_steps, guidance_scale=guidance_scale,
                order=dpm_solver_order
            )
        elif sampler == "ddim" or (self.is_ddim_sampling and sampler != "ddpm"):
            x = self.ddim_sample_loop(x, condition=condition, speaker_embedding=speaker_embedding, return_intermediate=return_intermediate, override_sampling_steps=sampling_steps, guidance_scale=guidance_scale)
        else:
            x = self.p_sample_loop(x, condition=condition, speaker_embedding=speaker_embedding, return_intermediate=return_intermediate, guidance_scale=guidance_scale)

        if isinstance(x, tuple):
            audio = x[0]
            noise_preds = x[1]
            x_start_preds = x[2]
        else:
            audio = x
            noise_preds = None
            x_start_preds = None

        if self.is_normalize:
            audio = self.unnormalize(audio)
            # Also unnormalize intermediate x_start predictions
            if x_start_preds is not None:
                x_start_preds = [self.unnormalize(x_start) for x_start in x_start_preds]

        if return_intermediate:
            return audio, noise_preds, x_start_preds
        return audio


def create_unet(config: configuration.MegaTransformerConfig, context_dim: Optional[int] = None):
    return diffusion.ConvDenoisingUNet(
        activation=config.audio_decoder_activation if hasattr(config, 'audio_decoder_activation') else "silu",
        stride=(2, 2),
        self_attn_class=AudioLinearSelfAttentionBlock,
        cross_attn_class=AudioDiffusionCrossAttentionBlock,
        norm_class=nn.LayerNorm,
        in_channels=1,
        model_channels=config.audio_decoder_model_channels,
        out_channels=1,
        channel_multipliers=config.audio_decoder_model_channels,
        time_embedding_dim=config.audio_decoder_time_embedding_dim,
        attention_levels=config.audio_decoder_attention_levels,
        num_res_blocks=config.audio_decoder_num_res_blocks,
        has_condition=True,
        context_dim=context_dim,
        dropout_p=config.audio_decoder_unet_dropout_p,
        down_block_self_attn_n_heads=config.audio_decoder_down_block_self_attn_n_heads,
        down_block_self_attn_d_queries=config.audio_decoder_down_block_self_attn_d_queries,
        down_block_self_attn_d_values=config.audio_decoder_down_block_self_attn_d_values,
        down_block_self_attn_use_flash_attention=config.audio_decoder_down_block_self_attn_use_flash_attention,
        up_block_self_attn_n_heads=config.audio_decoder_up_block_self_attn_n_heads,
        up_block_self_attn_d_queries=config.audio_decoder_up_block_self_attn_d_queries,
        up_block_self_attn_d_values=config.audio_decoder_up_block_self_attn_d_values,
        up_block_self_attn_use_flash_attention=config.audio_decoder_up_block_self_attn_use_flash_attention,
        cross_attn_n_heads=config.audio_decoder_cross_attn_n_heads,
        cross_attn_d_queries=config.audio_decoder_cross_attn_d_queries,
        cross_attn_d_values=config.audio_decoder_cross_attn_d_values,
        cross_attn_use_flash_attention=config.audio_decoder_cross_attn_use_flash_attention,
    )


# =============================================================================
# Audio DiT (Diffusion Transformer) with Factorized Attention
# =============================================================================
# Uses timestep tokenization with factorized time/frequency attention.
# Each token represents the full frequency spectrum for a time window.
# Attention alternates between time-wise and frequency-wise for efficiency.
# =============================================================================


class AudioDiTModulation(nn.Module):
    """
    adaLN-Zero modulation layer for Audio DiT.
    Produces scale and shift parameters from timestep embeddings.
    """

    def __init__(self, hidden_size: int, n_modulations: int = 6):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_size, n_modulations * hidden_size)
        self.n_modulations = n_modulations
        self.hidden_size = hidden_size

        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, t_emb: torch.Tensor) -> tuple:
        """
        Args:
            t_emb: [B, hidden_size] timestep embeddings

        Returns:
            Tuple of n_modulations tensors, each [B, hidden_size]
        """
        out = self.linear(self.silu(t_emb))
        return out.chunk(self.n_modulations, dim=-1)


class AudioDiTBlock(nn.Module):
    """
    Audio DiT block with factorized attention (time then frequency).

    Unlike image DiT which uses 2D patches, this uses:
    - Timestep tokens: each token = [n_freq_bands, time_window_size]
    - Time attention: attend across time positions for each frequency band
    - Frequency attention: attend across frequency bands for each time position

    Includes cross-attention for text conditioning.
    """

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        n_freq_tokens: int,
        mlp_ratio: float = 4.0,
        dropout_p: float = 0.0,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_freq_tokens = n_freq_tokens
        self.head_dim = hidden_size // n_heads

        # Norms (weights/biases absorbed into modulation)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Time-wise self-attention (attend across time for each freq band)
        self.time_qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.time_attn_proj = nn.Linear(hidden_size, hidden_size)
        self.time_attn_dropout = nn.Dropout(dropout_p)

        # Frequency-wise self-attention (attend across freq for each timestep)
        self.freq_qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.freq_attn_proj = nn.Linear(hidden_size, hidden_size)
        self.freq_attn_dropout = nn.Dropout(dropout_p)

        # Cross-attention for conditioning
        self.has_cross_attn = context_dim is not None
        if self.has_cross_attn:
            self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.q_cross = nn.Linear(hidden_size, hidden_size)
            self.kv_cross = nn.Linear(context_dim, 2 * hidden_size)
            self.cross_attn_proj = nn.Linear(hidden_size, hidden_size)
            self.cross_attn_dropout = nn.Dropout(dropout_p)

        # MLP
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout_p),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout_p),
        )

        # adaLN-Zero modulation
        # 6 modulations: time_scale, time_shift, freq_scale, freq_shift, mlp_scale, mlp_shift
        # + 3 gates for time, freq, mlp
        n_mods = 9 if not self.has_cross_attn else 12  # +3 for cross-attn
        self.adaLN_modulation = AudioDiTModulation(hidden_size, n_modulations=n_mods)

        self._init_weights()

    def _init_weights(self):
        # Initialize projections
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'attn_proj' in name or name.endswith('_proj'):
                    # Output projections: zero init for residual
                    nn.init.zeros_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def _attention(self, q, k, v, dropout: nn.Dropout):
        """Standard scaled dot-product attention with flash attention."""
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=dropout.p if self.training else 0.0,
            is_causal=False,
        )
        return attn_out

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, n_time_tokens, n_freq_tokens, hidden_size] - tokenized audio
            t_emb: [B, hidden_size] - timestep embeddings
            context: [B, N, context_dim] - conditioning (e.g., text embeddings)

        Returns:
            [B, n_time_tokens, n_freq_tokens, hidden_size]
        """
        B, T, F, H = x.shape

        # Get modulation parameters
        mods = self.adaLN_modulation(t_emb)
        if self.has_cross_attn:
            (time_scale, time_shift, time_gate,
             freq_scale, freq_shift, freq_gate,
             cross_scale, cross_shift, cross_gate,
             mlp_scale, mlp_shift, mlp_gate) = mods
        else:
            (time_scale, time_shift, time_gate,
             freq_scale, freq_shift, freq_gate,
             mlp_scale, mlp_shift, mlp_gate) = mods

        # Reshape modulations for broadcasting: [B, 1, 1, H]
        def expand_mod(m):
            return m.unsqueeze(1).unsqueeze(1)

        # === Time-wise self-attention ===
        # Attend across time for each frequency band
        x_norm = self.norm1(x) * (1 + expand_mod(time_scale)) + expand_mod(time_shift)

        # Reshape for time attention: [B*F, T, H]
        x_time = x_norm.permute(0, 2, 1, 3).reshape(B * F, T, H)

        qkv = self.time_qkv(x_time).reshape(B * F, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*F, n_heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_out = self._attention(q, k, v, self.time_attn_dropout)
        attn_out = attn_out.transpose(1, 2).reshape(B * F, T, H)
        attn_out = self.time_attn_proj(attn_out)

        # Reshape back: [B, F, T, H] -> [B, T, F, H]
        attn_out = attn_out.reshape(B, F, T, H).permute(0, 2, 1, 3)
        x = x + expand_mod(time_gate) * attn_out

        # === Frequency-wise self-attention ===
        # Attend across frequency bands for each time position
        x_norm = self.norm2(x) * (1 + expand_mod(freq_scale)) + expand_mod(freq_shift)

        # Reshape for frequency attention: [B*T, F, H]
        x_freq = x_norm.reshape(B * T, F, H)

        qkv = self.freq_qkv(x_freq).reshape(B * T, F, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*T, n_heads, F, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_out = self._attention(q, k, v, self.freq_attn_dropout)
        attn_out = attn_out.transpose(1, 2).reshape(B * T, F, H)
        attn_out = self.freq_attn_proj(attn_out)

        # Reshape back: [B, T, F, H]
        attn_out = attn_out.reshape(B, T, F, H)
        x = x + expand_mod(freq_gate) * attn_out

        # === Cross-attention (if conditioning provided) ===
        if self.has_cross_attn and context is not None:
            x_norm = self.norm_cross(x) * (1 + expand_mod(cross_scale)) + expand_mod(cross_shift)

            # Flatten time and freq for cross-attention: [B, T*F, H]
            x_flat = x_norm.reshape(B, T * F, H)
            N = context.shape[1]

            q = self.q_cross(x_flat).reshape(B, T * F, self.n_heads, self.head_dim).transpose(1, 2)
            kv = self.kv_cross(context).reshape(B, N, 2, self.n_heads, self.head_dim)
            kv = kv.permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn_out = self._attention(q, k, v, self.cross_attn_dropout)
            attn_out = attn_out.transpose(1, 2).reshape(B, T * F, H)
            attn_out = self.cross_attn_proj(attn_out)

            # Reshape back: [B, T, F, H]
            attn_out = attn_out.reshape(B, T, F, H)
            x = x + expand_mod(cross_gate) * attn_out

        # === MLP ===
        x_norm = self.norm2(x) * (1 + expand_mod(mlp_scale)) + expand_mod(mlp_shift)
        x = x + expand_mod(mlp_gate) * self.mlp(x_norm)

        return x


class AudioDiTFinalLayer(nn.Module):
    """
    Final layer for Audio DiT.
    Projects from hidden dim back to mel spectrogram space.
    """

    def __init__(self, hidden_size: int, n_mels: int, time_window: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, n_mels * time_window)
        self.adaLN_modulation = AudioDiTModulation(hidden_size, n_modulations=2)

        self.n_mels = n_mels
        self.time_window = time_window

        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, n_time_tokens, n_freq_tokens, hidden_size]
            t_emb: [B, hidden_size]

        Returns:
            [B, 1, n_mels, T] - reconstructed mel spectrogram
        """
        scale, shift = self.adaLN_modulation(t_emb)
        x = self.norm(x) * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)
        x = self.linear(x)  # [B, n_time_tokens, n_freq_tokens, n_mels * time_window]

        B, T_tokens, F_tokens, _ = x.shape

        # Reshape to mel spectrogram
        # x: [B, T_tokens, F_tokens, n_mels * time_window]
        # -> [B, T_tokens, F_tokens, n_mels, time_window]
        x = x.reshape(B, T_tokens, F_tokens, self.n_mels, self.time_window)

        # Rearrange: [B, n_mels, F_tokens, T_tokens, time_window]
        # Then merge frequency and time dimensions
        x = x.permute(0, 3, 2, 1, 4)  # [B, n_mels, F_tokens, T_tokens, time_window]

        # For now, F_tokens should be 1 (full spectrum per token)
        # x: [B, n_mels, 1, T_tokens, time_window] -> [B, n_mels, T_tokens * time_window]
        x = x.squeeze(2)  # [B, n_mels, T_tokens, time_window]
        x = x.reshape(B, self.n_mels, T_tokens * self.time_window)

        return x.unsqueeze(1)  # [B, 1, n_mels, T]


class AudioDiTBackbone(nn.Module):
    """
    Audio DiT Backbone with factorized time/frequency attention.

    Tokenization strategy:
    - Time tokenization: each token represents [n_mels, time_window] of the mel spectrogram
    - Frequency tokenization: optionally split frequency into bands (default: 1 band = full spectrum)

    Architecture:
    - Timestep embedding via sinusoidal + MLP
    - Positional embeddings for time and frequency positions
    - Stack of AudioDiTBlocks with factorized attention
    - Final layer to project back to mel space
    """

    def __init__(
        self,
        n_mels: int = 80,
        max_frames: int = 1875,
        time_window: int = 15,  # Tokens per 15 frames = 125 tokens for 1875 frames
        n_freq_bands: int = 1,  # 1 = full spectrum per token
        hidden_size: int = 256,
        n_layers: int = 12,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout_p: float = 0.0,
        context_dim: Optional[int] = None,
        speaker_embedding_dim: int = 0,
    ):
        super().__init__()

        self.n_mels = n_mels
        self.max_frames = max_frames
        self.time_window = time_window
        self.n_freq_bands = n_freq_bands
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.speaker_embedding_dim = speaker_embedding_dim

        # Calculate token grid size
        self.n_time_tokens = max_frames // time_window
        self.n_freq_tokens = n_freq_bands  # For now, 1 = full spectrum
        self.mels_per_band = n_mels // n_freq_bands

        # Input projection: [n_mels_per_band * time_window] -> hidden_size
        input_dim = self.mels_per_band * time_window
        self.input_proj = nn.Linear(input_dim, hidden_size)

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        # Speaker embedding projection (added to timestep embedding for AdaLN conditioning)
        if speaker_embedding_dim > 0:
            self.speaker_proj = nn.Sequential(
                nn.Linear(speaker_embedding_dim, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
        else:
            self.speaker_proj = None

        # Positional embeddings
        self.time_pos_embed = nn.Parameter(torch.zeros(1, self.n_time_tokens, 1, hidden_size))
        self.freq_pos_embed = nn.Parameter(torch.zeros(1, 1, self.n_freq_tokens, hidden_size))

        # Conditioning projection
        self.context_dim = context_dim
        if context_dim is not None:
            self.context_proj = nn.Linear(context_dim, hidden_size)
        else:
            self.context_proj = None

        # Transformer blocks
        self.blocks = nn.ModuleList([
            AudioDiTBlock(
                hidden_size=hidden_size,
                n_heads=n_heads,
                n_freq_tokens=self.n_freq_tokens,
                mlp_ratio=mlp_ratio,
                dropout_p=dropout_p,
                context_dim=hidden_size if context_dim else None,
            )
            for _ in range(n_layers)
        ])

        # Final layer
        self.final_layer = AudioDiTFinalLayer(
            hidden_size=hidden_size,
            n_mels=self.mels_per_band,
            time_window=time_window,
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.time_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.freq_pos_embed, std=0.02)

        # Initialize input projection
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        # Initialize timestep embedding
        for module in self.time_embed:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _get_timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.

        Args:
            timesteps: [B] or [B, 1] tensor of timestep indices

        Returns:
            [B, hidden_size] embeddings
        """
        if timesteps.dim() == 2:
            timesteps = timesteps.squeeze(1)

        half_dim = self.hidden_size // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb_scale)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if self.hidden_size % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert mel spectrogram to tokens.

        Args:
            x: [B, 1, n_mels, T] mel spectrogram

        Returns:
            [B, n_time_tokens, n_freq_tokens, hidden_size]
        """
        B, C, F, T = x.shape

        # Ensure T is divisible by time_window
        n_time_tokens = T // self.time_window
        x = x[..., :n_time_tokens * self.time_window]

        # Reshape into time windows: [B, 1, n_mels, n_time_tokens, time_window]
        x = x.reshape(B, C, F, n_time_tokens, self.time_window)

        # Split into frequency bands: [B, n_freq_bands, mels_per_band, n_time_tokens, time_window]
        x = x.reshape(B, self.n_freq_bands, self.mels_per_band, n_time_tokens, self.time_window)

        # Rearrange to [B, n_time_tokens, n_freq_bands, mels_per_band * time_window]
        x = x.permute(0, 3, 1, 2, 4)  # [B, n_time_tokens, n_freq_bands, mels_per_band, time_window]
        x = x.reshape(B, n_time_tokens, self.n_freq_bands, self.mels_per_band * self.time_window)

        # Project to hidden size
        x = self.input_proj(x)  # [B, n_time_tokens, n_freq_bands, hidden_size]

        return x

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        speaker_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, 1, n_mels, T] noisy mel spectrogram
            timesteps: [B] or [B, 1] diffusion timesteps
            condition: [B, N, context_dim] conditioning (e.g., text embeddings)
            speaker_embedding: [B, speaker_embedding_dim] or [B, 1, speaker_embedding_dim] speaker embedding

        Returns:
            [B, 1, n_mels, T] predicted noise or velocity
        """
        B = x.shape[0]
        original_T = x.shape[-1]

        # Ensure input dtype matches model
        model_dtype = self.input_proj.weight.dtype
        x = x.to(dtype=model_dtype)
        if condition is not None:
            condition = condition.to(dtype=model_dtype)

        # Tokenize input
        x = self.tokenize(x)  # [B, n_time_tokens, n_freq_tokens, hidden_size]

        # Timestep embedding
        t_emb = self._get_timestep_embedding(timesteps)
        t_emb = self.time_embed(t_emb.to(dtype=model_dtype))  # [B, hidden_size]

        # Add speaker embedding to timestep embedding for AdaLN conditioning
        if speaker_embedding is not None and self.speaker_proj is not None:
            # Handle [B, 1, D] -> [B, D]
            if speaker_embedding.dim() == 3:
                speaker_embedding = speaker_embedding.squeeze(1)
            speaker_embedding = speaker_embedding.to(dtype=model_dtype)
            spk_emb = self.speaker_proj(speaker_embedding)  # [B, hidden_size]
            t_emb = t_emb + spk_emb  # Combine with timestep embedding

        # Add positional embeddings
        x = x + self.time_pos_embed + self.freq_pos_embed

        # Project conditioning
        if condition is not None and self.context_proj is not None:
            condition = self.context_proj(condition)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, t_emb, condition)

        # Final layer
        x = self.final_layer(x, t_emb)  # [B, 1, n_mels, n_time_tokens * time_window]

        # Pad or truncate to original length
        current_T = x.shape[-1]
        if current_T < original_T:
            x = F.pad(x, (0, original_T - current_T))
        elif current_T > original_T:
            x = x[..., :original_T]

        return x


def create_audio_dit(
    config: configuration.MegaTransformerConfig,
    context_dim: Optional[int] = None,
    speaker_embedding_dim: int = 0,
) -> AudioDiTBackbone:
    """Create an Audio DiT backbone from config."""
    return AudioDiTBackbone(
        n_mels=config.audio_n_mels if hasattr(config, 'audio_n_mels') else 80,
        max_frames=config.audio_max_frames if hasattr(config, 'audio_max_frames') else 1875,
        time_window=config.audio_dit_time_window if hasattr(config, 'audio_dit_time_window') else 15,
        n_freq_bands=config.audio_dit_n_freq_bands if hasattr(config, 'audio_dit_n_freq_bands') else 1,
        hidden_size=config.hidden_size,
        n_layers=config.audio_decoder_num_res_blocks if hasattr(config, 'audio_decoder_num_res_blocks') else 12,
        n_heads=config.audio_decoder_down_block_self_attn_n_heads if hasattr(config, 'audio_decoder_down_block_self_attn_n_heads') else 8,
        mlp_ratio=4.0,
        dropout_p=config.audio_decoder_unet_dropout_p if hasattr(config, 'audio_decoder_unet_dropout_p') else 0.0,
        context_dim=context_dim,
        speaker_embedding_dim=speaker_embedding_dim,
    )


# =============================================================================
# Config factory functions - create configs with customizable audio parameters
# =============================================================================

def _create_config(
    base_config: dict,
    audio_n_fft: int = 1024,
    audio_hop_length: int = 256,
    audio_sample_rate: int = 16000,
    audio_n_mels: int = 80,
) -> configuration.MegaTransformerConfig:
    """Create a config with custom audio parameters."""
    config_dict = base_config.copy()
    config_dict["audio_n_fft"] = audio_n_fft
    config_dict["audio_hop_length"] = audio_hop_length
    config_dict["audio_sample_rate"] = audio_sample_rate
    config_dict["audio_n_mels"] = audio_n_mels
    return configuration.MegaTransformerConfig(**config_dict)


def _get_config(base_config: dict, audio_n_fft: int = None, audio_hop_length: int = None,
                audio_sample_rate: int = None, audio_n_mels: int = None) -> configuration.MegaTransformerConfig:
    """Get config with optional audio parameter overrides."""
    if audio_n_fft is None and audio_hop_length is None and audio_sample_rate is None and audio_n_mels is None:
        # Use defaults from base config
        return _create_config(base_config)
    return _create_config(
        base_config,
        audio_n_fft=audio_n_fft if audio_n_fft is not None else base_config.get("audio_n_fft", 1024),
        audio_hop_length=audio_hop_length if audio_hop_length is not None else base_config.get("audio_hop_length", 256),
        audio_sample_rate=audio_sample_rate if audio_sample_rate is not None else base_config.get("audio_sample_rate", 16000),
        audio_n_mels=audio_n_mels if audio_n_mels is not None else base_config.get("audio_n_mels", 80),
    )


# Base config templates (without audio params that will be overridden)
# DiT config for audio - ~12M params
_tiny_dit_base = {
    "hidden_size": 256,
    "audio_n_mels": 80,
    "audio_n_fft": 1024,
    "audio_hop_length": 256,
    "audio_sample_rate": 16000,
    "audio_max_frames": 1875,
    "audio_dit_time_window": 15,  # 1875/15 = 125 time tokens
    "audio_dit_n_freq_bands": 1,  # Full spectrum per token
    "audio_decoder_num_res_blocks": 8,  # Number of transformer layers
    "audio_decoder_down_block_self_attn_n_heads": 8,
    "audio_decoder_unet_dropout_p": 0.0,
}

# DiT config for audio - ~25M params
_small_dit_base = {
    "hidden_size": 384,
    "audio_n_mels": 80,
    "audio_n_fft": 1024,
    "audio_hop_length": 256,
    "audio_sample_rate": 16000,
    "audio_max_frames": 1875,
    "audio_dit_time_window": 15,
    "audio_dit_n_freq_bands": 1,
    "audio_decoder_num_res_blocks": 12,
    "audio_decoder_down_block_self_attn_n_heads": 8,
    "audio_decoder_unet_dropout_p": 0.0,
}

# UNet-based config
_small_base = {
    "hidden_size": 192,
    "audio_n_mels": 80,
    "audio_n_fft": 1024,
    "audio_hop_length": 256,
    "audio_sample_rate": 16000,
    "audio_decoder_model_channels": 80,
    "audio_decoder_time_embedding_dim": 192,
    "audio_decoder_attention_levels": [False, True, True],
    "audio_decoder_num_res_blocks": 2,
    "audio_decoder_down_block_self_attn_n_heads": 4,
    "audio_decoder_down_block_self_attn_d_queries": 48,
    "audio_decoder_down_block_self_attn_d_values": 48,
    "audio_decoder_down_block_self_attn_use_flash_attention": True,
    "audio_decoder_up_block_self_attn_n_heads": 4,
    "audio_decoder_up_block_self_attn_d_queries": 48,
    "audio_decoder_up_block_self_attn_d_values": 48,
    "audio_decoder_up_block_self_attn_use_flash_attention": True,
    "audio_decoder_cross_attn_n_heads": 4,
    "audio_decoder_cross_attn_d_queries": 48,
    "audio_decoder_cross_attn_d_values": 48,
    "audio_decoder_cross_attn_use_flash_attention": True,
    "audio_decoder_channel_multipliers": [1, 2, 3],  # Gradual growth
}

def create_diffusion_model(
    config: configuration.MegaTransformerConfig,
    context_dim: int = 512,
    num_timesteps: int = 1000,
    sampling_timesteps: int = 50,
    betas_schedule: str = "cosine",
    normalize: bool = True,
    min_snr_loss_weight: bool = True,
    min_snr_gamma: float = 5.0,
    prediction_type: str = "epsilon",  # "epsilon" or "v"
    cfg_dropout_prob: float = 0.1,  # Default 10% dropout for CFG training
    zero_terminal_snr: bool = True,  # Recommended for audio generation
    offset_noise_strength: float = 0.0,  # Less common for audio, default off
    timestep_sampling: str = "logit_normal",  # "uniform" or "logit_normal"
    logit_normal_mean: float = 0.0,  # Mean for logit-normal (0 = centered on middle timesteps)
    logit_normal_std: float = 1.0,  # Std for logit-normal (lower = more peaked)
) -> AudioConditionalGaussianDiffusion:
    return AudioConditionalGaussianDiffusion(
        config,
        create_unet(config, context_dim),
        1,  # Single channel mel spectrogram
        num_timesteps=num_timesteps,
        betas_schedule=betas_schedule,
        min_snr_loss_weight=min_snr_loss_weight,
        min_snr_gamma=min_snr_gamma,
        normalize=normalize,
        sampling_timesteps=sampling_timesteps,
        prediction_type=prediction_type,
        cfg_dropout_prob=cfg_dropout_prob,
        zero_terminal_snr=zero_terminal_snr,
        offset_noise_strength=offset_noise_strength,
        timestep_sampling=timestep_sampling,
        logit_normal_mean=logit_normal_mean,
        logit_normal_std=logit_normal_std,
    )


class AudioConditionalFlowMatching(nn.Module):
    """
    Flow Matching for Audio latent diffusion.

    Adapts the base FlowMatching approach for audio mel spectrograms,
    working with [B, 1, n_mels, T] shaped inputs.
    """

    def __init__(
        self,
        config: configuration.MegaTransformerConfig,
        backbone: nn.Module,
        in_channels: int = 1,
        cfg_dropout_prob: float = 0.1,
        timestep_sampling: str = "logit_normal",
        logit_normal_mean: float = 0.0,
        logit_normal_std: float = 1.0,
    ):
        super().__init__()

        self.config = config
        self.backbone = backbone
        self.in_channels = in_channels
        self.cfg_dropout_prob = cfg_dropout_prob
        self.timestep_sampling = timestep_sampling
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std

        # For compatibility with training scripts
        self.unet = backbone

    def _sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps t ∈ [0, 1] for training."""
        if self.timestep_sampling == "logit_normal":
            u = torch.randn(batch_size, device=device)
            u = self.logit_normal_mean + self.logit_normal_std * u
            t = torch.sigmoid(u)
        else:
            t = torch.rand(batch_size, device=device)
        return t.clamp(min=1e-5, max=1.0 - 1e-5)

    def interpolate(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Linear interpolation: x_t = (1 - t) * x_0 + t * noise"""
        t = t.view(-1, 1, 1, 1)
        return (1 - t) * x_0 + t * noise

    def get_velocity(self, x_0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Target velocity: v = noise - x_0"""
        return noise - x_0

    def predict_x0_from_velocity(self, x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict x_0: x_0 = x_t - t * v"""
        t = t.view(-1, 1, 1, 1)
        return x_t - t * v

    def training_step(
        self,
        x_0: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        speaker_embedding: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute training loss for flow matching."""
        batch_size = x_0.shape[0]
        device = x_0.device

        t = self._sample_timesteps(batch_size, device)
        noise = torch.randn_like(x_0).to(x_0.dtype)
        x_t = self.interpolate(x_0, noise, t)
        velocity_target = self.get_velocity(x_0, noise)

        # CFG dropout
        if condition is not None and self.cfg_dropout_prob > 0 and self.training:
            dropout_mask = torch.rand(batch_size, device=device) < self.cfg_dropout_prob
            condition = condition.clone()
            condition[dropout_mask] = 0.0

        # Convert t from [0,1] to timestep format expected by backbone
        # For DiT, we pass t directly; for UNet-style, we'd convert to integer timesteps
        velocity_pred = self.backbone(x_t, t, condition, speaker_embedding=speaker_embedding)

        loss = F.mse_loss(velocity_pred, velocity_target)
        return velocity_pred, loss

    @torch.no_grad()
    def euler_step(
        self,
        x: torch.Tensor,
        t_curr: float,
        t_next: float,
        condition: Optional[torch.Tensor] = None,
        speaker_embedding: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Single Euler step for ODE integration."""
        batch_size = x.shape[0]
        device = x.device
        t = torch.full((batch_size,), t_curr, device=device, dtype=x.dtype)

        if guidance_scale != 1.0 and condition is not None:
            v_cond = self.backbone(x, t, condition, speaker_embedding=speaker_embedding)
            v_uncond = self.backbone(x, t, torch.zeros_like(condition), speaker_embedding=speaker_embedding)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            v = self.backbone(x, t, condition, speaker_embedding=speaker_embedding)

        dt = t_next - t_curr
        return x + dt * v

    @torch.no_grad()
    def heun_step(
        self,
        x: torch.Tensor,
        t_curr: float,
        t_next: float,
        condition: Optional[torch.Tensor] = None,
        speaker_embedding: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Heun's method (improved Euler) for ODE integration."""
        batch_size = x.shape[0]
        device = x.device
        dt = t_next - t_curr

        def get_velocity(x_in, t_val):
            t = torch.full((batch_size,), t_val, device=device, dtype=x.dtype)
            if guidance_scale != 1.0 and condition is not None:
                v_cond = self.backbone(x_in, t, condition, speaker_embedding=speaker_embedding)
                v_uncond = self.backbone(x_in, t, torch.zeros_like(condition), speaker_embedding=speaker_embedding)
                return v_uncond + guidance_scale * (v_cond - v_uncond)
            return self.backbone(x_in, t, condition, speaker_embedding=speaker_embedding)

        v1 = get_velocity(x, t_curr)
        x_pred = x + dt * v1
        v2 = get_velocity(x_pred, t_next)
        return x + 0.5 * dt * (v1 + v2)

    @torch.no_grad()
    def sample(
        self,
        device: torch.device,
        batch_size: int,
        condition: Optional[torch.Tensor] = None,
        speaker_embedding: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        guidance_scale: float = 1.0,
        solver: str = "euler",
        generator: Optional[torch.Generator] = None,
        return_intermediate: bool = False,
        override_sampling_steps: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples using ODE integration from noise to data."""
        n_mels = getattr(self.config, 'audio_n_mels', 80)
        max_frames = getattr(self.config, 'audio_max_frames', 1875)

        steps = override_sampling_steps if override_sampling_steps is not None else num_steps

        # Start from pure noise at t=1 [B, 1, n_mels, T]
        if generator is not None:
            x = torch.randn(batch_size, 1, n_mels, max_frames, device=device, generator=generator)
        else:
            x = torch.randn(batch_size, 1, n_mels, max_frames, device=device)

        timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)
        intermediates = [x.clone()] if return_intermediate else None
        x_start_preds = [] if return_intermediate else None

        step_fn = self.heun_step if solver == "heun" else self.euler_step

        for i in range(steps):
            t_curr = timesteps[i].item()
            t_next = timesteps[i + 1].item()
            x = step_fn(x, t_curr, t_next, condition, speaker_embedding=speaker_embedding, guidance_scale=guidance_scale)

            if return_intermediate:
                intermediates.append(x.clone())
                # Estimate x_0 from current state
                t_tensor = torch.full((batch_size,), t_next, device=device, dtype=x.dtype)
                v = self.backbone(x, t_tensor, condition, speaker_embedding=speaker_embedding) if t_next > 1e-4 else torch.zeros_like(x)
                x_0_pred = self.predict_x0_from_velocity(x, v, t_tensor)
                x_start_preds.append(x_0_pred.clone())

        if return_intermediate:
            return x, intermediates, x_start_preds
        return x

    def forward(
        self,
        x_0: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        speaker_embedding: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training."""
        # Ensure 4D input [B, C, H, W] -> [B, 1, n_mels, T]
        if x_0.dim() == 3:
            x_0 = x_0.unsqueeze(1)

        if len(x_0.shape) == 5:
            b, e, c, h, w = x_0.shape
            x_0 = x_0.view(-1, c, h, w)
        else:
            b, c, h, w = x_0.shape
            e = None

        if condition is not None and len(condition.shape) == 4:
            condition = condition.view(-1, *condition.shape[2:])

        model_output, loss = self.training_step(x_0, condition, speaker_embedding=speaker_embedding)

        if e is not None:
            model_output = model_output.view(b, e, c, h, w)

        if return_diagnostics:
            diagnostics = {
                "mel_stats": {
                    "mel_min": x_0.min().item(),
                    "mel_max": x_0.max().item(),
                    "mel_mean": x_0.mean().item(),
                    "mel_std": x_0.std().item(),
                }
            }
            return model_output, loss, diagnostics

        return model_output, loss


def create_audio_flow_matching_model(
    config: configuration.MegaTransformerConfig,
    backbone: nn.Module,
    cfg_dropout_prob: float = 0.1,
    timestep_sampling: str = "logit_normal",
) -> AudioConditionalFlowMatching:
    """Create an Audio Flow Matching model with DiT backbone."""
    return AudioConditionalFlowMatching(
        config=config,
        backbone=backbone,
        in_channels=1,
        cfg_dropout_prob=cfg_dropout_prob,
        timestep_sampling=timestep_sampling,
    )


model_config_lookup = {
    # UNet-based models
    "small": lambda audio_n_fft=None, audio_hop_length=None, audio_sample_rate=None, audio_n_mels=None, **kwargs: create_diffusion_model(
        config=_get_config(_small_base, audio_n_fft, audio_hop_length, audio_sample_rate, audio_n_mels),
        **kwargs
    ),
    # DiT-based models with Gaussian Diffusion
    "tiny_dit": lambda context_dim, speaker_embedding_dim=0, audio_n_fft=None, audio_hop_length=None, audio_sample_rate=None, audio_n_mels=None, **kwargs: AudioConditionalGaussianDiffusion(
        config=_get_config(_tiny_dit_base, audio_n_fft, audio_hop_length, audio_sample_rate, audio_n_mels),
        unet=create_audio_dit(_get_config(_tiny_dit_base, audio_n_fft, audio_hop_length, audio_sample_rate, audio_n_mels), context_dim=context_dim, speaker_embedding_dim=speaker_embedding_dim),
        in_channels=1,
        **kwargs
    ),
    "small_dit": lambda context_dim, speaker_embedding_dim=0, audio_n_fft=None, audio_hop_length=None, audio_sample_rate=None, audio_n_mels=None, **kwargs: AudioConditionalGaussianDiffusion(
        config=_get_config(_small_dit_base, audio_n_fft, audio_hop_length, audio_sample_rate, audio_n_mels),
        unet=create_audio_dit(_get_config(_small_dit_base, audio_n_fft, audio_hop_length, audio_sample_rate, audio_n_mels), context_dim=context_dim, speaker_embedding_dim=speaker_embedding_dim),
        in_channels=1,
        **kwargs
    ),
    # DiT-based models with Flow Matching
    "tiny_dit_flow": lambda context_dim, speaker_embedding_dim=0, cfg_dropout_prob=0.1, timestep_sampling="logit_normal", audio_n_fft=None, audio_hop_length=None, audio_sample_rate=None, audio_n_mels=None, **_: create_audio_flow_matching_model(
        config=_get_config(_tiny_dit_base, audio_n_fft, audio_hop_length, audio_sample_rate, audio_n_mels),
        backbone=create_audio_dit(_get_config(_tiny_dit_base, audio_n_fft, audio_hop_length, audio_sample_rate, audio_n_mels), context_dim=context_dim, speaker_embedding_dim=speaker_embedding_dim),
        cfg_dropout_prob=cfg_dropout_prob,
        timestep_sampling=timestep_sampling,
    ),
    "small_dit_flow": lambda context_dim, speaker_embedding_dim=0, cfg_dropout_prob=0.1, timestep_sampling="logit_normal", audio_n_fft=None, audio_hop_length=None, audio_sample_rate=None, audio_n_mels=None, **_: create_audio_flow_matching_model(
        config=_get_config(_small_dit_base, audio_n_fft, audio_hop_length, audio_sample_rate, audio_n_mels),
        backbone=create_audio_dit(_get_config(_small_dit_base, audio_n_fft, audio_hop_length, audio_sample_rate, audio_n_mels), context_dim=context_dim, speaker_embedding_dim=speaker_embedding_dim),
        cfg_dropout_prob=cfg_dropout_prob,
        timestep_sampling=timestep_sampling,
    ),
}
