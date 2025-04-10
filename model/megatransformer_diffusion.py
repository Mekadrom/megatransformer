from functools import partial
from torch.amp import autocast
from typing import Optional, Union

from model import megatransformer_modules

import math
import megatransformer_utils
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, activation, norm):
        super().__init__()

        activation_type = megatransformer_utils.get_activation_type(activation)

        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = norm(out_channels)
        self.act = activation_type() if activation_type is not megatransformer_modules.SwiGLU else megatransformer_modules.SwiGLU(in_channels)

    def forward(self, x, time_embedding=None):
        x = self.proj(x)
        x = self.norm(x)

        if time_embedding is not None:
            x = x + time_embedding

        x = self.act(x)

        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim, activation, norm):
        super().__init__()
        self.time_embedding_dim = time_embedding_dim

        activation_type = megatransformer_utils.get_activation_type(activation)

        if time_embedding_dim is not None:
            self.time_mlp = nn.Sequential(
                activation_type() if activation_type is not megatransformer_modules.SwiGLU else megatransformer_modules.SwiGLU(time_embedding_dim),
                nn.Linear(time_embedding_dim, out_channels),
            )

        self.block1 = Block(in_channels, out_channels, activation, norm=norm)
        self.block2 = Block(out_channels, out_channels, activation, norm=norm)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.GroupNorm):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0.2)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(self.shortcut, nn.Conv2d):
            nn.init.kaiming_normal_(self.shortcut.weight, a=0.2)
            self.shortcut.bias.data.zero_()

    def forward(self, x: torch.Tensor, time_embedding: Optional[torch.Tensor]=None) -> torch.Tensor:
        time_embedding = None
        if self.time_embedding_dim is not None and time_embedding is not None:
            time_embedding = self.time_mlp(time_embedding)

        h = self.block1(x, time_embedding=time_embedding)
        h = self.block2(h)
        return h + self.shortcut(x)

class DownBlock(nn.Module):
    def __init__(self,
                 stride,
                 in_channels: int,
                 out_channels: int,
                 time_embedding_dim: int,
                 activation,
                 self_attn_class,
                 norm_class,
                 has_attn: bool=False,
                 num_res_blocks: int=2,
                 dropout: float=0.1,
                 self_attn_n_heads=6,
                 self_attn_d_queries=64,
                 self_attn_d_values=64,
                 self_attn_use_flash_attention=True):
        super().__init__()

        self.norms = nn.ModuleList([
            megatransformer_modules.RMSNorm(in_channels if i == 0 else out_channels)
            for i in range(num_res_blocks)
        ])

        self.res_blocks = nn.ModuleList([
            ResidualBlock(in_channels if i == 0 else out_channels, out_channels, time_embedding_dim, activation, norm_class)
            for i in range(num_res_blocks)
        ])

        self.attn_blocks = nn.ModuleList([
            self_attn_class(
                out_channels, self_attn_n_heads, self_attn_d_queries, self_attn_d_values, use_flash_attention=self_attn_use_flash_attention, dropout=dropout
            ) if has_attn else nn.Identity()
            for _ in range(num_res_blocks)
        ])

        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.downsample.weight, a=0.2)
        self.downsample.bias.data.zero_()

    def forward(self, x: torch.Tensor, time_embedding: Optional[torch.Tensor]=None, condition=None) -> tuple[torch.Tensor, torch.Tensor]:
        for norm, res_block, attn_block in zip(self.norms, self.res_blocks, self.attn_blocks):
            # switch channel and last dimension
            x = x.permute(0, 2, 3, 1).contiguous()
            x = norm(x)
            # switch back
            x = x.permute(0, 3, 1, 2).contiguous()
            x = res_block(x, time_embedding)
            x = attn_block(x)
        return self.downsample(x), x

class UpBlock(nn.Module):
    def __init__(self,
                 hidden_size,
                 in_channels: int,
                 out_channels: int,
                 time_embedding_dim: int,
                 activation,
                 scale_factor,
                 self_attn_class,
                 cross_attn_class,
                 norm_class,
                 has_attn: bool=False,
                 has_condition: bool=False,
                 num_res_blocks: int=2,
                 dropout: float=0.1,
                 self_attn_n_heads=6,
                 self_attn_d_queries=64,
                 self_attn_d_values=64,
                 self_attn_use_flash_attention=True,
                 cross_attn_n_heads=6,
                 cross_attn_d_queries=64,
                 cross_attn_d_values=64,
                 cross_attn_use_flash_attention=True):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        )

        self.norms = nn.ModuleList([
            megatransformer_modules.RMSNorm(in_channels*2 if i == 0 else out_channels)
            for i in range(num_res_blocks)
        ])

        self.res_blocks = nn.ModuleList([
            ResidualBlock(in_channels*2 if i == 0 else out_channels, out_channels, time_embedding_dim, activation, norm_class)
            for i in range(num_res_blocks)
        ])

        self.attn_blocks = nn.ModuleList([
            self_attn_class(
                out_channels, self_attn_n_heads, self_attn_d_queries, self_attn_d_values, use_flash_attention=self_attn_use_flash_attention, dropout=dropout
            ) if has_attn else nn.Identity()
            for _ in range(num_res_blocks)
        ])

        self.cross_attn_blocks = nn.ModuleList([
            cross_attn_class(
                out_channels, cross_attn_n_heads, cross_attn_d_queries, cross_attn_d_values, context_dim=hidden_size, use_flash_attention=cross_attn_use_flash_attention, dropout=dropout
            ) if has_attn and has_condition else nn.Identity()
            for _ in range(num_res_blocks)
        ])

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.upsample[1].weight, a=0.2)
        self.upsample[1].bias.data.zero_()

    def forward(self, x: torch.Tensor, skip: list[torch.Tensor], time_embedding: torch.Tensor, condition=None) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        for norm, res_block, attn_block, cross_attn_block in zip(self.norms, self.res_blocks, self.attn_blocks, self.cross_attn_blocks):
            # switch channel and last dimension
            x = x.permute(0, 2, 3, 1).contiguous()
            x = norm(x)
            # switch back
            x = x.permute(0, 3, 1, 2).contiguous()
            x = res_block(x, time_embedding)
            x = attn_block(x)
            if condition is not None and not isinstance(cross_attn_block, nn.Identity):
                x = cross_attn_block(x, condition)
        return x

class UNet(nn.Module):
    def __init__(
            self,
            hidden_size,
            activation,
            self_attn_class,
            cross_attn_class,
            norm_class,
            scale_factor: Union[int, tuple[int, int]] = 2,
            stride: Union[int, tuple[int, int]] = 1,
            in_channels: int = 3,
            model_channels: int = 64,
            out_channels: int = 3,
            channel_multipliers: list[int] = [2, 4, 8],
            time_embedding_dim: int = 256,
            attention_levels: list[bool] = [False, False, True, True],
            num_res_blocks: int = 2,
            dropout: float = 0.1,
            has_condition: bool = False,
            down_block_self_attn_n_heads=6,
            down_block_self_attn_d_queries=64,
            down_block_self_attn_d_values=64,
            down_block_self_attn_use_flash_attention=True,
            up_block_self_attn_n_heads=6,
            up_block_self_attn_d_queries=64,
            up_block_self_attn_d_values=64,
            up_block_self_attn_use_flash_attention=True,
            cross_attn_n_heads=6,
            cross_attn_d_queries=64,
            cross_attn_d_values=64,
            cross_attn_use_flash_attention=True,
    ):
        super().__init__()

        activation_type = megatransformer_utils.get_activation_type(activation)
        self.time_embedding = nn.Sequential(
            megatransformer_modules.SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_embedding_dim),
            activation_type() if activation_type is not megatransformer_modules.SwiGLU else megatransformer_modules.SwiGLU(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        channels = [model_channels]
        for multiplier in channel_multipliers:
            channels.append(model_channels * multiplier)

        self.down_blocks = nn.ModuleList()
        for i in range(len(channel_multipliers)):
            self.down_blocks.append(
                DownBlock(
                    stride,
                    channels[i],
                    channels[i + 1],
                    time_embedding_dim,
                    activation,
                    self_attn_class,
                    norm_class,
                    has_attn=attention_levels[i],
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    self_attn_n_heads=down_block_self_attn_n_heads,
                    self_attn_d_queries=down_block_self_attn_d_queries,
                    self_attn_d_values=down_block_self_attn_d_values,
                    self_attn_use_flash_attention=down_block_self_attn_use_flash_attention,
                )
            )

        self.middle_res_block = ResidualBlock(
            channels[-1], channels[-1], time_embedding_dim, activation, norm_class
        )
        self.middle_attn_norm = megatransformer_modules.RMSNorm(channels[-1])
        self.middle_attn_block = self_attn_class(
            channels[-1], down_block_self_attn_n_heads, down_block_self_attn_d_queries, down_block_self_attn_d_values, use_flash_attention=down_block_self_attn_use_flash_attention, dropout=dropout, is_linear_attention=False
        )
        self.middle_res_block2 = ResidualBlock(
            channels[-1], channels[-1], time_embedding_dim, activation, norm_class
        )

        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(channel_multipliers))):
            self.up_blocks.append(
                UpBlock(
                    hidden_size,
                    channels[i + 1],
                    channels[i],
                    time_embedding_dim,
                    activation,
                    scale_factor,
                    self_attn_class,
                    cross_attn_class,
                    norm_class,
                    has_attn=attention_levels[i],
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    has_condition=has_condition,
                    self_attn_n_heads=up_block_self_attn_n_heads,
                    self_attn_d_queries=up_block_self_attn_d_queries,
                    self_attn_d_values=up_block_self_attn_d_values,
                    self_attn_use_flash_attention=up_block_self_attn_use_flash_attention,
                    cross_attn_n_heads=cross_attn_n_heads,
                    cross_attn_d_queries=cross_attn_d_queries,
                    cross_attn_d_values=cross_attn_d_values,
                    cross_attn_use_flash_attention=cross_attn_use_flash_attention,
                )
            )

        self.final_res_block = ResidualBlock(
            model_channels*2, model_channels, time_embedding_dim, activation, norm_class
        )
        self.final_conv = nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        self.init_conv.weight.data.normal_(0.0, 0.02)
        self.init_conv.bias.data.zero_()
        self.final_conv.weight.data.normal_(0.0, 0.02)
        self.final_conv.bias.data.zero_()

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, condition=None) -> torch.Tensor:
        time_embedding = self.time_embedding(timesteps)

        h = self.init_conv(x)
        initial_h = h

        skips = []
        for i, down_block in enumerate(self.down_blocks):
            h, skip = down_block(h, time_embedding, condition=condition)
            skips.append(skip)
        
        h = self.middle_res_block(h, time_embedding)
        residual = h
        h = h.permute(0, 2, 3, 1).contiguous()
        h = self.middle_attn_norm(h)
        h = h.permute(0, 3, 1, 2).contiguous()
        h = residual + self.middle_attn_block(h)
        h = self.middle_res_block2(h, time_embedding)

        for i, (up_block, skip) in enumerate(zip(self.up_blocks, reversed(skips))):
            h = up_block(h, skip, time_embedding, condition=condition)

        h = torch.cat([h, initial_h], dim=1)
        h = self.final_res_block(h, time_embedding)
        h = self.final_conv(h)

        return h

class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            config: megatransformer_utils.MegaTransformerConfig,
            hidden_size,
            activation,
            scale_factor,
            stride,
            self_attn_class,
            cross_attn_class,
            norm_class,
            in_channels: int,
            model_channels: int,
            out_channels: int,
            time_embedding_dim: int,
            num_res_blocks: int,
            beta_start: float = 1e-4,
            beta_end: float = 0.02,
            num_timesteps: int = 1000,
            has_condition: bool = False,
            unet_dropout: float = 0.1,
            betas_schedule="linear",
            down_block_self_attn_n_heads=6,
            down_block_self_attn_d_queries=64,
            down_block_self_attn_d_values=64,
            down_block_self_attn_use_flash_attention=True,
            up_block_self_attn_n_heads=6,
            up_block_self_attn_d_queries=64,
            up_block_self_attn_d_values=64,
            up_block_self_attn_use_flash_attention=True,
            cross_attn_n_heads=6,
            cross_attn_d_queries=64,
            cross_attn_d_values=64,
            cross_attn_use_flash_attention=True,
            min_snr_loss_weight: bool = False,
            min_snr_gamma: float = 5.0,
            normalize: bool = True,
            ddim_sampling_eta = 0.0,
            sampling_timesteps=None,
    ):
        super().__init__()

        self.config = config

        self.num_timesteps = num_timesteps
        self.normalize = normalize
        self.ddim_sampling_eta = ddim_sampling_eta

        if sampling_timesteps is None:
            self.sampling_timesteps = num_timesteps
            self.is_ddim_sampling = False
        else:
            self.sampling_timesteps = sampling_timesteps
            assert sampling_timesteps <= num_timesteps, f"Sampling timesteps {sampling_timesteps} must be less than or equal to total timesteps {num_timesteps}."
            self.is_ddim_sampling = sampling_timesteps < num_timesteps

        self.unet = UNet(
            hidden_size,
            activation,
            scale_factor=scale_factor,
            stride=stride,
            self_attn_class=self_attn_class,
            cross_attn_class=cross_attn_class,
            norm_class=norm_class,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            time_embedding_dim=time_embedding_dim,
            num_res_blocks=num_res_blocks,
            dropout=unet_dropout,
            has_condition=has_condition,
            down_block_self_attn_n_heads=down_block_self_attn_n_heads,
            down_block_self_attn_d_queries=down_block_self_attn_d_queries,
            down_block_self_attn_d_values=down_block_self_attn_d_values,
            down_block_self_attn_use_flash_attention=down_block_self_attn_use_flash_attention,
            up_block_self_attn_n_heads=up_block_self_attn_n_heads,
            up_block_self_attn_d_queries=up_block_self_attn_d_queries,
            up_block_self_attn_d_values=up_block_self_attn_d_values,
            up_block_self_attn_use_flash_attention=up_block_self_attn_use_flash_attention,
            cross_attn_n_heads=cross_attn_n_heads,
            cross_attn_d_queries=cross_attn_d_queries,
            cross_attn_d_values=cross_attn_d_values,
            cross_attn_use_flash_attention=cross_attn_use_flash_attention,
        )

        if betas_schedule == "cosine":
            betas = self.cosine_beta_schedule(num_timesteps)
        elif betas_schedule == "sigmoid":
            betas = self.sigmoid_beta_schedule(num_timesteps)
        else:
            betas = torch.linspace(beta_start, beta_end, num_timesteps)

        self.register_buffer("betas", betas)

        alphas = 1.0 - betas
        self.register_buffer("alphas", alphas)

        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1. / alphas_cumprod))
        self.register_buffer("sqrt_recip1m_alphas_cumprod", torch.sqrt(1. / (1. - alphas_cumprod)))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        loss_weight = maybe_clipped_snr / snr

        self.register_buffer('loss_weight', loss_weight)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps) / timesteps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def sigmoid_beta_schedule(self, timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps) / timesteps
        v_start = torch.tensor(start / tau).sigmoid()
        v_end = torch.tensor(end / tau).sigmoid()
        alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def normalize_to_neg_one_to_one(self, x_0):
        return x_0 * 2 - 1

    def unnormalize_to_zero_to_one(self, x):
        return (x + 1) * 0.5

    def _extract(self, a: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        b, *_ = t.shape
        out = a.gather(-1, t.long())
        return out.reshape(b, *((1,) * (len(shape) - 1))).to(t.device)

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - 
            self._extract(self.sqrt_recip1m_alphas_cumprod, t, x_t.shape) * noise
        )

    def model_predictions(self, x, t, condition=None, clip_x_start=False):
        model_output = self.unet(x, t, condition)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else nn.Identity()

        pred_noise = model_output
        x_start = self.predict_start_from_noise(x, t, pred_noise)
        x_start = maybe_clip(x_start)

        return pred_noise, x_start

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, condition=None, clip_denoised=True):
        _, x_start = self.model_predictions(x, t, condition=condition)

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: int, condition=None):
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)

        model_mean, variance, model_log_variance, x_start = self.p_mean_variance(
            x = x, t = batched_times, condition=condition, clip_denoised=True
        )

        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0

        pred_x0 = model_mean + torch.exp(0.5 * model_log_variance) * noise

        return pred_x0, x_start

    @torch.no_grad()
    def p_sample_loop(self, x, condition=None, return_intermediate: bool=False):
        noise_preds = []
        x_start_preds = []
        for time_step in reversed(range(0, self.num_timesteps)):
            x, x_start = self.p_sample(x, time_step, condition=condition)
            if return_intermediate:
                noise_preds.append(x)
                x_start_preds.append(x_start)
        return x, noise_preds, x_start_preds if return_intermediate else x

    @torch.no_grad()
    def ddim_sample_loop(self, x, condition=None, return_intermediate=False, override_sampling_steps=None):
        sampling_timesteps = override_sampling_steps if override_sampling_steps is not None else self.sampling_timesteps

        times = torch.linspace(-1, self.num_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn_like(x, device=x.device)

        noise_preds = []
        x_start_preds = []

        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((x.shape[0],), time, device=x.device, dtype = torch.long)
            pred_noise, x_start = self.model_predictions(img, time_cond, condition=condition, clip_x_start=True)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            
            if return_intermediate:
                noise_preds.append(img)
                x_start_preds.append(x_start)

        return img, noise_preds, x_start_preds if return_intermediate else img

    @torch.no_grad()
    def sample(self, device, batch_size: int, image_size: int, condition: Optional[torch.Tensor]=None, return_intermediate: bool=False, override_ddim_sampling_steps: Optional[int]=None, generator=None) -> torch.Tensor:
        x = torch.randn(batch_size, 3, image_size, image_size, device=device, generator=generator)

        if self.is_ddim_sampling or override_ddim_sampling_steps is not None:
            x = self.ddim_sample_loop(x, condition=condition, return_intermediate=return_intermediate, override_sampling_steps=override_ddim_sampling_steps)
        else:
            x = self.p_sample_loop(x, condition=condition, return_intermediate=return_intermediate)

        if return_intermediate:
            img = x[0]
        else:
            img = x

        if self.normalize:
            img = self.unnormalize_to_zero_to_one(img)

        return (img, *x[1:]) if return_intermediate else img
    
    @autocast('cuda', enabled = False)
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor]=None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, noise=None, condition=None):
        b, c, h, w = x_start.shape
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        predicted_noise  = self.unet(x_noisy, t, condition)

        loss = F.mse_loss(predicted_noise , noise, reduction='none')
        loss = loss.mean(dim=[1, 2, 3])

        loss = loss * self._extract(self.loss_weight, t, loss.shape)

        return predicted_noise, loss.mean()

    def forward(self, x_0, condition=None):
        """
        default impl is image diffusion
        returns model output noises and noise reconstruction loss from `p_losses` function
        """

        if len(x_0.shape) == 5:
            # squish batch and example dimensions if necessary
            b, e, c, h, w = x_0.shape
            x_0 = x_0.view(-1, c, h, w)
        else:
            b, c, h, w = x_0.shape
            e = None

        if condition is not None:
            if len(condition.shape) == 5:
                # squish batch and example dimensions if necessary
                *_, c, h, w = condition.shape
                condition = condition.view(-1, c, h, w)

        t = torch.randint(0, self.num_timesteps, (x_0.shape[0],), device= x_0.device).long()

        if self.normalize:
            x_0 = self.normalize_to_neg_one_to_one(x_0)

        model_output, mel_l1_loss = self.p_losses(x_0, t, condition=condition)
        if e is not None:
            # restore model example dimension
            model_output = model_output.view(b, e, c, h, w)
        return model_output, mel_l1_loss
