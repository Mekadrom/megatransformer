from typing import Optional

from model import megatransformer_modules

import megatransformer_utils
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageUpsampleConv2dGenerator(nn.Module):
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig):
        super().__init__()
        self.config = config

        activation = config.image_decoder_activation
        dropout = config.image_decoder_dropout

        self.conv_layers = nn.ModuleList([
            nn.Unflatten(-1, (768, 1, 1)),
        ])

        activation_type = megatransformer_utils.get_activation_type(activation)

        channels = [768, 384, 96, 48, 24, 3]
        image_sizes = [1, 4, 16, 64, 128, 224]
        
        for i in range(len(channels) - 1):
            out_channels = channels[i+1]
            upsample_target = image_sizes[i+1]

            self.conv_layers.append(nn.Sequential(
                nn.Upsample(size=(upsample_target, upsample_target), mode="bilinear", align_corners=False),
                nn.Conv2d(channels[i], out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                activation_type() if activation_type is not megatransformer_modules.SwiGLU else megatransformer_modules.SwiGLU(out_channels),
                nn.Dropout2d(dropout)
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is of the shape [batch_size, sequence_length, hidden_size]
        x = x.permute(0, 2, 1) # [batch_size, hidden_size, sequence_length]

        # do mean pooling to get one feature for now
        x = x.mean(dim=-1) # [batch_size, hidden_size]

        for layer in self.conv_layers:
            x = layer(x) # unflatten makes it [batch_size, hidden_size, 1, 1] and it upsamples size and downsamples featuers from there
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim, activation, dropout):
        super().__init__()
        self.time_embedding_dim = time_embedding_dim

        activation_type = megatransformer_utils.get_activation_type(activation)
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            activation_type() if activation_type is not megatransformer_modules.SwiGLU else megatransformer_modules.SwiGLU(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        if time_embedding_dim is not None:
            self.time_mlp = nn.Sequential(
                activation_type() if activation_type is not megatransformer_modules.SwiGLU else megatransformer_modules.SwiGLU(time_embedding_dim),
                nn.Linear(time_embedding_dim, out_channels),
            )

        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            activation_type() if activation_type is not megatransformer_modules.SwiGLU else megatransformer_modules.SwiGLU(out_channels),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, time_embedding: Optional[torch.Tensor]=None) -> torch.Tensor:
        h = self.conv1(x)

        if self.time_embedding_dim is not None and time_embedding is not None:
            time_embedding = self.time_mlp(time_embedding)
            h = h + time_embedding.unsqueeze(-1).unsqueeze(-1)

        h = self.conv2(h)
        return h + self.shortcut(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embedding_dim: int, activation, has_attn: bool=False, num_res_blocks: int=2, dropout: float=0.1):
        super().__init__()

        self.res_blocks = nn.ModuleList([
            ResidualBlock(in_channels if i == 0 else out_channels, out_channels, time_embedding_dim, activation, dropout)
            for i in range(num_res_blocks)
        ])

        self.attn_blocks = nn.ModuleList([
            megatransformer_modules.SimpleSelfAttentionBlock(out_channels) if has_attn else nn.Identity()
            for _ in range(num_res_blocks) if has_attn
        ])

        self.cross_attn_blocks = nn.ModuleList([
            megatransformer_modules.SimpleCrossAttentionBlock(out_channels, out_channels) if has_attn else nn.Identity()
            for _ in range(num_res_blocks) if has_attn
        ])

        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, time_embedding: Optional[torch.Tensor]=None, condition=None) -> torch.Tensor:
        skips = []
        for res_block, attn_block, cross_attn_block in zip(self.res_blocks, self.attn_blocks, cross_attn_block):
            x = res_block(x, time_embedding)
            x = attn_block(x)
            if condition is not None:
                x = cross_attn_block(x, condition)
            skips.append(x)

        return self.downsample(x), skips

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embedding_dim: int, activation, has_attn: bool=False, num_res_blocks: int=2, dropout: float=0.1):
        super().__init__()

        self.res_blocks = nn.ModuleList([
            ResidualBlock(in_channels if i == 0 else out_channels, out_channels, time_embedding_dim, activation, dropout)
            for i in range(num_res_blocks)
        ])

        self.attn_blocks = nn.ModuleList([
            megatransformer_modules.SimpleSelfAttentionBlock(out_channels) if has_attn else nn.Identity()
            for _ in range(num_res_blocks) if has_attn
        ])

        self.cross_attn_blocks = nn.ModuleList([
            megatransformer_modules.SimpleCrossAttentionBlock(out_channels, out_channels) if has_attn else nn.Identity()
            for _ in range(num_res_blocks) if has_attn
        ])

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, skip_connections: list[torch.Tensor], time_embedding: torch.Tensor, condition=None) -> torch.Tensor:
        for res_block, attn_block, cross_attn_block, skip in zip(self.res_blocks, self.attn_blocks, self.cross_attn_blocks, skip_connections):
            x = torch.cat([x, skip], dim=1)
            x = res_block(x, time_embedding)
            x = attn_block(x)
            if condition is not None:
                x = cross_attn_block(x, condition)

        x = self.upsample(x)
        return x

class UNet(nn.Module):
    def __init__(
            self,
            activation,
            hidden_size,
            in_channels: int = 3,
            model_channels: int = 64,
            out_channels: int = 3,
            channel_multipliers: list[int] = [1, 2, 4, 8],
            time_embedding_dim: int = 256,
            attention_levels: list[bool] = [False, False, True, True],
            num_res_blocks: int = 2,
            dropout: float = 0.1,
    ):
        super().__init__()

        activation_type = megatransformer_utils.get_activation_type(activation)
        self.time_embedding = nn.Sequential(
            megatransformer_modules.SinusoidalPositionalEmbeddings(model_channels),
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
                    channels[i],
                    channels[i + 1],
                    time_embedding_dim,
                    activation,
                    has_attn=attention_levels[i],
                    num_res_blocks=num_res_blocks,
                    dropout=dropout
                )
            )

        self.middle_res_block = ResidualBlock(
            channels[-1], channels[-1], time_embedding_dim, activation, dropout
        )
        self.middle_attn_block = megatransformer_modules.SimpleSelfAttentionBlock(channels[-1])
        self.middle_res_block2 = ResidualBlock(
            channels[-1], channels[-1], time_embedding_dim, activation, dropout
        )

        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(channel_multipliers))):
            self.up_blocks.append(
                UpBlock(
                    channels[i + 1],
                    channels[i],
                    time_embedding_dim,
                    activation,
                    has_attn=attention_levels[i],
                    num_res_blocks=num_res_blocks,
                    dropout=dropout
                )
            )

        self.final_res_block = ResidualBlock(
            model_channels * 2, model_channels * 2, time_embedding_dim, activation, dropout
        )
        self.final_conv = nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, condition=None) -> torch.Tensor:
        time_embedding = self.time_embedding(timesteps)

        h = self.init_conv(x)
        initial_h = h

        skip_connections = []
        for down_block in self.down_blocks:
            h, skips = down_block(h, time_embedding, condition=condition)
            skip_connections.append(skips)

        h = self.middle_res_block(h, time_embedding)
        h = self.middle_attn_block(h)
        h = self.middle_res_block2(h, time_embedding)

        for up_block, down_skips in zip(self.up_blocks, reversed(skip_connections)):
            h = up_block(h, down_skips, time_embedding, condition=condition)

        h = torch.cat([h, initial_h], dim=1)
        h = self.final_res_block(h, time_embedding)
        h = self.final_conv(h)

        return h

class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            activation,
            hidden_size,
            beta_start: float = 1e-4,
            beta_end: float = 0.02,
            num_timesteps: int = 1000,
            predict_epsilon: bool = True,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.predict_epsilon = predict_epsilon

        self.unet = UNet(
            activation,
            hidden_size,
        )

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod.clone().detach()) / (1. - self.alphas_cumprod.clone().detach())
        self.posterior_variance = torch.cat([torch.tensor([0.0]), self.posterior_variance])
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod.clone().detach()) / (1. - self.alphas_cumprod.clone().detach())
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod.clone().detach()) * torch.sqrt(self.alphas_cumprod.clone().detach()) / (1. - self.alphas_cumprod.clone().detach())

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recip1m_alphas_cumprod = torch.sqrt(1.0 / (1.0 - self.alphas_cumprod))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor]=None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, t_index, condition=None):
        """Single step of the reverse diffusion process with conditioning"""
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x.shape)
        
        # Model forward pass with condition
        model_output = self.unet(x, t, condition=condition)
        
        if self.predict_epsilon:
            # Model predicts noise ε
            pred_epsilon = model_output
            pred_x0 = sqrt_recip_alphas_t * x - sqrt_recip_alphas_t * sqrt_one_minus_alphas_cumprod_t * pred_epsilon
            pred_x0 = torch.clamp(pred_x0, -1., 1.)
            
            # Calculate posterior mean using betas_t
            posterior_mean = (
                x * (1 - betas_t) / torch.sqrt(1 - self._extract(self.alphas_cumprod, t, x.shape)) +
                pred_x0 * betas_t / torch.sqrt(1 - self._extract(self.alphas_cumprod, t, x.shape))
            )
        else:
            # Model directly predicts x_0
            pred_x0 = model_output
            pred_x0 = torch.clamp(pred_x0, -1., 1.)
            
            # Calculate posterior mean using betas_t
            posterior_mean = (
                x * (1 - betas_t) / torch.sqrt(1 - self._extract(self.alphas_cumprod, t, x.shape)) +
                pred_x0 * betas_t / torch.sqrt(1 - self._extract(self.alphas_cumprod, t, x.shape))
            )
        
        # Calculate posterior variance using betas_t
        posterior_variance = betas_t * (1 - self._extract(self.alphas_cumprod, t-1, x.shape)) / (1 - self._extract(self.alphas_cumprod, t, x.shape))
        
        if t_index == 0:
            # No noise at the last step (t=0)
            return posterior_mean
        else:
            noise = torch.randn_like(x)
            return posterior_mean + torch.sqrt(posterior_variance) * noise
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        b, *_ = t.shape
        out = a.gather(-1, t.long())
        return out.reshape(b, *((1,) * (len(shape) - 1))).to(t.device)
    
    @torch.no_grad()
    def sample(self, device, batch_size: int, image_size: int) -> torch.Tensor:
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)

        for time_step in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), time_step, device=device, dtype=torch.long)
            x = self.p_sample(x, t, time_step)

        x = (x + 1) / 2
        x = torch.clamp(x, 0.0, 1.0)

        return x
    
    def forward(self, x_0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_0.device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)

        model_output = self.unet(x_t, t)

        if self.predict_epsilon:
            return model_output, noise
        else:
            return model_output, x_0

class ConditionalGaussianDiffusion(GaussianDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def q_sample(self, x_start, t, noise=None, condition=None):
        # Same as before, condition not needed for forward process
        return super().q_sample(x_start, t, noise)
    
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, t_index, condition=None):
        """Single step of the reverse diffusion process with conditioning"""
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x.shape)
        
        # Model forward pass with condition
        model_output = self.unet(x, t, condition=condition)
        
        if self.predict_epsilon:
            # Model predicts noise ε
            pred_epsilon = model_output
            pred_x0 = sqrt_recip_alphas_t * x - sqrt_recip_alphas_t * sqrt_one_minus_alphas_cumprod_t * pred_epsilon
            pred_x0 = torch.clamp(pred_x0, -1., 1.)
            
            # Calculate posterior mean using betas_t
            posterior_mean = (
                x * (1 - betas_t) / torch.sqrt(1 - self._extract(self.alphas_cumprod, t, x.shape)) +
                pred_x0 * betas_t / torch.sqrt(1 - self._extract(self.alphas_cumprod, t, x.shape))
            )
        else:
            # Model directly predicts x_0
            pred_x0 = model_output
            pred_x0 = torch.clamp(pred_x0, -1., 1.)
            
            # Calculate posterior mean using betas_t
            posterior_mean = (
                x * (1 - betas_t) / torch.sqrt(1 - self._extract(self.alphas_cumprod, t, x.shape)) +
                pred_x0 * betas_t / torch.sqrt(1 - self._extract(self.alphas_cumprod, t, x.shape))
            )
        
        # Calculate posterior variance using betas_t
        posterior_variance = betas_t * (1 - self._extract(self.alphas_cumprod, t-1, x.shape)) / (1 - self._extract(self.alphas_cumprod, t, x.shape))
        
        if t_index == 0:
            # No noise at the last step (t=0)
            return posterior_mean
        else:
            noise = torch.randn_like(x)
            return posterior_mean + torch.sqrt(posterior_variance) * noise
    
    @torch.no_grad()
    def sample(self, device, batch_size=1, image_size=224, condition=None):
        """Sample with conditioning"""
        # Start from pure noise
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)
        
        # Iteratively denoise with conditioning
        for time_step in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), time_step, device=device, dtype=torch.long)
            x = self.p_sample(x, t, time_step, condition=condition)
            
        # Scale to [0, 1] range
        x = (x + 1) / 2
        x = torch.clamp(x, 0.0, 1.0)
        
        return x
    
    def forward(self, x_0: torch.Tensor, condition=None):
        """Training forward pass with conditioning"""
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_0.device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        
        # Model forward pass with condition
        model_output = self.unet(x_t, t, condition=condition)
        
        if self.predict_epsilon:
            # Loss to the added noise
            loss = F.mse_loss(model_output, noise)
        else:
            # Loss to the original image
            loss = F.mse_loss(model_output, x_0)
        return model_output, loss
