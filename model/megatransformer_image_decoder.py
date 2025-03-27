from model import megatransformer_diffusion, megatransformer_modules

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

class ImageConditionalGaussianDiffusion(megatransformer_diffusion.GaussianDiffusion):
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
