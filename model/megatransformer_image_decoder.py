from typing import Union

from model import megatransformer_diffusion

import megatransformer_utils
import torch
import torch.nn.functional as F


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
        model_output = self.unet(x, t.to(x.dtype), condition=condition)
        
        if self.predict_epsilon:
            # Model predicts noise ε
            pred_epsilon = model_output
            pred_x0 = sqrt_recip_alphas_t * x - sqrt_recip_alphas_t * sqrt_one_minus_alphas_cumprod_t * pred_epsilon
            pred_x0 = torch.clamp(pred_x0, -1., 1.)
            
            # Calculate posterior mean using betas_t
            denominator = torch.sqrt(1 - self._extract(self.alphas_cumprod, t, x.shape) + 1e-8)
            posterior_mean = (
                x * (1 - betas_t) / denominator +
                pred_x0 * betas_t / denominator
            )
        else:
            # Model directly predicts x_0
            pred_x0 = model_output
            pred_x0 = torch.clamp(pred_x0, -1., 1.)
            
            # Calculate posterior mean using betas_t
            posterior_mean = (
                x * (1 - betas_t) / denominator +
                pred_x0 * betas_t / denominator
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
    def sample(self, device, condition, batch_size=1, image_size=224, return_intermediate=False) -> Union[tuple[torch.Tensor, list[torch.Tensor]], torch.Tensor]:
        """Sample with conditioning"""
        # Start from pure noise
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)
        
        # Iteratively denoise with conditioning
        time_step_outputs = []
        for time_step in reversed(range(1, self.num_timesteps)):
            t = torch.full((batch_size,), time_step, device=device, dtype=torch.long)
            x = self.p_sample(x, t, time_step, condition=condition)
            if return_intermediate:
                # perform same image normalization per intermediate step
                intermediate_output = (x + 1) / 2
                intermediate_output = torch.clamp(intermediate_output, 0.0, 1.0)
                time_step_outputs.append(intermediate_output)
            
        # Scale to [0, 1] range
        x = (x + 1) / 2
        x = torch.clamp(x, 0.0, 1.0)
        
        return x, time_step_outputs if return_intermediate else x
    
    def forward(self, x_0: torch.Tensor, condition=None):
        """Training forward pass with conditioning"""
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_0.device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        
        # Model forward pass with condition
        model_output = self.unet(x_t, t.to(x_0.dtype), condition=condition)
        
        loss_dict = {}
        if self.predict_epsilon:
            # Loss to the added noise
            loss_dict["noise_mse"] = F.mse_loss(model_output, noise)

            pred_x0 = self.predict_start_from_noise(x_t, t, model_output)
            # megatransformer_utils.print_debug_tensor('initial pred_x0', pred_x0)
            pred_x0 = torch.clamp(pred_x0, -1., 1.)
            model_output = pred_x0
            
            # Loss to the original image
            loss_dict["pred_x0_mse"] = F.mse_loss(model_output, x_0)
        else:
            # Loss to the original image
            loss_dict["direct"] = F.mse_loss(model_output, x_0)

        total_loss = (
            1.0 * loss_dict.get("noise_mse", 0.0) +
            1.0 * loss_dict.get("pred_x0_mse", 0.0) +
            1.0 * loss_dict.get("direct", 0.0)
        )

        return model_output, total_loss
