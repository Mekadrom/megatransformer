import torch
import torch.nn as nn

from model.image.vae import VAE
from model.megatransformer_diffusion import GaussianDiffusion


class VAEDiffusionWrapper(nn.Module):
    """Wrapper for VAE to be used as a diffuser model."""
    def __init__(self, latent_scale_factor: float, vae: VAE, diffusion: GaussianDiffusion):
        super().__init__()
        
        self.latent_scale_factor = latent_scale_factor
        self.vae = vae
        self.diffusion = diffusion

        # freeze VAE
        self.vae.eval()
        self.vae.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.vae.eval()
        self.vae.requires_grad_(False)
        return self

    def forward(self, x: torch.Tensor, **kwargs):
        # for training, encode input to latent space
        mu, _ = self.vae.encoder(x)
        mu = mu * self.latent_scale_factor

        # take only mean for diffusion
        latent_diffusion_output, losses = self.diffusion.forward(mu, **kwargs)
        mse_loss = losses[0]

        return latent_diffusion_output, mse_loss

    def sample(self, device, batch_size: int, **kwargs) -> torch.Tensor:
        # sample in latent space (pass latent image dimensions as "image_size" in kwargs)
        latent_samples = self.diffusion.sample(device, batch_size, **kwargs)
        latent_samples = latent_samples / self.latent_scale_factor        

        # decode to image space
        recon_images = self.vae.decoder(latent_samples)

        return recon_images
