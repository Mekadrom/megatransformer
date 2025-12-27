import torch.nn as nn

from model import activations
from model.vae import VAE
from utils.model_utils import get_activation_type


class AudioVAEEncoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        latent_channels=4,
        intermediate_channels=[32, 64, 128],
        kernel_sizes=[(3, 5), (3, 5), (3, 5)],
        strides=[(2, 3), (2, 5), (2, 5)],
        activation_fn: str = "silu"
    ):
        super().__init__()

        activation_type = get_activation_type(activation_fn)
        if activation_type not in [activations.SwiGLU, activations.Snake]:
            activation = lambda _: activation_type()  # drop unused arg
        else:
            activation = activation_type

        channels = [in_channels] + intermediate_channels

        self.channel_upsample = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=(kernel_size[0]//2, kernel_size[1]//2)),
                nn.GroupNorm(out_c // 4, out_c),
                activation(out_c)
            )
            for in_c, out_c, kernel_size, stride in zip(channels[:-1], channels[1:], kernel_sizes, strides)
        ])

        self.fc_mu = nn.Conv2d(channels[-1], latent_channels, kernel_size=(3, 5), padding=(1, 2))
        self.fc_logvar = nn.Conv2d(channels[-1], latent_channels, kernel_size=(3, 5), padding=(1, 2))

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Initialize mu/logvar with smaller variance for stable latent space
        nn.init.normal_(self.fc_mu.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.normal_(self.fc_logvar.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc_logvar.bias)

    def forward(self, x):
        for upsample in self.channel_upsample:
            x = upsample(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class AudioVAEDecoder(nn.Module):
    def __init__(
            self,
            latent_channels=4,
            out_channels=3,
            intermediate_channels=[128, 64, 32],
            scale_factors=[(2, 3), (2, 5), (2, 5)],
            kernel_sizes=[(3, 5), (3, 5), (3, 5)],
            activation_fn: str = "silu",
            speaker_embedding_dim: int = 0,  # 0 = no speaker conditioning
            normalize_speaker_embedding: bool = True,  # L2 normalize speaker embeddings before FiLM
            film_scale_bound: float = 0.5,  # Max absolute value for FiLM scale (0 = unbounded)
            film_shift_bound: float = 0.5,  # Max absolute value for FiLM shift (0 = unbounded)
        ):
        super().__init__()

        activation_type = get_activation_type(activation_fn)
        if activation_type not in [activations.SwiGLU, activations.Snake]:
            activation = lambda _: activation_type()  # drop unused arg
        else:
            activation = activation_type

        channels = [latent_channels] + intermediate_channels
        self.speaker_embedding_dim = speaker_embedding_dim
        self.normalize_speaker_embedding = normalize_speaker_embedding
        self.film_scale_bound = film_scale_bound
        self.film_shift_bound = film_shift_bound

        # Speaker embedding projection - project to each intermediate channel size for FiLM conditioning
        # Uses a hidden layer for more expressive conditioning (as in StyleGAN's mapping network)
        if speaker_embedding_dim > 0:
            self.speaker_projections = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(speaker_embedding_dim, speaker_embedding_dim),
                    nn.SiLU(),
                    nn.Linear(speaker_embedding_dim, out_c * 2),  # *2 for scale and shift (FiLM)
                )
                for out_c in intermediate_channels
            ])

        self.channel_upsample = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=(kernel_size[0]//2, kernel_size[1]//2)),
                nn.GroupNorm(out_c // 4, out_c),
                activation(out_c),
            )
            for in_c, out_c, scale_factor, kernel_size in zip(channels[:-1], channels[1:], scale_factors, kernel_sizes)
        ])

        self.final_conv = nn.Conv2d(channels[-1], out_channels, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Initialize FiLM projections with small normal init (std=0.02, as in GPT-2/BERT)
        # Hidden layer uses PyTorch default (Kaiming uniform), output layer uses small normal
        # This provides signal from the start while keeping initial conditioning small
        if self.speaker_embedding_dim > 0:
            for proj in self.speaker_projections:
                if isinstance(proj, nn.Sequential):
                    linear_layers = [m for m in proj if isinstance(m, nn.Linear)]
                    for i, linear in enumerate(linear_layers):
                        if i < len(linear_layers) - 1:
                            # Hidden layers: use Kaiming (PyTorch default) - already initialized
                            # Just zero the bias for cleaner start
                            if linear.bias is not None:
                                nn.init.zeros_(linear.bias)
                        else:
                            # Output layer: small normal init (std=0.02)
                            nn.init.normal_(linear.weight, mean=0.0, std=0.02)
                            if linear.bias is not None:
                                nn.init.zeros_(linear.bias)
                elif isinstance(proj, nn.Linear):
                    nn.init.normal_(proj.weight, mean=0.0, std=0.02)
                    if proj.bias is not None:
                        nn.init.zeros_(proj.bias)

    def forward(self, z, speaker_embedding=None):
        """
        Forward pass through decoder.

        Args:
            z: Latent tensor [B, latent_channels, H, W]
            speaker_embedding: Optional speaker embedding [B, 1, speaker_embedding_dim] or [B, speaker_embedding_dim]
                              Used for FiLM conditioning if speaker_embedding_dim > 0

        Returns:
            Reconstructed output [B, out_channels, H', W']
        """
        # Process speaker embedding if provided
        if speaker_embedding is not None and self.speaker_embedding_dim > 0:
            # Handle shape: [B, 1, dim] -> [B, dim]
            if speaker_embedding.dim() == 3:
                speaker_embedding = speaker_embedding.squeeze(1)

            # L2 normalize speaker embeddings for more stable FiLM conditioning
            # ECAPA-TDNN embeddings can have varying magnitudes (typically L2 norm ~5-15)
            # Normalizing ensures consistent conditioning strength across samples
            if self.normalize_speaker_embedding:
                speaker_embedding = nn.functional.normalize(speaker_embedding, p=2, dim=-1)

        for i, upsample in enumerate(self.channel_upsample):
            z_pre_film = upsample(z)

            # Apply FiLM conditioning from speaker embedding
            if speaker_embedding is not None and self.speaker_embedding_dim > 0:
                # Project speaker embedding to scale and shift
                film_params = self.speaker_projections[i](speaker_embedding)  # [B, out_c * 2]
                out_c = z_pre_film.shape[1]
                scale = film_params[:, :out_c].unsqueeze(-1).unsqueeze(-1)  # [B, out_c, 1, 1]
                shift = film_params[:, out_c:].unsqueeze(-1).unsqueeze(-1)  # [B, out_c, 1, 1]

                # Bound FiLM parameters using tanh to prevent extreme values
                # This prevents scale from approaching -1 (which would zero out activations)
                # and prevents shift from dominating the signal
                if self.film_scale_bound > 0:
                    scale = self.film_scale_bound * nn.functional.tanh(scale)
                if self.film_shift_bound > 0:
                    shift = self.film_shift_bound * nn.functional.tanh(shift)

                z = z_pre_film * (1 + scale) + shift  # FiLM: y = gamma * x + beta
            else:
                z = z_pre_film

        recon_x = self.final_conv(z)
        return recon_x


model_config_lookup = {
    "tiny": lambda latent_channels, speaker_embedding_dim=0, normalize_speaker_embedding=True, film_scale_bound=0.5, film_shift_bound=0.5, **kwargs: VAE(
        encoder=AudioVAEEncoder(
            in_channels=1,
            latent_channels=latent_channels,
            intermediate_channels=[32, 64],
            kernel_sizes=[(3, 5), (3, 5)],
            strides=[(2, 3), (2, 5)],
            activation_fn="silu"
        ),
        decoder=AudioVAEDecoder(
            latent_channels=latent_channels,
            out_channels=1,
            intermediate_channels=[64, 32],
            scale_factors=[(2, 3), (2, 5)],
            kernel_sizes=[(3, 5), (3, 5)],
            activation_fn="silu",
            speaker_embedding_dim=speaker_embedding_dim,
            normalize_speaker_embedding=normalize_speaker_embedding,
            film_scale_bound=film_scale_bound,
            film_shift_bound=film_shift_bound,
        ),
        **kwargs
    ),
    "mini": lambda latent_channels, speaker_embedding_dim=0, normalize_speaker_embedding=True, film_scale_bound=0.5, film_shift_bound=0.5, **kwargs: VAE(
        encoder=AudioVAEEncoder(
            in_channels=1,
            latent_channels=latent_channels,
            intermediate_channels=[64, 128],
            kernel_sizes=[(3, 5), (3, 5)],
            strides=[(2, 3), (2, 5)],
            activation_fn="silu"
        ),
        decoder=AudioVAEDecoder(
            latent_channels=latent_channels,
            out_channels=1,
            intermediate_channels=[128, 64],
            scale_factors=[(2, 3), (2, 5)],
            kernel_sizes=[(3, 5), (3, 5)],
            activation_fn="silu",
            speaker_embedding_dim=speaker_embedding_dim,
            normalize_speaker_embedding=normalize_speaker_embedding,
            film_scale_bound=film_scale_bound,
            film_shift_bound=film_shift_bound,
        ),
        **kwargs
    ),
    "mini_deep": lambda latent_channels, speaker_embedding_dim=0, normalize_speaker_embedding=True, film_scale_bound=0.5, film_shift_bound=0.5, **kwargs: VAE(
        encoder=AudioVAEEncoder(
            in_channels=1,
            latent_channels=latent_channels,
            intermediate_channels=[32, 64, 128],
            kernel_sizes=[(3, 5), (3, 5), (3, 5)],
            strides=[(2, 3), (2, 5), (2, 5)],
            activation_fn="silu"
        ),
        decoder=AudioVAEDecoder(
            latent_channels=latent_channels,
            out_channels=1,
            intermediate_channels=[128, 64, 32],
            scale_factors=[(2, 3), (2, 5), (2, 5)],
            kernel_sizes=[(3, 5), (3, 5), (3, 5)],
            activation_fn="silu",
            speaker_embedding_dim=speaker_embedding_dim,
            normalize_speaker_embedding=normalize_speaker_embedding,
            film_scale_bound=film_scale_bound,
            film_shift_bound=film_shift_bound,
        ),
        **kwargs
    ),
    # ~75x compression config
    # For [1, 80, 1875] mel spec with 8 latent channels:
    #   Height: 80 / 8 = 10, Time: 1875 / 75 = 25 (exact!)
    #   Latent: [8, 10, 25] = 2,000 elements
    #   Compression: 150,000 / 2,000 = 75x
    # Use 12 channels for ~50x, 16 channels for ~37x
    "small": lambda latent_channels, speaker_embedding_dim=0, normalize_speaker_embedding=True, film_scale_bound=0.5, film_shift_bound=0.5, **kwargs: VAE(
        encoder=AudioVAEEncoder(
            in_channels=1,
            latent_channels=latent_channels,
            intermediate_channels=[64, 128, 256],
            kernel_sizes=[(3, 5), (3, 5), (3, 5)],
            strides=[(2, 3), (2, 5), (2, 5)],  # 8x height, 75x time
            activation_fn="silu"
        ),
        decoder=AudioVAEDecoder(
            latent_channels=latent_channels,
            out_channels=1,
            intermediate_channels=[256, 128, 64],
            scale_factors=[(2, 3), (2, 5), (2, 5)],
            kernel_sizes=[(3, 5), (3, 5), (3, 5)],
            activation_fn="silu",
            speaker_embedding_dim=speaker_embedding_dim,
            normalize_speaker_embedding=normalize_speaker_embedding,
            film_scale_bound=film_scale_bound,
            film_shift_bound=film_shift_bound,
        ),
        **kwargs
    ),
    # Higher capacity version for better reconstruction at high compression
    "small_wide": lambda latent_channels, speaker_embedding_dim=0, normalize_speaker_embedding=True, film_scale_bound=0.5, film_shift_bound=0.5, **kwargs: VAE(
        encoder=AudioVAEEncoder(
            in_channels=1,
            latent_channels=latent_channels,
            intermediate_channels=[128, 256, 512],
            kernel_sizes=[(3, 5), (3, 5), (3, 5)],
            strides=[(2, 3), (2, 5), (2, 5)],  # 8x height, 75x time
            activation_fn="silu"
        ),
        decoder=AudioVAEDecoder(
            latent_channels=latent_channels,
            out_channels=1,
            intermediate_channels=[512, 256, 128],
            scale_factors=[(2, 3), (2, 5), (2, 5)],
            kernel_sizes=[(3, 5), (3, 5), (3, 5)],
            activation_fn="silu",
            speaker_embedding_dim=speaker_embedding_dim,
            normalize_speaker_embedding=normalize_speaker_embedding,
            film_scale_bound=film_scale_bound,
            film_shift_bound=film_shift_bound,
        ),
        **kwargs
    ),
}
