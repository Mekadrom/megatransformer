import torch
import torch.nn as nn
import torch.nn.functional as F

from config.image.vae.vae import IMAGE_DECODER_CONFIGS, IMAGE_ENCODER_CONFIGS, IMAGE_VAE_CONFIGS, ImageVAEConfig, ImageVAEDecoderConfig, ImageVAEEncoderConfig
from model import activations
from model.activations import get_activation_type
from model.image.vae.criteria import DINOv2PerceptualLoss, LPIPSLoss, VGGPerceptualLoss


class ImageVAEEncoder(nn.Module):
    """
    Image VAE encoder with optional bottleneck attention.

    Args:
        in_channels: Number of input channels (3 for RGB)
        latent_channels: Number of latent channels
        intermediate_channels: List of channel counts for each downsampling stage
        activation_fn: Activation function name
        use_attention: Whether to use bottleneck attention
        attention_heads: Number of attention heads (if use_attention=True)
    """
    def __init__(self, config: ImageVAEEncoderConfig):
        super().__init__()

        self.config = config

        activation_type = get_activation_type(config.activation)
        if activation_type not in [activations.SwiGLU, activations.Snake]:
            activation = lambda _: activation_type()  # drop unused arg
        else:
            activation = activation_type

        channels = [config.in_channels] + config.intermediate_channels

        self.channel_upsample = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(out_c // 4, out_c),
                activation(out_c)
            )
            for in_c, out_c in zip(channels[:-1], channels[1:])
        ])

        self.fc_mu = nn.Conv2d(channels[-1], config.latent_channels, kernel_size=3, padding=1)
        self.fc_logvar = nn.Conv2d(channels[-1], config.latent_channels, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if module in [self.fc_mu, self.fc_logvar]:
                    continue  # Skip mu/logvar, handled below
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

    @classmethod
    def from_config(cls, config_name: str, **overrides) -> "ImageVAEEncoder":
        """
        Create model from predefined config with optional overrides.

        Args:
            config_name: One of predefined configs
            **overrides: Override any config parameter

        Example:
            model = ImageVAEEncoder.from_config("small", latent_dim=6)
        """
        if config_name not in IMAGE_ENCODER_CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(IMAGE_ENCODER_CONFIGS.keys())}")

        config = IMAGE_ENCODER_CONFIGS[config_name]
        # Apply overrides
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(overrides)
        config = ImageVAEEncoderConfig(**config_dict)

        return cls(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] input image

        Returns:
            mu: [B, latent_channels, H', W'] latent mean
            logvar: [B, latent_channels, H', W'] latent log variance
        """
        for upsample in self.channel_upsample:
            x = upsample(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        logvar = torch.clamp(logvar, min=-10.0, max=self.config.logvar_clamp_max)

        return mu, logvar


class ImageVAEDecoder(nn.Module):
    """
    Image VAE decoder with optional bottleneck attention.

    Args:
        latent_channels: Number of latent channels
        out_channels: Number of output channels (3 for RGB)
        intermediate_channels: List of channel counts for each upsampling stage
        activation_fn: Activation function name
    """
    def __init__(self, config: ImageVAEDecoderConfig):
        super().__init__()
        
        self.config = config

        activation_type = get_activation_type(config.activation)
        if activation_type not in [activations.SwiGLU, activations.Snake]:
            activation = lambda _: activation_type()  # drop unused arg
        else:
            activation = activation_type

        channels = [config.latent_channels] + config.intermediate_channels

        self.channel_upsample = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.GroupNorm(out_c // 4, out_c),
                activation(out_c),
            )
            for in_c, out_c in zip(channels[:-1], channels[1:])
        ])

        self.final_conv = nn.Conv2d(channels[-1], config.out_channels, kernel_size=3, padding=1)
        self.final_act: nn.Module
        if config.use_final_tanh:
            self.final_act = nn.Tanh()
        else:
            self.final_act = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @classmethod
    def from_config(cls, config_name: str, **overrides) -> "ImageVAEDecoder":
        """
        Create model from predefined config with optional overrides.

        Args:
            config_name: One of predefined configs
            **overrides: Override any config parameter

        Example:
            model = ImageVAEDecoder.from_config("small", latent_dim=6)
        """
        if config_name not in IMAGE_DECODER_CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(IMAGE_DECODER_CONFIGS.keys())}")

        config = IMAGE_DECODER_CONFIGS[config_name]
        # Apply overrides
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(overrides)
        config = ImageVAEDecoderConfig(**config_dict)

        return cls(config)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, latent_channels, H, W] latent tensor

        Returns:
            recon_x: [B, out_channels, H', W'] reconstructed image
        """
        # speaker_embedding is ignored for image VAE (only used for audio VAE)

        for upsample in self.channel_upsample:
            z = upsample(z)

        recon_x = self.final_conv(z)
        recon_x = self.final_act(recon_x)

        return recon_x


class ImageVAE(nn.Module):
    def __init__(
        self,
        config: ImageVAEConfig,
        encoder: ImageVAEEncoder,
        decoder: ImageVAEDecoder,
    ):
        super().__init__()

        self.config = config
        self.encoder = encoder
        self.decoder = decoder

        if config.perceptual_loss_type == "vgg":
            self.perceptual_loss = VGGPerceptualLoss()
        elif config.perceptual_loss_type == "lpips":
            self.perceptual_loss = LPIPSLoss(net=config.lpips_net)
        else:
            self.perceptual_loss = None

        # DINO perceptual loss (semantic features)
        self.dino_loss = None
        if config.dino_loss_weight > 0:
            self.dino_loss = DINOv2PerceptualLoss(model_name=config.dino_model)

    @classmethod
    def from_config(cls, config_name: str, **overrides) -> "ImageVAE":
        """
        Create model from predefined config with optional overrides.

        Args:
            config_name: One of predefined configs
            **overrides: Override any config parameter

        Example:
            model = ImageVAE.from_config("small", encoder_config=custom_encoder_config, decoder_config=custom_decoder_config)
        """
        if config_name not in IMAGE_VAE_CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(IMAGE_VAE_CONFIGS.keys())}")

        config = IMAGE_VAE_CONFIGS[config_name]
        # Apply overrides
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(overrides)
        config = ImageVAEConfig(**config_dict)

        return cls(config)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space.

        Returns:
            mu: Latent mean
            logvar: Latent log variance
            learned_speaker_emb (optional): [B, learned_speaker_dim] if encoder.learn_speaker_embedding=True
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        image: torch.Tensor,
        kl_weight_multiplier: float = 1.0,
    ):
        """
        Forward pass through VAE.

        Args:
            x: Input tensor [B, C, H, W] or [B, C, H, T] for audio
            image: Alternative name for x (for compatibility with data collators)
            mask: Optional mask tensor [B, T] where 1 = valid, 0 = padding.
                  If provided, reconstruction loss is only computed on valid regions.
                  The mask is in the time dimension (last dim of x).
            kl_weight_multiplier: Multiplier for KL divergence weight (for KL annealing).
                                  Default 1.0 means use full kl_divergence_loss_weight.
                                  Set to 0.0 at start of training and anneal to 1.0.

        Returns:
            recon_x: Reconstructed input
            mu: Latent mean
            logvar: Latent log variance
            losses: Dictionary of loss components (includes film_stats, learned_speaker_embedding if applicable)
        """
        # Support both 'x' and 'image' as input parameter names
        if x is None and image is not None:
            x = image
        elif x is None and image is None:
            raise ValueError("Either 'x' or 'image' must be provided")

        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        recon_x = self.decode(z)

        # Compute KL divergence with optional free bits
        # Per-element KL: [B, C, H, W] for images, [B, C, T] for audio
        kl_per_element = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        if self.config.free_bits > 0:
            # Free bits: apply minimum KL per channel to prevent posterior collapse
            # Sum over spatial dims, mean over batch -> per-channel KL: [C]
            spatial_dims = list(range(2, mu.dim()))  # [2, 3] for 4D, [2] for 3D
            kl_per_channel = kl_per_element.sum(dim=spatial_dims).mean(dim=0)  # [C]

            # Clamp each channel's KL to at least free_bits
            kl_per_channel = torch.clamp(kl_per_channel, min=self.config.free_bits)

            # Sum over channels for total KL
            kl_divergence = kl_per_channel.sum()
        else:
            # Original behavior: sum over all latent dims, mean over batch
            latent_dims = list(range(1, mu.dim()))  # [1, 2, 3] for 4D, [1, 2] for 3D
            kl_divergence = kl_per_element.sum(dim=latent_dims).mean()

        # standard recon losses
        mse_loss = F.mse_loss(recon_x, x, reduction='mean')
        l1_loss = F.l1_loss(recon_x, x, reduction='mean')

        recon_loss = self.config.mse_loss_weight * mse_loss + self.config.l1_loss_weight * l1_loss

        # perceptual loss
        perceptual_loss = torch.tensor(0.0, device=x.device)
        if self.perceptual_loss is not None:
            perceptual_loss = self.perceptual_loss(recon_x, x)

        # DINO perceptual loss (semantic features, complementary to VGG/LPIPS)
        dino_loss = torch.tensor(0.0, device=x.device)
        if self.dino_loss is not None:
            dino_loss = self.dino_loss(recon_x, x)

        # Apply KL weight multiplier for KL annealing
        effective_kl_weight = self.config.kl_divergence_loss_weight * kl_weight_multiplier

        total_loss = (
            self.config.recon_loss_weight * recon_loss
            + self.config.perceptual_loss_weight * perceptual_loss
            + self.config.dino_loss_weight * dino_loss
            + effective_kl_weight * kl_divergence
        )

        losses = {
            "total_loss": total_loss,
            "kl_divergence": kl_divergence,
            "recon_loss": recon_loss,
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
            "perceptual_loss": perceptual_loss,
            "dino_loss": dino_loss,
            "kl_weight_multiplier": torch.tensor(kl_weight_multiplier, device=x.device),
        }

        return recon_x, mu, logvar, losses
