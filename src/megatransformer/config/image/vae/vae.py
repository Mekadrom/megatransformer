

import dataclasses
import json

from dataclasses import dataclass
from typing import Literal


@dataclass
class ImageVAEEncoderConfig:
    in_channels: int = 3
    latent_channels: int = 4
    intermediate_channels: list = dataclasses.field(default_factory=lambda: [32, 64, 128])
    activation: str = "silu"
    logvar_clamp_max: float = 4.0


    def __post_init__(self):
        pass

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)


IMAGE_ENCODER_CONFIGS = {
    "default": ImageVAEEncoderConfig(),
    "medium": ImageVAEEncoderConfig(
        in_channels=3,
        latent_channels=4,
        intermediate_channels=[320, 480, 640],
        activation="silu",
        logvar_clamp_max=4.0,
    ),
}


@dataclass
class ImageVAEDecoderConfig:
    latent_channels: int = 4
    out_channels: int = 3
    intermediate_channels: list = dataclasses.field(default_factory=lambda: [128, 64, 32])
    activation: str = "silu"
    use_final_tanh: bool = False


    def __post_init__(self):
        pass

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)


IMAGE_DECODER_CONFIGS = {
    "default": ImageVAEDecoderConfig(),
    "medium": ImageVAEDecoderConfig(
        latent_channels=4,
        out_channels=3,
        intermediate_channels=[640, 480, 320],
        activation="silu",
        use_final_tanh=True,
    ),
}


@dataclass
class ImageVAEConfig:
    encoder_config: ImageVAEEncoderConfig = dataclasses.field(default_factory=ImageVAEEncoderConfig)
    decoder_config: ImageVAEDecoderConfig = dataclasses.field(default_factory=ImageVAEDecoderConfig)

    perceptual_loss_type: Literal["vgg", "lpips", "none"] = "none"
    lpips_net: Literal["alex", "vgg"] = "vgg"

    recon_loss_weight: float = 1.0
    mse_loss_weight: float = 1.0
    l1_loss_weight: float = 0.0
    perceptual_loss_weight: float = 0.1
    kl_divergence_loss_weight: float = 0.01

    free_bits: float = 0.0  # Minimum KL per channel (0 = disabled)
    
    # DINO perceptual loss (semantic features, complementary to VGG/LPIPS)
    dino_loss_weight: float = 0.0  # 0 = disabled
    dino_model: str = "dinov2_vits14"  # dinov2_vits14, dinov2_vitb14, dinov2_vitl14


    def __post_init__(self):
        if self.perceptual_loss_type == "none":
            self.perceptual_loss_weight = 0.0
        if self.perceptual_loss_weight == 0.0:
            self.perceptual_loss_type = "none"

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)


IMAGE_VAE_CONFIGS = {
    "default": ImageVAEConfig(),
    "medium": ImageVAEConfig(
        encoder_config=IMAGE_ENCODER_CONFIGS["medium"],
        decoder_config=IMAGE_DECODER_CONFIGS["medium"],
    ),
}
