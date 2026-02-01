import dataclasses
import json


from dataclasses import dataclass


@dataclass
class MultiScalePatchDiscriminatorConfig:
    in_channels: int = 3
    base_channels: int = 64
    n_layers: int = 3
    n_scales: int = 3
    use_spectral_norm: bool = True


    def __post_init__(self):
        pass

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)
    

MULTI_SCALE_PATCH_DISCRIMINATOR_CONFIGS = {
    "default": MultiScalePatchDiscriminatorConfig(),
    "medium": MultiScalePatchDiscriminatorConfig(
        in_channels=3,
        base_channels=96,
        n_layers=3,
        n_scales=3,
        use_spectral_norm=False,
    ),
}
