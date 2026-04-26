import dataclasses
import json


from dataclasses import dataclass
from typing import Optional


@dataclass
class MelDomainMultiScaleDiscriminatorConfig:
    in_channels: int = 1
    base_channels: int = 32
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


@dataclass
class MelDomainMultiPeriodDiscriminatorConfig:
    in_channels: int = 1
    base_channels: int = 32
    periods: list = None
    use_spectral_norm: bool = True


    def __post_init__(self):
        if self.periods is None:
            self.periods = [2, 3, 5, 7, 11]

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class MelDomainCombinedDiscriminatorConfig:
    # Multi-scale discriminator settings
    multi_scale_config: Optional[MelDomainMultiScaleDiscriminatorConfig] = dataclasses.field(
        default_factory=lambda: MelDomainMultiScaleDiscriminatorConfig(
            base_channels=48,
            n_layers=4,
            n_scales=4,
        )
    )
    multi_period_config: Optional[MelDomainMultiPeriodDiscriminatorConfig] = dataclasses.field(
        default_factory=lambda: MelDomainMultiPeriodDiscriminatorConfig(
            base_channels=48,
            periods=[2, 3, 5, 7, 11],
        )
    )

    def __post_init__(self):
        pass

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)
    
    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)


MEL_COMBINED_DISCRIMINATOR_CONFIGS = {
    "default": MelDomainCombinedDiscriminatorConfig(),
}
