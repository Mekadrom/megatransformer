import dataclasses
import json


from dataclasses import dataclass
from typing import Optional


@dataclass
class WaveformDomainMultiPeriodDiscriminatorConfig:
    periods: list[int] = [2, 3, 5, 7, 11, 13, 17]
    base_channels: Optional[list[int]] = None


@dataclass
class WaveformDomainMultiScaleDiscriminatorConfig:
    base_channels: Optional[list[int]] = None


@dataclass
class WaveformDomainMultiResolutionDiscriminatorConfig:
    resolutions: list[tuple[int, int, int]] = [
        (1024, 256),
        (2048, 512),
        (512, 128),
    ]
    base_channels: list[int] = [16, 32, 64, 128, 256]


@dataclass
class WaveformDomainDiscriminatorConfig:
    mpd_config: WaveformDomainMultiPeriodDiscriminatorConfig = dataclasses.field(
        default_factory=WaveformDomainMultiPeriodDiscriminatorConfig
    )
    msd_config: WaveformDomainMultiScaleDiscriminatorConfig = dataclasses.field(
        default_factory=WaveformDomainMultiScaleDiscriminatorConfig
    )
    mrsd_config: WaveformDomainMultiResolutionDiscriminatorConfig = dataclasses.field(
        default_factory=WaveformDomainMultiResolutionDiscriminatorConfig
    )

    def __post_init__(self):
        pass

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)
    
    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)
    

DISCRIMINATOR_CONFIGS = {
    "default": WaveformDomainDiscriminatorConfig(),
    "small": WaveformDomainDiscriminatorConfig(
        WaveformDomainMultiPeriodDiscriminatorConfig(
            periods=[2, 3, 5, 7, 11, 13, 17],
        ),
        WaveformDomainMultiScaleDiscriminatorConfig(
            base_channels=[64] * 4,
        ),
        WaveformDomainMultiResolutionDiscriminatorConfig(
            resolutions=[
                (2048, 512),
                (1024, 256),
                (512, 128),
                (256, 64),
            ],
            base_channels=[8, 16, 32, 48, 64, 96],
        ),
    )
}
