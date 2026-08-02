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
    # Per-layer (freq_stride, time_stride) and kernels. None -> the PatchDiscriminator's
    # waveform-style default (huge time downsample). Pass explicit strides for a mel-aware,
    # axis-preserving discriminator (freq-heavy strides keep TIME; time-heavy keep FREQ).
    strides: Optional[list] = None
    kernel_sizes: Optional[list] = None
    max_channels: int = 512                 # per-layer channel cap
    # Between-scale input pool. Default (2,3) coarsens BOTH axes (-> global). Use (1,2) to
    # vary only TIME across scales (freq branch: (2,1) to vary only FREQ).
    scale_pool: tuple = (2, 3)


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
    time_stride: int = 3          # per-layer time stride (was hardcoded 3); 2 keeps more time
    max_channels: int = 512


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
    # Optional SECOND multi-scale group oriented for the FREQUENCY axis (time-heavy strides,
    # freq preserved), complementary to multi_scale_config's time-resolved view. None = off
    # (the `default` preset has no freq branch, so it is byte-identical to before).
    multi_scale_freq_config: Optional[MelDomainMultiScaleDiscriminatorConfig] = None

    def __post_init__(self):
        pass

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)
    
    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)


MEL_COMBINED_DISCRIMINATOR_CONFIGS = {
    # Full combined disc: multi-scale (temporal resolutions) + multi-period (harmonic
    # periodicities). The multi-period sub-discs are what target harmonic structure, so
    # this is the right choice for de-robotifying the SMG. ~13M params.
    "default": MelDomainCombinedDiscriminatorConfig(),
    # Cheaper multi-scale-ONLY disc (no periods). Faster/lighter but weaker on harmonics.
    "mini_multi_scale": MelDomainCombinedDiscriminatorConfig(
        multi_scale_config=MelDomainMultiScaleDiscriminatorConfig(
            base_channels=32, n_layers=3, n_scales=3,
        ),
        multi_period_config=None,
    ),
    # SHARP: mel-aware, axis-resolved disc for 100-mel @ 93.75Hz. The `default` collapses time
    # to ~3 patches and freq to ~7 bins (a HiFi-GAN waveform disc misapplied to an already-256x-
    # downsampled mel) -> blind to the local structure it must correct. This resolves both axes:
    #   * TIME branch (freq-heavy strides, time kept): freq /16, time /4 -> ~235 time patches at
    #     ~43ms each -> sees phonetic transitions, texture, ~6Hz warble modulation.
    #   * FREQ branch (time-heavy strides, freq kept): freq /4 (25 bins), time /16 -> sees
    #     inharmonic partials / formant errors (the ring-mod "alien").
    #   * PERIOD branch: harmonic periodicity, time stride 3->2 (keeps more time).
    # base_channels 48->64. Much heavier (feature maps no longer collapse) -> if it OOMs, drop
    # n_scales, max_channels, or batch. Returns the standard (outputs, features) contract.
    "sharp": MelDomainCombinedDiscriminatorConfig(
        multi_scale_config=MelDomainMultiScaleDiscriminatorConfig(
            base_channels=64, n_layers=4, n_scales=3, max_channels=512,
            strides=[(2, 2), (2, 2), (2, 1), (2, 1), (1, 1)],   # 4 layers (freq /16, time /4) + final
            kernel_sizes=[(3, 7), (3, 7), (3, 7), (3, 7), (3, 7)],
            scale_pool=(1, 2),                                  # vary TIME across scales
        ),
        multi_scale_freq_config=MelDomainMultiScaleDiscriminatorConfig(
            base_channels=64, n_layers=4, n_scales=2, max_channels=512,
            strides=[(2, 2), (1, 2), (2, 2), (1, 2), (1, 1)],   # 4 layers (freq /4, time /16) + final
            kernel_sizes=[(7, 3), (7, 3), (7, 3), (7, 3), (7, 3)],
            scale_pool=(2, 1),                                  # vary FREQ across scales
        ),
        multi_period_config=MelDomainMultiPeriodDiscriminatorConfig(
            base_channels=64, periods=[2, 3, 5, 7, 11], time_stride=2, max_channels=512,
        ),
    ),
}
