from config.audio.generator import (
    AUDIO_CODA_CONFIGS,
    AudioCodaConfig,
)
from model.voice.generator import VoiceCodaAndSMGWithLoss

class AudioCodaWithLoss(VoiceCodaAndSMGWithLoss):
    """Placeholder: currently delegates to the voice SMG-wrapping implementation.

    Will be replaced with an audio-specific architecture (non-SMG decoder) in a
    future refactor. See VoiceCodaAndSMGWithLoss for the current behavior.
    """

    def __init__(self, prefix: str, config: AudioCodaConfig):
        # AudioCodaConfig is a structural copy of VoiceCodaAndSMGConfig, so the
        # parent's init accepts its fields directly.
        super().__init__(prefix, config)

    @classmethod
    def from_config(cls, prefix: str, config_name: str, **overrides) -> "AudioCodaWithLoss":
        if config_name not in AUDIO_CODA_CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(AUDIO_CODA_CONFIGS.keys())}")

        config = AUDIO_CODA_CONFIGS[config_name]
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(overrides)
        config = AudioCodaConfig(**config_dict)

        return cls(prefix, config)
