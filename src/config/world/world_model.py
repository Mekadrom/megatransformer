import dataclasses
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

from config.audio.feature_extractor import AudioVAEPreludeFeatureExtractorConfig
from config.audio.generator import AudioCodaAndVAEConfig
from config.common import MegaTransformerBlockConfig
from config.image.feature_extractor import ImageVAEPreludeFeatureExtractorConfig
from config.image.generator import ImageCodaAndVAEConfig
from config.text.feature_extractor import TextFeatureExtractorConfig
from config.text.generator import TextCodaClassifierConfig
from utils.constants import (
    AUDIO_PLACEHOLDER_TOKEN_ID,
    VOICE_PLACEHOLDER_TOKEN_ID,
    IMAGE_PLACEHOLDER_TOKEN_ID,
)


@dataclass
class TokenInterleaverConfig:
    """Configuration for TokenInterleaver with placeholder token IDs.

    These token IDs should match your tokenizer's vocabulary for the special
    placeholder tokens that mark where media examples should be inserted.

    For example, if your tokenizer has:
        - <audio> at token ID 32000
        - <voice> at token ID 32001
        - <image> at token ID 32002

    Then configure accordingly. The interleaver will scan for these tokens
    in the input sequence and replace them with the corresponding media embeddings.
    """
    audio_placeholder_token_id: Optional[int] = AUDIO_PLACEHOLDER_TOKEN_ID
    voice_placeholder_token_id: Optional[int] = VOICE_PLACEHOLDER_TOKEN_ID
    image_placeholder_token_id: Optional[int] = IMAGE_PLACEHOLDER_TOKEN_ID


    def __post_init__(self):
        pass

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class MegaTransformerRecurrentConfig:
    """Configuration for the recurrent (thought vector) block.

    The recurrent block operates at 2*d_model internally to concatenate
    input embeddings with thought state, then projects back to d_model.
    """
    block_config: MegaTransformerBlockConfig = dataclasses.field(
        default_factory=lambda: MegaTransformerBlockConfig(d_model=1024, d_inner=4096)
    )
    mean_thinking_steps: int = 32
    backprop_depth: int = 8
    thought_initialization_method: str = "like-init"
    exit_criteria: str = "kl_divergence"
    exit_criteria_threshold: float = 1e-4
    lockstep_n: bool = False
    lockstep_k: bool = False

    
    def __post_init__(self):
        pass

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class MegaTransformerWorldModelConfig:
    """
    Configuration for the Megatransformer world model, integrating audio, image, and text modalities.

    include_modes controls which modality-specific preludes and codas are instantiated.
    Text is always included. Omitted modalities save memory and parameters.
    """
    # Which modalities to instantiate (text is always included)
    include_modes: List[str] = dataclasses.field(
        default_factory=lambda: ["text", "audio", "voice", "image"]
    )

    # Feature extractor configs
    text_feature_config: TextFeatureExtractorConfig = dataclasses.field(
        default_factory=TextFeatureExtractorConfig
    )
    audio_prelude_config: AudioVAEPreludeFeatureExtractorConfig = dataclasses.field(
        default_factory=AudioVAEPreludeFeatureExtractorConfig
    )
    voice_prelude_config: AudioVAEPreludeFeatureExtractorConfig = dataclasses.field(
        default_factory=AudioVAEPreludeFeatureExtractorConfig
    )
    image_prelude_config: ImageVAEPreludeFeatureExtractorConfig = dataclasses.field(
        default_factory=ImageVAEPreludeFeatureExtractorConfig
    )
    # Token interleaver config
    token_interleaver_config: TokenInterleaverConfig = dataclasses.field(
        default_factory=TokenInterleaverConfig
    )
    # Main transformer config
    recurrent_block_config: MegaTransformerRecurrentConfig = dataclasses.field(
        default_factory=MegaTransformerRecurrentConfig
    )
    # Coda/generator configs
    text_coda_config: TextCodaClassifierConfig = dataclasses.field(
        default_factory=TextCodaClassifierConfig
    )
    audio_coda_config: AudioCodaAndVAEConfig = dataclasses.field(
        default_factory=AudioCodaAndVAEConfig
    )
    voice_coda_config: AudioCodaAndVAEConfig = dataclasses.field(
        default_factory=AudioCodaAndVAEConfig
    )
    image_coda_config: ImageCodaAndVAEConfig = dataclasses.field(
        default_factory=ImageCodaAndVAEConfig
    )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# Slim prelude/coda block: lightweight adapter — just enough to translate
# between modality-specific representations and the shared d_model space.
_slim_block = lambda: MegaTransformerBlockConfig(
    d_model=512,
    n_heads=4,
    d_queries=64,
    d_values=64,
    n_query_groups=2,
    d_inner=1024,
)

WORLD_MODEL_CONFIGS = {
    # Original config: equal-weight preludes/codas/recurrent.
    "default": MegaTransformerWorldModelConfig(),

    # Researcher config: recurrent block dominates (~40-45% of params).
    # Preludes and codas use slim single-block transformers.
    # The recurrent block gets a wide FFN (12288) and 16 GQA heads,
    # since it iterates many times per token — every parameter is reused.
    "researcher": MegaTransformerWorldModelConfig(
        # Slim preludes
        audio_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            prelude_config=_slim_block(),
        ),
        voice_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            prelude_config=_slim_block(),
        ),
        image_prelude_config=ImageVAEPreludeFeatureExtractorConfig(
            prelude_config=_slim_block(),
        ),
        # Heavy recurrent block: wide FFN, more heads, GQA
        recurrent_block_config=MegaTransformerRecurrentConfig(
            block_config=MegaTransformerBlockConfig(
                d_model=1024,      # 2 * base d_model (concat input + thought)
                n_heads=16,
                d_queries=64,
                d_values=64,
                n_query_groups=4,  # GQA: 4 KV groups for 16 heads
                d_inner=12288,     # wide FFN — bulk of the parameters
            ),
        ),
        # Slim codas
        text_coda_config=TextCodaClassifierConfig(
            coda_config=_slim_block(),
        ),
        audio_coda_config=AudioCodaAndVAEConfig(
            coda_config=_slim_block(),
            output_mode="conv_refine",
        ),
        voice_coda_config=AudioCodaAndVAEConfig(
            coda_config=_slim_block(),
            output_mode="conv_refine",
        ),
        image_coda_config=ImageCodaAndVAEConfig(
            coda_config=_slim_block(),
            unpatchify_mode="pixel_shuffle",
        ),
    ),
}
