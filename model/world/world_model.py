from typing import Optional
import torch
import torch.nn as nn

from transformers import PretrainedConfig

from model.audio.feature_extractors import AudioVAEPreludeFeatureExtractor, AudioVAEPreludeFeatureExtractorConfig
from model.audio.generators import AudioCodaAndVAEConfig, AudioCodaAndVAEWithLoss
from model.audio.vae import AudioVAEEncoder, AudioVAEDecoder
from model.image.feature_extractors import ImageVAEPreludeFeatureExtractor, ImageVAEPreludeFeatureExtractorConfig
from model.image.generators import ImageCodaAndVAEConfig, ImageCodaAndVAEWithLoss
from model.image.vae import ImageVAEEncoder, ImageVAEDecoder
from model.text.feature_extractors import TextFeatureExtractor, TextFeatureExtractorConfig
from model.text.generators import TextCodaClassifierConfig, TextCodaClassifierWithLoss
from model.world.recurrent import MegatransformerRecurrentBlock, MegatransformerRecurrentConfig
from model.world.token_alignment import TokenInterleaver, TokenInterleaverConfig, TokenUninterleaver
from utils import configuration
from utils.configuration import TransformerBlockConfig


class MegatransformerWorldModelConfig(PretrainedConfig):
    """
    Configuration for the Megatransformer world model, integrating audio, image, and text modalities.
    """

    def __init__(
        self,
        # Feature extractor configs
        text_feature_config: TextFeatureExtractorConfig = None,
        audio_prelude_config: AudioVAEPreludeFeatureExtractorConfig = None,
        voice_prelude_config: AudioVAEPreludeFeatureExtractorConfig = None,
        image_prelude_config: ImageVAEPreludeFeatureExtractorConfig = None,
        # Token interleaver config
        token_interleaver_config: TokenInterleaverConfig = None,
        # Main transformer config
        recurrent_block_config: MegatransformerRecurrentConfig = None,
        # Coda/generator configs
        text_coda_config: TextCodaClassifierConfig = None,
        audio_coda_config: AudioCodaAndVAEConfig = None,
        voice_coda_config: AudioCodaAndVAEConfig = None,
        image_coda_config: ImageCodaAndVAEConfig = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.text_feature_config = text_feature_config
        self.audio_prelude_config = audio_prelude_config
        self.voice_prelude_config = voice_prelude_config
        self.image_prelude_config = image_prelude_config
        self.token_interleaver_config = token_interleaver_config
        self.recurrent_block_config = recurrent_block_config
        self.text_coda_config = text_coda_config
        self.audio_coda_config = audio_coda_config
        self.voice_coda_config = voice_coda_config
        self.image_coda_config = image_coda_config


class MegatransformerWorldModel(nn.Module):
    """
    Multimodal autoregressive world model combining text, audio, voice, and image.

    Supports two modes:
    1. Training: Uses precomputed VAE latents, computes loss in latent space
    2. Inference: Uses live VAE encoding/decoding for real inputs/outputs

    Architecture:
    - Modality-specific feature extractors (with optional VAE encoders)
    - Token interleaving based on placeholder positions in text
    - Recurrent transformer block with thought vector mechanism
    - Token uninterleaving back to modality-specific sequences
    - Modality-specific codas (with optional VAE decoders)
    """

    def __init__(self, config: MegatransformerWorldModelConfig):
        super(MegatransformerWorldModel, self).__init__()

        self.config = config

        # Feature extractors (VAE encoders optional - for inference only)
        self.text_feature_extractor = TextFeatureExtractor(config.text_feature_config)
        self.audio_feature_extractor = AudioVAEPreludeFeatureExtractor(config.audio_prelude_config)
        self.voice_feature_extractor = AudioVAEPreludeFeatureExtractor(config.voice_prelude_config)
        self.image_feature_extractor = ImageVAEPreludeFeatureExtractor(config.image_prelude_config)

        # Token alignment
        self.token_interleaver = TokenInterleaver(config.token_interleaver_config)
        self.token_uninterleaver = TokenUninterleaver()

        # Main transformer
        self.recurrent_block = MegatransformerRecurrentBlock(config.recurrent_block_config)

        # Generators/codas (VAE decoders optional - for inference only)
        self.text_generator = TextCodaClassifierWithLoss(config.text_coda_config)
        self.audio_generator = AudioCodaAndVAEWithLoss("audio", config.audio_coda_config)
        self.voice_generator = AudioCodaAndVAEWithLoss("voice", config.voice_coda_config)
        self.image_generator = ImageCodaAndVAEWithLoss(config.image_coda_config)

    # -------------------------------------------------------------------------
    # VAE Management (for inference mode)
    # -------------------------------------------------------------------------

    def load_audio_vae(self, encoder: AudioVAEEncoder, decoder: AudioVAEDecoder):
        """Load audio VAE encoder/decoder for live encoding/decoding."""
        self.audio_feature_extractor.vae_encoder = encoder
        self.audio_generator.vae_decoder = decoder

    def load_voice_vae(self, encoder: AudioVAEEncoder, decoder: AudioVAEDecoder):
        """Load voice VAE encoder/decoder for live encoding/decoding."""
        self.voice_feature_extractor.vae_encoder = encoder
        self.voice_generator.vae_decoder = decoder

    def load_image_vae(self, encoder: ImageVAEEncoder, decoder: ImageVAEDecoder):
        """Load image VAE encoder/decoder for live encoding/decoding."""
        self.image_feature_extractor.vae_encoder = encoder
        self.image_generator.vae_decoder = decoder

    def unload_vaes(self):
        """Unload all VAE encoders/decoders to free memory."""
        self.audio_feature_extractor.vae_encoder = None
        self.audio_generator.vae_decoder = None
        self.voice_feature_extractor.vae_encoder = None
        self.voice_generator.vae_decoder = None
        self.image_feature_extractor.vae_encoder = None
        self.image_generator.vae_decoder = None

    # -------------------------------------------------------------------------
    # Forward Pass
    # -------------------------------------------------------------------------

    def forward(
        self,
        text_input_ids: torch.Tensor,
        # Audio inputs (either raw mel specs or precomputed latents)
        audio_inputs: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None,
        audio_latent_labels: Optional[torch.Tensor] = None,
        # Voice inputs
        voice_inputs: Optional[torch.Tensor] = None,
        voice_lengths: Optional[torch.Tensor] = None,
        voice_latent_labels: Optional[torch.Tensor] = None,
        # Image inputs
        image_inputs: Optional[torch.Tensor] = None,
        image_latent_labels: Optional[torch.Tensor] = None,
        # Text targets
        text_targets: Optional[torch.Tensor] = None,
        # Mode flags
        precomputed_latents: bool = True,
        decode_outputs: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the world model.

        Args:
            text_input_ids: Token IDs for text, shape (batch, text_seq_len).
                Contains placeholder tokens for media positions.

            audio_inputs: Audio input, shape depends on mode:
                - precomputed_latents=True: (batch, n_audio, latent_channels, mel_bins, timesteps)
                - precomputed_latents=False: (batch, n_audio, mel_bins, timesteps) raw mel specs
            audio_lengths: Actual lengths per audio example, shape (batch, n_audio).
            audio_latent_labels: Target latents for audio loss, same shape as latent inputs.

            voice_inputs: Same format as audio_inputs.
            voice_lengths: Same format as audio_lengths.
            voice_latent_labels: Target latents for voice loss.

            image_inputs: Image input, shape depends on mode:
                - precomputed_latents=True: (batch, n_images, latent_channels, latent_h, latent_w)
                - precomputed_latents=False: (batch, n_images, 3, image_h, image_w) raw images
            image_latent_labels: Target latents for image loss.

            text_targets: Target token IDs for text loss (usually shifted input_ids).

            precomputed_latents: If True, media inputs are VAE latents. If False, raw inputs.
            decode_outputs: If True, decode latent predictions to mel specs/images (requires VAEs).

        Returns:
            Dictionary containing predictions and losses for each modality.
        """
        # -----------------------------------------------------------------
        # Feature Extraction
        # -----------------------------------------------------------------
        text_hidden_states = self.text_feature_extractor(text_input_ids)

        audio_hidden_states = None
        if audio_inputs is not None:
            # audio_inputs: (batch, n_audio, ...) -> need to process each example
            batch_size, n_audio = audio_inputs.shape[:2]
            # Flatten batch and n_audio for processing
            audio_flat = audio_inputs.view(batch_size * n_audio, *audio_inputs.shape[2:])
            audio_hidden_flat = self.audio_feature_extractor(
                audio_flat, precomputed_latents=precomputed_latents
            )
            # Reshape back: (batch * n_audio, seq, d_model) -> (batch, n_audio, seq, d_model)
            seq_len, d_model = audio_hidden_flat.shape[1], audio_hidden_flat.shape[2]
            audio_hidden_states = audio_hidden_flat.view(batch_size, n_audio, seq_len, d_model)

        voice_hidden_states = None
        if voice_inputs is not None:
            batch_size, n_voice = voice_inputs.shape[:2]
            voice_flat = voice_inputs.view(batch_size * n_voice, *voice_inputs.shape[2:])
            voice_hidden_flat = self.voice_feature_extractor(
                voice_flat, precomputed_latents=precomputed_latents
            )
            seq_len, d_model = voice_hidden_flat.shape[1], voice_hidden_flat.shape[2]
            voice_hidden_states = voice_hidden_flat.view(batch_size, n_voice, seq_len, d_model)

        image_hidden_states = None
        if image_inputs is not None:
            batch_size, n_images = image_inputs.shape[:2]
            image_flat = image_inputs.view(batch_size * n_images, *image_inputs.shape[2:])
            image_hidden_flat = self.image_feature_extractor(
                image_flat, precomputed_latents=precomputed_latents
            )
            seq_len, d_model = image_hidden_flat.shape[1], image_hidden_flat.shape[2]
            image_hidden_states = image_hidden_flat.view(batch_size, n_images, seq_len, d_model)

        # -----------------------------------------------------------------
        # Token Interleaving
        # -----------------------------------------------------------------
        interleaved_tokens, attn_mask, modality_map = self.token_interleaver(
            text_hidden_states=text_hidden_states,
            text_token_ids=text_input_ids,
            audio_hidden_states=audio_hidden_states,
            audio_lengths=audio_lengths,
            voice_hidden_states=voice_hidden_states,
            voice_lengths=voice_lengths,
            image_hidden_states=image_hidden_states,
        )

        # -----------------------------------------------------------------
        # Main Transformer (Recurrent Block)
        # -----------------------------------------------------------------
        recurrent_output = self.recurrent_block(
            interleaved_tokens,
            attention_mask=attn_mask  # True for attend, False for padding
        )

        # -----------------------------------------------------------------
        # Token Uninterleaving
        # -----------------------------------------------------------------
        uninterleaved = self.token_uninterleaver(recurrent_output, modality_map)

        text_batch = uninterleaved["text"]
        audio_batch = uninterleaved["audio"]
        voice_batch = uninterleaved["voice"]
        image_batch = uninterleaved["image"]

        # -----------------------------------------------------------------
        # Generators/Codas
        # -----------------------------------------------------------------
        outputs = {}

        # Text
        if text_batch is not None:
            text_outputs = self.text_generator(text_batch, targets=text_targets)
            outputs.update(text_outputs)

        # Audio
        if audio_batch is not None:
            audio_outputs = self.audio_generator(
                audio_batch,
                latent_labels=audio_latent_labels,
                lengths=uninterleaved["audio_lengths"],
                decode_to_mel=decode_outputs,
            )
            outputs.update(audio_outputs)

        # Voice
        if voice_batch is not None:
            voice_outputs = self.voice_generator(
                voice_batch,
                latent_labels=voice_latent_labels,
                lengths=uninterleaved["voice_lengths"],
                decode_to_mel=decode_outputs,
            )
            outputs.update(voice_outputs)

        # Image
        if image_batch is not None:
            image_outputs = self.image_generator(
                image_batch,
                latent_labels=image_latent_labels,
                decode_to_image=decode_outputs,
            )
            outputs.update(image_outputs)

        return outputs


def get_wm_config(
    d_model,
    vocab_size,
    max_position_embeddings,
    prelude_n_heads,
    prelude_n_layers,
    prelude_d_queries,
    prelude_d_values,
    recurrent_n_heads,
    recurrent_n_layers,
    recurrent_d_queries,
    recurrent_d_values,
    coda_n_heads,
    coda_n_layers,
    coda_d_queries,
    coda_d_values,
    audio_placeholder_token_id: int,
    voice_placeholder_token_id: int,
    image_placeholder_token_id: int,
) -> MegatransformerWorldModelConfig:
    small_text_feature_config = TextFeatureExtractorConfig(
        d_model=d_model,
        vocab_size=vocab_size
    )

    audio_config = configuration.AudioConfig()
    voice_config = configuration.AudioConfig()
    image_config = configuration.ImageConfig()

    prelude_transformer_config = TransformerBlockConfig(
        d_model=d_model,
        n_heads=prelude_n_heads,
        d_queries=prelude_d_queries,
        d_values=prelude_d_values,
        n_query_groups=prelude_n_heads,
        d_inner=d_model * 4,
        n_layers=prelude_n_layers,
        max_position_embeddings=max_position_embeddings,
    )

    audio_prelude_config = AudioVAEPreludeFeatureExtractorConfig(
        prelude_config=prelude_transformer_config,
        audio_config=audio_config,
    )

    voice_prelude_config = AudioVAEPreludeFeatureExtractorConfig(
        prelude_config=prelude_transformer_config,
        audio_config=voice_config,
    )

    image_prelude_config = ImageVAEPreludeFeatureExtractorConfig(
        prelude_config=prelude_transformer_config,
        image_config=image_config,
    )

    token_interleaver_config = TokenInterleaverConfig(
        audio_placeholder_token_id=audio_placeholder_token_id,
        voice_placeholder_token_id=voice_placeholder_token_id,
        image_placeholder_token_id=image_placeholder_token_id,
    )

    recurrent_block_config = MegatransformerRecurrentConfig(
        block_config=TransformerBlockConfig(
            d_model=d_model * 2,
            n_heads=recurrent_n_heads,
            d_queries=recurrent_d_queries,
            d_values=recurrent_d_values,
            n_query_groups=recurrent_n_heads,
            d_inner=d_model * 4 * 2,
            n_layers=recurrent_n_layers,
            max_position_embeddings=max_position_embeddings,
        )
    )

    coda_transformer_config = TransformerBlockConfig(
        d_model=d_model,
        n_heads=coda_n_heads,
        d_queries=coda_d_queries,
        d_values=coda_d_values,
        n_query_groups=coda_n_heads,
        d_inner=d_model * 4,
        n_layers=coda_n_layers,
        max_position_embeddings=max_position_embeddings,
    )

    text_coda_config = TextCodaClassifierConfig(
        coda_config=coda_transformer_config,
        vocab_size=vocab_size,
    )

    audio_coda_config = AudioCodaAndVAEConfig(
        coda_config=coda_transformer_config,
        audio_config=audio_config,
    )

    voice_coda_config = AudioCodaAndVAEConfig(
        coda_config=coda_transformer_config,
        audio_config=voice_config,
    )

    image_coda_config = ImageCodaAndVAEConfig(
        coda_config=coda_transformer_config,
        image_config=image_config,
    )

    wm_config = MegatransformerWorldModelConfig(
        text_feature_config=small_text_feature_config,
        audio_prelude_config=audio_prelude_config,
        voice_prelude_config=voice_prelude_config,
        image_prelude_config=image_prelude_config,
        token_interleaver_config=token_interleaver_config,
        recurrent_block_config=recurrent_block_config,
        text_coda_config=text_coda_config,
        audio_coda_config=audio_coda_config,
        voice_coda_config=voice_coda_config,
        image_coda_config=image_coda_config
    )

    return wm_config

vocab_size = 32_000 + 9  # mistralai/Mistral-7B-v0.1 tokenizer vocab + 9 special tokens (BOA, EOA, BOV, EOV, BOI, EOI, AUDIO_PLACEHOLDER, VOICE_PLACEHOLDER, IMAGE_PLACEHOLDER)

# for testing and quick experiments
tiny_world_model_config = get_wm_config(
    d_model=128,
    vocab_size=vocab_size,
    max_position_embeddings=512,
    prelude_n_heads=2,
    prelude_n_layers=1,
    prelude_d_queries=32,
    prelude_d_values=32,
    recurrent_n_heads=4,
    recurrent_n_layers=2,
    reurrent_d_queries=64,
    recurrent_d_values=64,
    coda_n_heads=2,
    coda_n_layers=1,
    coda_d_queries=32,
    coda_d_values=32,
    audio_placeholder_token_id=32_006,
    voice_placeholder_token_id=32_007,
    image_placeholder_token_id=32_008,
)

small_world_model_config = get_wm_config(
    d_model=256,
    vocab_size=vocab_size,
    max_position_embeddings=1024,
    prelude_n_heads=4,
    prelude_n_layers=2,
    prelude_d_queries=64,
    prelude_d_values=64,
    recurrent_n_heads=8,
    recurrent_n_layers=4,
    reurrent_d_queries=128,
    recurrent_d_values=128,
    coda_n_heads=4,
    coda_n_layers=2,
    coda_d_queries=64,
    coda_d_values=64,
    audio_placeholder_token_id=32_006,
    voice_placeholder_token_id=32_007,
    image_placeholder_token_id=32_008,
)

medium_world_model_config = get_wm_config(
    d_model=512,
    vocab_size=vocab_size,
    max_position_embeddings=2048,
    prelude_n_heads=8,
    prelude_n_layers=4,
    prelude_d_queries=128,
    prelude_d_values=128,
    recurrent_n_heads=16,
    recurrent_n_layers=8,
    reurrent_d_queries=256,
    recurrent_d_values=256,
    coda_n_heads=8,
    coda_n_layers=4,
    coda_d_queries=128,
    coda_d_values=128,
    audio_placeholder_token_id=32_006,
    voice_placeholder_token_id=32_007,
    image_placeholder_token_id=32_008,
)