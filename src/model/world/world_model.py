import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.world.world_model import WORLD_MODEL_CONFIGS, MegaTransformerWorldModelConfig
from model.audio.feature_extractor import AudioVAEPreludeFeatureExtractor
from model.audio.generator import AudioCodaAndVAEWithLoss
from model.image.feature_extractor import ImageVAEPreludeFeatureExtractor
from model.image.vae.vae import ImageVAEDecoder, ImageVAEEncoder
from model.sinusoidal_positional_encoding import Sinusoidal2DPositionalEmbedding, SinusoidalPositionalEncoding
from model.text.feature_extractor import TextPreludeFeatureExtractor
from model.text.generator import TextCodaClassifierWithLoss
from config.image.decoder import (
    DiffusionBridgeImageDecoderConfig,
    ImageDecoderConfig,
)
from model.image.decoder import ImageDecoder
from model.image.diffusion_decoder import DiffusionBridgeImageDecoder
from model.world.kv_cache import RecurrentKVCache
from model.world.recurrent import MegatransformerRecurrentBlock
from model.world.token_alignment import MODALITY_TEXT, TokenInterleaver, TokenUninterleaver
from utils import constants, megatransformer_utils


class MegaTransformerWorldModel(nn.Module):
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

    def __init__(self, config: MegaTransformerWorldModelConfig):
        super(MegaTransformerWorldModel, self).__init__()

        self.config = config
        self.include_modes = set(config.include_modes)

        # Feature extractors — text is always required
        self.text_feature_extractor = TextPreludeFeatureExtractor(config.text_prelude_config)

        # Modality-specific preludes (only instantiate if included)
        self.audio_feature_extractor = (
            AudioVAEPreludeFeatureExtractor(config.audio_prelude_config)
            if "audio" in self.include_modes else None
        )
        self.voice_feature_extractor = (
            AudioVAEPreludeFeatureExtractor(config.voice_prelude_config)
            if "voice" in self.include_modes else None
        )
        self.image_feature_extractor = (
            ImageVAEPreludeFeatureExtractor(config.image_prelude_config)
            if "image" in self.include_modes else None
        )

        # Token alignment
        self.token_interleaver = TokenInterleaver(config.token_interleaver_config)
        self.token_uninterleaver = TokenUninterleaver()

        # Main transformer
        self.recurrent_block = MegatransformerRecurrentBlock(config.recurrent_block_config)

        # Generators/codas — text is always required
        self.text_generator = TextCodaClassifierWithLoss(config.text_coda_config)

        self.audio_generator = (
            AudioCodaAndVAEWithLoss("audio", config.audio_coda_config)
            if "audio" in self.include_modes else None
        )
        self.voice_generator = (
            AudioCodaAndVAEWithLoss("voice", config.voice_coda_config)
            if "voice" in self.include_modes else None
        )
        # Image decoder (optional). Dispatch on the actual config type:
        #   - ImageDecoderConfig            → ImageDecoder (direct latent prediction)
        #   - DiffusionBridgeImageDecoderConfig → DiffusionBridgeImageDecoder (flow matching)
        self.image_generator = None
        if config.image_coda_config is not None and "image" in self.include_modes:
            if isinstance(config.image_coda_config, DiffusionBridgeImageDecoderConfig):
                self.image_generator = DiffusionBridgeImageDecoder(config.image_coda_config)
            elif isinstance(config.image_coda_config, ImageDecoderConfig):
                self.image_generator = ImageDecoder(config.image_coda_config)
            else:
                raise TypeError(
                    f"Unknown image_coda_config type: {type(config.image_coda_config).__name__}. "
                    f"Expected ImageDecoderConfig or DiffusionBridgeImageDecoderConfig."
                )
            # Normalize recurrent output before the decoder to prevent
            # activation growth from saturating the decoder's attention.
            self.image_coda_input_norm = nn.LayerNorm(config.text_prelude_config.d_model)

            # Image generation queries for synthesis tasks.
            # The gen query count is decoupled from the prelude's patch count:
            #   - Prelude patches = how finely to tokenize the image for transcription
            #   - Gen queries = how many "slots" the recurrent block gets for synthesis
            # If n_image_gen_positions is None, fall back to the prelude's patch
            # count for backward compatibility.
            d_model = config.text_prelude_config.d_model
            if config.n_image_gen_positions is not None:
                n_gen = config.n_image_gen_positions
                nps_gen = int(n_gen ** 0.5)
                if nps_gen * nps_gen != n_gen:
                    raise ValueError(
                        f"n_image_gen_positions={n_gen} is not a perfect square. "
                        f"The 2D sinusoidal PE requires a square grid."
                    )
            else:
                n_gen = self.image_num_patches
                nps_gen = self.image_feature_extractor.num_patches_per_side
            self._n_image_gen_positions = n_gen

            self.gen_query_mode = config.gen_query_mode
            if config.gen_query_mode == "learned":
                self.image_gen_queries = nn.Parameter(
                    torch.randn(1, n_gen, d_model) * 3.0
                )
            # Frozen 2D sinusoidal positional encoding for image gen queries.
            # Sized to the gen query grid (nps_gen × nps_gen), which may differ
            # from the prelude's patch grid.
            self.image_gen_pos_embedding = Sinusoidal2DPositionalEmbedding(nps_gen, d_model)

        # Voice/audio use teacher forcing during training (ground-truth SIVE
        # through the prelude) and autoregressive generation at inference
        # (re-encoding each coda prediction through the prelude). No generation
        # queries are needed — the recurrent block KV cache provides text context
        # and the causal coda provides sequential voice/audio context.

        # Huginn-style embedding scale: multiply embeddings by sqrt(d_model) so
        # the injected input x_0 matches the thought state initialization variance.
        self.embed_scale = math.sqrt(config.text_prelude_config.d_model) if config.scale_embeddings else 1.0

        # Weight tying: share embedding matrix between input and output
        if getattr(config, 'tie_word_embeddings', False):
            self.text_generator.lm_head.weight = self.text_feature_extractor.wte.weight

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing on all sub-modules that support it."""
        modules = [
            self.text_feature_extractor,
            self.audio_feature_extractor,
            self.voice_feature_extractor,
            self.image_feature_extractor,
            self.recurrent_block,
            self.text_generator,
            self.audio_generator,
            self.voice_generator,
            self.image_generator,
        ]
        for mod in modules:
            if mod is not None and hasattr(mod, 'gradient_checkpointing'):
                mod.gradient_checkpointing = True
        # Propagate to nested modules (bridge, dit inside image_generator)
        if self.image_generator is not None:
            for child in self.image_generator.modules():
                if hasattr(child, 'gradient_checkpointing'):
                    child.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing on all sub-modules."""
        for mod in self.modules():
            if hasattr(mod, 'gradient_checkpointing'):
                mod.gradient_checkpointing = False

    def load_image_vae(self, encoder: ImageVAEEncoder, decoder: ImageVAEDecoder):
        """Load image VAE encoder/decoder for live encoding/decoding."""
        if self.image_feature_extractor is not None:
            self.image_feature_extractor.vae_encoder = encoder

    @classmethod
    def from_config(cls, config_name: str, **overrides) -> "MegaTransformerWorldModel":
        """
        Create model from predefined config with optional overrides.

        Args:
            config_name: One of predefined configs
            **overrides: Override any config parameter

        Example:
            model = ImageVAEPreludeFeatureExtractor.from_config("small", vae_encoder=my_vae_encoder, prelude_config=custom_prelude_config)
        """
        if config_name not in WORLD_MODEL_CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(WORLD_MODEL_CONFIGS.keys())}")

        config = WORLD_MODEL_CONFIGS[config_name]
        # Apply overrides
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(overrides)
        config = MegaTransformerWorldModelConfig(**config_dict)

        return cls(config)

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
        # Per-sample direction: True = synthesis (text→media), False = transcription
        is_synthesis: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the world model.

        Args:
            text_input_ids: Token IDs for text, shape (batch, text_seq_len).
                Contains placeholder tokens for media positions.

            audio_inputs: SIVE features, shape (batch, n_audio, feature_channels, timesteps).
            audio_lengths: Actual lengths per audio example, shape (batch, n_audio).
            audio_latent_labels: Target latents for audio loss, same shape as latent inputs.

            voice_inputs: SIVE features, same format as audio_inputs.
            voice_lengths: Same format as audio_lengths.
            voice_latent_labels: Target SIVE features for voice loss.

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
        # Handle mixed-modality batches: null out modalities with mismatched batch size
        if text_input_ids is not None:
            B = text_input_ids.shape[0]
            if voice_inputs is not None and voice_inputs.shape[0] != B:
                voice_inputs = None
                voice_lengths = None
                voice_latent_labels = None
            if audio_inputs is not None and audio_inputs.shape[0] != B:
                audio_inputs = None
                audio_lengths = None
                audio_latent_labels = None
            if image_inputs is not None and image_inputs.shape[0] != B:
                image_inputs = None
                image_latent_labels = None

        text_hidden_states = self.text_feature_extractor(text_input_ids)

        # Audio and voice generation:
        #   - Synthesis (is_synthesis=True): shifted teacher forcing. The prelude
        #     encodes frames [0..T-2] and these are placed at positions [1..T-1].
        #     Position 0 gets a zero vector. The coda target at position t is
        #     frame t (so position 0 predicts frame 0 from "nothing", position 1
        #     predicts frame 1 from prelude(frame 0), etc.). This matches how BOV
        #     predicts the first frame in text-style shifted loss.
        #   - Transcription (is_synthesis=False): prelude runs normally on the
        #     full SIVE features. No shift — the model sees the audio and the
        #     text coda predicts the transcript.
        #   - At inference: autoregressive generation re-encodes each coda
        #     prediction through the prelude (no teacher forcing).
        audio_hidden_states = None
        if audio_inputs is not None and self.audio_feature_extractor is not None:
            batch_size, n_audio = audio_inputs.shape[:2]
            audio_flat = audio_inputs.view(batch_size * n_audio, *audio_inputs.shape[2:])

            if is_synthesis is not None and is_synthesis.any():
                d_model = self.config.text_prelude_config.d_model
                # Shift: encode frames [0..T-2], prepend zero at position 0
                shifted_input = audio_flat[:, :, :-1]  # (B*N, C, T-1)
                shifted_hidden = self.audio_feature_extractor(shifted_input)  # (B*N, T-1, d_model)
                zero_prefix = torch.zeros(shifted_hidden.shape[0], 1, d_model, device=shifted_hidden.device, dtype=shifted_hidden.dtype)
                synth_hidden = torch.cat([zero_prefix, shifted_hidden], dim=1)  # (B*N, T, d_model)

                if is_synthesis.all():
                    audio_hidden_states = synth_hidden.view(batch_size, n_audio, synth_hidden.shape[1], d_model)
                else:
                    # Mixed batch: run prelude normally for transcription samples
                    normal_hidden = self.audio_feature_extractor(audio_flat)
                    seq_len = normal_hidden.shape[1]
                    audio_hidden_states = normal_hidden.view(batch_size, n_audio, seq_len, d_model).clone()
                    synth_view = synth_hidden.view(batch_size, n_audio, synth_hidden.shape[1], d_model)
                    for b in range(batch_size):
                        if is_synthesis[b]:
                            audio_hidden_states[b] = synth_view[b]
            else:
                audio_hidden_flat = self.audio_feature_extractor(audio_flat)
                seq_len, d_model = audio_hidden_flat.shape[1], audio_hidden_flat.shape[2]
                audio_hidden_states = audio_hidden_flat.view(batch_size, n_audio, seq_len, d_model)

        voice_hidden_states = None
        if voice_inputs is not None and self.voice_feature_extractor is not None:
            batch_size, n_voice = voice_inputs.shape[:2]
            voice_flat = voice_inputs.view(batch_size * n_voice, *voice_inputs.shape[2:])

            if is_synthesis is not None and is_synthesis.any():
                d_model = self.config.text_prelude_config.d_model
                shifted_input = voice_flat[:, :, :-1]  # (B*N, C, T-1)
                shifted_hidden = self.voice_feature_extractor(shifted_input)  # (B*N, T-1, d_model)
                zero_prefix = torch.zeros(shifted_hidden.shape[0], 1, d_model, device=shifted_hidden.device, dtype=shifted_hidden.dtype)
                synth_hidden = torch.cat([zero_prefix, shifted_hidden], dim=1)  # (B*N, T, d_model)

                if is_synthesis.all():
                    voice_hidden_states = synth_hidden.view(batch_size, n_voice, synth_hidden.shape[1], d_model)
                else:
                    normal_hidden = self.voice_feature_extractor(voice_flat)
                    seq_len = normal_hidden.shape[1]
                    voice_hidden_states = normal_hidden.view(batch_size, n_voice, seq_len, d_model).clone()
                    synth_view = synth_hidden.view(batch_size, n_voice, synth_hidden.shape[1], d_model)
                    for b in range(batch_size):
                        if is_synthesis[b]:
                            voice_hidden_states[b] = synth_view[b]
            else:
                voice_hidden_flat = self.voice_feature_extractor(voice_flat)
                seq_len, d_model = voice_hidden_flat.shape[1], voice_hidden_flat.shape[2]
                voice_hidden_states = voice_hidden_flat.view(batch_size, n_voice, seq_len, d_model)

        image_hidden_states = None
        if image_inputs is not None and self.image_feature_extractor is not None:
            batch_size, n_images = image_inputs.shape[:2]
            image_flat = image_inputs.view(batch_size * n_images, *image_inputs.shape[2:])

            # Build generation queries (used at all synthesis-direction image
            # positions). Either learned + 2D PE or PE-only depending on mode.
            if is_synthesis is not None and is_synthesis.any():
                if hasattr(self, 'image_gen_queries'):
                    raw_gen_queries = self.image_gen_pos_embedding(self.image_gen_queries).expand(batch_size, -1, -1)
                else:
                    raw_gen_queries = self.image_gen_pos_embedding.pe.expand(batch_size, -1, -1)
                gen_queries = raw_gen_queries.unsqueeze(1)  # (B, 1, n_patches, d_model)

            if is_synthesis is not None and is_synthesis.all():
                # All-synthesis batch: skip the prelude entirely. The image
                # feature extractor's output isn't used for anything in this
                # path — gen queries replace the image positions outright.
                image_hidden_states = gen_queries
            elif is_synthesis is not None and is_synthesis.any():
                # Mixed batch: run the prelude for the transcription samples,
                # then overwrite the synthesis samples' positions with gen queries.
                image_hidden_flat = self.image_feature_extractor(
                    image_flat, precomputed_latents=precomputed_latents
                )
                seq_len, d_model = image_hidden_flat.shape[1], image_hidden_flat.shape[2]
                image_hidden_states = image_hidden_flat.view(batch_size, n_images, seq_len, d_model).clone()
                for b in range(batch_size):
                    if is_synthesis[b]:
                        image_hidden_states[b] = gen_queries[b]
            else:
                # No direction info or all transcription: run the prelude normally.
                image_hidden_flat = self.image_feature_extractor(
                    image_flat, precomputed_latents=precomputed_latents
                )
                seq_len, d_model = image_hidden_flat.shape[1], image_hidden_flat.shape[2]
                image_hidden_states = image_hidden_flat.view(batch_size, n_images, seq_len, d_model)

        # print("\tInputs to token interleaver:")
        # megatransformer_utils.print_debug_tensor("\t\ttext_hidden_states", text_hidden_states)
        # if audio_hidden_states is not None:
        #     megatransformer_utils.print_debug_tensor("\t\taudio_hidden_states", audio_hidden_states)
        # if voice_hidden_states is not None:
        #     megatransformer_utils.print_debug_tensor("\t\tvoice_hidden_states", voice_hidden_states)
        # if image_hidden_states is not None:
        #     megatransformer_utils.print_debug_tensor("\t\timage_hidden_states", image_hidden_states)

        # Token Interleaving
        interleaved_tokens, attn_mask, modality_map = self.token_interleaver(
            text_hidden_states=text_hidden_states,
            text_token_ids=text_input_ids,
            audio_hidden_states=audio_hidden_states,
            audio_lengths=audio_lengths,
            voice_hidden_states=voice_hidden_states,
            voice_lengths=voice_lengths,
            image_hidden_states=image_hidden_states,
        )

        # Scale embeddings by sqrt(d_model) to match thought state initialization
        interleaved_tokens = interleaved_tokens * self.embed_scale

        # print("\tInputs to recurrent block:")
        # megatransformer_utils.print_debug_tensor("\t\tinterleaved_tokens", interleaved_tokens)

        # Main Transformer (Recurrent Block)
        recurrent_output, _, _, _, iteration_stats = self.recurrent_block(
            interleaved_tokens,
            attention_mask=attn_mask  # True for attend, False for padding
        )

        # print("\tInputs to uninterleaver:")
        # megatransformer_utils.print_debug_tensor("\t\trecurrent_output", recurrent_output)
        # megatransformer_utils.print_debug_tensor("\t\tmodality_map", modality_map)

        # Token Uninterleaving
        uninterleaved = self.token_uninterleaver(recurrent_output, modality_map)

        text_batch = uninterleaved["text"]
        audio_batch = uninterleaved["audio"]
        voice_batch = uninterleaved["voice"]
        image_batch = uninterleaved["image"]

        # print("\tOutputs of token uninterleaver:")
        # if text_batch is not None:
        #     megatransformer_utils.print_debug_tensor("\t\ttext_batch", text_batch)
        # if audio_batch is not None:
        #     megatransformer_utils.print_debug_tensor("\t\taudio_batch", audio_batch)
        # if voice_batch is not None:
        #     megatransformer_utils.print_debug_tensor("\t\tvoice_batch", voice_batch)
        # if image_batch is not None:
        #     megatransformer_utils.print_debug_tensor("\t\timage_batch", image_batch)

        # Generators/Codas
        outputs = {}

        if iteration_stats is not None:
            outputs["iteration_stats"] = iteration_stats

        # Log variance and entropy of recurrent outputs per modality (coda inputs)
        for name, batch in [("text", text_batch), ("voice", voice_batch),
                            ("audio", audio_batch), ("image", image_batch)]:
            if batch is not None:
                # Per-token activation variance across d_model, averaged over batch+seq
                outputs[f"recurrent_out/{name}_token_var"] = batch.var(dim=-1).mean()
                # Entropy of softmax over d_model (activation spread per token)
                probs = torch.softmax(batch, dim=-1)
                entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
                outputs[f"recurrent_out/{name}_entropy"] = entropy
                # Cross-token variance: how different tokens are from each other
                # var across seq_len per feature dim, averaged over batch+d_model.
                # For image, split by direction: synthesis positions are gen
                # queries (near-identical across positions → low cross-token var)
                # while transcription positions are prelude-encoded patches
                # (high cross-token var). Averaging them together produces a
                # meaningless mean of two very different regimes.
                if name == "image" and is_synthesis is not None:
                    syn_mask = is_synthesis.bool().to(batch.device)
                    if syn_mask.any():
                        outputs["recurrent_out/image_syn_seq_var"] = batch[syn_mask].var(dim=1).mean()
                    if (~syn_mask).any():
                        outputs["recurrent_out/image_trans_seq_var"] = batch[~syn_mask].var(dim=1).mean()
                else:
                    outputs[f"recurrent_out/{name}_seq_var"] = batch.var(dim=1).mean()

        # Text
        if text_batch is not None:
            text_outputs = self.text_generator(text_batch, targets=text_targets)
            outputs.update(text_outputs)

        # Audio
        if audio_batch is not None and self.audio_generator is not None:
            audio_outputs = self.audio_generator(
                audio_batch,
                latent_labels=audio_latent_labels,
                lengths=uninterleaved["audio_lengths"],
                decode_to_mel=decode_outputs,
            )
            outputs.update(audio_outputs)

        # Voice
        if voice_batch is not None and self.voice_generator is not None:
            voice_outputs = self.voice_generator(
                voice_batch,
                latent_labels=voice_latent_labels,
                lengths=uninterleaved["voice_lengths"],
                decode_to_mel=decode_outputs,
            )
            outputs.update(voice_outputs)

        # Surface the recurrent block's image-position outputs for diagnostic
        # use (e.g. the per-sample diff check in scripts/debug/diagnose_checkpoint.py).
        # NOT used as a training target — that experiment is dead.
        if image_batch is not None:
            outputs["image_recurrent_tokens"] = image_batch

        # Image decoder: reconstruct image latents from content tokens.
        # Either ImageDecoder (direct prediction) or DiffusionBridgeImageDecoder
        # (flow matching). We propagate whichever loss keys the decoder produces.
        if image_batch is not None and hasattr(self, 'image_generator') and self.image_generator is not None:
            cross_input = self.image_coda_input_norm(image_batch)
            cross_outputs = self.image_generator(
                encoder_hidden_states=cross_input,
                latent_labels=image_latent_labels,
            )
            if "image_latent_preds" in cross_outputs:
                outputs["image_latent_preds"] = cross_outputs["image_latent_preds"]
            # Direct decoder pre-computes L1/MSE losses on the latent preds.
            if "image_latent_l1_loss" in cross_outputs:
                outputs["image_l1_loss"] = cross_outputs["image_latent_l1_loss"]
                outputs["image_mse_loss"] = cross_outputs["image_latent_mse_loss"]
            # Diffusion decoder computes a flow-matching loss internally.
            if "image_diffusion_loss" in cross_outputs:
                outputs["image_diffusion_loss"] = cross_outputs["image_diffusion_loss"]
            if "image_diffusion_loss_raw" in cross_outputs:
                outputs["image_diffusion_loss_raw"] = cross_outputs["image_diffusion_loss_raw"]

        return outputs

    @property
    def image_num_patches(self) -> int:
        """Number of image gen query positions used during synthesis.

        If `n_image_gen_positions` was set on the config, this returns that
        value (decoupled from the prelude). Otherwise falls back to the
        prelude's patch count for backward compatibility.
        """
        if hasattr(self, '_n_image_gen_positions'):
            return self._n_image_gen_positions
        if self.image_feature_extractor is None:
            return 0
        nps = self.image_feature_extractor.num_patches_per_side
        return nps * nps

    def _encode_prompt(
        self,
        text_input_ids: torch.Tensor,
        audio_inputs: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None,
        voice_inputs: Optional[torch.Tensor] = None,
        voice_lengths: Optional[torch.Tensor] = None,
        image_inputs: Optional[torch.Tensor] = None,
        precomputed_latents: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Encode a prompt with optional pre-encoded media into hidden states.

        If media inputs are provided, placeholder tokens in text_input_ids are
        replaced with actual media embeddings via the TokenInterleaver (same as
        in forward()).

        Returns:
            Tuple of (prompt_hidden, attention_mask, modality_map):
            - prompt_hidden: (batch, seq_len, d_model)
            - attention_mask: (batch, seq_len) or None
            - modality_map: (batch, seq_len) or None — integer tensor marking
              each position's modality (0=text, 1=audio, 2=voice, 3=image).
              None when no media is present (all positions are text).
        """
        has_media = (
            audio_inputs is not None
            or voice_inputs is not None
            or image_inputs is not None
        )

        if not has_media:
            # Text-only prompt — no interleaving needed
            prompt_hidden = self.text_feature_extractor(text_input_ids)
            return prompt_hidden, None, None

        # Encode text
        text_hidden = self.text_feature_extractor(text_input_ids)

        # Encode media through feature extractors
        audio_hidden = None
        if audio_inputs is not None and self.audio_feature_extractor is not None:
            batch_size, n_audio = audio_inputs.shape[:2]
            audio_flat = audio_inputs.view(batch_size * n_audio, *audio_inputs.shape[2:])
            audio_hidden_flat = self.audio_feature_extractor(audio_flat)
            seq_len, d_model = audio_hidden_flat.shape[1], audio_hidden_flat.shape[2]
            audio_hidden = audio_hidden_flat.view(batch_size, n_audio, seq_len, d_model)

        voice_hidden = None
        if voice_inputs is not None and self.voice_feature_extractor is not None:
            batch_size, n_voice = voice_inputs.shape[:2]
            voice_flat = voice_inputs.view(batch_size * n_voice, *voice_inputs.shape[2:])
            voice_hidden_flat = self.voice_feature_extractor(voice_flat)
            seq_len, d_model = voice_hidden_flat.shape[1], voice_hidden_flat.shape[2]
            voice_hidden = voice_hidden_flat.view(batch_size, n_voice, seq_len, d_model)

        image_hidden = None
        if image_inputs is not None and self.image_feature_extractor is not None:
            batch_size, n_images = image_inputs.shape[:2]
            image_flat = image_inputs.view(batch_size * n_images, *image_inputs.shape[2:])
            image_hidden_flat = self.image_feature_extractor(
                image_flat, precomputed_latents=precomputed_latents
            )
            seq_len, d_model = image_hidden_flat.shape[1], image_hidden_flat.shape[2]
            image_hidden = image_hidden_flat.view(batch_size, n_images, seq_len, d_model)

        # Interleave — replaces placeholder tokens with media embeddings
        interleaved, attn_mask, modality_map = self.token_interleaver(
            text_hidden_states=text_hidden,
            text_token_ids=text_input_ids,
            audio_hidden_states=audio_hidden,
            audio_lengths=audio_lengths,
            voice_hidden_states=voice_hidden,
            voice_lengths=voice_lengths,
            image_hidden_states=image_hidden,
        )

        return interleaved, attn_mask, modality_map

    def _finalize_image_sequence(
        self,
        image_sequences_b: List[torch.Tensor],
        decode_outputs: bool,
    ) -> Optional[torch.Tensor]:
        """Finalize an image sequence through the cross-attention image decoder.

        Pads or truncates to exactly image_num_patches tokens so the
        decoder can reshape to a square spatial grid.
        """
        if not image_sequences_b or self.image_generator is None:
            return None
        image_hidden = torch.cat(image_sequences_b, dim=0)  # (seq, d_model)
        expected = self.image_num_patches
        actual = image_hidden.shape[0]
        if actual < expected:
            # Pad with zeros to reach expected patch count
            pad = torch.zeros(
                expected - actual, image_hidden.shape[1],
                device=image_hidden.device, dtype=image_hidden.dtype,
            )
            image_hidden = torch.cat([image_hidden, pad], dim=0)
        elif actual > expected:
            image_hidden = image_hidden[:expected]
        image_hidden = image_hidden.unsqueeze(0)  # (1, num_patches, d_model)
        cross_input = self.image_coda_input_norm(image_hidden)
        image_out = self.image_generator(
            encoder_hidden_states=cross_input,
        )
        preds = image_out.get("image_latent_preds")
        if preds is not None:
            preds = preds.squeeze(0)  # (1, C, H, W) -> (C, H, W)
        return preds

    # Generation with KV Caching
    @torch.no_grad()
    def generate(
        self,
        text_input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        kv_cache_strategy: str = "huginn",
        kv_cache_budget: int = 16,
        decode_outputs: bool = False,
        # Token budgets for media generation (fixed-length, like image patches)
        audio_token_budget: int = 209,
        voice_token_budget: int = 209,
        # Pre-encoded media for transcription / cross-modal tasks
        audio_inputs: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None,
        voice_inputs: Optional[torch.Tensor] = None,
        voice_lengths: Optional[torch.Tensor] = None,
        image_inputs: Optional[torch.Tensor] = None,
        precomputed_latents: bool = True,
        share_kv_cache: bool = False,
        image_iteration_override: Optional[int] = None,
        image_num_inference_steps: Optional[int] = None,
        image_sampler: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate tokens autoregressively with KV caching.

        This method implements autoregressive generation through the recurrent block
        with efficient KV caching using the Huginn approach (circular buffer).
        If share_kv_cache=True, all recurrent blocks share a single KV cache slot
        per iteration (4x memory savings, Huginn-style).

        The generation flow:
        1. Process initial text input through text feature extractor
        2. If media inputs are provided, replace placeholder tokens with actual
           media embeddings via the TokenInterleaver (for transcription tasks)
        3. Autoregressively sample tokens through the recurrent block
        4. Media generation uses fixed token budgets: audio/voice generate exactly
           audio_token_budget/voice_token_budget tokens, images generate exactly
           num_patches tokens. After reaching the budget, the corresponding EO*
           token is auto-emitted and the sequence is finalized through the coda.
        5. Continue generating until max_new_tokens or end-of-sequence

        Args:
            text_input_ids: Initial text prompt token IDs, shape (batch, prompt_len).
                Can contain placeholder tokens for media that will be generated,
                or for pre-encoded media that replaces the placeholders.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature. Higher = more random.
            top_k: If set, only sample from top k most likely tokens.
            top_p: If set, use nucleus sampling with this probability mass.
            kv_cache_strategy: "huginn" (shared cache, efficient) or "per_iteration".
            kv_cache_budget: Number of cache slots for Huginn strategy.
            decode_outputs: If True, decode generated latents via VAE decoders.
            audio_inputs: Pre-encoded SIVE features, shape (batch, n_audio, C, T).
                Requires corresponding AUDIO_PLACEHOLDER tokens in text_input_ids.
            audio_lengths: Actual lengths per audio clip, shape (batch, n_audio).
            voice_inputs: Pre-encoded SIVE voice features, same format as audio_inputs.
            voice_lengths: Actual lengths per voice clip, shape (batch, n_voice).
            image_inputs: Pre-encoded images, shape (batch, n_images, C, H, W).
                Requires corresponding IMAGE_PLACEHOLDER tokens in text_input_ids.
            precomputed_latents: If True, media inputs are VAE latents.

        Returns:
            Dictionary containing:
            - "generated_token_ids": Generated text token IDs, shape (batch, seq_len)
            - "text_logits": Logits for text tokens, shape (batch, seq_len, vocab_size)
            - "audio_latent_preds": Padded tensor of shape (batch, max_n_audio, C, max_T)
            - "audio_counts": Number of audio clips per batch item, shape (batch,)
            - "audio_lengths": Actual time length of each audio, shape (batch, max_n_audio).
                Use these lengths to slice before VAE decoding to avoid decoding padding.
            - "voice_latent_preds": Padded tensor of shape (batch, max_n_voice, C, max_T)
            - "voice_counts": Number of voice clips per batch item, shape (batch,)
            - "voice_lengths": Actual time length of each voice, shape (batch, max_n_voice)
            - "image_latent_preds": Padded tensor of shape (batch, max_n_image, C, H, W)
            - "image_counts": Number of images per batch item, shape (batch,)
            - Decoded outputs if decode_outputs=True
        """
        batch_size = text_input_ids.shape[0]
        device = text_input_ids.device

        # Required token count for image generation
        image_token_budget = self.image_num_patches

        # Initialize KV cache
        kv_cache = RecurrentKVCache(
            strategy=kv_cache_strategy,
            cache_budget=kv_cache_budget,
        )

        # Track generation state per batch item
        generated_tokens: List[List[int]] = [[] for _ in range(batch_size)]
        all_logits: List[torch.Tensor] = []

        # Track modality sequences being built (current in-progress sequence)
        audio_sequences: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
        voice_sequences: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
        image_sequences: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]

        # Voice/audio prelude + coda KV caches for autoregressive generation.
        # The causal prelude needs KV caching so frame t's self-attention sees
        # frames 0..t-1 (matching training where the full shifted sequence is
        # processed with causal masking).
        voice_prelude_kv_caches: List[Optional[List]] = [None for _ in range(batch_size)]
        audio_prelude_kv_caches: List[Optional[List]] = [None for _ in range(batch_size)]
        voice_prelude_position_offsets: List[int] = [0 for _ in range(batch_size)]
        audio_prelude_position_offsets: List[int] = [0 for _ in range(batch_size)]
        voice_coda_kv_caches: List[Optional[List]] = [None for _ in range(batch_size)]
        audio_coda_kv_caches: List[Optional[List]] = [None for _ in range(batch_size)]
        voice_coda_position_offsets: List[int] = [0 for _ in range(batch_size)]
        audio_coda_position_offsets: List[int] = [0 for _ in range(batch_size)]
        # Last coda prediction per batch item for autoregressive re-encoding.
        # Shape: (feature_channels, 1) — single SIVE frame.
        last_voice_pred: List[Optional[torch.Tensor]] = [None for _ in range(batch_size)]
        last_audio_pred: List[Optional[torch.Tensor]] = [None for _ in range(batch_size)]

        # Track which modality we're currently generating for each batch item
        # None = text, "audio", "voice", "image"
        current_modality: List[Optional[str]] = [None for _ in range(batch_size)]
        # Per-batch-item EOS flag — stops text generation once EOS is sampled
        finished: List[bool] = [False for _ in range(batch_size)]

        # Completed modality outputs (list of tensors per batch item to support multiple media)
        completed_audio: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
        completed_voice: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
        completed_image: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
        # Recurrent iterations actually performed per generated image (one list per
        # batch item, one entry per image block).
        image_recurrent_iterations: List[List[int]] = [[] for _ in range(batch_size)]

        # Per-frame stop-head logit traces for diagnostics. One list per batch
        # item, appended to on every voice/audio generation iter. Returned in
        # the outputs dict so callers can visualize the stop head's signal and
        # diagnose why a budget-hit occurred (flat at -5 → head not learning;
        # rising but < 0 → distribution-shift; crosses 0 late → positional bias).
        voice_stop_logit_trace: List[List[float]] = [[] for _ in range(batch_size)]
        audio_stop_logit_trace: List[List[float]] = [[] for _ in range(batch_size)]

        # Process initial prompt — replace placeholders with media if provided.
        # Also initialize text prelude KV caches so subsequent tokens get the
        # same causal context the prelude sees during training.
        prompt_hidden, prompt_attn_mask, prompt_modality_map = self._encode_prompt(
            text_input_ids,
            audio_inputs=audio_inputs,
            audio_lengths=audio_lengths,
            voice_inputs=voice_inputs,
            voice_lengths=voice_lengths,
            image_inputs=image_inputs,
            precomputed_latents=precomputed_latents,
        )

        # Prime the text prelude KV cache from the prompt. The prelude's causal
        # self-attention needs to see all previous text tokens when encoding each
        # new generated token — without this, each token would be encoded in
        # isolation (seq_len=1), making the prelude's self-attention a no-op.
        text_prelude_result = self.text_feature_extractor(
            text_input_ids, use_cache=True, position_offset=0,
        )
        # Unpack: forward returns (hidden_states, kv_caches) when use_cache=True
        _, text_prelude_kv_caches = text_prelude_result
        text_prelude_position_offset = text_input_ids.shape[1]

        # Process through recurrent block to get initial context
        # Track recurrent iteration counts and KL divergences per generated token
        recurrent_iteration_counts: List[int] = []
        recurrent_kl_final: List[float] = []  # final KL per token (convergence measure)

        current_hidden, kv_cache, prompt_iters, prompt_kls, _ = self.recurrent_block(
            prompt_hidden * self.embed_scale,
            attention_mask=prompt_attn_mask,
            kv_cache=kv_cache,
            position_offset=0,
            use_cache=True,
            share_kv_cache=share_kv_cache,
        )

        position_offset = prompt_hidden.shape[1]

        # Check if prompt ends with a BO* token — if so, initialize modality mode
        # so the first generated hidden states are accumulated correctly.
        for b in range(batch_size):
            last_token = text_input_ids[b, -1].item()
            if last_token == constants.BOA_TOKEN_ID:
                current_modality[b] = "audio"
            elif last_token == constants.BOV_TOKEN_ID:
                current_modality[b] = "voice"
            elif last_token == constants.BOI_TOKEN_ID:
                current_modality[b] = "image"

        # Process the prompt through the text coda with KV caching so
        # the coda's self-attention has the complete prompt context.
        #
        # CRITICAL: only pass TEXT positions to the text coda. During training,
        # the uninterleaver strips media positions — the text coda never sees
        # voice/audio/image hidden states. If we pass all prompt positions
        # (including media) here, the coda's KV cache gets polluted with
        # out-of-distribution entries, breaking transcription tasks.
        if prompt_modality_map is not None:
            # Prompt contains media — extract only text positions per batch item.
            # All batch items in a prompt share the same modality layout, so we
            # use the first item's map as the mask.
            text_mask = prompt_modality_map[0] == MODALITY_TEXT  # (seq_len,)
            text_coda_input = current_hidden[:, text_mask, :]  # (batch, n_text, d_model)
        else:
            # Text-only prompt — all positions are text.
            text_coda_input = current_hidden
        text_coda_output = self.text_generator(
            text_coda_input, use_cache=True, position_offset=0,
        )
        text_coda_kv_caches = text_coda_output.get("kv_caches")
        text_coda_position_offset = text_coda_input.shape[1]
        logits = text_coda_output["logits"][:, -1, :]  # last position
        all_logits.append(logits.unsqueeze(1))

        # Sample first token
        next_token_ids = self._sample_tokens(logits, temperature, top_k, top_p)

        for b in range(batch_size):
            generated_tokens[b].append(next_token_ids[b].item())

        # Autoregressive generation loop
        for _ in range(max_new_tokens - 1):
            # Check for modality transitions and handle accordingly
            next_hidden_list = []

            # Per-batch forced next-token override. When a media block finalizes
            # in this iteration, we set forced_next_token[b] = EO*_TOKEN_ID so
            # that EO* becomes the sampled token for this step — replacing the
            # text coda's free-sampled token from the BO* position. This lets
            # the next iteration feed EO* through text_prelude → recurrent →
            # text_coda, advancing all three KV caches in the same sequence the
            # uninterleaver produced at training time ([..., BO*, EO*, text]).
            # Without this override, EO* would only exist as a marker in
            # generated_tokens and the actual token driving iter N+1 would be
            # whatever BO*'s logits sampled — breaking train/inference parity.
            forced_next_token: List[Optional[int]] = [None for _ in range(batch_size)]
            # just_entered_streaming[b] = "voice"/"audio" when BO* transitioned
            # current_modality from None this iter. For image (single-shot) the
            # shared text_coda call naturally processes BOI because the image
            # branch resets current_modality in the same iter, so any_media
            # becomes False. For voice/audio, any_media stays True throughout
            # streaming, so the shared call never runs — BO* never enters the
            # text coda's KV cache. We fix this by making a separate one-item
            # text coda call at the entry iter.
            just_entered_streaming: List[Optional[str]] = [None for _ in range(batch_size)]
            # just_finalized_streaming[b] = "voice"/"audio" when the voice or
            # audio branch finalizes this iter. At the finalizing iter,
            # `current_hidden` holds the LAST streaming-frame's recurrent
            # output (a MODALITY_VOICE/AUDIO position). At training the text
            # coda never sees these positions, so we must SKIP the shared
            # text_coda call at finalizing iters and emit EO* directly as
            # the next token.
            just_finalized_streaming: List[Optional[str]] = [None for _ in range(batch_size)]

            for b in range(batch_size):
                token_id = next_token_ids[b].item()

                # During media generation, ignore sampled tokens — the budget
                # controls finalization, not sampled EO* tokens. The text coda
                # runs every step but its output is only meaningful in text mode.
                if current_modality[b] in ("audio", "voice", "image"):
                    mod = current_modality[b]
                    if mod in ("voice", "audio"):
                        # Autoregressive: re-encode previous coda prediction
                        # through the causal prelude with KV caching, or use
                        # zeros for position 0 (shifted, like text).
                        last_pred = last_voice_pred[b] if mod == "voice" else last_audio_pred[b]
                        prelude = self.voice_feature_extractor if mod == "voice" else self.audio_feature_extractor
                        if last_pred is not None and prelude is not None:
                            # last_pred: (C, 1) → (1, C, 1) for prelude
                            p_kv = voice_prelude_kv_caches[b] if mod == "voice" else audio_prelude_kv_caches[b]
                            p_off = voice_prelude_position_offsets[b] if mod == "voice" else audio_prelude_position_offsets[b]
                            embed, new_p_kv = prelude(
                                last_pred.unsqueeze(0),
                                kv_caches=p_kv,
                                position_offset=p_off,
                                use_cache=True,
                            )  # embed: (1, 1, d_model)
                            if mod == "voice":
                                voice_prelude_kv_caches[b] = new_p_kv
                                voice_prelude_position_offsets[b] += 1
                            else:
                                audio_prelude_kv_caches[b] = new_p_kv
                                audio_prelude_position_offsets[b] += 1
                        else:
                            # First position: zero input (shifted, like text's
                            # BOV predicting the first SIVE frame)
                            embed = torch.zeros(1, 1, self.config.text_prelude_config.d_model, device=device, dtype=current_hidden.dtype)
                        next_hidden_list.append(embed[0])
                    else:
                        # Image: handled separately (single-shot, not autoregressive)
                        next_hidden_list.append(torch.zeros(1, self.config.text_prelude_config.d_model, device=device, dtype=current_hidden.dtype))

                # Text mode: embed via text prelude with KV caching so the
                # prelude's causal self-attention sees all previous text tokens.
                else:
                    if token_id == constants.BOA_TOKEN_ID:
                        current_modality[b] = "audio"
                        just_entered_streaming[b] = "audio"
                    elif token_id == constants.BOV_TOKEN_ID:
                        current_modality[b] = "voice"
                        just_entered_streaming[b] = "voice"
                    elif token_id == constants.BOI_TOKEN_ID:
                        current_modality[b] = "image"
                        # No flag — image is single-shot and resets current_modality
                        # in the same iter, so the shared text_coda call below
                        # naturally processes BOI's current_hidden.

                    token_embed, text_prelude_kv_caches = self.text_feature_extractor(
                        torch.tensor([[token_id]], device=device),
                        kv_caches=text_prelude_kv_caches,
                        position_offset=text_prelude_position_offset,
                        use_cache=True,
                    )
                    text_prelude_position_offset += 1
                    next_hidden_list.append(token_embed[0])

            # Stack embeddings for all batch items: (batch, 1, d_model)
            next_hidden = torch.stack(next_hidden_list, dim=0)

            # Process through recurrent block with KV cache
            current_hidden, kv_cache, n_iters, kl_trace, _ = self.recurrent_block(
                next_hidden * self.embed_scale,
                attention_mask=None,
                kv_cache=kv_cache,
                position_offset=position_offset,
                use_cache=True,
                share_kv_cache=share_kv_cache,
            )
            recurrent_iteration_counts.append(n_iters)
            recurrent_kl_final.append(kl_trace[-1] if kl_trace else 0.0)
            position_offset += 1

            # Accumulate hidden states and run codas for non-text modalities
            for b in range(batch_size):
                if current_modality[b] == "audio":
                    # Run audio coda autoregressively with KV cache.
                    # On mid-generation BOA entry (just_entered_streaming=="audio"),
                    # the coda's first input must be recurrent(zero_vec), not
                    # recurrent(BOA_embed) — matching training where position 0
                    # of a voice/audio block is a literal zero vector in d_model
                    # space (world_model.py:291-294 for voice; same shape for
                    # audio). For trailing-BOA prompts, this path is already
                    # handled naturally by the line-896 zero_vec in iter 0;
                    # here we emulate the same behavior when BOA is sampled
                    # mid-generation so both entry paths converge.
                    if just_entered_streaming[b] == "audio":
                        d_model_ = self.config.text_prelude_config.d_model
                        zero_hidden = torch.zeros(
                            1, 1, d_model_, device=device, dtype=current_hidden.dtype,
                        )
                        entry_hidden, kv_cache, _, _, _ = self.recurrent_block(
                            zero_hidden * self.embed_scale,
                            attention_mask=None,
                            kv_cache=kv_cache,
                            position_offset=position_offset,
                            use_cache=True,
                            share_kv_cache=share_kv_cache,
                        )
                        position_offset += 1
                        hidden_b = entry_hidden  # (1, 1, d_model)
                    else:
                        hidden_b = current_hidden[b:b+1]  # (1, 1, d_model)
                    should_stop_audio = False
                    if self.audio_generator is not None:
                        coda_out = self.audio_generator(
                            hidden_b,
                            kv_caches=audio_coda_kv_caches[b],
                            position_offset=audio_coda_position_offsets[b],
                            use_cache=True,
                        )
                        audio_coda_kv_caches[b] = coda_out.get("kv_caches")
                        audio_coda_position_offsets[b] += 1
                        frame_pred = coda_out["audio_latent_preds"]  # (1, C, 1)
                        audio_sequences[b].append(frame_pred.squeeze(0))  # (C, 1)
                        last_audio_pred[b] = frame_pred.squeeze(0)  # (C, 1)
                        # Check stop probability
                        stop_logit = coda_out["audio_stop_logits"]  # (1, 1)
                        stop_logit_val = stop_logit[0, 0].item()
                        audio_stop_logit_trace[b].append(stop_logit_val)
                        if torch.sigmoid(torch.tensor(stop_logit_val)).item() > 0.5:
                            should_stop_audio = True
                    else:
                        audio_sequences[b].append(current_hidden[b])

                    # Stop on predicted stop or hard budget
                    if should_stop_audio or len(audio_sequences[b]) >= audio_token_budget:
                        current_modality[b] = None
                        if self.audio_generator is not None:
                            audio_pred = torch.cat(audio_sequences[b], dim=-1)  # (C, T)
                            completed_audio[b].append(audio_pred)
                        audio_sequences[b] = []
                        audio_prelude_kv_caches[b] = None
                        audio_prelude_position_offsets[b] = 0
                        audio_coda_kv_caches[b] = None
                        audio_coda_position_offsets[b] = 0
                        last_audio_pred[b] = None
                        # Inject AUDIO_PLACEHOLDER into text_prelude KV cache so
                        # EOA's causal attention in iter N+1 sees APH between
                        # BOA and EOA (matching training [..., BOA, APH, EOA]).
                        _, text_prelude_kv_caches = self.text_feature_extractor(
                            torch.tensor([[constants.AUDIO_PLACEHOLDER_TOKEN_ID]], device=device),
                            kv_caches=text_prelude_kv_caches,
                            position_offset=text_prelude_position_offset,
                            use_cache=True,
                        )
                        text_prelude_position_offset += 1
                        forced_next_token[b] = constants.EOA_TOKEN_ID
                        just_finalized_streaming[b] = "audio"

                elif current_modality[b] == "voice":
                    # Run voice coda autoregressively with KV cache.
                    # On mid-generation BOV entry (just_entered_streaming=="voice"),
                    # the coda's first input must be recurrent(zero_vec), not
                    # recurrent(BOV_embed) — matching training where position 0
                    # of a voice block is a literal zero vector in d_model space
                    # (world_model.py:291-294). For trailing-BOV prompts, this
                    # path is already handled naturally by the line-896 zero_vec
                    # in iter 0; here we emulate the same behavior when BOV is
                    # sampled mid-generation so both entry paths converge.
                    if just_entered_streaming[b] == "voice":
                        d_model_ = self.config.text_prelude_config.d_model
                        zero_hidden = torch.zeros(
                            1, 1, d_model_, device=device, dtype=current_hidden.dtype,
                        )
                        entry_hidden, kv_cache, _, _, _ = self.recurrent_block(
                            zero_hidden * self.embed_scale,
                            attention_mask=None,
                            kv_cache=kv_cache,
                            position_offset=position_offset,
                            use_cache=True,
                            share_kv_cache=share_kv_cache,
                        )
                        position_offset += 1
                        hidden_b = entry_hidden  # (1, 1, d_model)
                    else:
                        hidden_b = current_hidden[b:b+1]  # (1, 1, d_model)
                    should_stop_voice = False
                    if self.voice_generator is not None:
                        coda_out = self.voice_generator(
                            hidden_b,
                            kv_caches=voice_coda_kv_caches[b],
                            position_offset=voice_coda_position_offsets[b],
                            use_cache=True,
                        )
                        voice_coda_kv_caches[b] = coda_out.get("kv_caches")
                        voice_coda_position_offsets[b] += 1
                        frame_pred = coda_out["voice_latent_preds"]  # (1, C, 1)
                        voice_sequences[b].append(frame_pred.squeeze(0))  # (C, 1)
                        last_voice_pred[b] = frame_pred.squeeze(0)  # (C, 1)
                        # Check stop probability
                        stop_logit = coda_out["voice_stop_logits"]  # (1, 1)
                        stop_logit_val = stop_logit[0, 0].item()
                        voice_stop_logit_trace[b].append(stop_logit_val)
                        if torch.sigmoid(torch.tensor(stop_logit_val)).item() > 0.5:
                            should_stop_voice = True
                    else:
                        voice_sequences[b].append(current_hidden[b])

                    # Stop on predicted stop or hard budget
                    if should_stop_voice or len(voice_sequences[b]) >= voice_token_budget:
                        current_modality[b] = None
                        if self.voice_generator is not None:
                            voice_pred = torch.cat(voice_sequences[b], dim=-1)  # (C, T)
                            completed_voice[b].append(voice_pred)
                        voice_sequences[b] = []
                        voice_prelude_kv_caches[b] = None
                        voice_prelude_position_offsets[b] = 0
                        voice_coda_kv_caches[b] = None
                        voice_coda_position_offsets[b] = 0
                        last_voice_pred[b] = None
                        # Inject VOICE_PLACEHOLDER into text_prelude KV cache so
                        # EOV's causal attention in iter N+1 sees VPH between
                        # BOV and EOV (matching training [..., BOV, VPH, EOV]).
                        _, text_prelude_kv_caches = self.text_feature_extractor(
                            torch.tensor([[constants.VOICE_PLACEHOLDER_TOKEN_ID]], device=device),
                            kv_caches=text_prelude_kv_caches,
                            position_offset=text_prelude_position_offset,
                            use_cache=True,
                        )
                        text_prelude_position_offset += 1
                        forced_next_token[b] = constants.EOV_TOKEN_ID
                        just_finalized_streaming[b] = "voice"

                elif current_modality[b] == "image":
                    # Single-shot image generation: feed generation queries through
                    # recurrent block and decode with cross-attention decoder.
                    if hasattr(self, 'image_gen_pos_embedding'):
                        if hasattr(self, 'image_gen_queries'):
                            image_input = self.image_gen_pos_embedding(self.image_gen_queries)[:1].to(device=device, dtype=current_hidden.dtype)
                        else:
                            image_input = self.image_gen_pos_embedding.pe[:1].to(device=device, dtype=current_hidden.dtype)
                    else:
                        d_model = current_hidden.shape[-1]
                        image_input = torch.zeros(
                            1, image_token_budget, d_model,
                            device=device, dtype=current_hidden.dtype,
                        )
                    # Run through recurrent block (single pass, all 256 at once)
                    image_hidden, _, image_iters, _, _ = self.recurrent_block(
                        image_input * self.embed_scale,
                        attention_mask=None,
                        kv_cache=kv_cache,
                        position_offset=position_offset,
                        use_cache=True,
                        share_kv_cache=share_kv_cache,
                        max_iterations_override=image_iteration_override,
                    )
                    image_recurrent_iterations[b].append(int(image_iters))
                    position_offset += image_token_budget

                    # Decode through cross-attention decoder
                    if self.image_generator is not None:
                        cross_input = self.image_coda_input_norm(image_hidden)
                        gen_kwargs = {}
                        if isinstance(self.image_generator, DiffusionBridgeImageDecoder):
                            if image_num_inference_steps is not None:
                                gen_kwargs["num_inference_steps"] = image_num_inference_steps
                            if image_sampler is not None:
                                gen_kwargs["sampler"] = image_sampler
                        cross_out = self.image_generator(
                            encoder_hidden_states=cross_input,
                            **gen_kwargs,
                        )
                        image_pred = cross_out["image_latent_preds"].squeeze(0)  # (C, H, W)
                    else:
                        image_pred = None

                    if image_pred is not None:
                        completed_image[b].append(image_pred)
                    current_modality[b] = None
                    # Defer EOI emission to the shared text-coda path below,
                    # and first inject IMAGE_PLACEHOLDER into text_prelude's KV
                    # cache so EOI's causal self-attention in iter N+1 sees
                    # IPH between BOI and EOI — matching training's text
                    # sequence [..., BOI, IPH, EOI, ...]. The prelude's IPH
                    # embedding is discarded (the interleaver strips it at
                    # training too); we only need it in the causal KV cache.
                    _, text_prelude_kv_caches = self.text_feature_extractor(
                        torch.tensor([[constants.IMAGE_PLACEHOLDER_TOKEN_ID]], device=device),
                        kv_caches=text_prelude_kv_caches,
                        position_offset=text_prelude_position_offset,
                        use_cache=True,
                    )
                    text_prelude_position_offset += 1
                    forced_next_token[b] = constants.EOI_TOKEN_ID

            # Voice/audio ENTRY: run text_coda once on BO*'s current_hidden to
            # add BO* to the text coda's KV cache, matching training (where the
            # uninterleaver feeds the text coda the BO* position). Without this,
            # the shared call below is skipped (any_media=True during streaming)
            # and BO* is silently dropped from the text coda's view.
            if any(just_entered_streaming[b] is not None for b in range(batch_size)):
                entry_coda_out = self.text_generator(
                    current_hidden,  # BO*'s recurrent output
                    kv_caches=text_coda_kv_caches,
                    position_offset=text_coda_position_offset,
                    use_cache=True,
                )
                text_coda_kv_caches = entry_coda_out.get("kv_caches")
                text_coda_position_offset += 1
                # Logits from this call are the BO*-position predictions (trained
                # to predict EO*). We don't sample; we're staying in media mode.
                # Note: text_coda KV is shared across batch — correct handling
                # for mixed batches (some entering, some not) would require
                # per-item KV slices. This path effectively assumes batch=1.

            # Get logits for next token. Only run the text coda when ALL batch
            # items are in text mode. During voice/audio/image generation the
            # text coda should NOT see these positions — during training it only
            # sees text positions (the uninterleaver strips media). Feeding media
            # hidden states through the text coda would pollute its KV cache with
            # out-of-distribution entries.
            any_media = any(current_modality[b] is not None for b in range(batch_size))
            # Skip shared text_coda at voice/audio finalization iters:
            # current_hidden holds the LAST streaming-frame's recurrent output,
            # which is a MODALITY_VOICE/AUDIO position — OOD for the text coda.
            # EO* will be emitted directly via forced_next_token below and
            # processed by text_coda in the next iter (when EO* is fed as the
            # regular next token).
            skip_shared_coda = any(just_finalized_streaming[b] is not None for b in range(batch_size))
            if not any_media and not skip_shared_coda:
                text_coda_output = self.text_generator(
                    current_hidden,
                    kv_caches=text_coda_kv_caches,
                    position_offset=text_coda_position_offset,
                    use_cache=True,
                )
                text_coda_kv_caches = text_coda_output.get("kv_caches")
                text_coda_position_offset += 1
                logits = text_coda_output["logits"][:, 0, :]  # (batch, vocab_size)
                all_logits.append(logits.unsqueeze(1))

            # Sample / emit next tokens
            if not any_media:
                if skip_shared_coda:
                    # Finalizing iter: emit forced EO* directly, no sampling.
                    forced_ids = [forced_next_token[b] if forced_next_token[b] is not None else constants.EOS_TOKEN_ID for b in range(batch_size)]
                    next_token_ids = torch.tensor(forced_ids, device=device)
                    for b in range(batch_size):
                        generated_tokens[b].append(next_token_ids[b].item())
                else:
                    sampled = self._sample_tokens(logits, temperature, top_k, top_p)
                    # Override sample with forced EO* for batch items that just
                    # finalized a media block (e.g. image) — this turns EO*
                    # into the actual next token driving iter N+1, so
                    # text_prelude → recurrent → text_coda all process EO* as
                    # a real position matching training.
                    for b in range(batch_size):
                        if forced_next_token[b] is not None:
                            sampled[b] = forced_next_token[b]
                    next_token_ids = sampled
                    for b in range(batch_size):
                        generated_tokens[b].append(next_token_ids[b].item())

            # Check for EOS — stop generation for batch items that produced it
            if not any_media:
                all_done = True
                for b in range(batch_size):
                    if next_token_ids[b].item() == constants.EOS_TOKEN_ID:
                        finished[b] = True
                    if not finished[b]:
                        all_done = False
                if all_done:
                    break

        # Compile outputs
        outputs: Dict[str, torch.Tensor] = {}

        # Convert generated tokens to tensor
        max_gen_len = max(len(tokens) for tokens in generated_tokens)
        gen_token_tensor = torch.zeros(batch_size, max_gen_len, dtype=torch.long, device=device)
        for b in range(batch_size):
            gen_token_tensor[b, :len(generated_tokens[b])] = torch.tensor(
                generated_tokens[b], device=device
            )
        outputs["generated_token_ids"] = gen_token_tensor

        # Stack logits
        outputs["text_logits"] = torch.cat(all_logits, dim=1)  # (batch, seq, vocab)

        # Recurrent iteration counts and KL divergences per generated token
        outputs["recurrent_iteration_counts"] = recurrent_iteration_counts
        outputs["recurrent_kl_final"] = recurrent_kl_final
        outputs["prompt_recurrent_iterations"] = prompt_iters
        outputs["prompt_recurrent_kl"] = prompt_kls

        # Collect completed modality outputs
        # Stack into padded tensors with counts and lengths for proper unpadding before VAE decoding
        # Audio/voice: variable time dimension needs lengths
        # Image: fixed spatial size, just needs counts
        if any(len(audios) > 0 for audios in completed_audio):
            stacked, counts, lengths = self._stack_variable_length_media(
                completed_audio, device, time_dim=-1
            )
            outputs["audio_latent_preds"] = stacked  # (batch, max_n, C, H, max_T)
            outputs["audio_counts"] = counts  # (batch,)
            outputs["audio_lengths"] = lengths  # (batch, max_n)

        if any(len(voices) > 0 for voices in completed_voice):
            stacked, counts, lengths = self._stack_variable_length_media(
                completed_voice, device, time_dim=-1
            )
            outputs["voice_latent_preds"] = stacked  # (batch, max_n, C, H, max_T)
            outputs["voice_counts"] = counts  # (batch,)
            outputs["voice_lengths"] = lengths  # (batch, max_n)

        if any(len(images) > 0 for images in completed_image):
            stacked, counts, lengths = self._stack_variable_length_media(
                completed_image, device, time_dim=None  # Images have fixed spatial size
            )
            outputs["image_latent_preds"] = stacked  # (batch, max_n, C, H, W)
            outputs["image_counts"] = counts  # (batch,)
            # No lengths needed for images since spatial dims are fixed

        # Per-image recurrent iteration counts (list-of-list; one entry per
        # completed image per batch item). Empty list if no images generated.
        outputs["image_recurrent_iterations"] = image_recurrent_iterations

        # Per-frame stop-head logit traces (diagnostic — see init comment).
        # List-of-list Python objects (not tensors) since lengths vary per
        # batch item. Empty list when no voice/audio was generated.
        outputs["voice_stop_logit_trace"] = voice_stop_logit_trace
        outputs["audio_stop_logit_trace"] = audio_stop_logit_trace

        return outputs

    def _sample_tokens(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> torch.Tensor:
        """Sample tokens from logits with temperature, top-k, and top-p."""
        if temperature != 1.0:
            logits = logits / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Scatter sorted tensors back to original order
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _stack_optional_tensors(
        self,
        tensors: List[Optional[torch.Tensor]],
        device: torch.device,
    ) -> torch.Tensor:
        """Stack list of optional tensors, padding None entries with zeros."""
        # Find a non-None tensor to get the shape
        ref_tensor = None
        for t in tensors:
            if t is not None:
                ref_tensor = t
                break

        if ref_tensor is None:
            return torch.tensor([], device=device)

        result = []
        for t in tensors:
            if t is None:
                result.append(torch.zeros_like(ref_tensor))
            else:
                result.append(t)

        return torch.stack(result, dim=0)

    def _stack_variable_length_media(
        self,
        media_lists: List[List[torch.Tensor]],
        device: torch.device,
        time_dim: Optional[int] = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Stack variable-length media tensors into a padded batch tensor.

        Args:
            media_lists: List of lists, where media_lists[b] contains tensors for batch item b.
                Each tensor has shape (C, H, T) for audio/voice or (C, H, W) for images.
            device: Device for output tensors.
            time_dim: Dimension that varies in length (-1 for audio/voice time dim).
                If None, assumes fixed size (images) and no length tracking needed.

        Returns:
            Tuple of:
            - stacked: Padded tensor of shape (batch, max_n, C, H, max_T) or (batch, max_n, C, H, W)
            - counts: Tensor of shape (batch,) with number of media per batch item
            - lengths: Tensor of shape (batch, max_n) with actual length of each media along time_dim.
                For fixed-size media (time_dim=None), returns zeros.
        """
        batch_size = len(media_lists)
        counts = torch.tensor([len(m) for m in media_lists], dtype=torch.long, device=device)
        max_n = max(len(m) for m in media_lists) if any(media_lists) else 0

        if max_n == 0:
            # No media generated
            return (
                torch.tensor([], device=device),
                counts,
                torch.zeros(batch_size, 0, dtype=torch.long, device=device),
            )

        # Find reference tensor for shape and compute max length along time_dim
        ref_tensor = None
        max_time = 0
        all_lengths = []

        for b in range(batch_size):
            batch_lengths = []
            for tensor in media_lists[b]:
                if ref_tensor is None:
                    ref_tensor = tensor
                if time_dim is not None:
                    length = tensor.shape[time_dim]
                    max_time = max(max_time, length)
                    batch_lengths.append(length)
                else:
                    batch_lengths.append(0)  # Fixed size, no length tracking
            # Pad batch_lengths to max_n
            while len(batch_lengths) < max_n:
                batch_lengths.append(0)
            all_lengths.append(batch_lengths)

        lengths = torch.tensor(all_lengths, dtype=torch.long, device=device)  # (batch, max_n)

        # Determine output shape
        if time_dim is not None:
            # Variable length (audio/voice): pad time dimension
            # ref_tensor shape: (C, H, T) -> output: (batch, max_n, C, H, max_T)
            base_shape = list(ref_tensor.shape)
            base_shape[time_dim] = max_time
            output_shape = [batch_size, max_n] + base_shape
        else:
            # Fixed size (images): no padding needed
            # ref_tensor shape: (C, H, W) -> output: (batch, max_n, C, H, W)
            output_shape = [batch_size, max_n] + list(ref_tensor.shape)

        stacked = torch.zeros(output_shape, dtype=ref_tensor.dtype, device=device)

        # Fill in the tensors
        for b in range(batch_size):
            for n, tensor in enumerate(media_lists[b]):
                if time_dim is not None:
                    # Pad along time dimension
                    length = tensor.shape[time_dim]
                    # Create slice for the time dimension
                    slices = [slice(None)] * tensor.dim()
                    slices[time_dim] = slice(0, length)
                    # stacked[b, n, :, :, :length] = tensor
                    stacked[b, n][tuple(slices)] = tensor
                else:
                    stacked[b, n] = tensor

        return stacked, counts, lengths
