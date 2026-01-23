import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import Dict, List, Optional

from config.world.world_model import WORLD_MODEL_CONFIGS, MegaTransformerWorldModelConfig
from model.audio.feature_extractor import AudioVAEPreludeFeatureExtractor
from model.audio.generator import AudioCodaAndVAEWithLoss
from model.audio.vae.vae import AudioVAEDecoder, AudioVAEEncoder
from model.image.feature_extractor import ImageVAEPreludeFeatureExtractor
from model.image.generator import ImageCodaAndVAEWithLoss
from model.image.vae.vae import ImageVAEDecoder, ImageVAEEncoder
from model.text import TextFeatureExtractor
from model.text.generator import TextCodaClassifierWithLoss
from model.world.kv_cache import RecurrentKVCache
from model.world.recurrent import MegatransformerRecurrentBlock
from model.world.token_alignment import TokenInterleaver, TokenUninterleaver
from utils import constants


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

        # Main Transformer (Recurrent Block)
        recurrent_output, _ = self.recurrent_block(
            interleaved_tokens,
            attention_mask=attn_mask  # True for attend, False for padding
        )

        # Token Uninterleaving
        uninterleaved = self.token_uninterleaver(recurrent_output, modality_map)

        text_batch = uninterleaved["text"]
        audio_batch = uninterleaved["audio"]
        voice_batch = uninterleaved["voice"]
        image_batch = uninterleaved["image"]

        # Generators/Codas
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
    ) -> Dict[str, torch.Tensor]:
        """
        Generate tokens autoregressively with KV caching.

        This method implements autoregressive generation through the recurrent block
        with efficient KV caching using the Huginn approach (circular buffer).

        The generation flow:
        1. Process initial text input through text feature extractor
        2. Autoregressively sample tokens through the recurrent block
        3. When end-of-modality tokens (EOA/EOV/EOI) are detected, collect the
           complete modality sequence and pass it to the appropriate coda
        4. Continue generating until max_new_tokens or end-of-sequence

        Args:
            text_input_ids: Initial text prompt token IDs, shape (batch, prompt_len).
                Can contain placeholder tokens for media that will be generated.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature. Higher = more random.
            top_k: If set, only sample from top k most likely tokens.
            top_p: If set, use nucleus sampling with this probability mass.
            kv_cache_strategy: "huginn" (shared cache, efficient) or "per_iteration".
            kv_cache_budget: Number of cache slots for Huginn strategy.
            decode_outputs: If True, decode generated latents via VAE decoders.

        Returns:
            Dictionary containing:
            - "generated_token_ids": Generated text token IDs, shape (batch, seq_len)
            - "text_logits": Logits for text tokens, shape (batch, seq_len, vocab_size)
            - "audio_latent_preds": Padded tensor of shape (batch, max_n_audio, C, H, max_T)
            - "audio_counts": Number of audio clips per batch item, shape (batch,)
            - "audio_lengths": Actual time length of each audio, shape (batch, max_n_audio).
                Use these lengths to slice before VAE decoding to avoid decoding padding.
            - "voice_latent_preds": Padded tensor of shape (batch, max_n_voice, C, H, max_T)
            - "voice_counts": Number of voice clips per batch item, shape (batch,)
            - "voice_lengths": Actual time length of each voice, shape (batch, max_n_voice)
            - "image_latent_preds": Padded tensor of shape (batch, max_n_image, C, H, W)
            - "image_counts": Number of images per batch item, shape (batch,)
            - Decoded outputs if decode_outputs=True
        """
        batch_size = text_input_ids.shape[0]
        device = text_input_ids.device

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

        # Track which modality we're currently generating for each batch item
        # None = text, "audio", "voice", "image"
        current_modality: List[Optional[str]] = [None for _ in range(batch_size)]

        # Completed modality outputs (list of tensors per batch item to support multiple media)
        completed_audio: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
        completed_voice: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
        completed_image: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]

        # Process initial prompt
        # Get embeddings for prompt
        prompt_hidden = self.text_feature_extractor(text_input_ids)  # (batch, prompt_len, d_model)

        # Process through recurrent block to get initial context
        current_hidden, kv_cache = self.recurrent_block(
            prompt_hidden,
            attention_mask=None,
            kv_cache=kv_cache,
            position_offset=0,
            use_cache=True,
        )

        position_offset = text_input_ids.shape[1]

        # Get initial logits from text coda for the last position
        last_hidden = current_hidden[:, -1:, :]  # (batch, 1, d_model)
        text_output = self.text_generator(last_hidden)
        logits = text_output["logits"][:, 0, :]  # (batch, vocab_size)
        all_logits.append(logits.unsqueeze(1))

        # Sample first token
        next_token_ids = self._sample_tokens(logits, temperature, top_k, top_p)

        for b in range(batch_size):
            generated_tokens[b].append(next_token_ids[b].item())

        # Autoregressive generation loop
        for _ in range(max_new_tokens - 1):
            # Check for modality transitions and handle accordingly
            next_hidden_list = []

            for b in range(batch_size):
                token_id = next_token_ids[b].item()

                # Check for begin-of-modality tokens
                if token_id == constants.BOA_TOKEN_ID:
                    current_modality[b] = "audio"
                    # Embed the BOA token
                    token_embed = self.text_feature_extractor(
                        torch.tensor([[token_id]], device=device)
                    )  # (1, 1, d_model)
                    next_hidden_list.append(token_embed[0])

                elif token_id == constants.BOV_TOKEN_ID:
                    current_modality[b] = "voice"
                    token_embed = self.text_feature_extractor(
                        torch.tensor([[token_id]], device=device)
                    )
                    next_hidden_list.append(token_embed[0])

                elif token_id == constants.BOI_TOKEN_ID:
                    current_modality[b] = "image"
                    token_embed = self.text_feature_extractor(
                        torch.tensor([[token_id]], device=device)
                    )
                    next_hidden_list.append(token_embed[0])

                # Check for end-of-modality tokens
                elif token_id == constants.EOA_TOKEN_ID and current_modality[b] == "audio":
                    current_modality[b] = None
                    # Process accumulated audio sequence through audio coda
                    if audio_sequences[b]:
                        audio_hidden = torch.cat(audio_sequences[b], dim=0)  # (seq, d_model)
                        audio_hidden = audio_hidden.unsqueeze(0)  # (1, seq, d_model)
                        audio_out = self.audio_generator(
                            audio_hidden,
                            decode_to_mel=decode_outputs,
                        )
                        audio_pred = audio_out.get("audio_latent_preds")
                        if audio_pred is not None:
                            completed_audio[b].append(audio_pred)
                        audio_sequences[b] = []
                    # Embed EOA token
                    token_embed = self.text_feature_extractor(
                        torch.tensor([[token_id]], device=device)
                    )
                    next_hidden_list.append(token_embed[0])

                elif token_id == constants.EOV_TOKEN_ID and current_modality[b] == "voice":
                    current_modality[b] = None
                    if voice_sequences[b]:
                        voice_hidden = torch.cat(voice_sequences[b], dim=0)
                        voice_hidden = voice_hidden.unsqueeze(0)
                        voice_out = self.voice_generator(
                            voice_hidden,
                            decode_to_mel=decode_outputs,
                        )
                        voice_pred = voice_out.get("voice_latent_preds")
                        if voice_pred is not None:
                            completed_voice[b].append(voice_pred)
                        voice_sequences[b] = []
                    token_embed = self.text_feature_extractor(
                        torch.tensor([[token_id]], device=device)
                    )
                    next_hidden_list.append(token_embed[0])

                elif token_id == constants.EOI_TOKEN_ID and current_modality[b] == "image":
                    current_modality[b] = None
                    if image_sequences[b]:
                        image_hidden = torch.cat(image_sequences[b], dim=0)
                        image_hidden = image_hidden.unsqueeze(0)
                        image_out = self.image_generator(
                            image_hidden,
                            decode_to_image=decode_outputs,
                        )
                        image_pred = image_out.get("image_latent_preds")
                        if image_pred is not None:
                            completed_image[b].append(image_pred)
                        image_sequences[b] = []
                    token_embed = self.text_feature_extractor(
                        torch.tensor([[token_id]], device=device)
                    )
                    next_hidden_list.append(token_embed[0])

                else:
                    # Regular token - embed it
                    token_embed = self.text_feature_extractor(
                        torch.tensor([[token_id]], device=device)
                    )
                    next_hidden_list.append(token_embed[0])

            # Stack embeddings for all batch items: (batch, 1, d_model)
            next_hidden = torch.stack(next_hidden_list, dim=0)

            # Process through recurrent block with KV cache
            current_hidden, kv_cache = self.recurrent_block(
                next_hidden,
                attention_mask=None,
                kv_cache=kv_cache,
                position_offset=position_offset,
                use_cache=True,
            )
            position_offset += 1

            # Accumulate hidden states for non-text modalities
            for b in range(batch_size):
                if current_modality[b] == "audio":
                    audio_sequences[b].append(current_hidden[b])  # (1, d_model)
                elif current_modality[b] == "voice":
                    voice_sequences[b].append(current_hidden[b])
                elif current_modality[b] == "image":
                    image_sequences[b].append(current_hidden[b])

            # Get logits for next token
            text_output = self.text_generator(current_hidden)
            logits = text_output["logits"][:, 0, :]  # (batch, vocab_size)
            all_logits.append(logits.unsqueeze(1))

            # Sample next tokens
            next_token_ids = self._sample_tokens(logits, temperature, top_k, top_p)

            for b in range(batch_size):
                generated_tokens[b].append(next_token_ids[b].item())

            # Check for EOS (could add EOS_TOKEN_ID check here)

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
