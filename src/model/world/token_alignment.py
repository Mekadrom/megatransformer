from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn


# Modality type constants for the modality map
MODALITY_TEXT = 0
MODALITY_AUDIO = 1
MODALITY_VOICE = 2
MODALITY_IMAGE = 3
MODALITY_PAD = -1


class TokenInterleaver(nn.Module):
    """Interleaves text and media token embeddings into a single sequence.

    Text is the primary driver sequence with placeholder tokens that get replaced
    by the actual media embeddings (audio, voice, image). Placeholder positions
    are automatically detected from the token IDs.

    Args:
        config: TokenInterleaverConfig with placeholder token IDs for each modality.

    Returns:
        Tuple of (interleaved_tokens, attention_mask, modality_map):
        - interleaved_tokens: (batch_size, final_seq_len, d_model)
        - attention_mask: (batch_size, final_seq_len) - True for real tokens, False for padding
        - modality_map: (batch_size, final_seq_len) - indicates modality type per position
    """

    def __init__(self, config: TokenInterleaverConfig):
        super().__init__()
        self.config = config

    def _find_placeholder_positions(
        self,
        token_ids: torch.Tensor,
        placeholder_token_id: int,
    ) -> list:
        """Find positions of placeholder tokens for each batch item.

        Args:
            token_ids: (batch_size, seq_len) tensor of token IDs
            placeholder_token_id: The token ID to search for

        Returns:
            List of lists, where each inner list contains the positions of
            placeholder tokens for that batch item, in order.
        """
        batch_size = token_ids.shape[0]
        positions_per_batch = []

        for batch_idx in range(batch_size):
            # Find all positions where token_id == placeholder_token_id
            mask = token_ids[batch_idx] == placeholder_token_id
            positions = mask.nonzero(as_tuple=True)[0].tolist()
            positions_per_batch.append(positions)

        return positions_per_batch

    def _assertions(
        self,
        text_hidden_states: torch.Tensor,
        text_token_ids: torch.Tensor,
        audio_hidden_states: Optional[torch.Tensor],
        audio_lengths: Optional[torch.Tensor],
        voice_hidden_states: Optional[torch.Tensor],
        voice_lengths: Optional[torch.Tensor],
        image_hidden_states: Optional[torch.Tensor],
    ):
        if text_hidden_states is None:
            raise ValueError("Text hidden states must be provided for interleaving.")
        if text_token_ids is None:
            raise ValueError("Text token IDs must be provided to find placeholder positions.")

        assert text_hidden_states.shape[0] == text_token_ids.shape[0], \
            "Text hidden states and token IDs must have the same batch size."
        assert text_hidden_states.shape[1] == text_token_ids.shape[1], \
            "Text hidden states and token IDs must have the same sequence length."

        if audio_hidden_states is not None:
            assert audio_lengths is not None, \
                "Audio lengths must be provided if audio hidden states are given."
            assert audio_hidden_states.shape[0] == text_hidden_states.shape[0], \
                "Audio hidden states batch size must match text hidden states batch size."
            assert audio_lengths.shape[0] == text_hidden_states.shape[0], \
                "Audio lengths batch size must match text hidden states batch size."
            assert audio_lengths.shape[-1] == audio_hidden_states.shape[1], \
                "Audio lengths must have the same number of examples as audio hidden states."
            if self.config.audio_placeholder_token_id is None:
                raise ValueError("Audio placeholder token ID must be configured to use audio.")

        if voice_hidden_states is not None:
            assert voice_lengths is not None, \
                "Voice lengths must be provided if voice hidden states are given."
            assert voice_hidden_states.shape[0] == text_hidden_states.shape[0], \
                "Voice hidden states batch size must match text hidden states batch size."
            assert voice_lengths.shape[0] == text_hidden_states.shape[0], \
                "Voice lengths batch size must match text hidden states batch size."
            assert voice_lengths.shape[-1] == voice_hidden_states.shape[1], \
                "Voice lengths must have the same number of examples as voice hidden states."
            if self.config.voice_placeholder_token_id is None:
                raise ValueError("Voice placeholder token ID must be configured to use voice.")

        if image_hidden_states is not None:
            assert image_hidden_states.shape[0] == text_hidden_states.shape[0], \
                "Image hidden states batch size must match text hidden states batch size."
            if self.config.image_placeholder_token_id is None:
                raise ValueError("Image placeholder token ID must be configured to use images.")

    def forward(
        self,
        text_hidden_states: torch.Tensor,
        text_token_ids: torch.Tensor,
        audio_hidden_states: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None,
        voice_hidden_states: Optional[torch.Tensor] = None,
        voice_lengths: Optional[torch.Tensor] = None,
        image_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Interleave text and media token embeddings into a single sequence.

        Placeholder positions are automatically detected from text_token_ids by
        scanning for the configured placeholder token IDs. Each placeholder is
        replaced with the corresponding media embeddings in order.

        Args:
            text_hidden_states: Text token embeddings, shape (batch_size, text_seq_len, d_model).
                This is the driver sequence containing placeholder tokens for media.
            text_token_ids: Token IDs for the text sequence, shape (batch_size, text_seq_len).
                Used to find placeholder positions.
            audio_hidden_states: Audio token embeddings, shape (batch_size, n_audio_examples, audio_seq_len, d_model)
            audio_lengths: Actual lengths of each audio example, shape (batch_size, n_audio_examples)
            voice_hidden_states: Voice token embeddings, shape (batch_size, n_voice_examples, voice_seq_len, d_model)
            voice_lengths: Actual lengths of each voice example, shape (batch_size, n_voice_examples)
            image_hidden_states: Image patch embeddings, shape (batch_size, n_image_examples, n_patches, d_model)
                Images are fixed size, so no lengths parameter is needed.

        Returns:
            Tuple of:
            - interleaved_tokens: Token embeddings, shape (batch_size, final_seq_len, d_model).
                Padded to max length within batch.
            - attention_mask: Boolean mask, shape (batch_size, final_seq_len).
                True for real tokens, False for padding.
            - modality_map: Integer tensor, shape (batch_size, final_seq_len).
                Indicates modality type: 0=text, 1=audio, 2=voice, 3=image, -1=padding.
        """

        # Initial assertions
        self._assertions(
            text_hidden_states,
            text_token_ids,
            audio_hidden_states,
            audio_lengths,
            voice_hidden_states,
            voice_lengths,
            image_hidden_states,
        )

        batch_size = text_hidden_states.shape[0]
        d_model = text_hidden_states.shape[-1]
        device = text_hidden_states.device
        dtype = text_hidden_states.dtype

        # Find placeholder positions for each modality
        audio_positions = (
            self._find_placeholder_positions(text_token_ids, self.config.audio_placeholder_token_id)
            if audio_hidden_states is not None else [[] for _ in range(batch_size)]
        )
        voice_positions = (
            self._find_placeholder_positions(text_token_ids, self.config.voice_placeholder_token_id)
            if voice_hidden_states is not None else [[] for _ in range(batch_size)]
        )
        image_positions = (
            self._find_placeholder_positions(text_token_ids, self.config.image_placeholder_token_id)
            if image_hidden_states is not None else [[] for _ in range(batch_size)]
        )

        # Verify placeholder counts match example counts
        for batch_idx in range(batch_size):
            if audio_hidden_states is not None:
                expected = audio_hidden_states.shape[1]
                found = len(audio_positions[batch_idx])
                assert found == expected, \
                    f"Batch {batch_idx}: found {found} audio placeholders but have {expected} audio examples"
            if voice_hidden_states is not None:
                expected = voice_hidden_states.shape[1]
                found = len(voice_positions[batch_idx])
                assert found == expected, \
                    f"Batch {batch_idx}: found {found} voice placeholders but have {expected} voice examples"
            if image_hidden_states is not None:
                expected = image_hidden_states.shape[1]
                found = len(image_positions[batch_idx])
                assert found == expected, \
                    f"Batch {batch_idx}: found {found} image placeholders but have {expected} image examples"

        # Shortcut if no media
        if audio_hidden_states is None and voice_hidden_states is None and image_hidden_states is None:
            seq_len = text_hidden_states.shape[1]
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
            modality_map = torch.full((batch_size, seq_len), MODALITY_TEXT, dtype=torch.long, device=device)
            return text_hidden_states, attention_mask, modality_map

        # Process each batch entry
        interleaved_sequences = []
        sequence_lengths = []
        modality_maps = []

        for batch_idx in range(batch_size):
            tokens_to_concat = []
            modalities_to_concat = []

            # Text sequence for this batch: (text_seq_len, d_model)
            text_sequence = text_hidden_states[batch_idx]
            text_seq_len = text_sequence.shape[0]

            # Get media examples for this batch if present
            batch_audio = audio_hidden_states[batch_idx] if audio_hidden_states is not None else None
            batch_voice = voice_hidden_states[batch_idx] if voice_hidden_states is not None else None
            batch_image = image_hidden_states[batch_idx] if image_hidden_states is not None else None

            batch_audio_lens = audio_lengths[batch_idx] if audio_lengths is not None else None
            batch_voice_lens = voice_lengths[batch_idx] if voice_lengths is not None else None

            # Build mapping of position -> (modality, example_idx)
            all_placeholders = {}

            for ex_idx, pos in enumerate(audio_positions[batch_idx]):
                all_placeholders[pos] = ("audio", ex_idx)
            for ex_idx, pos in enumerate(voice_positions[batch_idx]):
                all_placeholders[pos] = ("voice", ex_idx)
            for ex_idx, pos in enumerate(image_positions[batch_idx]):
                all_placeholders[pos] = ("image", ex_idx)

            sorted_positions = sorted(all_placeholders.keys())

            # Interleave text and media
            text_cursor = 0
            for placeholder_pos in sorted_positions:
                # Add text tokens from cursor to placeholder position (excluding placeholder)
                if placeholder_pos > text_cursor:
                    text_chunk = text_sequence[text_cursor:placeholder_pos]  # (chunk_len, d_model)
                    tokens_to_concat.append(text_chunk)
                    modalities_to_concat.append(
                        torch.full((text_chunk.shape[0],), MODALITY_TEXT, dtype=torch.long, device=device)
                    )

                # Skip the placeholder token
                text_cursor = placeholder_pos + 1

                # Add the media tokens
                modality, ex_idx = all_placeholders[placeholder_pos]
                if modality == "audio":
                    length = batch_audio_lens[ex_idx].item()
                    media_chunk = batch_audio[ex_idx, :length]  # (length, d_model)
                    tokens_to_concat.append(media_chunk)
                    modalities_to_concat.append(
                        torch.full((length,), MODALITY_AUDIO, dtype=torch.long, device=device)
                    )
                elif modality == "voice":
                    length = batch_voice_lens[ex_idx].item()
                    media_chunk = batch_voice[ex_idx, :length]  # (length, d_model)
                    tokens_to_concat.append(media_chunk)
                    modalities_to_concat.append(
                        torch.full((length,), MODALITY_VOICE, dtype=torch.long, device=device)
                    )
                elif modality == "image":
                    # Images are fixed size (all patches)
                    media_chunk = batch_image[ex_idx]  # (n_patches, d_model)
                    tokens_to_concat.append(media_chunk)
                    modalities_to_concat.append(
                        torch.full((media_chunk.shape[0],), MODALITY_IMAGE, dtype=torch.long, device=device)
                    )

            # Add any remaining text after the last placeholder
            if text_cursor < text_seq_len:
                text_chunk = text_sequence[text_cursor:]  # (remaining_len, d_model)
                tokens_to_concat.append(text_chunk)
                modalities_to_concat.append(
                    torch.full((text_chunk.shape[0],), MODALITY_TEXT, dtype=torch.long, device=device)
                )

            # Concatenate all chunks for this batch entry
            if tokens_to_concat:
                interleaved_seq = torch.cat(tokens_to_concat, dim=0)  # (seq_len, d_model)
                modality_seq = torch.cat(modalities_to_concat, dim=0)  # (seq_len,)
            else:
                # Edge case: empty sequence
                interleaved_seq = torch.zeros(0, d_model, device=device, dtype=dtype)
                modality_seq = torch.zeros(0, dtype=torch.long, device=device)

            interleaved_sequences.append(interleaved_seq)
            modality_maps.append(modality_seq)
            sequence_lengths.append(interleaved_seq.shape[0])

        # Pad all sequences to max length
        max_seq_len = max(sequence_lengths) if sequence_lengths else 0

        if max_seq_len == 0:
            # Handle empty batch
            return (
                torch.zeros(batch_size, 0, d_model, device=device, dtype=dtype),
                torch.zeros(batch_size, 0, dtype=torch.bool, device=device),
                torch.zeros(batch_size, 0, dtype=torch.long, device=device),
            )

        padded_tokens = []
        padded_modalities = []
        attention_masks = []

        for seq, mod_map, seq_len in zip(interleaved_sequences, modality_maps, sequence_lengths):
            pad_len = max_seq_len - seq_len

            if pad_len > 0:
                # Pad tokens with zeros
                token_pad = torch.zeros(pad_len, d_model, device=device, dtype=dtype)
                padded_seq = torch.cat([seq, token_pad], dim=0)

                # Pad modality map with PAD indicator
                mod_pad = torch.full((pad_len,), MODALITY_PAD, dtype=torch.long, device=device)
                padded_mod = torch.cat([mod_map, mod_pad], dim=0)
            else:
                padded_seq = seq
                padded_mod = mod_map

            padded_tokens.append(padded_seq)
            padded_modalities.append(padded_mod)

            # Create attention mask: True for real tokens, False for padding
            mask = torch.zeros(max_seq_len, dtype=torch.bool, device=device)
            mask[:seq_len] = True
            attention_masks.append(mask)

        # Stack into batch tensors
        interleaved_tokens = torch.stack(padded_tokens, dim=0)  # (batch_size, max_seq_len, d_model)
        attention_mask = torch.stack(attention_masks, dim=0)  # (batch_size, max_seq_len)
        modality_map = torch.stack(padded_modalities, dim=0)  # (batch_size, max_seq_len)

        return interleaved_tokens, attention_mask, modality_map


class TokenUninterleaver(nn.Module):
    """
    Separates interleaved tokens back into modality-specific sequences.

    Uses the modality_map from TokenInterleaver to extract tokens belonging to
    each modality and re-batch them with padding for the respective coda modules.
    """

    def __init__(self):
        super().__init__()

    def _pad_and_stack(
        self,
        tensors: list[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Pad a list of variable-length tensors and stack into a batch.

        Args:
            tensors: List of (seq_len, d_model) tensors, one per batch item.
            device: Device for output tensors.
            dtype: Dtype for output tensors.

        Returns:
            Tuple of:
            - Padded batch tensor (batch_size, max_seq_len, d_model), or None if all empty.
            - Lengths tensor (batch_size,) with actual lengths, or None if all empty.
        """
        lengths = [t.size(0) for t in tensors]
        max_len = max(lengths) if lengths else 0

        if max_len == 0:
            return None, None

        d_model = tensors[0].size(1) if tensors[0].numel() > 0 else 0
        if d_model == 0:
            # Find first non-empty tensor to get d_model
            for t in tensors:
                if t.numel() > 0:
                    d_model = t.size(1)
                    break
            if d_model == 0:
                return None, None

        padded = []
        for t, length in zip(tensors, lengths):
            if length < max_len:
                # Pad (left, right) for last dim, (top, bottom) for second-to-last
                # For 2D tensor (seq, d_model): pad format is (left, right, top, bottom)
                # We want to pad rows (sequence dim), so (0, 0, 0, pad_amount)
                pad_amount = max_len - length
                if length == 0:
                    # Empty tensor - create zeros
                    padded_t = torch.zeros(max_len, d_model, device=device, dtype=dtype)
                else:
                    padded_t = nn.functional.pad(t, (0, 0, 0, pad_amount), value=0.0)
            else:
                padded_t = t
            padded.append(padded_t)

        batch_tensor = torch.stack(padded, dim=0)  # (batch_size, max_seq_len, d_model)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=device)

        return batch_tensor, lengths_tensor

    def forward(
        self,
        interleaved_tokens: torch.Tensor,
        modality_map: torch.Tensor,
    ) -> dict[str, Optional[torch.Tensor]]:
        """
        Separate interleaved tokens back into modality-specific sequences.

        Args:
            interleaved_tokens: (batch_size, seq_len, d_model)
            modality_map: (batch_size, seq_len) - indicates modality type per position

        Returns:
            Dictionary with keys for each modality:
            - 'text', 'audio', 'voice', 'image': Padded batch tensors (batch, max_len, d_model) or None
            - 'text_lengths', 'audio_lengths', etc.: Length tensors (batch,) or None
        """
        batch_size = interleaved_tokens.shape[0]
        device = interleaved_tokens.device
        dtype = interleaved_tokens.dtype

        text_list: list[torch.Tensor] = []
        audio_list: list[torch.Tensor] = []
        voice_list: list[torch.Tensor] = []
        image_list: list[torch.Tensor] = []

        for batch_idx in range(batch_size):
            batch_tokens = interleaved_tokens[batch_idx]  # (seq_len, d_model)
            batch_modality = modality_map[batch_idx]      # (seq_len,)

            # Boolean indexing gives (filtered_len, d_model)
            text_list.append(batch_tokens[batch_modality == MODALITY_TEXT])
            audio_list.append(batch_tokens[batch_modality == MODALITY_AUDIO])
            voice_list.append(batch_tokens[batch_modality == MODALITY_VOICE])
            image_list.append(batch_tokens[batch_modality == MODALITY_IMAGE])

        # Pad and stack each modality
        text_batch, text_lengths = self._pad_and_stack(text_list, device, dtype)
        audio_batch, audio_lengths = self._pad_and_stack(audio_list, device, dtype)
        voice_batch, voice_lengths = self._pad_and_stack(voice_list, device, dtype)
        image_batch, image_lengths = self._pad_and_stack(image_list, device, dtype)

        return {
            "text": text_batch,
            "text_lengths": text_lengths,
            "audio": audio_batch,
            "audio_lengths": audio_lengths,
            "voice": voice_batch,
            "voice_lengths": voice_lengths,
            "image": image_batch,
            "image_lengths": image_lengths,
        }
