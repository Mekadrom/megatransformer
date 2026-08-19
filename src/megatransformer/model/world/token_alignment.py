import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn

from megatransformer.config.world.world_model import TokenInterleaverConfig


# Modality type constants for the modality map
MODALITY_TEXT = 0
MODALITY_AUDIO = 1
MODALITY_VOICE = 2
MODALITY_IMAGE = 3
MODALITY_PAD = -1


# ── NAR→AR voice-attention curriculum (Variant B) ──────────────────────────────
# The world-TTS voice AR leans on the voice-history CRUTCH (earlier voice frames
# predict the next one via local acoustic smoothness), so the text is under-used.
# The curriculum starves that crutch by DOWN-SCALING voice→voice attention by a
# schedulable factor alpha in the prelude, recurrent trunk, and coda: at alpha=0 a
# voice position attends to text (and its own shifted-TF input frame) ONLY; alpha
# ramps 0→1 to hand history back once the text pathway is established.
#
# Mechanism: add log(alpha) to the QK score of every VOICE-query × VOICE-key pair
# (off-diagonal only — a position always keeps its own diagonal so no softmax row is
# fully masked). Adding log(alpha) multiplies that pair's pre-normalization softmax
# weight by alpha. alpha>=1 => bias 0 (identity, callers pass None for the fast path).

# "Fully severed" sentinel for alpha==0. A large FINITE negative (not -inf) so it is
# representable in fp16/bf16 and never produces NaN from inf-inf inside a fused
# attention kernel; exp(-1e4) underflows to 0 all the same.
VOICE_ATTN_MASK_NEG = -1e4


def voice_attn_bias_value(alpha: Optional[float]) -> float:
    """Per-pair additive logit that scales a voice→voice softmax weight by ``alpha``."""
    if alpha is None or alpha >= 1.0:
        return 0.0
    if alpha <= 0.0:
        return VOICE_ATTN_MASK_NEG
    return math.log(alpha)


def build_voice_voice_attn_bias(
    modality_map: torch.Tensor,
    alpha: Optional[float],
    dtype: torch.dtype,
    is_synthesis: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """(B, S, S) additive attention bias down-scaling VOICE-query × VOICE-key attention
    (off-diagonal) by ``alpha`` for the interleaved recurrent trunk.

    The diagonal is left at 0 so a voice row is never fully masked (a fully-masked row
    would NaN the softmax). ``is_synthesis`` (B,) gates it per row: transcription rows
    read real audio as INPUT, so their voice attention is left untouched (bias 0).
    Returns None when ``alpha >= 1`` (no-op) so callers keep the fast attention path.
    """
    if alpha is None or alpha >= 1.0:
        return None
    val = voice_attn_bias_value(alpha)
    voice = modality_map == MODALITY_VOICE                       # (B, S)
    pair = voice.unsqueeze(2) & voice.unsqueeze(1)              # (B, S, S): voice_i & voice_j
    S = modality_map.shape[1]
    eye = torch.eye(S, dtype=torch.bool, device=modality_map.device).unsqueeze(0)
    offdiag = pair & ~eye
    bias = offdiag.to(dtype) * val                              # (B, S, S)
    if is_synthesis is not None:
        gate = is_synthesis.to(dtype).view(-1, 1, 1)           # 0 for transcription rows
        bias = bias * gate
    return bias


def build_all_voice_attn_bias(
    seq_len: int,
    alpha: Optional[float],
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 1,
    is_synthesis: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """(B, T, T) additive bias for an all-voice sequence (prelude / coda): off-diagonal
    entries get log(alpha), the diagonal stays 0. With ``is_synthesis`` (B,) the bias is
    zeroed on transcription rows. Returns None when ``alpha >= 1`` (no-op).

    Padding positions need no special handling: these attentions are causal, so a valid
    query never attends to trailing padding, and padded query rows are discarded.
    """
    if alpha is None or alpha >= 1.0:
        return None
    val = voice_attn_bias_value(alpha)
    m = torch.full((seq_len, seq_len), val, device=device, dtype=dtype)
    m.fill_diagonal_(0.0)
    bias = m.unsqueeze(0).expand(batch_size, -1, -1)           # (B, T, T)
    if is_synthesis is not None:
        bias = bias * is_synthesis.to(dtype).view(-1, 1, 1)
    return bias.contiguous()


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
        # Vectorized masked left-pack. The former per-batch boolean-index loop cost ~4*B
        # GPU->CPU syncs per step (one per modality per sample, because a boolean index's
        # output size is data-dependent) and stalled the pipeline. This is one scatter per
        # modality with a SINGLE host sync total (all four max-lengths read at once), and is
        # bit-identical: boolean indexing returns elements in ascending-position order, which
        # is exactly the order a cumsum-derived slot assigns.
        mods = (
            ("text", MODALITY_TEXT),
            ("audio", MODALITY_AUDIO),
            ("voice", MODALITY_VOICE),
            ("image", MODALITY_IMAGE),
        )
        masks = {name: (modality_map == mod) for name, mod in mods}
        lengths = {name: m.sum(1) for name, m in masks.items()}         # each (B,), long
        # One sync for all four max lengths instead of 4*B.
        max_lens = {
            name: int(v) for name, v in
            zip([n for n, _ in mods], torch.stack([lengths[n].max() for n, _ in mods]).tolist())
        }

        packed = {}
        for name, _ in mods:
            packed[name] = self._masked_left_pack(
                interleaved_tokens, masks[name], lengths[name], max_lens[name])

        return {
            "text": packed["text"], "text_lengths": lengths["text"] if packed["text"] is not None else None,
            "audio": packed["audio"], "audio_lengths": lengths["audio"] if packed["audio"] is not None else None,
            "voice": packed["voice"], "voice_lengths": lengths["voice"] if packed["voice"] is not None else None,
            "image": packed["image"], "image_lengths": lengths["image"] if packed["image"] is not None else None,
        }

    @staticmethod
    def _masked_left_pack(tokens: torch.Tensor, mask: torch.Tensor,
                          length: torch.Tensor, max_len: int) -> Optional[torch.Tensor]:
        """Left-pack the mask-selected rows of ``tokens`` into (B, max_len, d), zeros
        beyond each row's length. Returns None when no sample has any such token (matching
        the old _pad_and_stack, which returned None for an all-empty modality).

        Fully vectorized: a per-True-position destination slot is the running count of
        selected elements (cumsum), and a single scatter places every token; unselected
        positions scatter to a throwaway column that is then sliced off. No per-sample loop
        and no ``nonzero`` (whose output size would itself force a sync)."""
        if max_len == 0:
            return None
        B, S, d = tokens.shape
        slots = mask.long().cumsum(1) - 1                              # (B, S); >=0 at True
        # Unselected positions -> the throwaway column (index max_len), discarded below.
        target = torch.where(mask, slots, slots.new_full((), max_len))
        out = tokens.new_zeros(B, max_len + 1, d)
        out.scatter_(1, target.unsqueeze(-1).expand(-1, -1, d), tokens)
        return out[:, :max_len].contiguous()


def build_mrope_position_ids(modality_map: torch.Tensor, voice_rate: float = 6.0,
                             scale_side: str = "voice") -> torch.Tensor:
    """Per-token (GLOBAL, LOCAL) coordinates for M-RoPE, from the interleaved modality map.

    GLOBAL: strictly increasing over non-pad positions. This is what keeps multiple media
    examples distinguishable — in `<text> <voice_0> <text> <voice_1> <text>` the two voice
    segments would otherwise collide on the local axis (voice_0 frame 5 and voice_1 frame 5
    are positionally identical), so voice->voice attention could not tell them apart and a
    "compare the two samples" finetune would have no positional handle.

    LOCAL: index WITHIN the current contiguous same-modality segment, and for media segments
    scaled by 1/voice_rate. That is the point of the whole exercise: RoPE encodes position
    DIFFERENCES, not ratios, so putting a voice frame at frame_index/rate on the text's clock
    makes an aligned (text token j, voice frame t) pair sit at relative distance ~0, where
    RoPE's locality bias can act. Under the current single global axis that pair is separated
    by L_text + 0.83*t — large, growing with t, and different for every utterance, which is
    why only one attention head (block 0 head 4) manages to align at all.

    scale_side picks WHICH stream absorbs the rate, and the two are different RoPE regimes:
      "voice" — text j, voice t/rate. Aligned pairs at ~0, but voice frames end up at
                FRACTIONAL sub-unit spacing (1/6 apart), which is a regime RoPE is
                essentially never used in, for the 250-position stream being generated.
      "text"  — text j*rate, voice t. Same alignment property, but both streams stay at
                INTEGER spacing; text simply strides by `rate`. High-frequency aliasing at
                stride 6 is ordinary RoPE behaviour (its fast dims alias constantly at normal
                lengths and the slow dims disambiguate).

    Returns: (batch, seq_len, 2) float — [..., 0] global, [..., 1] local.
    """
    B, S = modality_map.shape
    dev = modality_map.device
    valid = modality_map != MODALITY_PAD
    # GLOBAL: running count of valid positions (pad contributes nothing).
    g = (valid.long().cumsum(dim=1) - 1).clamp(min=0).float()
    # LOCAL: restart at each modality change; media positions advance by 1/voice_rate.
    change = torch.ones_like(modality_map, dtype=torch.bool)
    change[:, 1:] = modality_map[:, 1:] != modality_map[:, :-1]
    seg_id = change.long().cumsum(dim=1)                       # (B, S) segment index
    idx = torch.arange(S, device=dev).unsqueeze(0).expand(B, S)
    # start index of each segment, broadcast back to its positions. cummax over
    # "index where a segment starts, else 0" gives the most recent start at every position --
    # vectorized on purpose: the previous per-batch loop called .max() per row, and each of
    # those is a GPU->CPU sync in what would be the training hot path.
    seg_start = torch.cummax(torch.where(change, idx, torch.zeros_like(idx)), dim=1).values
    within = (idx - seg_start).float()
    is_media = (modality_map != MODALITY_TEXT) & valid
    r = max(voice_rate, 1e-3)
    if scale_side == "text":
        l = torch.where(is_media, within, within * r)
    else:
        l = torch.where(is_media, within / r, within)
    return torch.stack([g, l], dim=-1)
