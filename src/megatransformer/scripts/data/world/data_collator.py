import random
from typing import Optional

import torch

from megatransformer.scripts.data.data_collator import DataCollator
from megatransformer.utils import constants
from megatransformer.utils.megatransformer_utils import pad_and_mask, trim


def _as_int(v):
    """voice_feature_length arrives as an int OR a 0-d/1-elem tensor depending on the shard."""
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        return int(v.reshape(-1)[0].item()) if v.numel() else None
    return int(v)


class MultimodalDataCollator(DataCollator):
    """
    Data collator for multimodal training that combines text, audio, voice, and image samples.

    Handles two critical tasks:
    1. Collates per-modality tensors with padding/masking
    2. Injects boundary tokens (BOA/EOA, BOV/EOV, BOI/EOI) and placeholder tokens
       into the text sequence so that the TokenInterleaver can replace placeholders
       with media embeddings during the forward pass

    Each sample is randomly assigned a direction:
    - Synthesis (text → media):  [text_tokens] [BO*] [PLACEHOLDER] [EO*]
    - Transcription (media → text):  [BO*] [PLACEHOLDER] [EO*] [text_tokens]

    For samples with multiple media types, media blocks are chained in fixed
    order: audio, voice, image.
    """

    def __init__(
        self,
        # text
        max_seq_len: int = 2048,
        # audio
        max_waveforms: int = 160000,
        max_mel_spec_frames: int = 625,
        max_sive_feature_frames: int = 157,
        # End-of-voice / end-of-audio terminal unit ids. When set (discrete path), a
        # terminal EOV/EOA token is appended right after each utterance's last content
        # frame -- the discrete-vocab replacement for the stop head. Value is the
        # codebook size K (the id of the (K+1)-th class). None => continuous path,
        # no terminal token inserted (backward compatible; the stop head governs).
        voice_eov_id: Optional[int] = None,
        audio_eov_id: Optional[int] = None,
        # Base id for the 9 control tokens + the native eos id. Default (32000 / 2) = Mistral;
        # in pretrained-LLM mode training.py passes the LLM's native vocab size + native eos so
        # the injected boundary/placeholder/eos tokens match the model and the tokenized data.
        special_token_base: int = constants.SPECIAL_TOKEN_BASE,
        eos_token_id: int = constants.EOS_TOKEN_ID,
        # NAR voice: emit a DUR_* bucket token between BOV and the voice placeholder, so the
        # decoder knows its block length before masked refinement starts. Off => the AR
        # layout is byte-identical.
        emit_duration_token: bool = False,
        # BISTREAM chunk interleaving (CosyVoice 2 style). Instead of all text then all
        # speech, alternate k text tokens with s speech frames so the text a frame realizes
        # is ADJACENT rather than hundreds of positions back:
        #
        #   [text 0:k][BOV][PH -> frames 0:s][EOV] [text k:2k][BOV][PH -> frames s:2s][EOV] ...
        #
        # Each voice block is delimited exactly as the unistream one is, so a 1-chunk plan is
        # BYTE-IDENTICAL to today's layout -- unistream is literally the m=1 case.
        # 0 disables (default), and every tensor this collator emits is then unchanged.
        bistream_text_chunk: int = 0,      # k
        bistream_voice_chunk: int = 0,     # s
        # Per-sample mix, as in CosyVoice 2: this fraction of eligible samples go bistream,
        # the rest stay unistream. One model does both, and the unistream half stays directly
        # comparable to what is trained today.
        bistream_prob: float = 0.5,
        # Terminal unit id ending a CHUNK ("I have said everything the text so far supports;
        # send more text"), as distinct from voice_eov_id which ends the UTTERANCE. Required
        # when bistream is on. Value is K+1 (EOV is K), so the unit head must be K+2 wide.
        voice_fill_id: Optional[int] = None,
    ):
        self.max_seq_len = max_seq_len
        self.max_waveforms = max_waveforms
        self.max_mel_spec_frames = max_mel_spec_frames
        self.max_sive_feature_frames = max_sive_feature_frames
        self.voice_eov_id = voice_eov_id
        self.audio_eov_id = audio_eov_id
        self._sp = constants.special_token_ids(special_token_base)
        self._eos = eos_token_id
        self.emit_duration_token = bool(emit_duration_token)
        self.bistream_text_chunk = int(bistream_text_chunk or 0)
        self.bistream_voice_chunk = int(bistream_voice_chunk or 0)
        self.bistream_prob = float(bistream_prob)
        self.voice_fill_id = voice_fill_id
        self.bistream_enabled = self.bistream_text_chunk > 0 and self.bistream_voice_chunk > 0
        if self.bistream_enabled and self.voice_fill_id is None:
            raise ValueError(
                "bistream needs voice_fill_id (the chunk terminator, K+1). Without it a chunk "
                "boundary is unmarked, the model has no 'send more text' target, and the "
                "layout silently degrades to unistream-with-extra-delimiters.")
        self.force_direction = None  # Set to "synthesis" or "transcription" to override random direction

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        valid_examples = [ex for ex in examples if ex is not None]
        if not valid_examples:
            return {}

        batch = {}

        # Detect which modalities are present. Check all samples (not just first)
        # to handle mixed batches from eval dataloaders.
        has_text = any("text_token_ids" in ex for ex in valid_examples)
        has_audio = any(any(k.startswith("audio_") for k in ex) for ex in valid_examples)
        has_voice = any(any(k.startswith("voice_") for k in ex) for ex in valid_examples)
        has_image = any("image_image" in ex for ex in valid_examples)

        # Bistream plans are decided ONCE, before either collation, because the chunk
        # boundaries shape both the voice tensors and the token sequence and the two must
        # agree exactly. Disabled => None, and every path below is byte-identical to before.
        plans = None
        if self.bistream_enabled and has_voice and has_text:
            plans = self._build_plans(valid_examples, self.force_direction)

        # Collate non-text modalities (only from samples that have them)
        if has_audio:
            batch.update(self._collate_audio(valid_examples))
        if has_voice:
            batch.update(self._collate_voice(valid_examples, plans=plans))
        if has_image:
            batch.update(self._collate_image(valid_examples))

        # Collate text with boundary/placeholder token injection
        if has_text:
            batch.update(self._collate_text(
                valid_examples,
                has_audio=has_audio,
                has_voice=has_voice,
                has_image=has_image,
                force_direction=self.force_direction,
                plans=plans,
            ))

        return batch

    # ------------------------------------------------------------------ bistream planning

    def _resolve_direction(self, ex, force_direction):
        """Direction, decided ONCE per example.

        Normally `_build_token_sequence` decides this inline. Bistream needs it earlier --
        the chunk plan shapes both the voice tensors and the token sequence, and only the
        SYNTHESIS direction is chunked (in transcription the speech is an input whose text is
        the target, so there is no 'send more text' decision to learn). Only called when
        bistream is enabled, so the non-bistream RNG consumption order is untouched.
        """
        d = force_direction or ex.get("_direction", None)
        if d == "synthesis":
            return True
        if d == "transcription":
            return False
        return random.random() < 0.5

    def _bistream_plan(self, text_length: int, n_frames: int):
        """Chunk plan, or None if this sample cannot be chunked.

        Returns a list of (tok_start, tok_len, frame_start, frame_len), text and speech
        advancing together: k text tokens, then s speech frames, until the text runs out --
        the final text chunk carries whatever speech remains.

        FEASIBILITY is exact, not a ratio. CosyVoice 2 gates on
        `speech_len/text_len > s/k`, which for our corpus (5.9 frames/token, s/k = 6) would
        reject about half the data for no reason: what actually has to hold is that the
        chunks BEFORE the last one fit, i.e. n_frames > (n_text_chunks - 1) * s. That is
        strictly weaker and is what makes the last chunk non-empty.
        """
        k, s = self.bistream_text_chunk, self.bistream_voice_chunk
        text_length = int(text_length); n_frames = int(n_frames)
        if text_length <= 0 or n_frames <= 0:
            return None
        n_text_chunks = (text_length + k - 1) // k
        if n_text_chunks < 2:
            return None                     # one chunk IS unistream; nothing gained
        if n_frames <= (n_text_chunks - 1) * s:
            return None                     # the last chunk would be empty or negative
        chunks = []
        ti = fi = 0
        for j in range(n_text_chunks):
            tl = min(k, text_length - ti)
            fl = (n_frames - fi) if j == n_text_chunks - 1 else min(s, n_frames - fi)
            chunks.append((ti, tl, fi, fl))
            ti += tl
            fi += fl
        assert ti == text_length and fi == n_frames, (ti, text_length, fi, n_frames)
        return chunks

    def _build_plans(self, examples, force_direction):
        """Per-example bistream decision. None entry = this sample stays unistream."""
        plans = []
        cap = self.max_sive_feature_frames
        for ex in examples:
            is_syn = self._resolve_direction(ex, force_direction)
            n = _as_int(ex.get("voice_feature_length"))
            chunks = None
            if is_syn and n is not None and random.random() < self.bistream_prob:
                chunks = self._bistream_plan(int(ex["text_text_length"]), min(int(n), cap))
            plans.append({"is_synthesis": is_syn, "chunks": chunks})
        return plans

    def _build_token_sequence(
        self,
        text_token_ids: torch.Tensor,
        text_length: int,
        has_audio: bool,
        has_voice: bool,
        has_image: bool,
        force_direction: str = None,  # None=random, "synthesis", "transcription"
        voice_frames: Optional[int] = None,
        plan: Optional[dict] = None,
    ) -> torch.Tensor:
        """Build a token sequence with boundary and placeholder tokens injected.

        Randomly chooses synthesis (text → media) or transcription (media → text)
        direction. For text-only samples, returns the original token IDs unchanged.

        Returns:
            1D tensor of token IDs with boundary/placeholder tokens inserted.
        """
        # Trim text to actual length (strips any preprocessor padding)
        text_tokens = text_token_ids[:text_length]
        eos = torch.tensor([self._eos], dtype=text_tokens.dtype)

        # Only append EOS when the text wasn't truncated — truncated samples
        # were cut off mid-content and didn't genuinely end.
        text_truncated = text_length >= self.max_seq_len

        has_any_media = has_audio or has_voice or has_image
        if not has_any_media:
            if text_truncated:
                return text_tokens, False
            return torch.cat([text_tokens, eos]), False

        # Direction is chosen BEFORE the blocks are built: under NAR the voice block carries
        # a duration token, and that token only makes sense in the SYNTHESIS direction (in
        # transcription the voice is an INPUT whose length is already known, so there is
        # nothing to predict and emitting it would train a head on a free variable).
        if plan is not None:
            # Decided in _build_plans, because the chunk layout depends on it.
            is_synthesis = plan["is_synthesis"]
        elif force_direction == "synthesis":
            is_synthesis = True
        elif force_direction == "transcription":
            is_synthesis = False
        else:
            is_synthesis = random.random() < 0.5

        # BISTREAM: text and speech alternate, each voice block delimited exactly as the
        # unistream one is. Only voice is chunked, and only in synthesis; a chunked sample
        # never carries audio/image blocks (the plan is only built for voice-bearing
        # synthesis examples), so the fixed audio/voice/image block order is not disturbed.
        if plan is not None and plan["chunks"] is not None and is_synthesis:
            parts = []
            for (ts, tl, _fs, _fl) in plan["chunks"]:
                parts.append(text_tokens[ts:ts + tl])
                parts.append(torch.tensor(
                    [self._sp.BOV, self._sp.VOICE_PLACEHOLDER, self._sp.EOV],
                    dtype=text_tokens.dtype))
            return torch.cat(parts + [eos]), True

        # Build media token blocks in fixed order: audio, voice, image
        media_blocks = []
        if has_audio:
            media_blocks.append(torch.tensor(
                [self._sp.BOA, self._sp.AUDIO_PLACEHOLDER, self._sp.EOA],
                dtype=text_tokens.dtype,
            ))
        if has_voice:
            # NAR: [BOV] [DUR_k] [VOICE_PH] [EOV]. The duration token sits between BOV and the
            # placeholder so that at generation the model emits it from the BOV hidden state --
            # the last position with full causal multimodal context and the last one before
            # gen queries must be allocated. AR layout ([BOV][VOICE_PH][EOV]) is unchanged
            # when the flag is off.
            _voice_block = [self._sp.BOV]
            if self.emit_duration_token and is_synthesis and voice_frames:
                _voice_block.append(
                    self._sp.base + 9 + constants.duration_bucket(int(voice_frames)))
            _voice_block += [self._sp.VOICE_PLACEHOLDER, self._sp.EOV]
            media_blocks.append(torch.tensor(_voice_block, dtype=text_tokens.dtype))
        if has_image:
            media_blocks.append(torch.tensor(
                [self._sp.BOI, self._sp.IMAGE_PLACEHOLDER, self._sp.EOI],
                dtype=text_tokens.dtype,
            ))

        media_sequence = torch.cat(media_blocks)

        if is_synthesis:
            # Synthesis: [text] [media] [EOS]
            return torch.cat([text_tokens, media_sequence, eos]), is_synthesis
        else:
            # Transcription: [media] [text] [EOS]
            return torch.cat([media_sequence, text_tokens, eos]), is_synthesis

    def _collate_text(
        self,
        examples: list[dict],
        has_audio: bool = False,
        has_voice: bool = False,
        has_image: bool = False,
        force_direction: str = None,
        plans: Optional[list] = None,
    ) -> dict:
        all_token_ids = []
        all_text_lengths = []
        all_texts = []
        all_is_synthesis = []

        for i, ex in enumerate(examples):
            raw_token_ids = trim(ex["text_token_ids"], self.max_seq_len, dim=-1)
            text_length = ex["text_text_length"]

            # Use sample's _direction if available, otherwise use force_direction or random
            direction = force_direction or ex.get("_direction", None)

            # Inject boundary + placeholder tokens
            injected, is_synthesis = self._build_token_sequence(
                raw_token_ids,
                text_length,
                has_audio=has_audio,
                has_voice=has_voice,
                has_image=has_image,
                force_direction=direction,
                voice_frames=_as_int(ex.get("voice_feature_length")),
                plan=(plans[i] if plans is not None else None),
            )

            all_token_ids.append(injected)
            all_text_lengths.append(torch.tensor(injected.shape[0], dtype=torch.long))
            all_texts.append(ex.get("text_text", None))
            all_is_synthesis.append(is_synthesis)

        padded_token_ids, token_masks = pad_and_mask(all_token_ids, all_text_lengths)

        return {
            "text_token_ids": torch.stack(padded_token_ids),
            "text_lengths": torch.stack(all_text_lengths),
            "text_token_masks": torch.stack(token_masks),
            "text_texts": all_texts,
            # Per-sample flag: True = synthesis (text→media), False = transcription (media→text)
            "is_synthesis": torch.tensor(all_is_synthesis, dtype=torch.bool),
        }

    def _collate_text_per_sample(
        self,
        examples: list[dict],
        per_sample_has_audio: list[bool],
        per_sample_has_voice: list[bool],
        per_sample_has_image: list[bool],
        force_direction: str = None,
    ) -> dict:
        """Collate text with per-sample modality awareness.

        Each sample gets placeholder tokens only for its own modalities.
        Text-only samples get no placeholders. Voice samples get BOV/VOICE_PH/EOV.
        Image samples get BOI/IMAGE_PH/EOI.
        """
        all_token_ids = []
        all_text_lengths = []
        all_texts = []
        all_is_synthesis = []

        for i, ex in enumerate(examples):
            if "text_token_ids" not in ex:
                # Skip samples without text (shouldn't happen, but defensive)
                continue

            raw_token_ids = trim(ex["text_token_ids"], self.max_seq_len, dim=-1)
            text_length = ex.get("text_text_length", raw_token_ids.shape[0])
            if isinstance(text_length, torch.Tensor):
                text_length = text_length.item()

            injected, is_synthesis = self._build_token_sequence(
                raw_token_ids,
                text_length,
                has_audio=per_sample_has_audio[i],
                has_voice=per_sample_has_voice[i],
                has_image=per_sample_has_image[i],
                force_direction=force_direction,
                voice_frames=_as_int(ex.get("voice_feature_length")),
            )

            all_token_ids.append(injected)
            all_text_lengths.append(torch.tensor(injected.shape[0], dtype=torch.long))
            all_texts.append(ex.get("text_text", None))
            all_is_synthesis.append(is_synthesis)

        padded_token_ids, token_masks = pad_and_mask(all_token_ids, all_text_lengths)

        return {
            "text_token_ids": torch.stack(padded_token_ids),
            "text_lengths": torch.stack(all_text_lengths),
            "text_token_masks": torch.stack(token_masks),
            "text_texts": all_texts,
            "is_synthesis": torch.tensor(all_is_synthesis, dtype=torch.bool),
        }

    def _collate_audio_like(self, examples: list[dict], prefix: str, eov_id: Optional[int] = None,
                            plans: Optional[list] = None) -> dict:
        """Collate audio-like modality (audio or voice) with the given key prefix."""
        all_waveforms = []
        all_waveform_lengths = []
        all_features = []
        all_feature_lengths = []
        all_mel_specs = []
        all_mel_lengths = []
        all_speaker_embeddings = []
        all_speaker_ids = []
        all_f0 = []
        all_vuv = []
        all_ctc_tokens = []
        all_ctc_lengths = []
        all_texts = []
        all_unit_ids = []
        all_f0_contour = []
        all_dedup_unit_ids = []
        all_durations = []

        for ex in examples:
            all_waveforms.append(trim(ex.get(f"{prefix}_waveform", None), self.max_waveforms, dim=-1))
            all_waveform_lengths.append(ex.get(f"{prefix}_waveform_length", None))
            all_features.append(trim(ex.get(f"{prefix}_features", None), self.max_sive_feature_frames, dim=-1))
            all_feature_lengths.append(ex.get(f"{prefix}_feature_length", None))
            all_mel_specs.append(trim(ex.get(f"{prefix}_mel_spec", None), self.max_mel_spec_frames, dim=-1))
            all_mel_lengths.append(ex.get(f"{prefix}_mel_length", None))
            all_speaker_embeddings.append(ex.get(f"{prefix}_speaker_embedding", None))
            all_speaker_ids.append(ex.get(f"{prefix}_speaker_id", None))
            all_f0.append(trim(ex.get(f"{prefix}_f0", None), self.max_mel_spec_frames, dim=-1))
            all_vuv.append(trim(ex.get(f"{prefix}_vuv", None), self.max_mel_spec_frames, dim=-1))
            all_unit_ids.append(trim(ex.get(f"{prefix}_unit_ids", None), self.max_sive_feature_frames, dim=-1))
            all_f0_contour.append(trim(ex.get(f"{prefix}_f0_contour", None), self.max_mel_spec_frames, dim=-1))
            # Dedup streams are their OWN length (M segments), NOT trimmed to a frame cap:
            # they are already collapsed (<= feature frames), so a frame-based trim would
            # never fire, and durations must sum to the real frame count to expand back.
            all_dedup_unit_ids.append(ex.get(f"{prefix}_dedup_unit_ids", None))
            all_durations.append(ex.get(f"{prefix}_durations", None))
            all_ctc_tokens.append(ex.get(f"{prefix}_ctc_tokens", None))
            all_ctc_lengths.append(ex.get(f"{prefix}_ctc_length", None))
            # Text key differs between audio and voice in dataset
            text_key = f"{prefix}_audio_text" if prefix == "audio" else f"{prefix}_voice_text"
            all_texts.append(ex.get(text_key, None))

        batch = {}

        if all_waveforms[0] is not None:
            padded, masks = pad_and_mask(all_waveforms, all_waveform_lengths)
            batch[f"{prefix}_waveforms"] = torch.stack(padded)
            batch[f"{prefix}_waveform_lengths"] = torch.stack(all_waveform_lengths)
            batch[f"{prefix}_waveform_masks"] = torch.stack(masks)

        # EOV/EOA terminal token (discrete path only). When an eov_id is configured AND
        # this batch carries unit ids, extend an utterance's supervised span by ONE
        # position: the frame at index `length` (the first slot after the last content
        # frame, which occupies indices 0..length-1) becomes the terminal token. Driving
        # the extension through the LENGTHS -- so pad_and_mask derives both the pad width
        # and each sample's mask from length+1 -- makes the terminal position a genuine
        # valid position everywhere downstream (the interleaver places length+1 media
        # tokens; the coda is supervised at index length) while real padding stays
        # strictly beyond it. The terminal token therefore can NEVER land in padding.
        #
        # PER-SAMPLE, and NOT for a maxed-out utterance: an utterance whose length reaches
        # the frame cap (max_sive_feature_frames, which is ALSO the inference generation
        # budget) was cut at the cap -- it did not genuinely end. Appending EOV there
        # would (a) teach a false "end" at a truncation boundary and (b) desync train from
        # inference, where generation budget-stops at the cap WITHOUT emitting EOV. So it
        # gets no EOV, exactly as the text collator withholds EOS from truncated text.
        #
        # feature_length can EXCEED the cap: features / unit_ids are trimmed to the cap
        # (above), but feature_length is the PRE-trim value. So clamp the effective content
        # length to the cap before building spans/targets from it -- otherwise the raw value
        # over-runs the trimmed tensors (out[:n] vs a cap-wide u). `eff < cap` then cleanly
        # separates an ended clip from a truncated/at-cap one. Discrete path only; the
        # continuous path keeps its original (un-clamped) feature_length handling untouched.
        cap = self.max_sive_feature_frames
        discrete = eov_id is not None and all_unit_ids[0] is not None
        if discrete:
            eff = [min(int(n), cap) for n in all_feature_lengths]
            eov_here = [e < cap for e in eff]
        else:
            eff = [int(n) for n in all_feature_lengths]
            eov_here = [False] * len(all_feature_lengths)
        any_eov = any(eov_here)
        span_lengths = [
            torch.as_tensor(e + (1 if h else 0), dtype=torch.long)
            for e, h in zip(eff, eov_here)
        ]

        # ------------------------------------------------------------------ discrete path
        # Everything the discrete path emits is built from ONE description: a list of
        # (content_start, content_len, terminal_unit_or_None) segments per sample.
        #
        #   unistream  -> [(0, e, EOV)]                         (or terminal None if truncated)
        #   bistream   -> [(fs_0, fl_0, FILL), ..., (fs_m, fl_m, EOV)]
        #
        # The emitted stream is those segments laid end to end WITH their terminals inline,
        # so an utterance occupies e + (#terminals) positions. Two consequences worth stating,
        # because they are what keep the rest of the model untouched:
        #   - `voice_feature_lengths` is that expanded length, exactly as it already counts
        #     the EOV slot today, so every downstream span/mask keeps working;
        #   - the unit TARGET tensor is still (B, T_expanded) per utterance, so the coda, the
        #     CE, EOV handling and speaker conditioning stay per-utterance concepts. Chunks
        #     are a slicing of this stream, NOT a new batch axis (see the `n`-axis note in
        #     docs/plans/bistream-inner-monologue.md).
        if discrete:
            segs = []
            for i in range(len(examples)):
                pl = plans[i] if plans is not None else None
                chunks = pl["chunks"] if pl is not None else None
                e, h = eff[i], eov_here[i]
                if chunks is None:
                    segs.append([(0, e, eov_id if h else None)])
                    continue
                m = len(chunks)
                # A truncated utterance gets no terminal on its LAST chunk, for the same
                # reason it gets no EOV today: it did not genuinely end, and teaching a
                # terminal at a truncation boundary desyncs train from inference. Interior
                # chunk boundaries are real regardless, so they keep their FILL.
                segs.append([
                    (fs, fl, (self.voice_fill_id if j < m - 1 else (eov_id if h else None)))
                    for j, (_ts, _tl, fs, fl) in enumerate(chunks)
                ])

            span_lengths = [
                torch.as_tensor(sum(cl + (1 if t is not None else 0) for _cs, cl, t in sg),
                                dtype=torch.long)
                for sg in segs
            ]
            T = max(int(x) for x in span_lengths)
            n_chunks = [len(sg) for sg in segs]
            M = max(n_chunks)
            chunk_starts = torch.zeros(len(segs), M, dtype=torch.long)
            chunk_lengths = torch.zeros(len(segs), M, dtype=torch.long)

            feats_out, masks_out, units_out = [], [], []
            for i, sg in enumerate(segs):
                u = all_unit_ids[i]
                f = all_features[i] if all_features[0] is not None else None
                uo = torch.full((T,), -100, dtype=torch.long)
                fo = torch.zeros(f.shape[:-1] + (T,), dtype=f.dtype) if f is not None else None
                mo = torch.zeros(T, dtype=torch.float32)   # 1.0 = valid, matching pad_and_mask
                cur = 0
                for j, (cs, cl, term) in enumerate(sg):
                    # Clamp against what is actually STORED, not just the recorded length:
                    # a shard can carry a feature/unit tensor shorter than feature_length,
                    # which pad_and_mask used to absorb silently.
                    avail = u.shape[-1] - cs
                    if f is not None:
                        avail = min(avail, f.shape[-1] - cs)
                    cl = max(0, min(cl, avail))
                    start = cur
                    if cl > 0:
                        uo[cur:cur + cl] = u[cs:cs + cl].to(torch.long)
                        if fo is not None:
                            fo[..., cur:cur + cl] = f[..., cs:cs + cl]
                        cur += cl
                    if term is not None:
                        # Terminal slot: supervised as `term`, feature left at ZERO. The coda
                        # never consumes it as input (its target is a token, not a feature),
                        # which is what makes a zero column correct rather than merely benign.
                        uo[cur] = int(term)
                        cur += 1
                    chunk_starts[i, j] = start
                    chunk_lengths[i, j] = cur - start
                mo[:cur] = 1.0
                units_out.append(uo)
                masks_out.append(mo)
                if fo is not None:
                    feats_out.append(fo)

            if all_features[0] is not None:
                batch[f"{prefix}_features"] = torch.stack(feats_out)
                batch[f"{prefix}_feature_masks"] = torch.stack(masks_out)
            batch[f"{prefix}_feature_lengths"] = torch.stack(span_lengths)
            batch[f"{prefix}_unit_ids"] = torch.stack(units_out)
            if plans is not None and any(p is not None and p["chunks"] is not None for p in plans):
                # Placeholder -> (utt_idx, start, length). utt_idx is 0 throughout because
                # this collator emits ONE utterance per example; it is carried explicitly so
                # the interleaver never has to infer it positionally.
                batch[f"{prefix}_chunk_starts"] = chunk_starts
                batch[f"{prefix}_chunk_lengths"] = chunk_lengths
                batch[f"{prefix}_chunk_counts"] = torch.tensor(n_chunks, dtype=torch.long)
                batch[f"{prefix}_chunk_utts"] = torch.zeros(len(segs), M, dtype=torch.long)

        if not discrete and all_features[0] is not None:
            padded, masks = pad_and_mask(all_features, span_lengths)
            # Zero the feature pad. VQ quantizes EVERY frame (pad included) to a nonzero
            # centroid, and pad_and_mask masks-but-doesn't-zero — so the prelude's conv would
            # bleed real-unit-like pad frames into the last valid frames, and the viz SMG
            # decodes the pad tail as babble (the "N real seconds + nonsense to the cap" the
            # transcription-input render shows). Mask is [T]; broadcasts over [D, T] / [L, D, T].
            # With EOV the terminal frame at index `length` is a zero column (no stored feature
            # there) that the coda never consumes as INPUT -- its target is the EOV unit, not a
            # feature -- so zeroing it is both harmless and consistent with the rest of the pad.
            padded = [f * m.to(f.dtype) for f, m in zip(padded, masks)]
            if any_eov:
                # Explicitly zero the terminal (EOV) frame at index `length`. pad_and_mask's
                # mask marks it VALID (so voice_lengths / the interleaver include it), which
                # leaves it UNzeroed by the mask-multiply above -- and a feature tensor stored
                # slightly longer than feature_length would otherwise leak a real frame there.
                # The coda never consumes it as input (its target is the EOV unit), so zeroing
                # is safe and keeps the terminal frame content-free. Only for samples that got
                # an EOV (effective length < cap <= width, so always in bounds).
                for f, e, h in zip(padded, eff, eov_here):
                    if h:
                        f[..., e] = 0.0
            batch[f"{prefix}_features"] = torch.stack(padded)
            batch[f"{prefix}_feature_lengths"] = torch.stack(span_lengths)
            batch[f"{prefix}_feature_masks"] = torch.stack(masks)

        if not discrete and all_unit_ids[0] is not None:
            # Pad with -100, NOT 0: 0 is a real unit id. pad_and_mask() pads with 0
            # (it is the shared waveform/mel helper), which would silently supervise the
            # coda to predict unit 0 across every padded frame — the exact bug the text
            # targets had. Pad by hand so padding is the CE ignore_index.
            if any_eov:
                # Width matches the EOV-extended features (max span) so the coda's logits
                # and this target align position-for-position.
                T = max(int(s) for s in span_lengths)
            else:
                T = max(int(u.shape[-1]) for u in all_unit_ids)
            padded_units = []
            for u, e, h in zip(all_unit_ids, eff, eov_here):
                out = torch.full((T,), -100, dtype=torch.long)
                out[:e] = u[:e].to(torch.long)
                if h:
                    # Terminal EOV at index `length` -- immediately after the last content
                    # frame (indices 0..length-1), before any -100 padding. Supervised (not
                    # -100) so the coda learns to emit it; the codebook has no row at id==K,
                    # so generation stops on it before any centroid lookup. Skipped for
                    # maxed-out (truncated) utterances -- see the eov_here note above.
                    out[e] = eov_id
                padded_units.append(out)
            batch[f"{prefix}_unit_ids"] = torch.stack(padded_units)

        if all_dedup_unit_ids[0] is not None:
            # Deduped (unit, duration) segment sequence, variable length M per sample.
            # dedup_lengths (M per sample) is the mask everything downstream uses — the
            # coda, the CE, the duration/F0/stop losses, and the expand-before-SMG all key
            # off it, since M is unrelated to feature/mel length.
            dedup_lengths = torch.tensor([int(u.shape[-1]) for u in all_dedup_unit_ids], dtype=torch.long)
            Md = int(dedup_lengths.max())
            d_units, d_dur = [], []
            for i, m in enumerate(dedup_lengths.tolist()):
                # unit ids: -100 pad (CE ignore_index), same reason as 50Hz unit_ids.
                u = torch.full((Md,), -100, dtype=torch.long)
                u[:m] = all_dedup_unit_ids[i][:m].to(torch.long)
                d_units.append(u)
                # durations: 0 pad. Real durations are >= 1, so 0 is an unambiguous pad
                # marker and a duration loss masks it out via dedup_lengths anyway.
                dr = torch.zeros(Md, dtype=torch.long)
                dr[:m] = all_durations[i][:m].to(torch.long)
                d_dur.append(dr)
            batch[f"{prefix}_dedup_unit_ids"] = torch.stack(d_units)
            batch[f"{prefix}_durations"] = torch.stack(d_dur)
            batch[f"{prefix}_dedup_lengths"] = dedup_lengths
            # F0/VUV are NOT deduped: they stay at 50Hz (batch[..._f0_contour] / _vuv) and
            # are predicted from the duration-expanded hidden state so within-segment pitch
            # survives.

        if all_mel_specs[0] is not None:
            padded, masks = pad_and_mask(all_mel_specs, all_mel_lengths)
            batch[f"{prefix}_mel_specs"] = torch.stack(padded)
            batch[f"{prefix}_mel_lengths"] = torch.stack(all_mel_lengths)
            batch[f"{prefix}_mel_spec_masks"] = torch.stack(masks)

        if all_speaker_embeddings[0] is not None:
            batch[f"{prefix}_speaker_embeddings"] = torch.stack(all_speaker_embeddings)

        if all_speaker_ids[0] is not None:
            batch[f"{prefix}_speaker_ids"] = torch.stack(all_speaker_ids)

        if all_f0_contour[0] is not None:
            # 0-pad is correct here (unlike unit ids): unvoiced frames are already 0 in the
            # contour, and the F0 loss is voicing-weighted, so padding contributes nothing.
            padded_c, _ = pad_and_mask(all_f0_contour, all_mel_lengths)
            batch[f"{prefix}_f0_contour"] = torch.stack(padded_c)

        if all_f0[0] is not None:
            padded_f0, _ = pad_and_mask(all_f0, all_mel_lengths)
            padded_vuv, _ = pad_and_mask(all_vuv, all_mel_lengths)
            batch[f"{prefix}_f0"] = torch.stack(padded_f0)
            batch[f"{prefix}_vuv"] = torch.stack(padded_vuv)

        if all_ctc_tokens[0] is not None:
            padded, masks = pad_and_mask(all_ctc_tokens, all_ctc_lengths)
            batch[f"{prefix}_ctc_tokens"] = torch.stack(padded)
            batch[f"{prefix}_ctc_lengths"] = torch.stack(all_ctc_lengths)
            batch[f"{prefix}_ctc_masks"] = torch.stack(masks)

        batch[f"{prefix}_texts"] = all_texts

        return batch

    def _collate_audio(self, examples: list[dict]) -> dict:
        filtered = [ex for ex in examples if any(k.startswith("audio_") for k in ex)]
        return self._collate_audio_like(filtered, "audio", eov_id=self.audio_eov_id) if filtered else {}

    def _collate_voice(self, examples: list[dict], plans: Optional[list] = None) -> dict:
        keep = [i for i, ex in enumerate(examples) if any(k.startswith("voice_") for k in ex)]
        filtered = [examples[i] for i in keep]
        # Plans are indexed against the FULL example list (that is what _collate_text sees),
        # so they must be filtered in lockstep or every chunk map lands on the wrong sample.
        fplans = [plans[i] for i in keep] if plans is not None else None
        return (self._collate_audio_like(filtered, "voice", eov_id=self.voice_eov_id, plans=fplans)
                if filtered else {})

    def _collate_image(self, examples: list[dict]) -> dict:
        images = [ex["image_image"] for ex in examples if "image_image" in ex]
        if not images:
            return {}
        return {"image_images": torch.stack(images)}
