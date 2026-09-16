import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatransformer.config.world.world_model import WORLD_MODEL_CONFIGS, MegaTransformerWorldModelConfig
from megatransformer.model.voice.feature_extractor import VoiceSIVEPreludeFeatureExtractor
from megatransformer.model.audio.generator import AudioCodaWithLoss
from megatransformer.model.voice.generator import VoiceCodaAndSMGWithLoss
from megatransformer.model.image.feature_extractor import ImageVAEPreludeFeatureExtractor
from megatransformer.model.image.vae.vae import ImageVAEDecoder, ImageVAEEncoder
from megatransformer.model.sinusoidal_positional_encoding import Sinusoidal2DPositionalEmbedding, SinusoidalPositionalEncoding
from megatransformer.model.text.feature_extractor import TextPreludeFeatureExtractor
from megatransformer.model.text.generator import TextCodaClassifierWithLoss
from megatransformer.config.image.decoder import (
    DiffusionBridgeImageDecoderConfig,
    ImageDecoderConfig,
    SDXLAdapterConfig,
    ZImageAdapterConfig,
)
from megatransformer.model.image.decoder import ImageDecoder
from megatransformer.model.image.diffusion_decoder import DiffusionBridgeImageDecoder
from megatransformer.model.image.sdxl_adapter import SDXLConditioningAdapter
from megatransformer.model.image.zimage_adapter import ZImageConditioningAdapter
from megatransformer.model.world.kv_cache import RecurrentKVCache
from megatransformer.model.world.recurrent import MegatransformerRecurrentBlock
from megatransformer.model.world.token_alignment import (
    MODALITY_TEXT,
    MODALITY_VOICE,
    TokenInterleaver,
    TokenUninterleaver,
    build_all_voice_attn_bias,
    build_voice_voice_attn_bias,
)
from megatransformer.utils import constants, megatransformer_utils


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

    # HF Trainer's _issue_warnings_after_load reads this PreTrainedModel attribute whenever a
    # resume has missing/unexpected keys. This is a plain nn.Module, so provide it (None) or the
    # access raises AttributeError. Harmless for our own save/load (we don't ignore any keys).
    _keys_to_ignore_on_save = None

    def __init__(self, config: MegaTransformerWorldModelConfig):
        super(MegaTransformerWorldModel, self).__init__()

        self.config = config
        self.include_modes = set(config.include_modes)

        # Control-token ids resolved for this config's base (32000 default / native LLM vocab in
        # pretrained mode). Use self._sp.<NAME> everywhere instead of the module-level constants so
        # a base swap threads through the model. self._eos is the native (not control) eos id.
        self._sp = constants.special_token_ids(getattr(config, "special_token_base", constants.SPECIAL_TOKEN_BASE))
        self._eos = getattr(config, "eos_token_id", constants.EOS_TOKEN_ID)

        # Feature extractors — text is always required. text_encoder (single gate, default None)
        # swaps the from-scratch prelude for a pretrained-LLM body + translator; None is unchanged.
        _text_encoder = getattr(config, "text_encoder", None)
        self.text_feature_extractor = TextPreludeFeatureExtractor(
            config.text_prelude_config, text_encoder=_text_encoder)

        # Classifier-free guidance: a learned "null text" embedding that replaces the text
        # hidden states on dropped SYNTHESIS examples during training, so the model also learns
        # an unconditional (text-free) voice distribution. At inference, cond vs. null-forced
        # forwards are combined as uncond + w*(cond-uncond). Zero-init: starts as "no signal".
        # GATED on voice_cfg_enabled: a non-CFG model gets NO extra param, so pre-CFG checkpoints
        # (and their optimizer state) resume without a size mismatch.
        if getattr(config, "voice_cfg_enabled", False):
            self.null_text_embed = nn.Parameter(torch.zeros(config.text_prelude_config.d_model))

        # NAR voice: the learned [MASK] token, in VOICE FEATURE space (it substitutes for a
        # codebook centroid in voice_inputs, upstream of the prelude, so it must live in the
        # same space the prelude consumes). Small random rather than zeros -- padded regions
        # are already zeros, and a MASK indistinguishable from padding would teach the model
        # that "unknown" and "past the end" are the same state.
        # GATED, so a non-NAR checkpoint gets no extra parameter.
        if getattr(config, "voice_nar", False):
            self.voice_mask_feature = nn.Parameter(
                torch.randn(config.voice_prelude_config.feature_channels) * 0.02)
        # Direct unit path into the coda, for the trunk-text-only variant. Mirrors
        # voice_coda_prev_proj: a projection from voice feature space into the coda's width,
        # added to the coda's input. Zero-init so the model starts from "no unit signal" and
        # has to learn to use it, rather than being perturbed at step 0.
        if getattr(config, "voice_nar_trunk_text_only", False):
            self.voice_coda_units_proj = nn.Linear(
                config.voice_coda_config.feature_channels,
                config.voice_coda_config.coda_config.d_model)
            nn.init.zeros_(self.voice_coda_units_proj.weight)
            nn.init.zeros_(self.voice_coda_units_proj.bias)

        # Modality-specific preludes (only instantiate if included)
        self.audio_feature_extractor = (
            VoiceSIVEPreludeFeatureExtractor(config.audio_prelude_config)
            if "audio" in self.include_modes else None
        )
        self.voice_feature_extractor = (
            VoiceSIVEPreludeFeatureExtractor(config.voice_prelude_config)
            if "voice" in self.include_modes else None
        )
        self.image_feature_extractor = (
            ImageVAEPreludeFeatureExtractor(config.image_prelude_config)
            if "image" in self.include_modes else None
        )

        # k-means codebook for the discrete voice path, set by the trainer via
        # set_voice_codebook(). generate() emits CENTROIDS rather than the regression
        # head's frames, so the SMG receives the same manifold it was trained on. A
        # buffer so it follows .to(device) and persists in the checkpoint -- the codebook
        # is not optional metadata, it defines what the predicted unit ids MEAN.
        self.register_buffer("voice_codebook", None, persistent=True)

        # Token alignment
        self.token_interleaver = TokenInterleaver(config.token_interleaver_config)
        self.token_uninterleaver = TokenUninterleaver()

        # Main transformer
        self.recurrent_block = MegatransformerRecurrentBlock(config.recurrent_block_config)

        # Generators/codas — text is always required
        self.text_generator = TextCodaClassifierWithLoss(
            config.text_coda_config, text_encoder=_text_encoder)

        self.audio_generator = (
            AudioCodaWithLoss("audio", config.audio_coda_config)
            if "audio" in self.include_modes else None
        )
        self.voice_generator = (
            VoiceCodaAndSMGWithLoss("voice", config.voice_coda_config)
            if "voice" in self.include_modes else None
        )
        # Image decoder (optional). Dispatch on the actual config type:
        #   - ImageDecoderConfig            → ImageDecoder (direct latent prediction)
        #   - DiffusionBridgeImageDecoderConfig → DiffusionBridgeImageDecoder (flow matching)
        self.image_generator = None
        if config.image_coda_config is not None and "image" in self.include_modes:
            if isinstance(config.image_coda_config, DiffusionBridgeImageDecoderConfig):
                self.image_generator = DiffusionBridgeImageDecoder(config.image_coda_config)
            elif isinstance(config.image_coda_config, SDXLAdapterConfig):
                # Frozen-SDXL path: predict CLIP conditioning instead of a latent.
                self.image_generator = SDXLConditioningAdapter(config.image_coda_config)
            elif isinstance(config.image_coda_config, ZImageAdapterConfig):
                # Frozen Z-Image-Turbo path: predict Qwen3-4B conditioning instead of a latent.
                self.image_generator = ZImageConditioningAdapter(config.image_coda_config)
            elif isinstance(config.image_coda_config, ImageDecoderConfig):
                self.image_generator = ImageDecoder(config.image_coda_config)
            else:
                raise TypeError(
                    f"Unknown image_coda_config type: {type(config.image_coda_config).__name__}. "
                    f"Expected ImageDecoderConfig, DiffusionBridgeImageDecoderConfig, "
                    f"SDXLAdapterConfig, or ZImageAdapterConfig."
                )
            # Normalize recurrent output before the decoder to prevent
            # activation growth from saturating the decoder's attention.
            self.image_coda_input_norm = nn.LayerNorm(config.text_prelude_config.d_model)
            self.image_gen_out_offset = None

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

            # Lever D (see ImageGenConfig.image_gen_out_offset): learned per-position offset,
            # zero-init, subtracted from the trunk's gen-query output BEFORE
            # image_coda_input_norm. Built here rather than beside that norm because n_gen is
            # only resolved at this point. Applied only when the sequence length matches n_gen,
            # so transcription-direction image tokens are untouched.
            if bool(getattr(config, "image_gen_out_offset", False)):
                self.image_gen_out_offset = nn.Parameter(
                    torch.zeros(n_gen, config.text_prelude_config.d_model))

            self.gen_query_mode = config.gen_query_mode
            if config.gen_query_mode == "learned":
                init_std = getattr(config, 'image_gen_query_init_std', 3.0)
                self.image_gen_queries = nn.Parameter(
                    torch.randn(1, n_gen, d_model) * init_std
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

        # Voice gen-query mode (opt-in via config.voice_gen_query_mode): the trunk's SYNTHESIS
        # voice input becomes a learned per-position gen query (text-only conditioning; text
        # arrives via the trunk's attention), removing the AR crutch from the shared trunk. The
        # coda keeps local coherence via a direct projection of the PREVIOUS frame's centroid.
        # Gated so a non-gen-query model has zero extra params. Input/transcription voice path is
        # untouched. Learned (not fixed-sinusoidal) queries — positional-only collapsed for image.
        self.voice_gen_query_mode = getattr(config, "voice_gen_query_mode", None)
        # voice_coda_prev IS a crutch: handing the coda the previous frame's centroid lets it
        # predict unit t by 1st-order extrapolation, with NO attention and NO text (a position-
        # wise MLP coda does exactly this). Default True keeps the original gen-query behavior;
        # set False for a TRULY crutch-free path (coda sees only the text-driven trunk output).
        self.voice_gen_query_coda_prev = getattr(config, "voice_gen_query_coda_prev", True)
        if self.voice_gen_query_mode is not None and "voice" in self.include_modes:
            vd = config.text_prelude_config.d_model
            _coda_block = getattr(config.voice_coda_config, "coda_config", None)
            max_pos = getattr(_coda_block, "max_position_embeddings", 1024) or 1024
            self.voice_gen_queries = nn.Embedding(max_pos, vd)
            if self.voice_gen_query_coda_prev:
                self.voice_coda_prev_proj = nn.Linear(config.voice_prelude_config.feature_channels, vd)

        # Huginn-style embedding scale: multiply embeddings by sqrt(d_model) so
        # the injected input x_0 matches the thought state initialization variance.
        self.embed_scale = math.sqrt(config.text_prelude_config.d_model) if config.scale_embeddings else 1.0

        # Weight tying: share embedding matrix between input and output. Skipped in pretrained-
        # LLM mode: there is no `wte` (the LLM owns its embeddings, already tied internally).
        if getattr(config, 'tie_word_embeddings', False) and _text_encoder is None:
            self.text_generator.lm_head.weight = self.text_feature_extractor.wte.weight
        elif _text_encoder is not None and not getattr(self.text_generator, "trainable_head", False):
            # Pretrained-LLM mode: the coda SHARES the FE's (tied) LM head — one embed/head weight,
            # no duplicate, tie survives an unfreeze. And tie the trainable control-token extension
            # (special_embed <-> special_head), mirroring the LLM's own embed/head tie.
            # Skipped under `trainable_head`: there the coda owns a full-width readout of its own,
            # so there is nothing to share and no special_head to tie (the FE keeps its
            # special_embed — that is the INPUT side, still needed either way).
            self.text_generator.lm_head = self.text_feature_extractor.lm_head
            if getattr(self.text_feature_extractor, "special_embed", None) is not None:
                self.text_generator.special_head.weight = self.text_feature_extractor.special_embed.weight

    @torch.no_grad()
    def generate_voice_nar_from_prompt(self, text_input_ids, n_rounds: int = 16,
                                       temperature: float = 1.0, sp=None,
                                       fallback_frames: int = 250,
                                       choice_temperature: float = 1.0,
                                       force_bucket: Optional[int] = None,
                                       reveal: str = "confidence"):
        """Full NAR voice synthesis from a prompt ENDING AT BOV. Returns (unit_ids, bucket).

        Shared by the training-time viz, the WER eval, the renders and the AR-diagnostics
        generation section, so they cannot drift apart -- the block length must be decided the
        same way everywhere or the numbers stop describing the same procedure.

        Two steps, because no voice position exists until the length is known:
          1. one forward over the prompt to read the next-token distribution, restricted to the
             DUR_* ids, and take the duration bucket;
          2. allocate from that bucket (upper edge + margin, so errors are one-sided) and run
             the MaskGIT sampler. EOV trims whatever is over-allocated.
        """
        from megatransformer.utils import constants as _C
        if sp is None:
            sp = _C.special_token_ids(getattr(self.config, "special_token_base",
                                              _C.SPECIAL_TOKEN_BASE))
        dur_lo = sp.base + 9
        dur_hi = dur_lo + _C.N_DURATION_BUCKETS
        bucket = None
        out = self(text_input_ids=text_input_ids, decode_outputs=False)
        logits = out.get("logits")
        # A model without the duration-token extension has a shorter head; fall back rather
        # than index off the end.
        if logits is not None and logits.shape[-1] >= dur_hi:
            bucket = int(torch.argmax(logits[0, -1, dur_lo:dur_hi]).item())
        # force_bucket: CAUSAL test of the duration mechanism. Substituting another
        # utterance's bucket answers whether the token actually CONTROLS length, or whether
        # length is being driven by something else and the token merely correlates with it.
        if force_bucket is not None:
            bucket = int(force_bucket)
        n_frames = (_C.duration_bucket_alloc(bucket) if bucket is not None
                    else int(fallback_frames))
        tail = [sp.VOICE_PLACEHOLDER, sp.EOV]
        if bucket is not None:
            tail = [dur_lo + bucket] + tail
        full = torch.cat([
            text_input_ids,
            torch.tensor([tail], dtype=text_input_ids.dtype, device=text_input_ids.device),
        ], dim=1)
        ids = self.generate_voice_nar(full, n_frames=n_frames, n_rounds=n_rounds,
                                      temperature=temperature,
                                      choice_temperature=choice_temperature,
                                      reveal=reveal)[0]
        return ids, bucket

    @torch.no_grad()
    def generate_voice_nar(self, text_input_ids, n_frames, n_rounds: int = 16,
                           temperature: float = 1.0, eov_id: Optional[int] = None,
                           choice_temperature: float = 1.0, reveal: str = "confidence"):
        """Masked-parallel (MaskGIT) voice decoding. Returns per-batch unit id lists.

        Built on forward(), NOT on generate(): every position is decoded in parallel, so
        there is no AR loop, no KV cache, and none of the cache-parity rules that govern the
        autoregressive path apply. K forward passes total, independent of n_frames.

        Each round samples every still-masked position, keeps the highest-confidence ones
        (cosine reveal schedule), and re-masks the rest. Confidence-ordered revelation is what
        makes this approximate the joint: the model commits first to the positions it is sure
        about, and later rounds condition on them. Decoding in one round (n_rounds=1) samples
        independent per-position marginals instead, which is a useful sanity check -- it
        should sound obviously worse.

        text_input_ids must already contain the [BOV (DUR_k) VOICE_PH EOV] block, as the
        collator lays it out, so the interleaver knows where the voice goes.
        """
        assert getattr(self.config, "voice_nar", False), \
            "generate_voice_nar requires a model built with voice_nar=True (it needs the MASK feature)"
        dev = next(self.parameters()).device
        text_input_ids = text_input_ids.to(dev)
        B = text_input_ids.shape[0]
        cb = self.voice_codebook.to(dev)
        K, C = int(cb.shape[0]), int(cb.shape[1])
        eov = K if eov_id is None else int(eov_id)

        mask_feat = self.voice_mask_feature.to(dev)
        ids = torch.zeros(B, n_frames, dtype=torch.long, device=dev)
        known = torch.zeros(B, n_frames, dtype=torch.bool, device=dev)
        # (B, 1) not (B,): the prelude/interleaver treat voice as (batch, n_segments, ...) and
        # index lengths per SEGMENT, so a flat (B,) collapses to a 0-dim tensor on indexing.
        lengths = torch.full((B, 1), n_frames, dtype=torch.long, device=dev)
        is_synth = torch.ones(B, dtype=torch.bool, device=dev)
        round_entropy: List[float] = []

        def _inputs():
            feats = mask_feat.view(1, 1, C).expand(B, n_frames, C).clone()
            if bool(known.any()):
                feats[known] = cb[ids[known].clamp(max=K - 1)].to(feats.dtype)
            return feats.permute(0, 2, 1).unsqueeze(1)          # (B, 1, C, T)

        for r in range(max(1, n_rounds)):
            out = self(text_input_ids=text_input_ids, voice_inputs=_inputs(),
                       voice_lengths=lengths, precomputed_latents=True,
                       decode_outputs=False, is_synthesis=is_synth)
            logits = out.get("voice_unit_logits")
            if logits is None:
                raise RuntimeError("model produced no voice_unit_logits; is the unit head built?")
            logits = logits.reshape(B, -1, logits.shape[-1])[:, :n_frames]
            # Mean raw entropy over the positions still undecided this round. Falling across
            # rounds = the model growing certain as it conditions on its own commitments.
            with torch.no_grad():
                _pe = torch.softmax(logits.float(), dim=-1)
                _ent = -(_pe.clamp_min(1e-12).log2() * _pe).sum(-1)      # (B, T)
                _open = ~known
                round_entropy.append(float(_ent[_open].mean().item()) if bool(_open.any())
                                     else float("nan"))
            probs = torch.softmax(logits.float() / max(1e-3, temperature), dim=-1)
            samp = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(B, n_frames)
            conf = probs.gather(-1, samp.unsqueeze(-1)).squeeze(-1)
            # Already-revealed positions keep their id and are treated as maximally confident
            # so the top-k below never evicts a committed token.
            samp = torch.where(known, ids, samp)
            # CONFIDENCE NOISE (MaskGIT's, and not optional in practice). Revealing strictly by
            # probability commits to the MODE first -- which for this model is repetition -- and
            # every later round re-conditions on those commitments, so the mode reinforces
            # itself. Measured without it at 23k: K=16 scored 2.7x WORSE than K=1 (truncated LCS
            # 0.0246 vs 0.0668, paired CI [+0.027,+0.058]) and hyp/ref collapsed 0.95 -> 0.37.
            # Gumbel noise on the log-confidence, annealed to 0 over the rounds, keeps the early
            # commitments diverse while letting the final rounds be decisive.
            if choice_temperature > 0.0:
                # Gumbel(0,1) = -log(-log(u)). Parenthesize the inner negation explicitly:
                # `-torch.log(x).clamp_min(e)` parses as `-(torch.log(x).clamp_min(e))`, which
                # clamps a NEGATIVE log up to +e and then takes log of a negative number -> NaN.
                # That silently turned the reveal order arbitrary instead of noisy-but-ordered,
                # and made choice_temperature a no-op (NaN does not scale).
                _u = torch.rand_like(conf).clamp(1e-9, 1.0 - 1e-7)
                _g = -torch.log((-torch.log(_u)).clamp_min(1e-9))
                _ann = float(choice_temperature) * (1.0 - (r + 1) / float(max(1, n_rounds)))
                conf = torch.log(conf.clamp_min(1e-9)) + _ann * _g
            # REVEAL ORDER. MaskGIT reveals by confidence, which assumes confidence tracks
            # correctness. For this model it tracks REPETITION -- the mode is "same unit again"
            # -- so confidence ordering commits to repeats first and then conditions on them.
            # Measured at 23k (decoding is deterministic here, so these are exact): sequential
            # reveal 0.0742 truncated LCS vs confidence reveal 0.0320, a 2.3x gap. 'sequential'
            # grows a left-to-right prefix like AR; 'random' is the unbiased control that
            # separates "not-confidence is better" from "sequence order specifically is better".
            if reveal == "sequential":
                conf = torch.linspace(1.0, 0.0, conf.shape[-1], device=conf.device
                                      ).unsqueeze(0).expand_as(conf).clone()
            elif reveal == "random":
                conf = torch.rand_like(conf)
            conf = torch.where(known, torch.full_like(conf, float("inf")), conf)

            # Cosine reveal: few commitments early (when context is thin), many late.
            frac = 1.0 - math.cos(math.pi * (r + 1) / (2 * max(1, n_rounds)))
            n_keep = n_frames if r == max(1, n_rounds) - 1 else int(math.ceil(n_frames * frac))
            n_keep = max(1, min(n_frames, n_keep))
            keep_idx = conf.topk(n_keep, dim=-1).indices
            new_known = torch.zeros_like(known)
            new_known.scatter_(1, keep_idx, True)
            ids = torch.where(new_known, samp, ids)
            known = known | new_known
            if bool(known.all()):
                break

        self.last_nar_round_entropy = round_entropy      # bits per refinement round
        # Truncate each row at its first EOV (the block is sized from a duration bucket, which
        # deliberately over-allocates -- see constants.duration_bucket_alloc).
        outs = []
        for b in range(B):
            row = ids[b].tolist()
            cut = next((i for i, u in enumerate(row) if u == eov), len(row))
            outs.append([int(u) for u in row[:cut]])
        return outs

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

    def set_voice_codebook(self, centroids):
        """Install the (K, D) k-means codebook used by the discrete voice path.

        Must be the SAME codebook the dataset quantizes with and the SMG was trained on:
        unit ids are only meaningful relative to a codebook, and a re-fit reorders the
        centroids, silently mapping every unit to the wrong phoneme.
        """
        self.voice_codebook = None if centroids is None else centroids.float()

    def _trunk_readout_voice(self, latents: torch.Tensor,
                             kv_caches: Optional[list] = None,
                             position_offset: int = 0) -> Optional[torch.Tensor]:
        """Latent -> VOICE UNIT logits, for `logit_kl` at voice positions.

        The text readout is meaningless at voice positions -- different head, different
        vocabulary -- so `forward()` restricts convergence to text and lets voice run the
        full budget. This is the voice analogue, so voice positions can be scored by the
        head that actually produces them.

        ⚠️ SNAPSHOT/RESTORE, and it is load-bearing. The voice coda is CAUSAL and
        KV-cached, and `KVCache.update()` does `self.key_cache = torch.cat([...])` -- it
        REBINDS on every call. A readout invoked once per recurrent iteration would
        therefore append one phantom frame per iteration, so at the default 32 iterations
        the coda's cache would grow 32x faster than the utterance and every subsequent
        frame would attend to garbage.

        Passing `kv_caches=None` instead would avoid that but score a distribution with no
        history, which for a causal coda is not the distribution the model would emit.
        Because update() REBINDS rather than mutating the tensor in place, saving the old
        references and restoring them afterwards is O(1) and exact: the readout sees the
        real context, and the cache is byte-identical when it returns.
        """
        if self.voice_generator is None:
            return None
        snap = None
        if kv_caches is not None:
            snap = [(getattr(c, "key_cache", None), getattr(c, "value_cache", None))
                    for c in kv_caches]
        try:
            out = self.voice_generator(latents, kv_caches=kv_caches,
                                       position_offset=position_offset, use_cache=False)
        finally:
            if snap is not None:
                for c, (k, v) in zip(kv_caches, snap):
                    c.key_cache, c.value_cache = k, v
        return out.get("voice_unit_logits")

    def _exit_readouts(self) -> dict:
        """{modality_key: readout} for `logit_kl` in forward().

        forward() has no KV caches (use_cache=False throughout), so both heads can be
        called plainly. Voice is included only when the DISCRETE path exists -- the
        criterion needs a categorical distribution, and `voice_unit_logits` is only
        produced when a codebook is configured.
        """
        outs = {"text": self._trunk_readout}
        if self.voice_generator is not None and getattr(self, "voice_codebook", None) is not None:
            outs["voice"] = lambda h: self.voice_generator(
                h, kv_caches=None, position_offset=0, use_cache=False,
            ).get("voice_unit_logits")
        return outs

    def _exit_eligibility(self, modality_map: torch.Tensor) -> dict:
        """Disjoint per-key position masks, matching `_exit_readouts()` keys."""
        elig = {"text": modality_map == MODALITY_TEXT}
        if self.voice_generator is not None and getattr(self, "voice_codebook", None) is not None:
            elig["voice"] = modality_map == MODALITY_VOICE
        return elig

    def _trunk_readout_text_cached(self, latents: torch.Tensor,
                                   kv_caches: Optional[list] = None,
                                   position_offset: int = 0) -> Optional[torch.Tensor]:
        """Latent -> TEXT logits during GENERATION, where the coda may be KV-cached.

        `_trunk_readout` below is the forward() version and passes `use_cache=False` with
        no caches, which is exact there because forward() has none. In generate() the text
        coda carries `text_coda_kv_caches`, and a from-scratch (transformer) coda attends
        over them -- so scoring with `kv_caches=None` would compare a distribution with no
        history, which is not what the model would emit.

        Same snapshot/restore as `_trunk_readout_voice`, and for the same reason:
        `KVCache.update()` REBINDS (`self.key_cache = torch.cat([...])`) rather than
        mutating in place, so a readout called once per recurrent iteration would append a
        phantom entry per iteration -- at 32 iterations the cache grows 32x faster than the
        text being generated. Saving and restoring the references is O(1) and exact.

        In pretrained/trainable-head mode the coda is stateless (norm -> MLP -> head) and
        ignores caches entirely, so this reduces to a plain forward; the guard costs
        nothing and keeps the from-scratch path correct.
        """
        if self.text_generator is None:
            return None
        snap = None
        if kv_caches is not None:
            snap = [(getattr(c, "key_cache", None), getattr(c, "value_cache", None))
                    for c in kv_caches]
        try:
            out = self.text_generator(latents, targets=None, kv_caches=kv_caches,
                                      position_offset=position_offset, use_cache=False)
        finally:
            if snap is not None:
                for c, (k, v) in zip(kv_caches, snap):
                    c.key_cache, c.value_cache = k, v
        return out.get("logits")

    def _trunk_readout(self, latents: torch.Tensor) -> torch.Tensor:
        """Latent -> text logits, for the `logit_kl` exit criterion only.

        Huginn's criterion compares successive POST-READOUT distributions, so the trunk
        needs a way to turn a thought state into logits. Its `predict_from_latents` is
        `ln_f` + `lm_head`; the analogue here is the text coda.

        MUST stay side-effect free: `use_cache=False` and `targets=None`, so no KV cache is
        written and no loss is computed. It is called once per iteration purely to score
        convergence and its output is discarded.

        Exactness caveat: in pretrained/trainable-head mode the text coda is STATELESS
        (norm -> MLP -> head), so this is exactly the distribution the model would emit.
        With a from-scratch transformer coda it self-attends across the interleaved
        sequence, which training never does (the uninterleaver hands it text only), so the
        scored distribution is an approximation of the real one. The criterion is a
        convergence heuristic either way -- this readout never contributes to the output.
        """
        return self.text_generator(latents, targets=None, use_cache=False)["logits"]

    def _mrope_ids(self, modality_map):
        """(batch, seq, 2) global+local coordinates for M-RoPE, or None when it is off.

        Only the TRUNK gets these: it is the one module that sees text and media together,
        and therefore the only place a text<->voice coordinate can exist. The preludes and
        codas already carry clean per-stream RoPE (voice-local 0..T) and are unchanged.
        """
        blk = getattr(getattr(self.config, "recurrent_block_config", None), "block_config", None)
        if modality_map is None or not bool(getattr(blk, "use_mrope", False)):
            return None
        from megatransformer.model.world.token_alignment import build_mrope_position_ids
        return build_mrope_position_ids(
            modality_map, voice_rate=float(getattr(self.config, "mrope_voice_rate", 6.0)),
            scale_side=str(getattr(self.config, "mrope_scale_side", "voice")))

    def _image_coda_input(self, x):
        """Trunk gen-query states -> adapter input. Optional learned per-position centering
        (lever D) BEFORE the LayerNorm, since LayerNorm cannot remove a per-position constant."""
        off = getattr(self, "image_gen_out_offset", None)
        if off is not None and x.dim() == 3 and x.shape[1] == off.shape[0]:
            x = x - off.unsqueeze(0).to(x.dtype)
        return self.image_coda_input_norm(x)

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
        # BISTREAM: (B, n_placeholders, 3) long, mapping the i-th VOICE placeholder to
        # (utt_idx, start, length) -- a SLICE of an utterance, so one utterance can be spread
        # over several placeholders interleaved with its transcript. None => placeholder i is
        # the whole of utterance i, the historical layout.
        voice_chunk_map: Optional[torch.Tensor] = None,
        # Image inputs
        image_inputs: Optional[torch.Tensor] = None,
        image_latent_labels: Optional[torch.Tensor] = None,
        # SDXL-adapter targets (frozen-SDXL path only; ignored by DiT/direct decoders).
        # CLIP conditioning of the caption: sequence (B, 77, 2048) + pooled (B, 1280).
        image_clip_seq_labels: Optional[torch.Tensor] = None,
        image_clip_pooled_labels: Optional[torch.Tensor] = None,
        image_cond_labels: Optional[torch.Tensor] = None,
        # T4 AR: (B, L) bool mask of real Qwen3 tokens, since native-length targets are
        # right-padded per batch. None for the fixed-K parallel head.
        image_cond_mask: Optional[torch.Tensor] = None,
        # T4 AR inference: how many conditioning tokens to emit (from the caption's
        # tokenization). None -> the adapter falls back to its fixed K.
        cond_length: Optional[int] = None,
        # Text targets
        text_targets: Optional[torch.Tensor] = None,
        # Mode flags
        precomputed_latents: bool = True,
        decode_outputs: bool = False,
        # Per-sample direction: True = synthesis (text→media), False = transcription
        is_synthesis: Optional[torch.Tensor] = None,
        # NAR→AR curriculum (Variant B): scalar in [0, 1] that down-scales voice→voice
        # attention in the prelude, recurrent trunk, and voice coda on SYNTHESIS rows.
        # alpha=0 severs voice history (position t sees text + its own shifted-TF frame
        # only); alpha=1 (or None) is the identity — no bias built, fast path preserved.
        voice_attn_alpha: Optional[float] = None,
        # Trunk-text-only NAR: revealed unit features (B, n, C, T) routed DIRECTLY to the coda,
        # bypassing the prelude and trunk. None => unchanged behavior.
        voice_coda_units: Optional[torch.Tensor] = None,
        # NAR→AR prenet curriculum: per-step Tacotron-2 prenet dropout on the AR voice path
        # (shifted teacher forcing). Overrides the prelude config's static prenet_dropout so
        # the trainer can RAMP it; None => use the config value. Attacks the shifted-input
        # crutch (the dominant one) that voice_attn_alpha leaves intact.
        voice_prenet_dropout: Optional[float] = None,
        # Classifier-free guidance. cfg_text_dropout_prob>0 (training only): per-synthesis-row
        # probability of replacing text with the learned null embedding, so the model learns the
        # unconditional voice distribution. cfg_force_null_text: unconditionally replace text with
        # null (the uncond forward at eval/inference, for the uncond+w*(cond-uncond) combine).
        cfg_text_dropout_prob: float = 0.0,
        cfg_force_null_text: bool = False,
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
                voice_chunk_map = None      # a map without its voice is a stale index
            if audio_inputs is not None and audio_inputs.shape[0] != B:
                audio_inputs = None
                audio_lengths = None
                audio_latent_labels = None
            if image_inputs is not None and image_inputs.shape[0] != B:
                image_inputs = None
                image_latent_labels = None

        text_hidden_states = self.text_feature_extractor(text_input_ids)

        # Classifier-free guidance text-drop / null-forcing. Replace text hidden states with the
        # learned null embedding; text_token_ids is left intact so the interleaver still finds
        # media placeholders and sequence positions -- only the text CONTENT goes null. Requires
        # a CFG-enabled model (null_text_embed exists); a no-op otherwise.
        if (cfg_force_null_text or (self.training and cfg_text_dropout_prob > 0.0)) \
                and getattr(self, "null_text_embed", None) is None:
            raise ValueError("CFG requested (cfg_force_null_text / cfg_text_dropout_prob) but the "
                             "model was built without voice_cfg_enabled=True (no null_text_embed).")
        if cfg_force_null_text:
            text_hidden_states = self.null_text_embed.to(text_hidden_states.dtype).view(1, 1, -1).expand_as(text_hidden_states)
        elif self.training and cfg_text_dropout_prob > 0.0:
            Bt = text_hidden_states.shape[0]
            drop = torch.rand(Bt, device=text_hidden_states.device) < cfg_text_dropout_prob
            # Only drop text on SYNTHESIS rows (text->voice); dropping text on a transcription
            # row (voice->text) would corrupt its target. is_synthesis may be None (all-synthesis).
            if is_synthesis is not None:
                drop = drop & is_synthesis.to(drop.device).bool()
            if drop.any():
                null = self.null_text_embed.to(text_hidden_states.dtype).view(1, 1, -1)
                text_hidden_states = torch.where(
                    drop.view(Bt, 1, 1), null.expand_as(text_hidden_states), text_hidden_states)

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
        voice_coda_prev = None  # gen-query mode: previous-unit signal routed to the coda (below)
        if voice_inputs is not None and self.voice_feature_extractor is not None:
            batch_size, n_voice = voice_inputs.shape[:2]
            voice_flat = voice_inputs.view(batch_size * n_voice, *voice_inputs.shape[2:])

            if is_synthesis is not None and is_synthesis.any():
                d_model = self.config.text_prelude_config.d_model
                if self.voice_gen_query_mode is not None:
                    # GEN-QUERY synthesis: the trunk's voice input is a learned per-position query
                    # (text-only; text arrives via the trunk's attention) -- the AR crutch is
                    # REMOVED from the shared trunk. Position 0..T-1 index the learned embedding.
                    T_v = voice_flat.shape[-1]
                    pos = torch.arange(T_v, device=voice_flat.device)
                    synth_hidden = self.voice_gen_queries(pos).to(voice_flat.dtype).unsqueeze(0).expand(
                        batch_size * n_voice, T_v, d_model)  # (B*N, T, d_model)
                    # The coda optionally keeps local coherence via the PREVIOUS frame's centroid
                    # (frame t-1 at position t, zero at 0), projected and routed to the coda after
                    # uninterleave -- NOT to the trunk. When voice_gen_query_coda_prev is False this
                    # is skipped entirely: the coda then sees ONLY the text-driven trunk output, so
                    # there is no neighbor-extrapolation crutch anywhere (trunk or coda).
                    if self.voice_gen_query_coda_prev:
                        prev_c = voice_flat[:, :, :-1].transpose(1, 2)  # (B*N, T-1, C)
                        prev_h = self.voice_coda_prev_proj(prev_c)      # (B*N, T-1, d_model)
                        zpref = torch.zeros(prev_h.shape[0], 1, d_model, device=prev_h.device, dtype=prev_h.dtype)
                        voice_coda_prev = torch.cat([zpref, prev_h], dim=1)  # (B*N, T, d_model)
                else:
                    shifted_input = voice_flat[:, :, :-1]  # (B*N, C, T-1)
                    # NAR→AR curriculum: sever the prelude's voice→voice self-attention so a
                    # frame encodes only its own (shifted) input, not the aggregated history.
                    # Built for every row here — non-synthesis rows' output is overwritten by
                    # the un-suppressed normal_hidden below, so no per-row gating is needed.
                    prelude_bias = build_all_voice_attn_bias(
                        shifted_input.shape[-1], voice_attn_alpha,
                        shifted_input.device, voice_flat.dtype,
                    )
                    # THE autoregressive crutch: these are the TRUE previous frames, and they
                    # predict frame t so well on their own that the text earns no gradient.
                    # prenet_dropout (config, default off) is the Tacotron-2 bottleneck.
                    shifted_hidden = self.voice_feature_extractor(
                        shifted_input, apply_prenet_dropout=True,
                        prenet_dropout_override=voice_prenet_dropout,
                        additive_attn_bias=prelude_bias,
                    )  # (B*N, T-1, d_model)
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
            voice_chunk_map=voice_chunk_map,
        )

        # Scale embeddings by sqrt(d_model) to match thought state initialization
        interleaved_tokens = interleaved_tokens * self.embed_scale

        # print("\tInputs to recurrent block:")
        # megatransformer_utils.print_debug_tensor("\t\tinterleaved_tokens", interleaved_tokens)

        # NAR→AR curriculum: down-scale voice→voice attention in the trunk on synthesis
        # rows so the recurrent block can't route the next-frame answer from voice history.
        trunk_voice_bias = build_voice_voice_attn_bias(
            modality_map, voice_attn_alpha, interleaved_tokens.dtype, is_synthesis=is_synthesis,
        )

        # Main Transformer (Recurrent Block)
        recurrent_output, _, recurrent_num_iters, recurrent_kls, iteration_stats = self.recurrent_block(
            interleaved_tokens,
            attention_mask=attn_mask,  # True for attend, False for padding
            additive_attn_bias=trunk_voice_bias,
            position_ids=self._mrope_ids(modality_map),
            # Per-modality exit: each head scores only the positions it produces, in its
            # own vocabulary. Scoring a voice latent with the text head compares the wrong
            # distribution, and the two vocabularies (49k-152k vs ~6.5k units) cannot share
            # one tensor anyway. Audio/image positions are in no mask, so they run the full
            # budget -- image deliberately so, see docs/findings/world-text.md.
            readout=self._exit_readouts(),
            converge_eligible=self._exit_eligibility(modality_map),
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

        # Recurrent-depth telemetry (scalar total iterations = when all tokens converged;
        # kl_per_iteration = mean-over-tokens KL curve). Surfaced so evals can inspect the
        # EFFECTIVE refinement depth per example (KL early-exit floor), not just the cap.
        outputs["recurrent_num_iterations"] = recurrent_num_iters
        outputs["recurrent_kl_per_iteration"] = recurrent_kls

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
                        # ACROSS-PROMPT conditionality at the gen queries. syn_seq_var above is a
                        # spread ACROSS POSITIONS, which learned per-position queries make large BY
                        # CONSTRUCTION -- it reads ~0.97 (healthy) even when every position emits
                        # nearly the same thing regardless of the prompt. This is the quantity that
                        # actually matters: decompose h = mu + r over the BATCH (mu = the
                        # prompt-invariant part: learned queries + biases) and report ||r||/||mu||.
                        # Measured ~0.04 at 10k steps, i.e. the prompt moves the gen-query output by
                        # ~4% of the constant it rides on. Needs >=2 synthesis samples for a mean.
                        syn = batch[syn_mask]
                        if syn.shape[0] >= 2:
                            mu = syn.mean(dim=0, keepdim=True)
                            outputs["recurrent_out/image_syn_cond_frac"] = (
                                (syn - mu).norm(dim=-1).mean()
                                / mu.norm(dim=-1).mean().clamp_min(1e-9))
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
            # Gen-query mode: the trunk got text-only queries, so inject the previous-unit signal
            # HERE so the coda's causal attention still has voice history for local coherence.
            # Row-for-row aligned (n_voice=1, uninterleave preserves order); trim to the common
            # length defensively in case padding differs.
            if voice_coda_prev is not None and voice_coda_prev.shape[0] == voice_batch.shape[0]:
                m = min(voice_coda_prev.shape[1], voice_batch.shape[1])
                voice_batch = voice_batch.clone()
                voice_batch[:, :m] = voice_batch[:, :m] + voice_coda_prev[:, :m].to(voice_batch.dtype)
            # NAR→AR curriculum: sever the coda's causal voice→voice self-attention on
            # synthesis rows so it can't re-mix voice history after the trunk. voice_batch
            # is all-voice (uninterleaved), one row per batch item, so is_synthesis aligns
            # row-for-row. Causal attention means trailing padding is never attended.
            coda_voice_bias = build_all_voice_attn_bias(
                voice_batch.shape[1], voice_attn_alpha, voice_batch.device, voice_batch.dtype,
                batch_size=voice_batch.shape[0], is_synthesis=is_synthesis,
            )
            # Trunk-text-only NAR: revealed units bypass the prelude/trunk entirely and are
            # added here, so the trunk's voice positions carry text-derived content only.
            if voice_coda_units is not None and getattr(self, "voice_coda_units_proj", None) is not None:
                _u = voice_coda_units
                if _u.dim() == 4:                      # (B, n, C, T) -> (B*n, T, C)
                    _b, _n, _c, _t = _u.shape
                    _u = _u.permute(0, 1, 3, 2).reshape(_b * _n, _t, _c)
                _T = min(_u.shape[1], voice_batch.shape[1])
                if _u.shape[0] == voice_batch.shape[0]:
                    _add = self.voice_coda_units_proj(_u[:, :_T].to(voice_batch.dtype))
                    voice_batch = voice_batch.clone()
                    voice_batch[:, :_T] = voice_batch[:, :_T] + _add
                elif not getattr(self, "_warned_coda_units", False):
                    self._warned_coda_units = True
                    print(f"[nar] coda unit injection skipped: {tuple(_u.shape)} vs "
                          f"{tuple(voice_batch.shape)}", flush=True)
            voice_outputs = self.voice_generator(
                voice_batch,
                latent_labels=voice_latent_labels,
                lengths=uninterleaved["voice_lengths"],
                decode_to_mel=decode_outputs,
                additive_attn_bias=coda_voice_bias,
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
            cross_input = self._image_coda_input(image_batch)
            if isinstance(self.image_generator, SDXLConditioningAdapter):
                # Frozen-SDXL path: predict CLIP conditioning; regress to caption CLIP.
                cross_outputs = self.image_generator(
                    encoder_hidden_states=cross_input,
                    clip_seq_labels=image_clip_seq_labels,
                    clip_pooled_labels=image_clip_pooled_labels,
                    sample_mask=is_synthesis,
                )
                outputs["image_clip_seq_pred"] = cross_outputs["image_clip_seq_pred"]
                outputs["image_clip_pooled_pred"] = cross_outputs["image_clip_pooled_pred"]
                if "image_clip_loss" in cross_outputs:
                    outputs["image_clip_loss"] = cross_outputs["image_clip_loss"]
                    outputs["image_clip_mse_loss"] = cross_outputs["image_clip_mse_loss"]
            elif isinstance(self.image_generator, ZImageConditioningAdapter):
                # Frozen Z-Image path: predict Qwen3-4B conditioning; regress to the
                # resampled caption target. No pooled vector. Shares the image_clip_* keys.
                cross_outputs = self.image_generator(
                    encoder_hidden_states=cross_input,
                    cond_labels=image_cond_labels,
                    cond_mask=image_cond_mask,
                    cond_length=cond_length,
                    sample_mask=is_synthesis,
                )
                outputs["image_clip_seq_pred"] = cross_outputs["image_clip_seq_pred"]
                if "image_clip_loss" in cross_outputs:
                    outputs["image_clip_loss"] = cross_outputs["image_clip_loss"]
                    # T4 AR emits no point-head MSE (its target is variable-length), so this
                    # key is conditional rather than assumed present.
                    if "image_clip_mse_loss" in cross_outputs:
                        outputs["image_clip_mse_loss"] = cross_outputs["image_clip_mse_loss"]
                if "image_contrastive_loss" in cross_outputs:
                    outputs["image_contrastive_loss"] = cross_outputs["image_contrastive_loss"]
                if "image_contrastive_negatives" in cross_outputs:
                    outputs["image_contrastive_negatives"] = cross_outputs["image_contrastive_negatives"]
                if "image_flow_loss" in cross_outputs:
                    outputs["image_flow_loss"] = cross_outputs["image_flow_loss"]
            else:
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
        cross_input = self._image_coda_input(image_hidden)
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
        # Stochastic voice sampling (only active if the voice coda emits a
        # log-variance). 0.0 => deterministic (mean only, == old behaviour);
        # ~0.5-0.7 => moderate. variance_floor clamps the per-frame std before
        # sampling (anti-collapse mitigation).
        voice_temperature: float = 0.0,
        voice_variance_floor: float = 0.0,
        # Decode-time EOV suppression: forbid the EOV token (discrete path) until at least this
        # many content frames have been emitted for a sequence. 0 (default) => no suppression,
        # byte-identical to prior behavior (training-viz generate() never sets this). A diagnostic
        # lever to test whether "under-speaking" is the model quitting early vs already-drifted.
        voice_min_frames: int = 0,
        voice_min_frame_ratio: float = 0.0,
        # ORACLE LENGTH (duration upper-bound experiment): force the voice segment to be
        # EXACTLY this many frames -- EOV is banned before it and the segment is cut at it.
        # None (default) => untouched behavior. Per-call scalar, so callers generating one
        # utterance at a time can pass that utterance's ground-truth frame count and ask
        # "what would perfect duration prediction be worth?" without training anything.
        voice_exact_frames: Optional[int] = None,
        # BISTREAM decoding. `voice_bistream_text` is the FULL transcript (B, L) whose first
        # chunk the prompt already contains; after each fill_token the next `..._chunk` tokens
        # are fed and a BOV forces the model back into speech. Leave None for a unistream
        # prompt: a fill_token is then treated as end-of-utterance, since there is no further
        # text for the model to have been asking for.
        voice_bistream_text: Optional[torch.Tensor] = None,
        voice_bistream_text_chunk: int = 0,
        voice_bistream_text_offset: int = 0,   # tokens of the transcript already in the prompt
        # Minimum frames before fill_token is allowed in a chunk. Always at least 1 (a fill on
        # a chunk's first step emits nothing, hands back to text, and burns the transcript
        # without producing audio).
        voice_min_chunk_frames: int = 0,
        # EVAL DECODING: ban the media CONTROL tokens from the text sampler -- the three BO*
        # boundary tokens and the three placeholders. Off by default (a speak-at-will model
        # must be able to emit BO* to start a media block); ON for eval, where the prompt
        # already supplies BO* and anything further is a hallucination.
        #
        # What it buys: a model that never emits EOV runs its block to the frame budget,
        # finalizes, is handed back to text, and can then sample BOV AGAIN and speak a second
        # time. Banning BO* makes that structurally impossible instead of merely unlikely.
        # The placeholders are banned with them because sampling one is meaningless in any
        # regime: they are a data-layout artifact the interleaver consumes at training time,
        # and at generation there is nothing to replace them with.
        suppress_media_control_tokens: bool = False,
        # Truncated sampling for the discrete voice unit head (only active when voice_temperature > 0).
        # top_k keeps the k highest-prob units; top_p (nucleus) keeps the smallest set whose cumulative
        # prob >= p. Both None/0 (default) => untruncated temperature sampling = prior behavior. These cut
        # the low-prob tail pure-temperature sampling can draw (wrong units) -- the standard AR-coherence fix.
        voice_top_k: Optional[int] = None,
        voice_top_p: Optional[float] = None,
        # Repetition-aware sampling (CosyVoice 2's ras_sampling defaults: win 10, tau_r 0.1).
        # Off by default so existing behaviour is byte-identical unless asked for.
        voice_ras_win: int = 0,
        voice_ras_tau: float = 0.1,
        voice_ras_temperature: Optional[float] = None,
        voice_step_stats: Optional[list] = None,
        voice_eov_min_prob: Optional[float] = None,
        voice_eov_max_entropy: Optional[float] = None,
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

        # Real TEXT tokens in the prompt, per batch row — the denominator for CV2's
        # text-proportional EOS floor (voice_min_frame_ratio). Control tokens (BOV/EOV/
        # placeholders) live at or above special_token_base and are excluded: they are not
        # speech to be realised, and counting them would inflate the floor by a fixed amount
        # that matters most on the short prompts where the floor is most delicate.
        _sp_base = int(getattr(self.config, "special_token_base", constants.SPECIAL_TOKEN_BASE))
        _n_text_tokens = (text_input_ids < _sp_base).sum(dim=1).tolist()

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
        # SDXL-adapter path: (seq 77x2048, pooled 1280) CLIP conditioning per generated image,
        # surfaced for the caller to render via frozen SDXL (generate() can't render itself).
        completed_image_cond: List[List[tuple]] = [[] for _ in range(batch_size)]
        # Recurrent iterations actually performed per generated image (one list per
        # batch item, one entry per image block).
        image_recurrent_iterations: List[List[int]] = [[] for _ in range(batch_size)]

        # Per-frame stop-head logit traces for diagnostics. One list per batch
        # item, appended to on every voice/audio generation iter. Returned in
        # the outputs dict so callers can visualize the stop head's signal and
        # diagnose why a budget-hit occurred (flat at -5 → head not learning;
        # rising but < 0 → distribution-shift; crosses 0 late → positional bias).
        voice_stop_logit_trace: List[List[float]] = [[] for _ in range(batch_size)]
        voice_unit_id_trace: List[List[int]] = [[] for _ in range(batch_size)]
        # SEGMENT BOUNDARIES within that trace.
        #
        # voice_unit_id_trace is allocated once and never reset, while the token budget is
        # checked against voice_sequences[b], which IS reset when a segment ends. So a
        # generation that emits EOV and then opens a SECOND voice segment appends both to
        # one flat trace while the budget restarts -- which is how 250-budget runs produced
        # 361/465/500-"frame" traces. Callers that scored the trace as a single utterance
        # were transcribing two concatenated utterances.
        # Recording the boundaries fixes the MEASUREMENT without changing generation: the
        # flat trace stays exactly as before for existing callers, and anyone who wants one
        # utterance takes segments[b][0].
        voice_unit_entropy_trace: List[List[float]] = [[] for _ in range(batch_size)]
        voice_unit_id_segments: List[List[List[int]]] = [[] for _ in range(batch_size)]
        voice_seg_start: List[int] = [0 for _ in range(batch_size)]
        # BISTREAM state. A CHUNK is a slice of an utterance, so its start is tracked
        # separately from the utterance's (voice_seg_start): fill ends a chunk, EOV ends the
        # utterance, and only the latter finalizes.
        voice_chunk_start: List[int] = [0 for _ in range(batch_size)]
        # BISTREAM: tokens to emit verbatim over the NEXT iterations (the following text
        # chunk, then BOV to re-enter speech).
        #
        # ⚠️ DECLARED HERE, OUTSIDE THE TOKEN LOOP, ON PURPOSE. `forced_next_token` is
        # re-created every iteration because it is set and consumed within one -- but this
        # queue must OUTLIVE the iteration that fills it: that iteration emits the forced
        # text-EOV, and the queue is drained over the iterations after it. Declaring it in
        # the loop wipes it every step, so the continuation silently never happens and every
        # utterance renders as exactly its first chunk.
        forced_token_queue: List[List[int]] = [[] for _ in range(batch_size)]
        should_continue_chunk: List[bool] = [False for _ in range(batch_size)]
        # Transcript tokens already consumed by the prompt; the next chunk starts here.
        voice_text_cursor: List[int] = [int(voice_bistream_text_offset) for _ in range(batch_size)]
        voice_f0_seq: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
        completed_voice_f0: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
        # Per-segment durations on the deduped path (empty otherwise). Each loop step is a
        # SEGMENT there; these expand the segment centroids/F0 back to 50Hz at flush so the
        # frozen SMG still receives frame-rate features.
        voice_duration_seq: List[List[int]] = [[] for _ in range(batch_size)]

        def _finalize_voice(seqs, f0s, durs):
            """Concat a voice's steps; on the deduped path expand the CENTROIDS by duration.

            F0 is NOT expanded here: on the deduped path each f0 entry is already a frame-
            rate chunk (the segment's hidden state expanded by its duration, run through the
            F0 head), so concatenating the chunks is already 50Hz and aligns with the
            expanded centroids. Off the deduped path f0 is one value per frame and durs is
            empty, so both branches just concatenate.
            """
            feat = torch.cat(seqs, dim=-1)                       # (C, S)
            if durs and len(durs) == feat.shape[-1]:
                rep = torch.tensor(durs, device=feat.device)
                feat = feat.repeat_interleave(rep, dim=-1)       # (C, T)
            f0 = torch.cat(f0s, dim=-1) if f0s else None         # (T,) already frame-rate
            return feat, f0
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

        # M-RoPE coordinates for the prompt. Generation MUST use the same coordinate system
        # training used, or the trunk sees positions it was never trained on — the exact
        # train/inference mismatch class this codebase has been bitten by before.
        # A TEXT-ONLY prompt yields modality_map=None from _encode_prompt, which would make
        # _mrope_ids return None and silently disengage M-RoPE at inference while training
        # used it — a train/inference coordinate mismatch. Synthesize an all-text map instead.
        _blk_cfg = getattr(getattr(self.config, "recurrent_block_config", None), "block_config", None)
        _mm = prompt_modality_map
        if _mm is None and bool(getattr(_blk_cfg, "use_mrope", False)):
            from megatransformer.model.world.token_alignment import MODALITY_TEXT
            _mm = torch.full(prompt_hidden.shape[:2], MODALITY_TEXT,
                             dtype=torch.long, device=prompt_hidden.device)
        mrope_on = self._mrope_ids(_mm) is not None
        mrope_local_next = None
        if mrope_on:
            _pids = self._mrope_ids(_mm)
            # per-batch running counters for the incremental steps below
            mrope_global_next = [float(_pids[b, :, 0].max().item()) + 1.0 for b in range(_pids.shape[0])]
            mrope_local_next = [float(_pids[b, -1, 1].item()) + 1.0 for b in range(_pids.shape[0])]
            _mrope_was_media = [False for _ in range(_pids.shape[0])]
        else:
            _pids, mrope_global_next, _mrope_was_media = None, None, None
        current_hidden, kv_cache, prompt_iters, prompt_kls, _ = self.recurrent_block(
            prompt_hidden * self.embed_scale,
            attention_mask=prompt_attn_mask,
            kv_cache=kv_cache,
            position_offset=0,
            use_cache=True,
            share_kv_cache=share_kv_cache,
            position_ids=_pids,
        )

        position_offset = prompt_hidden.shape[1]

        # Check if prompt ends with a BO* token — if so, initialize modality mode
        # so the first generated hidden states are accumulated correctly.
        for b in range(batch_size):
            last_token = text_input_ids[b, -1].item()
            if last_token == self._sp.BOA:
                current_modality[b] = "audio"
            elif last_token == self._sp.BOV:
                current_modality[b] = "voice"
            elif last_token == self._sp.BOI:
                current_modality[b] = "image"

        _banned_text_ids = None
        if suppress_media_control_tokens:
            _banned_text_ids = [self._sp.BOA, self._sp.BOV, self._sp.BOI,
                                self._sp.AUDIO_PLACEHOLDER, self._sp.VOICE_PLACEHOLDER,
                                self._sp.IMAGE_PLACEHOLDER]

        def _ban(lg):
            """-inf the banned ids so they cannot be sampled. EO* is deliberately NOT banned:
            generate() FORCES it after a media block, and forcing bypasses the sampler."""
            if not _banned_text_ids:
                return lg
            lg = lg.clone()
            lg[..., _banned_text_ids] = float("-inf")
            return lg

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
        next_token_ids = self._sample_tokens(_ban(logits), temperature, top_k, top_p)

        for b in range(batch_size):
            generated_tokens[b].append(next_token_ids[b].item())

        # Context capacity of the recurrent trunk: stop at the TRAINED context length rather
        # than generating indefinitely past it. This is a quality bound, not a crash guard --
        # the "Causal mask buffer too small" crash it was originally written for is NOT a
        # position-count problem and this check cannot catch it. The trunk's Huginn cache has
        # 16 slots but runs 32 iterations, so each slot takes two keys per token and the KEY
        # length grows ~2x faster than position_offset (512 tokens -> 1024 keys, while
        # position_offset is only ~512). That is handled where it belongs, by growing the mask
        # at inference in transformer.py.
        try:
            _trunk_limit = int(
                self.config.recurrent_block_config.block_config.max_position_embeddings)
        except AttributeError:
            _trunk_limit = None
        hit_context_limit = False

        # Autoregressive generation loop
        for _ in range(max_new_tokens - 1):
            if _trunk_limit is not None and position_offset >= _trunk_limit:
                # the next step would ask for key position _trunk_limit + 1
                hit_context_limit = True
                break
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
                    if mod == "voice" and self.voice_gen_query_mode is not None:
                        # GEN-QUERY synthesis: the trunk's voice input is a learned
                        # per-position query (text-only), NOT the prelude of the
                        # previous frame — the AR crutch is off the shared trunk.
                        # The segment position is the count of frames emitted so far
                        # in THIS voice block (0 at entry, handled by the entry pass
                        # below; this path covers positions >= 1). Mirrors the forward
                        # gen-query branch (world_model.py:410-424).
                        seg_pos = len(voice_sequences[b])
                        pos_idx = min(seg_pos, self.voice_gen_queries.num_embeddings - 1)
                        gq = self.voice_gen_queries(
                            torch.tensor([pos_idx], device=device)
                        ).to(current_hidden.dtype)  # (1, d_model)
                        next_hidden_list.append(gq)
                    elif mod in ("voice", "audio"):
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
                                # Same bottleneck as the training-time shifted path. Kept ON
                                # at generation deliberately (Tacotron 2): dropping it here
                                # would restore the over-reliance on the previous frame that
                                # training was regularized to avoid, i.e. a train/inference
                                # mismatch in exactly the wrong direction.
                                apply_prenet_dropout=True,
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
                    if token_id == self._sp.BOA:
                        current_modality[b] = "audio"
                        just_entered_streaming[b] = "audio"
                    elif token_id == self._sp.BOV:
                        current_modality[b] = "voice"
                        just_entered_streaming[b] = "voice"
                    elif token_id == self._sp.BOI:
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

            # M-RoPE ids for THIS step: global advances by 1 for every token; local advances
            # by 1 for a text token and by 1/rate for a media frame, and RESTARTS when the
            # modality changes (matching build_mrope_position_ids on the training path).
            step_pids = None
            if mrope_global_next is not None:
                rate = float(getattr(self.config, "mrope_voice_rate", 6.0))
                _side = str(getattr(self.config, "mrope_scale_side", "voice"))
                rows = []
                for b in range(next_hidden.shape[0]):
                    # current_modality[b] is the stream this token belongs to (None => text)
                    is_media_b = current_modality[b] in ("audio", "voice", "image")
                    if is_media_b != _mrope_was_media[b]:
                        mrope_local_next[b] = 0.0        # modality changed -> restart local
                        _mrope_was_media[b] = is_media_b
                    rows.append([mrope_global_next[b], mrope_local_next[b]])
                    mrope_global_next[b] += 1.0
                    if _side == "text":
                        mrope_local_next[b] += 1.0 if is_media_b else max(rate, 1e-3)
                    else:
                        mrope_local_next[b] += (1.0 / max(rate, 1e-3)) if is_media_b else 1.0
                step_pids = torch.tensor(rows, dtype=torch.float32,
                                         device=next_hidden.device).unsqueeze(1)  # (B,1,2)
            # VOICE-POSITION EXIT CRITERION. `logit_kl` is Huginn's real criterion --
            # KL between successive POST-READOUT distributions -- and Huginn applies it
            # during single-token generation, which is exactly here. Until now it was
            # wired only into forward(), so generation always ran the full budget no
            # matter what the config said.
            #
            # The readout must match the head that produces the position: scoring a voice
            # latent with the TEXT coda compares the wrong distribution. Items not
            # currently emitting voice get zeros and are masked out of the exit decision
            # by `converge_eligible`, so they simply run the full budget as before.
            _v_readout = None
            _v_eligible = None
            # voice_codebook gates the DISCRETE path, which is what produces
            # `voice_unit_logits`. Without it the readout would return None and the block
            # would call init_state(None) -- it only checks that the readout is not None,
            # not that it returns something.
            if (self.voice_generator is not None
                    and self.voice_codebook is not None
                    and getattr(self.recurrent_block.exit_criteria, "needs_readout", False)):
                _v_flags = [current_modality[b] == "voice" for b in range(batch_size)]
                if any(_v_flags):
                    _v_eligible = torch.tensor(
                        [[f] for f in _v_flags], dtype=torch.bool,
                        device=next_hidden.device)  # (B,1)

                    def _v_readout(h, _flags=tuple(_v_flags)):
                        outs = []
                        for _b in range(h.shape[0]):
                            lg = None
                            if _flags[_b]:
                                lg = self._trunk_readout_voice(
                                    h[_b:_b + 1],
                                    kv_caches=voice_coda_kv_caches[_b],
                                    position_offset=voice_coda_position_offsets[_b])
                            outs.append(lg)
                        _ref = next((o for o in outs if o is not None), None)
                        if _ref is None:
                            return None
                        outs = [o if o is not None else torch.zeros_like(_ref) for o in outs]
                        return torch.cat(outs, dim=0)

            # TEXT-POSITION EXIT. The voice wiring above covers items emitting voice;
            # items in TEXT mode (`current_modality[b] is None`) were left ineligible and
            # so ran the full budget. They get their own head here, scored in the text
            # vocabulary -- the two heads cannot share a readout because their vocabularies
            # differ, which is why the block takes a per-key dict.
            #
            # Eligibility mirrors the guard the text coda itself uses below: it only runs
            # when NO item is in a media modality (`any_media`), because its KV cache is
            # shared across the batch and feeding it media positions would pollute it. The
            # readout inherits that constraint exactly -- scoring under any_media would
            # read a cache the real call never builds.
            _t_readout = None
            _t_eligible = None
            if (self.text_generator is not None
                    and getattr(self.recurrent_block.exit_criteria, "needs_readout", False)):
                _t_flags = [current_modality[b] is None for b in range(batch_size)]
                if all(_t_flags):
                    _t_eligible = torch.ones((batch_size, 1), dtype=torch.bool,
                                             device=next_hidden.device)

                    def _t_readout(h):
                        return self._trunk_readout_text_cached(
                            h, kv_caches=text_coda_kv_caches,
                            position_offset=text_coda_position_offset)

            _readouts, _eligible = {}, {}
            if _v_readout is not None:
                _readouts["voice"], _eligible["voice"] = _v_readout, _v_eligible
            if _t_readout is not None:
                _readouts["text"], _eligible["text"] = _t_readout, _t_eligible

            # Process through recurrent block with KV cache
            current_hidden, kv_cache, n_iters, kl_trace, _ = self.recurrent_block(
                next_hidden * self.embed_scale,
                attention_mask=None,
                kv_cache=kv_cache,
                position_offset=position_offset,
                use_cache=True,
                share_kv_cache=share_kv_cache,
                position_ids=step_pids,
                readout=_readouts or None,
                converge_eligible=_eligible or None,
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
                            torch.tensor([[self._sp.AUDIO_PLACEHOLDER]], device=device),
                            kv_caches=text_prelude_kv_caches,
                            position_offset=text_prelude_position_offset,
                            use_cache=True,
                        )
                        text_prelude_position_offset += 1
                        forced_next_token[b] = self._sp.EOA
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
                    if just_entered_streaming[b] == "voice" and voice_sequences[b] and last_voice_pred[b] is not None:
                        # BISTREAM continuation: this BOV re-opens an utterance already in
                        # flight, so it is NOT position 0 and must not get the zero prefix.
                        # Training gives this position prelude(feature at the fill slot), and
                        # that feature is zero -- last_voice_pred was set to a zero FEATURE at
                        # the chunk boundary for exactly this step.
                        _emb, voice_prelude_kv_caches[b] = self.voice_feature_extractor(
                            last_voice_pred[b].unsqueeze(0),
                            kv_caches=voice_prelude_kv_caches[b],
                            position_offset=voice_prelude_position_offsets[b],
                            use_cache=True,
                            apply_prenet_dropout=True,
                        )
                        voice_prelude_position_offsets[b] += 1
                        entry_hidden, kv_cache, _, _, _ = self.recurrent_block(
                            _emb.to(current_hidden.dtype) * self.embed_scale,
                            attention_mask=None,
                            kv_cache=kv_cache,
                            position_offset=position_offset,
                            use_cache=True,
                            share_kv_cache=share_kv_cache,
                        )
                        position_offset += 1
                        hidden_b = entry_hidden
                    elif just_entered_streaming[b] == "voice":
                        d_model_ = self.config.text_prelude_config.d_model
                        if self.voice_gen_query_mode is not None:
                            # Gen-query position 0: the trunk's first voice input is
                            # gen_query(0), not the zero vector (matches forward, where
                            # synth_hidden[0] = voice_gen_queries(0), NOT a zero prefix).
                            entry_input = self.voice_gen_queries(
                                torch.tensor([0], device=device)
                            ).to(current_hidden.dtype).unsqueeze(0)  # (1, 1, d_model)
                        else:
                            entry_input = torch.zeros(
                                1, 1, d_model_, device=device, dtype=current_hidden.dtype,
                            )
                        entry_hidden, kv_cache, _, _, _ = self.recurrent_block(
                            entry_input * self.embed_scale,
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
                    if self.voice_gen_query_mode is not None and self.voice_gen_query_coda_prev:
                        # Route the previous frame's centroid to the coda for local
                        # coherence (forward: voice_coda_prev added to voice_batch,
                        # world_model.py:613-616). At entry (last_voice_pred None) the
                        # coda-prev is zero, so this is a no-op for the first frame.
                        # Skipped entirely when coda_prev is disabled (crutch-free path).
                        if last_voice_pred[b] is not None:
                            prev_c = last_voice_pred[b].transpose(0, 1).unsqueeze(0)  # (C,1)->(1,C)->(1,1,C)
                            prev_h = self.voice_coda_prev_proj(prev_c).to(hidden_b.dtype)  # (1,1,d_model)
                            hidden_b = hidden_b + prev_h
                    if self.voice_generator is not None:
                        coda_out = self.voice_generator(
                            hidden_b,
                            kv_caches=voice_coda_kv_caches[b],
                            position_offset=voice_coda_position_offsets[b],
                            use_cache=True,
                        )
                        voice_coda_kv_caches[b] = coda_out.get("kv_caches")
                        voice_coda_position_offsets[b] += 1
                        unit_logits = coda_out.get("voice_unit_logits")
                        if unit_logits is not None and self.voice_codebook is not None:
                            # DISCRETE path: sample a unit id, emit its CENTROID. The
                            # centroid (not the regression head's frame) is what gets fed
                            # back to the prelude and handed to the SMG, so generation
                            # stays on the same manifold the SMG was trained on.
                            logits = unit_logits[0, -1]  # (K+1,) -- includes the EOV token
                            # Decode-time EOV suppression: forbid the terminal token until at
                            # least `_floor` content frames have been emitted IN THIS SEGMENT
                            # (voice_min_frames, or voice_exact_frames for the oracle).
                            # Both default to 0/None => never triggers => unchanged behavior.
                            # The count is segment-relative because the budget below is too;
                            # this previously counted the whole cumulative trace, so the two
                            # length controls disagreed across a segment boundary.
                            # PER-STEP PREDICTIVE ENTROPY during free-running. The separate
                            # teacher-forced probe measures this with a PERFECT prefix; here the
                            # prefix is the model's own drifting output, which is the condition
                            # that actually produces the audio. Raw (untempered) so it is a
                            # property of the model rather than of the sampler, and directly
                            # comparable to the probe's raw column.
                            _pe = torch.softmax(logits.float(), dim=-1)
                            voice_unit_entropy_trace[b].append(
                                float(-(_pe.clamp_min(1e-12).log2() * _pe).sum().item()))
                            _seg_len = len(voice_unit_id_trace[b]) - voice_seg_start[b]
                            # CV2 makes its EOS floor TEXT-PROPORTIONAL, not a fixed count:
                            # `min_len = (text_len - prompt_text_len) * min_token_text_ratio`
                            # with min_token_text_ratio=2 against its own 3:1 token rate
                            # (5:15 chunks) -- i.e. two thirds of the expected length. Scaled
                            # to our measured 7.5 frames/token that ratio is 5.0. A fixed floor
                            # cannot do this: the same number is far too high for a 5-token
                            # prompt and far too low for a 40-token one.
                            _floor = max(int(voice_min_frames or 0), int(voice_exact_frames or 0))
                            if voice_min_frame_ratio and voice_min_frame_ratio > 0.0:
                                _floor = max(_floor, int(voice_min_frame_ratio * _n_text_tokens[b]))
                            if _floor > 0 and _seg_len < _floor:
                                logits = logits.clone()
                                logits[self.voice_codebook.shape[0]] = float("-inf")
                            # EOV CONFIDENCE GUARD. Premature collapse is NOT the model
                            # confidently deciding to stop -- measured at ck90000, n=96, the
                            # 5 collapsed utterances fired EOV at entropy 5.87 nats / p_max
                            # 0.064 / p(EOV) 0.031, against 2.59 / 0.287 / 0.219 for the 89
                            # that ended normally. The distribution has gone nearly flat
                            # (5.87 nats ~ 350 effective candidates of 6562) and EOV wins by
                            # accident, sometimes as "argmax" only because nothing is likely.
                            #
                            # So the discriminator is an ABSOLUTE floor. Relative truncation
                            # (min-p, top-a) cannot see this: their threshold is a multiple of
                            # p_max, which is itself tiny exactly when this happens.
                            #
                            # BOTH conditions are required. Either alone blocks 5/5 collapses
                            # but breaks 22-27 of 89 legitimate stops; together they still
                            # block 5/5 while breaking 13. They are not redundant -- some good
                            # stops are low-confidence-but-sharp, others diffuse-but-confident;
                            # only the collapses are both.
                            #
                            # Inert unless BOTH are set. Reads only the current distribution,
                            # so it carries to non-TTS voice targets unchanged.
                            if (voice_eov_min_prob is not None
                                    and voice_eov_max_entropy is not None):
                                _eov_id_g = self.voice_codebook.shape[0]
                                _pg = torch.softmax(logits.float(), dim=-1)
                                _Hg = float(-(_pg * torch.log(_pg.clamp_min(1e-12))).sum())
                                if (float(_pg[_eov_id_g]) < voice_eov_min_prob
                                        and _Hg > voice_eov_max_entropy):
                                    logits = logits.clone()
                                    logits[_eov_id_g] = float("-inf")
                            # BISTREAM fill_token (K+1), when the head is wide enough to have
                            # one. Suppressed while the CURRENT CHUNK is still empty: a fill on
                            # a chunk's first step would emit no frames, hand back to text, and
                            # -- with the transcript advancing each time -- burn the whole
                            # prompt without producing audio.
                            _fill_id = self.voice_codebook.shape[0] + 1
                            if logits.shape[-1] > _fill_id:
                                _chunk_len = len(voice_unit_id_trace[b]) - voice_chunk_start[b]
                                if _chunk_len < max(1, int(voice_min_chunk_frames or 0)):
                                    logits = logits.clone()
                                    logits[_fill_id] = float("-inf")
                            # `scaled` is the SINGLE scored tensor both the initial pick and
                            # the RAS resample draw from. Keeping one tensor is the point: the
                            # resample used to read raw `logits` while the pick read
                            # logits/temperature, so at T=0.6 every ban silently resampled at
                            # T=1.0 -- flatter, handing EOV and the tail more mass at exactly
                            # the moments RAS fires. CosyVoice 2 passes the same
                            # `weighted_scores` to nucleus_sampling and random_sampling
                            # (cosyvoice/utils/common.py), so one tensor is also what the
                            # reference does.
                            scaled = (logits.float() / voice_temperature
                                      if voice_temperature > 0.0 else logits.float())
                            if voice_temperature > 0.0:
                                # CV2-EXACT nucleus (cosyvoice/utils/common.py:nucleus_sampling):
                                # sort the FULL softmax and take the prefix while
                                # `cum_prob < top_p and len < top_k`. Ours previously applied
                                # top_k FIRST and then measured top_p against the renormalised
                                # top-k mass, so top_p meant "fraction of the top-k mass" rather
                                # than "fraction of total mass" -- a different, smaller set.
                                filt = scaled
                                _k = voice_top_k if (voice_top_k or 0) > 0 else None
                                _p = voice_top_p if (voice_top_p or 0) > 0 and voice_top_p < 1.0 else None
                                if _k is not None or _p is not None:
                                    sv, si = torch.sort(torch.softmax(scaled, dim=-1), descending=True)
                                    cum = torch.cumsum(sv, dim=-1)
                                    keep = torch.ones_like(sv, dtype=torch.bool)
                                    if _p is not None:
                                        # keep the token that CROSSES p, drop everything after
                                        keep &= torch.cat([torch.ones(1, dtype=torch.bool, device=sv.device),
                                                           cum[:-1] < _p])
                                    if _k is not None:
                                        keep &= (torch.arange(sv.numel(), device=sv.device) < _k)
                                    mask = torch.zeros_like(keep).scatter(0, si, keep)
                                    filt = scaled.masked_fill(~mask, float("-inf"))
                                unit_id = torch.multinomial(torch.softmax(filt, dim=-1), 1)[0]
                            else:
                                unit_id = logits.argmax(-1)
                            # REPETITION-AWARE SAMPLING (RAS), after CosyVoice 2's own decoder
                            # (cosyvoice/utils/common.py:ras_sampling). Free-running measurement
                            # at 44k: adj_repeat 0.103 (3.4x GT) and longest_run 56 -- the model
                            # loops, while the teacher's RAS decoding sits BELOW GT at 0.0070.
                            # If the chosen id already occurred within the last `win` emitted
                            # units at a rate >= tau_r, ban it and resample from the rest.
                            # NOTE the EOV id is exempt: banning it would worsen the very
                            # over-length problem this is meant to fix (EOV fires 10/32 at 44k).
                            if voice_ras_win > 0 and voice_ras_tau > 0.0:
                                eov_id = self.voice_codebook.shape[0]
                                recent = voice_unit_id_trace[b][-voice_ras_win:]
                                if recent and int(unit_id) != eov_id:
                                    rep = sum(1 for u in recent if u == int(unit_id))
                                    if rep >= voice_ras_win * voice_ras_tau:
                                        # UNTRUNCATED but temperature-scaled: CV2's
                                        # random_sampling is
                                        # `weighted_scores.softmax(0).multinomial(1)` over the
                                        # full vocab, i.e. it drops the nucleus on resample but
                                        # keeps the caller's scaling. Matching both halves.
                                        # RESAMPLE TEMPERATURE. `scaled` is the PICK's
                                        # tensor, and reusing it couples two decisions that
                                        # want different sharpness. Worse, it is discontinuous
                                        # at T=0: `scaled` is logits/T for T>0 but RAW logits
                                        # at T=0, so "greedy" resamples at an effective 1.0
                                        # while T=0.2 resamples 5x sharper than the pick ever
                                        # is. Measured at step 68000, n=12, ras_win=10: budget
                                        # caps ran 0/12 at T=0, 12/12 at T=0.2, then recovered
                                        # monotonically 7/12, 5/12, 3/12 at T=0.4/0.5/0.6 as
                                        # the resample re-flattened. A near-deterministic
                                        # second-best pick locks into cycles that never reach
                                        # EOV; a softer draw escapes them.
                                        #
                                        # None keeps the inherited behaviour EXACTLY, so this
                                        # is inert unless set.
                                        if voice_ras_temperature is not None and voice_ras_temperature > 0.0:
                                            banned = logits.float() / voice_ras_temperature
                                        else:
                                            banned = scaled.clone()
                                        banned[int(unit_id)] = float("-inf")
                                        if _floor > 0 and _seg_len < _floor:
                                            banned[eov_id] = float("-inf")
                                        p2 = torch.softmax(banned, dim=-1)
                                        if bool(torch.isfinite(p2).all()) and float(p2.sum()) > 0:
                                            unit_id = torch.multinomial(p2, 1)[0]
                            if voice_step_stats is not None:
                                # DIAGNOSTIC HOOK. Records the model's own belief at this
                                # step, from RAW logits -- NOT `scaled` -- so entropy and
                                # p_max describe the model rather than the sampler, and stay
                                # comparable across temperature arms. Appended only when a
                                # collector list is passed, so this is inert by default.
                                _pp = torch.softmax(logits.float(), dim=-1)
                                _H = float(-(_pp * torch.log(_pp.clamp_min(1e-12))).sum())
                                _top = torch.topk(_pp, 2)
                                voice_step_stats.append({
                                    "b": int(b),
                                    "t": len(voice_unit_id_trace[b]),
                                    "entropy": _H,
                                    "p_max": float(_top.values[0]),
                                    "p_2nd": float(_top.values[1]),
                                    "argmax_id": int(_top.indices[0]),
                                    "chosen_id": int(unit_id),
                                    "p_eov": float(_pp[self.voice_codebook.shape[0]]),
                                    "is_eov": bool(int(unit_id) == self.voice_codebook.shape[0]),
                                })
                            voice_unit_id_trace[b].append(int(unit_id))
                            # EOV token: the terminal unit (id == codebook size; the codebook
                            # has no row there). It is the discrete-vocab replacement for the
                            # old stop head -- terminate WITHOUT emitting a frame (no centroid,
                            # no F0), so the finalized utterance is exactly the content frames.
                            if int(unit_id) == self.voice_codebook.shape[0]:
                                should_stop_voice = True
                            elif int(unit_id) == self.voice_codebook.shape[0] + 1:
                                # BISTREAM chunk terminator: "I have said everything the text
                                # so far supports". Ends the voice BLOCK without ending the
                                # UTTERANCE -- no frame is emitted (no centroid at this id),
                                # and the utterance's frames, F0, and prelude/coda KV caches
                                # all survive into the next chunk.
                                should_stop_voice = True
                                should_continue_chunk[b] = True
                            else:
                                # The coda's F0/VUV for this frame, if the head exists. This is
                                # the contour the SMG will be conditioned on -- the whole point
                                # of predicting it here rather than letting the SMG infer it
                                # from prosody-free units.
                                dur_p = coda_out.get("voice_duration_preds")
                                if dur_p is not None:
                                    # Deduped path: this step is a SEGMENT. Record how many 50Hz
                                    # frames it becomes (exp(log-frames), round, clamp 1..100),
                                    # then predict F0 at frame rate by expanding THIS segment's
                                    # hidden state by that duration and running the F0 head --
                                    # matching training, where F0 comes off the expanded h.
                                    d = int(torch.exp(dur_p[0, -1].detach()).round().clamp(min=1, max=100))
                                    voice_duration_seq[b].append(d)
                                    hid = coda_out.get("voice_hidden")
                                    f0_head = getattr(self.voice_generator, "f0_head", None)
                                    if hid is not None and f0_head is not None:
                                        h_exp = hid[0, -1].unsqueeze(0).expand(d, -1)  # (d, d_model)
                                        voice_f0_seq[b].append(f0_head(h_exp).squeeze(-1).detach())  # (d,)
                                else:
                                    # Frame-rate path: one F0 value per frame, from the coda.
                                    f0_p = coda_out.get("voice_f0_preds")
                                    if f0_p is not None:
                                        voice_f0_seq[b].append(f0_p[0, -1].detach().reshape(1))
                                frame_pred = self.voice_codebook[unit_id].to(
                                    device=hidden_b.device, dtype=coda_out["voice_latent_preds"].dtype
                                ).view(1, -1, 1)  # (1, C, 1)
                                voice_sequences[b].append(frame_pred.squeeze(0))  # (C, 1)
                                last_voice_pred[b] = frame_pred.squeeze(0)  # (C, 1)
                        else:
                            frame_pred = coda_out["voice_latent_preds"]  # (1, C, 1) = Gaussian mean
                            # Heteroscedastic sampling: if the coda emitted a log-variance
                            # and temperature > 0, draw the frame from N(mu, (temp*std)^2).
                            # The sampled frame is BOTH emitted and fed back to the prelude
                            # at the next step (so downstream context sees the sample, not
                            # the mean). temperature 0 or a deterministic coda => mean.
                            voice_logvar = coda_out.get("voice_latent_logvar")
                            if voice_logvar is not None and voice_temperature > 0.0:
                                std = torch.exp(0.5 * voice_logvar)  # (1, C, 1)
                                if voice_variance_floor > 0.0:
                                    std = std.clamp_min(voice_variance_floor)
                                frame_pred = frame_pred + voice_temperature * std * torch.randn_like(frame_pred)
                            voice_sequences[b].append(frame_pred.squeeze(0))  # (C, 1)
                            last_voice_pred[b] = frame_pred.squeeze(0)  # (C, 1)
                            # Stop head (CONTINUOUS path only; the discrete path stops on the
                            # EOV unit above and has no stop head). Guarded: a discrete model
                            # whose codebook was not installed lands here with no stop head --
                            # degrade to budget-based stopping instead of a hard KeyError.
                            stop_logit = coda_out.get("voice_stop_logits")
                            if stop_logit is not None:
                                stop_logit_val = stop_logit[0, 0].item()
                                voice_stop_logit_trace[b].append(stop_logit_val)
                                if torch.sigmoid(torch.tensor(stop_logit_val)).item() > 0.5:
                                    should_stop_voice = True
                    else:
                        voice_sequences[b].append(current_hidden[b])

                    # Stop on predicted stop, hard budget, or the oracle-length cut.
                    _cut = (voice_exact_frames is not None
                            and (len(voice_unit_id_trace[b]) - voice_seg_start[b]) >= int(voice_exact_frames))
                    if should_stop_voice or _cut or len(voice_sequences[b]) >= voice_token_budget:
                        # A CHUNK boundary (fill_token) tears the voice block down but keeps
                        # the utterance alive: frames, F0, and the prelude/coda KV caches all
                        # carry across. The caches especially -- in training the coda runs
                        # causally over the whole utterance's expanded stream with no gap at a
                        # chunk boundary, so resetting them here would give inference a shorter
                        # acoustic history than training ever had.
                        _continue = bool(should_continue_chunk[b])
                        if _continue:
                            _nxt = None
                            if (voice_bistream_text is not None and voice_bistream_text_chunk > 0
                                    and voice_text_cursor[b] < int(voice_bistream_text.shape[1])):
                                _end = min(voice_text_cursor[b] + int(voice_bistream_text_chunk),
                                           int(voice_bistream_text.shape[1]))
                                _nxt = [int(v) for v in voice_bistream_text[b, voice_text_cursor[b]:_end]]
                                voice_text_cursor[b] = _end
                            if not _nxt:
                                # The model asked for text that does not exist (transcript
                                # exhausted, or unistream decoding of a bistream-capable head).
                                # Treat the fill as end-of-utterance rather than handing back to
                                # a text stream with nothing to say.
                                _continue = False
                            else:
                                forced_token_queue[b] = _nxt + [self._sp.BOV]
                        should_continue_chunk[b] = False
                        voice_chunk_start[b] = len(voice_unit_id_trace[b])
                        if not _continue and len(voice_unit_id_trace[b]) > voice_seg_start[b]:
                            voice_unit_id_segments[b].append(voice_unit_id_trace[b][voice_seg_start[b]:])
                            voice_seg_start[b] = len(voice_unit_id_trace[b])
                        current_modality[b] = None
                        # `and voice_sequences[b]`: EOV can fire on the FIRST step (an
                        # undertrained model), leaving no frames -- _finalize_voice would
                        # torch.cat([]) and crash. An empty utterance is a valid outcome
                        # (the model chose to emit nothing); skip finalizing it, mirroring
                        # the end-of-generation flush guard below.
                        if _continue:
                            # Next chunk's first trunk input is prelude(feature at the fill
                            # slot), and that feature is ZERO -- the collator zeroes every
                            # terminal column. So hand the prelude an explicit zero FEATURE,
                            # not None: None takes the position-0 branch, which is a zero in
                            # d_model space WITHOUT running the prelude, and only the
                            # utterance's very first position looks like that in training.
                            _C = self.voice_codebook.shape[1]
                            last_voice_pred[b] = torch.zeros(
                                _C, 1, device=device,
                                dtype=(last_voice_pred[b].dtype if last_voice_pred[b] is not None
                                       else current_hidden.dtype))
                        else:
                            if self.voice_generator is not None and voice_sequences[b]:
                                feat, f0 = _finalize_voice(voice_sequences[b], voice_f0_seq[b], voice_duration_seq[b])
                                completed_voice[b].append(feat)
                                if f0 is not None:
                                    completed_voice_f0[b].append(f0)
                            voice_sequences[b] = []
                            voice_f0_seq[b] = []
                            voice_duration_seq[b] = []
                            voice_prelude_kv_caches[b] = None
                            voice_prelude_position_offsets[b] = 0
                            voice_coda_kv_caches[b] = None
                            voice_coda_position_offsets[b] = 0
                            last_voice_pred[b] = None
                        # Inject VOICE_PLACEHOLDER into text_prelude KV cache so
                        # EOV's causal attention in iter N+1 sees VPH between
                        # BOV and EOV (matching training [..., BOV, VPH, EOV]).
                        _, text_prelude_kv_caches = self.text_feature_extractor(
                            torch.tensor([[self._sp.VOICE_PLACEHOLDER]], device=device),
                            kv_caches=text_prelude_kv_caches,
                            position_offset=text_prelude_position_offset,
                            use_cache=True,
                        )
                        text_prelude_position_offset += 1
                        forced_next_token[b] = self._sp.EOV
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
                    # An image consumes image_token_budget trunk positions in a SINGLE
                    # recurrent call, so the top-of-loop guard cannot catch an overflow here.
                    # Refuse the image rather than crash; the loop's guard then ends generation.
                    if _trunk_limit is not None and position_offset + image_token_budget > _trunk_limit:
                        hit_context_limit = True
                        break
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
                        cross_input = self._image_coda_input(image_hidden)
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
                        if "image_latent_preds" in cross_out:
                            image_pred = cross_out["image_latent_preds"].squeeze(0)  # (C, H, W)
                        else:
                            # SDXL adapter: predicts CLIP conditioning, not a latent. generate()
                            # can't render (SDXL isn't loaded), but we SURFACE the conditioning so
                            # a caller (chat UI / eval_sdxl_adapter.py) can render it via frozen SDXL.
                            image_pred = None
                            # pooled is None for the Z-Image adapter (Qwen3 has no pooled vector);
                            # SDXL provides a (1280,) pooled. Surface (seq, pooled|None) either way.
                            _pooled = cross_out.get("image_clip_pooled_pred")
                            completed_image_cond[b].append((
                                cross_out["image_clip_seq_pred"].squeeze(0).detach(),  # (seq_len, seq_dim)
                                _pooled.squeeze(0).detach() if _pooled is not None else None,
                            ))
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
                        torch.tensor([[self._sp.IMAGE_PLACEHOLDER]], device=device),
                        kv_caches=text_prelude_kv_caches,
                        position_offset=text_prelude_position_offset,
                        use_cache=True,
                    )
                    text_prelude_position_offset += 1
                    forced_next_token[b] = self._sp.EOI

            # The image branch above breaks out of the PER-BATCH loop when an image will not
            # fit in the trunk's remaining context. Propagate that to the generation loop --
            # without this the AR loop would retry the same image every step until
            # max_new_tokens ran out, since position_offset never advances.
            if hit_context_limit:
                break

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
                    forced_ids = [forced_next_token[b] if forced_next_token[b] is not None
                                  else (forced_token_queue[b].pop(0) if forced_token_queue[b]
                                        else self._eos)
                                  for b in range(batch_size)]
                    next_token_ids = torch.tensor(forced_ids, device=device)
                    for b in range(batch_size):
                        generated_tokens[b].append(next_token_ids[b].item())
                else:
                    sampled = self._sample_tokens(_ban(logits), temperature, top_k, top_p)
                    # Override sample with forced EO* for batch items that just
                    # finalized a media block (e.g. image) — this turns EO*
                    # into the actual next token driving iter N+1, so
                    # text_prelude → recurrent → text_coda all process EO* as
                    # a real position matching training.
                    for b in range(batch_size):
                        if forced_next_token[b] is not None:
                            sampled[b] = forced_next_token[b]
                        elif forced_token_queue[b]:
                            # Bistream TTS: the transcript is fed, not invented. (Under inner
                            # monologue the queue stays empty and these positions are sampled,
                            # which is the only difference between the two modes at decode.)
                            sampled[b] = forced_token_queue[b].pop(0)
                    next_token_ids = sampled
                    for b in range(batch_size):
                        generated_tokens[b].append(next_token_ids[b].item())

            # Check for EOS — stop generation for batch items that produced it
            if not any_media:
                all_done = True
                for b in range(batch_size):
                    if next_token_ids[b].item() == self._eos:
                        finished[b] = True
                    if not finished[b]:
                        all_done = False
                if all_done:
                    break

        # Flush media still in progress when the max_new_tokens budget ran out.
        #
        # The in-loop flush fires only on the stop head or voice_token_budget, so a voice
        # that is still streaming when the token budget expires would otherwise be dropped
        # on the floor -- generate() returns NO voice at all rather than a truncated one.
        # That is silent and it is worst exactly when you most want to look: an
        # undertrained stop head does not fire, so every eval sample comes back empty and
        # eval_voice_synthesis reports "No voice generated" for all of them. A truncated
        # voice is still measurable; nothing is not.
        for b in range(batch_size):
            if voice_sequences[b] and self.voice_generator is not None:
                feat, f0 = _finalize_voice(voice_sequences[b], voice_f0_seq[b], voice_duration_seq[b])
                completed_voice[b].append(feat)
                if f0 is not None:
                    completed_voice_f0[b].append(f0)
                voice_sequences[b], voice_f0_seq[b], voice_duration_seq[b] = [], [], []
            if len(voice_unit_id_trace[b]) > voice_seg_start[b]:
                voice_unit_id_segments[b].append(voice_unit_id_trace[b][voice_seg_start[b]:])
                voice_seg_start[b] = len(voice_unit_id_trace[b])
            if audio_sequences[b] and self.audio_generator is not None:
                completed_audio[b].append(torch.cat(audio_sequences[b], dim=-1))
                audio_sequences[b] = []

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

        # SDXL-adapter path: per-image (seq, pooled) CLIP conditioning for the caller to render.
        if any(len(c) > 0 for c in completed_image_cond):
            outputs["image_clip_cond"] = completed_image_cond  # List[List[(seq 77x2048, pooled 1280)]]

        # Per-image recurrent iteration counts (list-of-list; one entry per
        # completed image per batch item). Empty list if no images generated.
        outputs["image_recurrent_iterations"] = image_recurrent_iterations

        # Per-frame stop-head logit traces (diagnostic — see init comment).
        # List-of-list Python objects (not tensors) since lengths vary per
        # batch item. Empty list when no voice/audio was generated.
        outputs["voice_stop_logit_trace"] = voice_stop_logit_trace
        outputs["voice_unit_id_trace"] = voice_unit_id_trace
        # Same units, split at segment boundaries. Score segments[b][0] to measure ONE
        # utterance; the flat trace above may span several.
        outputs["voice_unit_id_segments"] = voice_unit_id_segments
        # Per-step entropy (bits) of the raw unit distribution at each generated frame.
        outputs["voice_unit_entropy_trace"] = voice_unit_entropy_trace
        # Padded like voice_latent_preds (same helper, same time_dim) so the contour lines
        # up frame-for-frame with the units it belongs to.
        #
        # This is a speaker-NORMALIZED contour, (log_f0 - mu_spk) / sd_spk, in sigma units
        # -- not log Hz. Pass it as SMG.decode(f0_contour=...) so the SMG's F0 predictor
        # denormalizes it with ECAPA. Do NOT hand it to SMG.f0_embedding(log_f0, voiced)
        # directly: that expects absolute log Hz and would read ~1.1 sigma as ~3 Hz. The
        # split exists because the speaker offset is the larger term (between-speaker
        # spread of mean log-F0 0.267 vs within-speaker contour spread 0.195) and is
        # exactly the part a text-only model cannot know.
        if any(len(f) > 0 for f in completed_voice_f0):
            stacked, _, _ = self._stack_variable_length_media(
                completed_voice_f0, device, time_dim=-1
            )
            outputs["voice_f0_preds"] = stacked
        outputs["audio_stop_logit_trace"] = audio_stop_logit_trace
        # True when generation stopped because the trunk ran out of context rather than
        # because it emitted EOS or exhausted max_new_tokens. Callers should surface this --
        # silently truncated output is indistinguishable from a model that chose to stop.
        outputs["hit_context_limit"] = hit_context_limit

        return outputs

    def _sample_tokens(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> torch.Tensor:
        """Sample tokens from logits with temperature, top-k, and top-p."""
        # temperature <= 0 means GREEDY. Without this the division below produces +-inf,
        # softmax turns that into NaN, and torch.multinomial fails with a device-side assert
        # whose traceback points at the sampler rather than at the argument -- so a perfectly
        # reasonable `temperature=0` looked like a CUDA/driver fault. The voice sampler has
        # taken 0 as greedy all along; this makes the text sampler agree.
        if temperature is None or temperature <= 0.0:
            return logits.argmax(-1)
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
