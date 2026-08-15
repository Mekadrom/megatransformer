from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from megatransformer.config.text.feature_extractor import TextPreludeFeatureExtractorConfig
from megatransformer.model.norms import create_norm
from megatransformer.model.sinusoidal_positional_encoding import SinusoidalPositionalEncoding
from megatransformer.model.transformer import MegaTransformerEncoderBlock
from megatransformer.utils import megatransformer_utils
from megatransformer.utils.megatransformer_utils import (
    apply_depth_scaled_residual_init,
    embedding_weight_init,
    linear_weight_init,
)


class TextPreludeFeatureExtractor(nn.Module):
    """
    Embeds text token IDs into dense vectors for the multimodal world model.

    Unlike audio and image modalities, text has no prelude transformer because
    processing text tokens in isolation (without interleaved modalities) provides
    limited benefit. The text embeddings are designed to be immediately interleaved
    with media embeddings via TokenInterleaver before entering the main transformer.

    The embedding includes optional layer normalization and dropout for regularization.
    """

    def __init__(self, config: TextPreludeFeatureExtractorConfig, text_encoder: Optional[dict] = None):
        super().__init__()

        self.config = config
        prelude_config = config.prelude_config

        # PRETRAINED-LLM MODE (opt-in). The from-scratch {wte + prelude} below is replaced by a
        # pretrained causal LM body + an input projection (LLM d_model -> trunk d_model) + a
        # small residual-MLP translator. The forward signature is preserved, so world_model.py
        # is unchanged; the LLM's HF KV cache rides through the same kv_caches slot.
        self._pretrained = text_encoder is not None
        if self._pretrained:
            from transformers import AutoModelForCausalLM
            from megatransformer.model.norms import create_norm as _cn
            # Load the full causal LM ONCE (fp32 at init; --bf16 autocast handles training). Keep
            # its body (encoder) AND its tied LM head — the text coda references this same head,
            # so there is one embed/head weight (no duplicate) and the tie survives an unfreeze.
            llm = AutoModelForCausalLM.from_pretrained(text_encoder["model"], dtype=torch.float32)
            self.llm_body = llm.model                     # transformer body (encoder)
            self.lm_head = llm.get_output_embeddings()    # tied head; coda references this
            self.llm_native_vocab = int(llm.config.vocab_size)
            llm_d = int(llm.config.hidden_size)
            del llm
            if text_encoder.get("freeze", True):
                for p in self.llm_body.parameters():
                    p.requires_grad = False
                for p in self.lm_head.parameters():
                    p.requires_grad = False
                self.llm_body.eval()
            # Trainable vocab-EXTENSION for the multimodal control tokens (BOV/EOV/placeholders):
            # a small embedding at ids >= native_vocab, spliced into inputs_embeds. Frozen-LLM-safe
            # (soft-prompt style); its weight is tied to the coda's special_head by world_model.
            self.n_special = int(text_encoder.get("n_special_tokens", 0))
            if self.n_special > 0:
                self.special_embed = nn.Embedding(self.n_special, llm_d)
            trunk_d = config.d_model
            hidden = max(1, int(trunk_d * text_encoder.get("translator_hidden_mult", 2.0)))
            self.input_proj = nn.Linear(llm_d, trunk_d)
            self.translator_norm = _cn(trunk_d, config.output_norm_type, config.norm_epsilon)
            self.translator = nn.Sequential(
                nn.Linear(trunk_d, hidden), nn.GELU(), nn.Linear(hidden, trunk_d))
            self.gradient_checkpointing = False
            return

        self.wte = nn.Embedding(config.vocab_size, config.d_model)

        self.prelude = nn.ModuleList([
            MegaTransformerEncoderBlock(prelude_config)
            for _ in range(config.n_layers)
        ])

        if (not prelude_config.use_rotary_embedding) or config.use_pos_emb_ovr:
            self.pos_encoding = SinusoidalPositionalEncoding(
                d_model=prelude_config.d_model,
                max_len=config.max_position_embeddings * 2 + 1,
                dropout=0.0
            )

        if config.use_output_norm:
            self.output_norm = create_norm(config.d_model, config.output_norm_type, config.norm_epsilon)

        self.gradient_checkpointing = False
        self._init_weights()

    def _init_weights(self):
        # Embeddings: N(0, 1/sqrt(d_model)) so embedding magnitude is ~1.
        self.apply(embedding_weight_init(self.config.d_model))
        # Prelude transformer: standard Xavier on linears + depth-scaled
        # residual outputs. Previously this was missing entirely (linear
        # layers fell through to PyTorch's kaiming_uniform default).
        init_linear = linear_weight_init(gain=1.0)
        for block in self.prelude:
            block.apply(init_linear)
        apply_depth_scaled_residual_init(self.prelude)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[List] = None,
        position_offset: int = 0,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """
        Embed text token IDs into dense vectors.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            kv_caches: Optional list of KVCache objects, one per prelude layer.
                Used for autoregressive generation so token t's prelude
                self-attention sees tokens 0..t-1 (matching training).
            position_offset: Position offset for sinusoidal PE and RoPE when
                using KV caching.
            use_cache: If True, return (embeddings, kv_caches) tuple.

        Returns:
            Hidden states of shape (batch_size, seq_len, d_model).
            If use_cache=True, returns (hidden_states, new_kv_caches) tuple.
        """
        if self._pretrained:
            # LLM body handles positions/causality internally via its HF cache; position_offset is
            # ignored (cache length gives position). kv_caches carries the HF Cache opaquely.
            if self.n_special > 0:
                # Control-token ids (>= native vocab) go through the trainable special_embed,
                # everything else through the frozen LLM embedding; splice into inputs_embeds.
                normal_ids = input_ids.clamp(max=self.llm_native_vocab - 1)
                embeds = self.llm_body.get_input_embeddings()(normal_ids)
                mask = input_ids >= self.llm_native_vocab
                if mask.any():
                    se = self.special_embed(input_ids[mask] - self.llm_native_vocab)
                    embeds = embeds.clone()
                    embeds[mask] = se.to(embeds.dtype)
                out = self.llm_body(inputs_embeds=embeds, past_key_values=kv_caches, use_cache=use_cache)
            else:
                out = self.llm_body(input_ids=input_ids, past_key_values=kv_caches, use_cache=use_cache)
            # cast to the translator's dtype (handles a bf16 LLM output vs fp32 proj, and autocast)
            h = self.input_proj(out.last_hidden_state.to(self.input_proj.weight.dtype))
            h = h + self.translator(self.translator_norm(h))
            if use_cache:
                return h, out.past_key_values
            return h

        projected = self.wte(input_ids)

        if hasattr(self, 'pos_encoding'):
            projected = self.pos_encoding(projected, offset=position_offset)

        # MegaTransformerEncoderBlock.forward already adds residuals internally,
        # so the loop just chains layers without re-adding the input.
        x = projected
        new_kv_caches = []
        for i, block in enumerate(self.prelude):
            block_cache = kv_caches[i] if kv_caches is not None else None
            if self.gradient_checkpointing and self.training and not use_cache:
                x, new_cache = torch_checkpoint(
                    block, x, None, None, block_cache, position_offset, use_cache,
                    use_reentrant=False,
                )
            else:
                x, new_cache = block(
                    x,
                    kv_cache=block_cache,
                    position_offset=position_offset,
                    use_cache=use_cache,
                )
            if use_cache:
                new_kv_caches.append(new_cache)

        if hasattr(self, 'output_norm'):
            x = self.output_norm(x)

        if use_cache:
            return x, new_kv_caches
        return x
