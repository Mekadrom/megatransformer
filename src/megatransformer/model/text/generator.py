from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from megatransformer.config.text.generator import TextCodaClassifierConfig
from megatransformer.model.norms import create_norm
from megatransformer.model.transformer import MegaTransformerEncoderBlock
from megatransformer.utils.megatransformer_utils import (
    apply_depth_scaled_residual_init,
    linear_weight_init,
)


class TextCodaClassifierWithLoss(nn.Module):
    """
    Text output head for the multimodal world model.

    Processes hidden states from the main transformer through a modality-specific
    coda (a small transformer stack) and projects to vocabulary logits for next-token
    prediction. The coda allows the model to specialize its final processing for
    text generation separately from other modalities.

    The architecture uses a residual connection around the coda transformer.
    """

    def __init__(self, config: TextCodaClassifierConfig, text_encoder: Optional[dict] = None):
        super(TextCodaClassifierWithLoss, self).__init__()

        self.config = config
        coda_config = config.coda_config
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

        # PRETRAINED-LLM MODE (opt-in). Replace {transformer coda + lm_head} with a small
        # residual-MLP translator (trunk d_model -> LLM d_model) + the pretrained LM head. The
        # coda is STATELESS here (no self-attention/KV cache); forward signature is preserved.
        self._pretrained = text_encoder is not None
        if self._pretrained:
            from transformers import AutoConfig
            # No LLM weights here: llm_d comes from the config, and the LM head is SHARED from the
            # feature extractor (assigned by world_model) so the pretrained embed/head tie holds.
            llm_cfg = AutoConfig.from_pretrained(text_encoder["model"])
            llm_d = int(llm_cfg.hidden_size)
            trunk_d = coda_config.d_model
            hidden = max(1, int(trunk_d * text_encoder.get("translator_hidden_mult", 2.0)))
            self.n_special = int(text_encoder.get("n_special_tokens", 0))
            # `trainable_head` (opt-in; default False leaves the shared-head path below untouched):
            # keep the frozen LLM on the INPUT side but give the coda its OWN readout at TRUNK
            # width. The shared path routes every output distribution through an llm_d bottleneck
            # AND a frozen basis — logit-matrix rank <= llm_d+1, output directions pinned to the
            # LLM's token geometry — which is a real ceiling once the trunk is wider than the LLM.
            # Here one trainable matrix spans native vocab + control tokens, so the control tokens
            # are native rows rather than a bolted-on `special_head`, and nothing is tied.
            self.trainable_head = bool(text_encoder.get("trainable_head", False))
            self.out_translator_norm = create_norm(trunk_d, config.input_norm_type, config.norm_epsilon)
            if self.trainable_head:
                # Residual at trunk width — the shared path cannot be residual (it changes dim).
                self.out_translator = nn.Sequential(
                    nn.Linear(trunk_d, hidden), nn.GELU(), nn.Linear(hidden, trunk_d))
                self.lm_head = nn.Linear(trunk_d, int(llm_cfg.vocab_size) + self.n_special)
                # Xavier on both, matching the from-scratch coda's convention (the pretrained
                # branch returns before _init_weights, so this has to be explicit).
                self.apply(linear_weight_init(gain=1.0))
                self.gradient_checkpointing = False
                return
            self.out_translator = nn.Sequential(
                nn.Linear(trunk_d, hidden), nn.GELU(), nn.Linear(hidden, llm_d))
            # Trainable head for the control tokens; its weight is tied to the FE's special_embed
            # by world_model (embed row == head row, mirroring the LLM's own embed/head tie).
            if self.n_special > 0:
                self.special_head = nn.Linear(llm_d, self.n_special, bias=False)
            self.lm_head = None  # SHARED from the feature extractor, injected by world_model
            self.gradient_checkpointing = False
            return

        if config.use_input_norm:
            self.input_norm = create_norm(coda_config.d_model, config.input_norm_type, config.norm_epsilon)

        self.coda = nn.ModuleList([
            MegaTransformerEncoderBlock(coda_config)
            for _ in range(config.n_layers)
        ])

        self.lm_head = nn.Linear(coda_config.d_model, config.vocab_size)

        self.gradient_checkpointing = False
        self._init_weights()

    def _init_weights(self):
        # Standard Xavier on every Linear (coda blocks + lm_head), then
        # depth-scaled init for the coda blocks' residual outputs.
        self.apply(linear_weight_init(gain=1.0))
        apply_depth_scaled_residual_init(self.coda)

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List] = None,
        position_offset: int = 0,
        use_cache: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Takes inputs in the shape (batch_size, seq_length, d_model) and processes them through the Coda and classification head.

        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model).
            targets: Target token IDs for computing classification loss.
            kv_caches: Optional list of KVCache objects, one per coda layer.
                When provided, the coda's self-attention attends to cached
                representations from previous positions (inference only).
            position_offset: RoPE position offset for cached generation.
            use_cache: If True, return updated KV caches in the output dict.

        Returns:
            dict with "logits" and optionally "text_classification_loss" and "kv_caches".
        """
        if self._pretrained:
            # Stateless: MLP translator, then a readout. Shared-head mode: translate trunk d_model
            # -> LLM d_model, apply the SHARED (tied) LM head over the native vocab, concatenate the
            # trainable special_head over the control tokens. trainable_head mode: stay at trunk
            # width (residual) and apply one owned head over vocab + control tokens.
            # No coda self-attention either way -> kv_caches passes through.
            h = self.out_translator(self.out_translator_norm(x))
            if self.trainable_head:
                h = x + h  # residual: trunk width in, trunk width out
            h = h.to(self.lm_head.weight.dtype)
            logits = self.lm_head(h)
            if self.n_special > 0 and not self.trainable_head:
                logits = torch.cat([logits, self.special_head(h)], dim=-1)
            cap = getattr(self.config, 'lm_head_logit_cap', None)
            if cap is not None:
                logits = cap * torch.tanh(logits / cap)
            output = {"logits": logits}
            if use_cache:
                output["kv_caches"] = kv_caches
            if targets is not None:
                B, T, V = logits.size()
                output["text_classification_loss"] = self.loss_fn(
                    logits.view(B * T, V), targets.view(B * T))
            return output

        if hasattr(self, 'input_norm'):
            x = self.input_norm(x)

        # MegaTransformerEncoderBlock.forward already adds residuals internally,
        # so the loop just chains layers without re-adding the input.
        h = x
        new_kv_caches = []
        for i, block in enumerate(self.coda):
            block_cache = kv_caches[i] if kv_caches is not None else None
            if self.gradient_checkpointing and self.training and not use_cache:
                h, new_cache = torch_checkpoint(
                    block, h, None, None, block_cache, position_offset, use_cache,
                    use_reentrant=False,
                )
            else:
                h, new_cache = block(
                    h,
                    kv_cache=block_cache,
                    position_offset=position_offset,
                    use_cache=use_cache,
                )
            if use_cache:
                new_kv_caches.append(new_cache)

        logits: torch.Tensor = self.lm_head(h)

        # Soft logit capping (Gemma 2-style): bounds logits within [-cap, cap]
        # to prevent overconfident predictions. Pairs with label smoothing.
        cap = getattr(self.config, 'lm_head_logit_cap', None)
        if cap is not None:
            logits = cap * torch.tanh(logits / cap)

        output = {"logits": logits}
        if use_cache:
            output["kv_caches"] = new_kv_caches

        if targets is not None:
            batch_size, seq_length, vocab_size = logits.size()
            loss = self.loss_fn(
                logits.view(batch_size * seq_length, vocab_size),
                targets.view(batch_size * seq_length)
            )
            output["text_classification_loss"] = loss

        return output
