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
            from transformers import AutoModelForCausalLM
            llm = AutoModelForCausalLM.from_pretrained(text_encoder["model"], dtype=torch.float32)
            tgt_vocab = text_encoder.get("vocab_size", None)
            if tgt_vocab and tgt_vocab != llm.config.vocab_size:
                llm.resize_token_embeddings(tgt_vocab)
            self.lm_head = llm.get_output_embeddings()  # keep only the head; body is GC'd
            if text_encoder.get("freeze", True):
                for p in self.lm_head.parameters():
                    p.requires_grad = False
            llm_d = llm.config.hidden_size
            trunk_d = coda_config.d_model
            hidden = max(1, int(trunk_d * text_encoder.get("translator_hidden_mult", 2.0)))
            self.out_translator_norm = create_norm(trunk_d, config.input_norm_type, config.norm_epsilon)
            self.out_translator = nn.Sequential(
                nn.Linear(trunk_d, hidden), nn.GELU(), nn.Linear(hidden, llm_d))
            del llm
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
            # Stateless: residual-MLP translator (trunk d_model -> LLM d_model) then the
            # pretrained LM head. No coda self-attention -> kv_caches passes through untouched.
            h = self.out_translator(self.out_translator_norm(x))
            logits = self.lm_head(h.to(self.lm_head.weight.dtype))
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
