"""Frozen CosyVoice 2 speech LM as a DISTILLATION TEACHER.

CosyVoice 2's Qwen2-0.5B LM solves the exact task the world model's voice path is trying to
learn -- text -> 25Hz speech tokens -- trained on ~30k hours of English (vs LibriTTS-R's
~585h). This exposes its per-position distribution over speech tokens so the student can be
trained with KL against it instead of (only) hard-label CE.

WHY KL AND NOT CE-ALONE: the target is one-to-many. Measured on this cache, the teacher's
predictive entropy is ~4.02 nats (~56 effectively-plausible continuations per position) and
its own top-1 is only 0.1245 -- yet it produces intelligible speech. Hard labels throw away
exactly the "which continuations are plausible" structure that distinguishes the two; the
soft targets carry it.

ONLY the TEACHER-FORCED forward is used (one full pass, no KV cache). That matters: the
CosyVoice AR decode path is BROKEN under transformers 5.x (it feeds a (1,1) attention mask
against a full cache), but the teacher-forced path is unaffected and correct in the project
venv -- verified against the isolated transformers-4.44 venv.

Sequence layout reproduced from Qwen2LM.forward (unistream):
    [sos_emb, text_token_emb, task_id_emb, speech_token_emb]
with logits at index (1 + n_text + i) predicting speech token i, and index
(1 + n_text + L) predicting eos -- which is id 6561, the SAME id as the student's EOV.
NOTE the speaker is absent by design: Qwen2LM overrides TransformerLM.forward and drops the
speaker slot, so this stream is purely content.
"""
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from megatransformer.model.voice.cosyvoice2_decoder import (
    DEFAULT_RUNTIME_DIR, _ensure_importable, _preserve_root_logging,
)

# Teacher head width (speech_token_size + 3). Ids 0..6560 content, 6561 eos, 6562/6563 are
# streaming-mode markers the unistream layout never emits.
TEACHER_VOCAB = 6564
CONTENT_PLUS_EOS = 6562          # == the student's unit_vocab (6561 content + EOV at 6561)


class CosyVoice2Teacher(nn.Module):
    def __init__(self, lm, tokenizer, student_vocab: int = CONTENT_PLUS_EOS):
        super().__init__()
        self.lm = lm
        self.tokenizer = tokenizer
        self.student_vocab = int(student_vocab)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @classmethod
    def from_pretrained(cls, model_dir: str, runtime_dir: Optional[str] = None,
                        device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                        student_vocab: int = CONTENT_PLUS_EOS):
        runtime_dir = runtime_dir or os.environ.get("COSYVOICE_RUNTIME", DEFAULT_RUNTIME_DIR)
        import functools
        with _preserve_root_logging():
            _ensure_importable(runtime_dir)
            from cosyvoice.llm.llm import Qwen2LM, Qwen2Encoder
            from cosyvoice.utils.common import ras_sampling
            from transformers import AutoTokenizer

            qwen_path = os.path.join(model_dir, "CosyVoice-BlankEN")
            lm = Qwen2LM(
                llm_input_size=896, llm_output_size=896, speech_token_size=6561,
                llm=Qwen2Encoder(pretrain_path=qwen_path),
                sampling=functools.partial(ras_sampling, top_p=0.8, top_k=25,
                                           win_size=10, tau_r=0.1),
                length_normalized_loss=True, lsm_weight=0, mix_ratio=[5, 15],
            )
            lm.load_state_dict(torch.load(os.path.join(model_dir, "llm.pt"),
                                          map_location="cpu", weights_only=False), strict=True)
            tok = AutoTokenizer.from_pretrained(qwen_path)
        # Cast the WHOLE module: the Qwen2 backbone loads bf16 while the bolted-on speech
        # modules are fp32, and the forward concatenates them -- mixed dtypes raise.
        lm = lm.to(device=device, dtype=dtype).eval()
        return cls(lm, tok, student_vocab).to(device=device, dtype=dtype)

    @torch.no_grad()
    def forward(
        self,
        texts: List[Optional[str]],
        unit_ids: torch.Tensor,
        unit_lengths: torch.Tensor,
        max_positions: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Teacher logits aligned 1:1 with the student's voice positions.

        Args:
            texts: per-sample transcripts (None entries are skipped).
            unit_ids: (B, T) GT content units (teacher forcing).
            unit_lengths: (B,) content-frame counts (the EOV frame sits at index length).
            max_positions: the student's voice time dim T.

        Returns:
            logits: (B, max_positions, student_vocab) -- teacher scores sliced to the
                student's label space (content + eos/EOV, which share id 6561).
            mask: (B, max_positions) bool, True where the teacher supervised a position
                (content frames plus the one EOV frame).
        """
        dev = next(self.lm.parameters()).device
        dt = next(self.lm.parameters()).dtype
        B = len(texts)
        out = torch.zeros(B, max_positions, self.student_vocab, device=dev, dtype=torch.float32)
        mask = torch.zeros(B, max_positions, dtype=torch.bool, device=dev)

        embed = self.lm.llm.model.model.embed_tokens
        sos = self.lm.llm_embedding.weight[self.lm.sos].reshape(1, 1, -1)
        task = self.lm.llm_embedding.weight[self.lm.task_id].reshape(1, 1, -1)

        rows, starts, spans = [], [], []
        for b in range(B):
            t = texts[b]
            L = int(unit_lengths[b]) if unit_lengths is not None else 0
            if not t or L <= 0:
                continue
            ids = self.tokenizer(t, return_tensors="pt")["input_ids"].to(dev)
            n_text = int(ids.shape[1])
            # +1 EOV position; clip so we never index past the student's tensor
            span = min(L + 1, max_positions)
            speech = unit_ids[b, :L].to(dev).long().clamp_(0, 6560).unsqueeze(0)
            seq = torch.cat([sos.to(dt), embed(ids).to(dt), task.to(dt),
                             self.lm.speech_embedding(speech).to(dt)], dim=1)
            rows.append(seq[0])
            starts.append(1 + n_text)
            spans.append(span)
        if not rows:
            return out, mask

        lens = torch.tensor([r.shape[0] for r in rows], dtype=torch.int32, device=dev)
        padded = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)
        hidden, _ = self.lm.llm(padded, lens)
        logits = self.lm.llm_decoder(hidden.to(dt)).float()   # (R, Tmax, 6564)

        r = 0
        for b in range(B):
            t = texts[b]
            L = int(unit_lengths[b]) if unit_lengths is not None else 0
            if not t or L <= 0:
                continue
            s, span = starts[r], spans[r]
            # index (1+n_text+i) predicts speech token i; (1+n_text+L) predicts eos(6561)
            sl = logits[r, s:s + span, :self.student_vocab]
            out[b, :sl.shape[0]] = sl
            mask[b, :sl.shape[0]] = True
            r += 1
        return out, mask
