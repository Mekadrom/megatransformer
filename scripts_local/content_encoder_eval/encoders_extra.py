"""Install-required content encoders, kept out of encoders.py so the core three
(SIVE / ContentVec / Mimi) import with zero extra deps.

- SpeechTokenizer: pip `speechtokenizer` (+beartype). 50 Hz RVQ, 8 codebooks;
  codebook 0 is the HuBERT-distilled SEMANTIC stream (the content-bearing one).
- GLM-4-Voice tokenizer: Whisper-large encoder + 4x avg-pool -> 12.5 Hz + single VQ
  (16384). Needs GLM-4-Voice's WhisperVQEncoder class; if it won't subclass against
  the installed transformers, extract it in an isolated venv from a raw-waveform dump
  (see extract_isolated.py) instead of here.
"""
from __future__ import annotations

import torch

from scripts_local.content_encoder_eval.encoders import ContentEncoder, EncodeOut


class SpeechTokenizerEncoder(ContentEncoder):
    is_discrete = True

    def __init__(self, device="cuda",
                 repo="fnlp/SpeechTokenizer", subdir="speechtokenizer_hubert_avg"):
        from huggingface_hub import hf_hub_download
        from speechtokenizer import SpeechTokenizer
        cfg = hf_hub_download(repo, f"{subdir}/config.json")
        ckpt = hf_hub_download(repo, f"{subdir}/SpeechTokenizer.pt")
        self.model = SpeechTokenizer.load_from_checkpoint(cfg, ckpt).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device
        self.frame_rate = 50.0            # 16 kHz, 320x hop
        self.dim = 1024                   # semantic codebook embedding width
        self.name = "speechtokenizer"

    @torch.no_grad()
    def encode(self, wav_16k):
        w = wav_16k.to(self.device).float().reshape(1, 1, -1)
        codes = self.model.encode(w)                      # [8, 1, T]
        emb0 = self.model.quantizer.decode(codes[:1])     # [1, 1024, T] semantic only
        feats = emb0[0].transpose(0, 1).float().cpu()     # [T, 1024]
        ids = codes[0, 0].long().cpu()                    # [T]
        return EncodeOut(feats=feats, codes=ids, frame_rate=self.frame_rate)


def build_extra(name, device="cuda", **kw):
    if name == "speechtokenizer":
        return SpeechTokenizerEncoder(device=device)
    if name == "glm":
        raise RuntimeError(
            "GLM-4-Voice cannot run in the training .venv (needs transformers==4.44). "
            "Extract it via the isolated venv: dump_raw_subset.py then "
            "`<glmvenv>/bin/python scripts_local/content_encoder_eval/extract_glm.py`.")
    raise ValueError(f"unknown encoder: {name}")
