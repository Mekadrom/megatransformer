"""Content-encoder comparison: one INTERFACE, several implementations.

Every encoder maps a 16 kHz mono waveform to a per-frame continuous content
embedding `feats` [L, D] (what a downstream SMG would consume) plus, when the
model is natively discrete, its integer token ids `codes` [L]. The eval harness
(extract.py + compare.py) is written against this interface only, so adding a
model = adding one subclass + a registry line, nothing else.

Frame rates differ on purpose (SIVE ~20.8, ContentVec 50, Mimi 12.5) — that IS
one of the axes under test (redundancy/dedup), so the harness never resamples
features to a common rate; it compares each at its native rate.

Sample-rate note: the CANONICAL input is 16 kHz (the dataset). Each encoder
resamples internally to whatever IT needs (Mimi wants 24 kHz).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torchaudio.functional as AF


@dataclass
class EncodeOut:
    feats: torch.Tensor            # [L, D] continuous per-frame content embedding (cpu fp32)
    codes: Optional[torch.Tensor]  # [L] native discrete token ids (cpu long), or None
    frame_rate: float              # Hz


class ContentEncoder:
    name: str = "base"
    dim: int = 0
    frame_rate: float = 0.0
    is_discrete: bool = False

    @torch.no_grad()
    def encode(self, wav_16k: torch.Tensor) -> EncodeOut:
        """wav_16k: 1D float tensor @16 kHz -> EncodeOut."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# SIVE (our model) — continuous features @ ~20.8 Hz. The baseline everything is
# measured against. Uses the same mel front-end as training / sive_cer_eval.
# --------------------------------------------------------------------------- #
class SIVEEncoder(ContentEncoder):
    is_discrete = False

    def __init__(self, checkpoint, config="small_deep_3xdownsample_conv2d_attentive",
                 num_speakers=3610, activation="swiglu", use_std_hinge=True, device="cuda"):
        from megatransformer.model.voice.sive.sive import SpeakerInvariantVoiceEncoder
        from megatransformer.utils.model_loading_utils import detect_sive_variant, load_model
        from megatransformer.utils.audio_utils import SharedWindowBuffer

        vq = detect_sive_variant(checkpoint)
        if vq.get("use_vq"):
            vq["vq_cosine"] = True
        overrides = {"num_speakers": num_speakers, "activation": activation,
                     "use_std_hinge": use_std_hinge, **vq}
        self.model = load_model(SpeakerInvariantVoiceEncoder, config,
                                checkpoint_path=checkpoint, overrides=overrides)
        self.model.to(device).eval()
        self.device = device
        self.swb = SharedWindowBuffer()
        self.dim = self.model.config.encoder_dim
        # 16 kHz, hop 256 -> 62.5 Hz mel, 3x temporal downsample -> ~20.83 Hz
        self.frame_rate = 16000 / 256 / 3
        self.is_discrete = bool(vq.get("use_vq", False))
        self.name = "sive"

    @torch.no_grad()
    def encode(self, wav_16k):
        from megatransformer.utils.audio_utils import extract_mels
        wav = wav_16k.to(self.device).float().reshape(-1)
        mel = extract_mels(self.swb, wav, sr=16000, n_mels=80, n_fft=1024, hop_length=256)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)  # [1, n_mels, T]
        mlen = torch.tensor([mel.shape[-1]], device=self.device)
        out = self.model(mel, lengths=mlen, grl_alpha=0.0)
        feats = out["features"][0].float().cpu()  # [T, D]
        codes = None
        if self.is_discrete and out.get("vq_codes") is not None:
            codes = out["vq_codes"][0].long().cpu()
        return EncodeOut(feats=feats, codes=codes, frame_rate=self.frame_rate)


# --------------------------------------------------------------------------- #
# ContentVec — off-the-shelf speaker-DISENTANGLED HuBERT @ 50 Hz, continuous.
# The "explicitly invariant but high-rate/redundant" reference point.
# --------------------------------------------------------------------------- #
class ContentVecEncoder(ContentEncoder):
    is_discrete = False

    def __init__(self, model_id="lengyue233/content-vec-best", dim=256, device="cuda"):
        from megatransformer.utils.contentvec_features import load_contentvec
        self.model = load_contentvec(model_id, device, dim=dim)
        self.device = device
        self.dim = dim
        self._final_proj = (dim == 256)
        self.frame_rate = 50.0
        self.name = "contentvec"

    @torch.no_grad()
    def encode(self, wav_16k):
        from megatransformer.utils.contentvec_features import contentvec_hidden
        h = contentvec_hidden(self.model, wav_16k.to(self.device).float(),
                              layer=-1, final_proj=self._final_proj)  # [T', D]
        return EncodeOut(feats=h.float().cpu(), codes=None, frame_rate=self.frame_rate)


# --------------------------------------------------------------------------- #
# Mimi (Kyutai/Moshi) — neural codec @ 12.5 Hz, 24 kHz. Split-RVQ: codebook 0 is
# the WavLM-distilled SEMANTIC stream (the content-bearing one); we take its ids
# and its dequantized embedding. NOT speaker-invariant (that's the whole point of
# the leakage axis). Native discrete.
# --------------------------------------------------------------------------- #
class MimiEncoder(ContentEncoder):
    is_discrete = True

    def __init__(self, model_id="kyutai/mimi", device="cuda"):
        from transformers import MimiModel
        self.model = MimiModel.from_pretrained(model_id).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device
        self.model_sr = self.model.config.sampling_rate  # 24000
        self.frame_rate = float(self.model.config.frame_rate)  # 12.5
        self.dim = self.model.config.hidden_size  # 512
        self.name = "mimi"

    @torch.no_grad()
    def encode(self, wav_16k):
        wav = wav_16k.to(self.device).float().reshape(1, 1, -1)
        if self.model_sr != 16000:
            wav = AF.resample(wav, 16000, self.model_sr)
        enc = self.model.encode(wav, num_quantizers=1)  # semantic codebook only
        codes = enc.audio_codes if hasattr(enc, "audio_codes") else enc[0]  # [1, 1, L]
        codes = codes[:, :1, :]  # keep codebook 0
        # dequantize codebook 0 -> continuous semantic embedding [1, D, L]
        emb = self.model.quantizer.decode(codes)
        feats = emb[0].transpose(0, 1).float().cpu()  # [L, D]
        ids = codes[0, 0].long().cpu()                 # [L]
        return EncodeOut(feats=feats, codes=ids, frame_rate=self.frame_rate)


# --------------------------------------------------------------------------- #
# Registry — extract.py resolves --encoder against this.
# --------------------------------------------------------------------------- #
def build_encoder(name: str, device="cuda", **kw) -> ContentEncoder:
    if name == "sive":
        return SIVEEncoder(checkpoint=kw["checkpoint"], device=device)
    if name == "contentvec":
        return ContentVecEncoder(dim=kw.get("dim", 256), device=device)
    if name == "mimi":
        return MimiEncoder(device=device)
    # SpeechTokenizer / GLM-4-Voice added in encoders_extra.py (needs installs)
    from scripts_local.content_encoder_eval import encoders_extra
    return encoders_extra.build_extra(name, device=device, **kw)
