"""Frozen CosyVoice 2 decoder: content unit ids -> 24 kHz waveform.

The world model's voice coda classifies over the CosyVoice 2 codebook (unit_head, K=6561
+ EOV), so its output is ALREADY int unit ids -- exactly CosyVoice 2's decoder input. This
wraps the two frozen pieces that turn those ids into audio:

    unit ids (25 Hz) -> flow (flow-matching) -> mel (80, 50 Hz) -> HiFT vocoder -> wav @24 kHz

Only `flow.pt` (112.6M) + `hift.pt` (20.8M) are loaded. Deliberately NOT loaded:
  - the Qwen LLM (641.9M)      -- the world-model trunk replaces it
  - speech_tokenizer_v2.onnx   -- built the cache; not needed to decode
  - campplus.onnx              -- speaker embeddings are already stored in the cache

Because the LLM is never constructed, the `llm:` block is stripped from cosyvoice2.yaml
before hyperpyyaml parses it (hyperpyyaml instantiates every top-level object eagerly).
That is what lets this run in the PROJECT venv (torch 2.6 / transformers 5.13) instead of
the isolated CosyVoice venv -- a transformers-version incompatibility in modeling code we
never call can't break the decode path.

The `cosyvoice` package is not pip-installed; it lives in a checkout (default
~/dev/projects/cosyvoice-runtime, override with $COSYVOICE_RUNTIME) whose `cv_extra`
directory holds `--no-deps` support packages. See that repo's README.md.
"""
import os
import sys
from typing import Optional

import torch


DEFAULT_RUNTIME_DIR = os.path.expanduser("~/dev/projects/cosyvoice-runtime")


def _ensure_importable(runtime_dir: str):
    """Put the CosyVoice checkout (+ Matcha, shims, no-deps extras) on sys.path."""
    if not os.path.isdir(runtime_dir):
        raise FileNotFoundError(
            f"CosyVoice runtime dir not found: {runtime_dir}. Set --voice_cosyvoice2_runtime_dir "
            f"or $COSYVOICE_RUNTIME."
        )
    for p in (
        os.path.join(runtime_dir, "cv_extra"),
        os.path.join(runtime_dir, "cv_shims"),
        os.path.join(runtime_dir, "CosyVoice", "third_party", "Matcha-TTS"),
        os.path.join(runtime_dir, "CosyVoice"),
    ):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def _load_configs_without_llm(model_dir: str):
    from hyperpyyaml import load_hyperpyyaml

    yaml_path = os.path.join(model_dir, "cosyvoice2.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"cosyvoice2.yaml not found in {model_dir}")
    with open(yaml_path) as f:
        raw = f.read()

    # Drop the top-level `llm:` block (its nested lines are indented; the block ends at the
    # next unindented line). hyperpyyaml would otherwise construct the Qwen2 wrapper.
    keep, skipping = [], False
    for ln in raw.split("\n"):
        if ln.startswith("llm:"):
            skipping = True
        elif skipping and ln and not ln[0].isspace():
            skipping = False
        if not skipping:
            keep.append(ln)
    return load_hyperpyyaml(
        "\n".join(keep),
        overrides={"qwen_pretrain_path": os.path.join(model_dir, "CosyVoice-BlankEN")},
    )


class CosyVoice2Decoder(torch.nn.Module):
    """Frozen flow + HiFT. `decode(unit_ids, speaker_embedding) -> (T,) waveform @ sample_rate`."""

    def __init__(self, flow, hift, sample_rate: int):
        super().__init__()
        self.flow = flow
        self.hift = hift
        self.sample_rate = int(sample_rate)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @classmethod
    def from_pretrained(cls, model_dir: str, runtime_dir: Optional[str] = None,
                        device: str = "cpu", dtype: torch.dtype = torch.float32):
        runtime_dir = runtime_dir or os.environ.get("COSYVOICE_RUNTIME", DEFAULT_RUNTIME_DIR)
        _ensure_importable(runtime_dir)
        configs = _load_configs_without_llm(model_dir)
        flow, hift = configs["flow"], configs["hift"]
        flow.load_state_dict(torch.load(os.path.join(model_dir, "flow.pt"),
                                        map_location="cpu", weights_only=False), strict=True)
        hift.load_state_dict(torch.load(os.path.join(model_dir, "hift.pt"),
                                        map_location="cpu", weights_only=False), strict=True)
        m = cls(flow, hift, int(configs["sample_rate"]))
        return m.to(device=device, dtype=dtype)

    @torch.no_grad()
    def decode(self, unit_ids: torch.Tensor, speaker_embedding: torch.Tensor) -> Optional[torch.Tensor]:
        """unit_ids: (T,) or (1,T) int content units (EOV/padding already stripped).
        speaker_embedding: (192,) or (1,192) campplus. Returns (T_wav,) float32 on CPU."""
        ids = unit_ids.reshape(1, -1).to(device=self._device, dtype=torch.int32)
        if ids.shape[1] == 0:
            return None
        emb = speaker_embedding.reshape(1, -1).to(device=self._device, dtype=self._dtype)

        kw = dict(
            token=ids,
            token_len=torch.tensor([ids.shape[1]], dtype=torch.int32, device=self._device),
            prompt_token=torch.zeros(1, 0, dtype=torch.int32, device=self._device),
            prompt_token_len=torch.tensor([0], dtype=torch.int32, device=self._device),
            prompt_feat=torch.zeros(1, 0, 80, device=self._device, dtype=self._dtype),
            prompt_feat_len=torch.tensor([0], dtype=torch.int32, device=self._device),
            embedding=emb,
        )
        # Signature drifts across CosyVoice revisions; pass only what this one declares.
        import inspect
        sig = inspect.signature(self.flow.inference).parameters
        if "flow_cache" in sig:
            kw["flow_cache"] = torch.zeros(1, 80, 0, 2, device=self._device, dtype=self._dtype)
        if "finalize" in sig:
            kw["finalize"] = True
        if "streaming" in sig:
            kw["streaming"] = False

        out = self.flow.inference(**kw)
        mel = out[0] if isinstance(out, tuple) else out
        if mel is None or mel.numel() == 0:
            return None
        wav, _ = self.hift.inference(
            speech_feat=mel.to(self._dtype),
            cache_source=torch.zeros(1, 1, 0, device=self._device, dtype=self._dtype),
        )
        return wav.reshape(-1).float().cpu()

    @property
    def _device(self):
        return next(self.parameters()).device

    @property
    def _dtype(self):
        return next(self.parameters()).dtype
