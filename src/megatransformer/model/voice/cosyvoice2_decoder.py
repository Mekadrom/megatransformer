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
import logging
import os
import sys
from contextlib import contextmanager
from typing import Optional

import torch


DEFAULT_RUNTIME_DIR = os.path.expanduser("~/dev/projects/cosyvoice-runtime")


@contextmanager
def _preserve_root_logging():
    """Undo CosyVoice's global logging hijack.

    cosyvoice/utils/file_utils.py runs `logging.basicConfig(level=logging.DEBUG)` at MODULE
    IMPORT TIME, which flips the ROOT logger to DEBUG for the whole process. In a training
    run that turns on httpx/httpcore/urllib3 debug spam for every subsequent HTTP call and
    buries the training log. Snapshot the root logger's level and handlers, and restore them
    once CosyVoice is imported.
    """
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    noisy = {n: logging.getLogger(n).level
             for n in ("httpx", "httpcore", "urllib3", "filelock", "fsspec", "matplotlib")}
    try:
        yield
    finally:
        root.setLevel(level)
        for h in list(root.handlers):
            if h not in handlers:      # drop any handler basicConfig installed
                root.removeHandler(h)
        for n, lv in noisy.items():
            logging.getLogger(n).setLevel(lv)


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
        # Everything that imports cosyvoice goes inside the guard — the yaml load is what
        # pulls in cosyvoice.utils.file_utils and its root-logger basicConfig(DEBUG).
        with _preserve_root_logging():
            _ensure_importable(runtime_dir)
            configs = _load_configs_without_llm(model_dir)
        flow, hift = configs["flow"], configs["hift"]
        flow.load_state_dict(torch.load(os.path.join(model_dir, "flow.pt"),
                                        map_location="cpu", weights_only=False), strict=True)
        hift.load_state_dict(torch.load(os.path.join(model_dir, "hift.pt"),
                                        map_location="cpu", weights_only=False), strict=True)
        m = cls(flow, hift, int(configs["sample_rate"]))
        return m.to(device=device, dtype=dtype)

    # CosyVoice 2's prompt mel. NOT our training mel: n_fft/hop/win and the 24 kHz rate are
    # the flow's own feat_extractor settings (conf/cosyvoice2.yaml), and a mismatch puts the
    # prompt off-manifold in a way that is worse than no prompt at all.
    PROMPT_MEL = dict(n_fft=1920, num_mels=80, sampling_rate=24000,
                      hop_size=480, win_size=1920, fmin=0, fmax=8000, center=False)

    @torch.no_grad()
    def prompt_mel(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Reference waveform -> (T_mel, 80) prompt feature for zero-shot cloning.

        24 kHz on purpose: the flow's mel is 24000/480 = 50 Hz while the FSQ tokenizer reads
        16 kHz, so a reference clip feeds the two halves of the prompt at DIFFERENT rates
        (frontend.py:_extract_speech_feat vs _extract_spk_embedding).
        """
        import torchaudio
        from matcha.utils.audio import mel_spectrogram
        wav = waveform.reshape(1, -1).float()
        if sr != 24000:
            wav = torchaudio.functional.resample(wav, sr, 24000)
        return mel_spectrogram(wav, **self.PROMPT_MEL).squeeze(0).transpose(0, 1)

    @torch.no_grad()
    def decode(self, unit_ids: torch.Tensor, speaker_embedding: torch.Tensor,
               prompt_ids: Optional[torch.Tensor] = None,
               prompt_feat: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """unit_ids: (T,) or (1,T) int content units (EOV/padding already stripped).
        speaker_embedding: (192,) or (1,192) campplus. Returns (T_wav,) float32 on CPU.

        prompt_ids / prompt_feat are OPTIONAL zero-shot conditioning: the reference clip's FSQ
        units and its 24 kHz 80-bin mel, concatenated ahead of the target inside the flow. The
        192-d embedding is a global summary that modulates via spk_embed_affine_layer; the
        prompt supplies PER-FRAME acoustic evidence (timbre detail, channel, style) that no
        fixed-size vector carries. Passing neither reproduces the previous embedding-only
        behaviour exactly.
        """
        ids = unit_ids.reshape(1, -1).to(device=self._device, dtype=torch.int32)
        if ids.shape[1] == 0:
            return None
        emb = speaker_embedding.reshape(1, -1).to(device=self._device, dtype=self._dtype)

        p_ids = torch.zeros(1, 0, dtype=torch.int32, device=self._device)
        p_feat = torch.zeros(1, 0, 80, device=self._device, dtype=self._dtype)
        if prompt_ids is not None and prompt_feat is not None:
            _pi = prompt_ids.reshape(1, -1).to(device=self._device, dtype=torch.int32)
            _pf = prompt_feat.reshape(-1, 80).unsqueeze(0).to(device=self._device, dtype=self._dtype)
            # force  mel_frames == 2 * token_frames  (frontend.py:176-178)
            n_tok = min(int(_pf.shape[1] // 2), int(_pi.shape[1]))
            if n_tok > 0:
                p_ids, p_feat = _pi[:, :n_tok], _pf[:, :2 * n_tok]

        kw = dict(
            token=ids,
            token_len=torch.tensor([ids.shape[1]], dtype=torch.int32, device=self._device),
            prompt_token=p_ids,
            prompt_token_len=torch.tensor([p_ids.shape[1]], dtype=torch.int32, device=self._device),
            prompt_feat=p_feat,
            prompt_feat_len=torch.tensor([p_feat.shape[1]], dtype=torch.int32, device=self._device),
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
