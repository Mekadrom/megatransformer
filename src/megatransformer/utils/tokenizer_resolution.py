"""Single source of truth for "which tokenizer do this checkpoint's text ids belong to?".

THE BUG THIS CLOSES. A checkpoint carried text ids without recording their vocabulary, so
every consumer re-derived the tokenizer from its own CLI flags -- and each one could be wrong
independently. An audit on 2026-09-18 (docs/findings/world-text.md) found EIGHT sites
defaulting to `mistralai/Mistral-7B-v0.1`, in two shapes: five that could not be overridden at
all, and three that honoured `--text_encoder_model` but not a from-scratch prelude trained on a
non-Mistral corpus. The symptom was never an error:

    Mistral decode:  'tern pentalal shink\\x1boun sum\\x19or historian`status displayeda har M...'
    SmolLM2 decode:  'hip and artistic sensibilities of their time. For instance, Egyptian...'

3 of the first 32 corpus ids were beyond Mistral's 32000 vocab, so they were out of range, not
merely mis-decoded -- and the transcription scripts computed WER against that.

THE FIX. `MegaTransformerWorldModelConfig.text_tokenizer_name` is written at training time.
Consumers call `resolve_tokenizer_name(model.config, args)` and get the recorded name, so a
flag can no longer disagree with the weights.

PRECEDENCE, and why the config wins over an explicit flag: the config is a FACT about how the
data was tokenized, while a flag is an assertion by whoever typed the command. When they
disagree the flag is wrong. Deliberately overriding (to measure the cost of a mismatch, as
`--mrope_scale_side off` does) is still possible via `force=`.

    1. force=          explicit override, for deliberate mismatch experiments
    2. checkpoint      megatransformer_meta.json beside the weights -- authoritative
    3. config          an in-memory config that carries the name (training time)
    4. args flags      --text_encoder_model, then --text_tokenizer (pre-2026-09-18 checkpoints)
    5. Mistral         the historical default, kept so old checkpoints behave as before
"""
import json
import os
from typing import Any, Optional

MISTRAL_DEFAULT = "mistralai/Mistral-7B-v0.1"
META_FILENAME = "megatransformer_meta.json"


def read_checkpoint_meta(checkpoint_path: Optional[str]) -> dict:
    """Read `megatransformer_meta.json` from a checkpoint dir, or {} if absent.

    A checkpoint is weights + optimizer + scheduler + RNG and carries NO config, so this file
    is the only place a checkpoint records its own tokenizer. Absent for everything written
    before 2026-09-18, which is why every caller still falls back to flags.
    """
    if not checkpoint_path:
        return {}
    path = checkpoint_path
    if os.path.isfile(path):
        path = os.path.dirname(path)
    try:
        with open(os.path.join(path, META_FILENAME)) as fh:
            return json.load(fh)
    except Exception:
        return {}


def resolve_tokenizer_name(config: Any = None, args: Any = None,
                           force: Optional[str] = None,
                           default: str = MISTRAL_DEFAULT,
                           checkpoint_path: Optional[str] = None) -> str:
    """Return the tokenizer name to load. See module docstring for precedence."""
    if force:
        return force
    ckpt = checkpoint_path or (getattr(args, "checkpoint_path", None) if args is not None else None)
    recorded = read_checkpoint_meta(ckpt).get("text_tokenizer_name")
    if recorded:
        return recorded
    recorded = getattr(config, "text_tokenizer_name", None) if config is not None else None
    if recorded:
        return recorded
    if args is not None:
        for flag in ("text_encoder_model", "text_tokenizer"):
            val = getattr(args, flag, None)
            if val:
                return val
    return default


def describe_resolution(config: Any = None, args: Any = None,
                        force: Optional[str] = None) -> str:
    """Human-readable account of WHICH source won, for logging at load time.

    Worth printing: the failure mode this module exists for is silent, so a line naming the
    tokenizer and its provenance is the cheapest possible tripwire.
    """
    name = resolve_tokenizer_name(config, args, force)
    ckpt = getattr(args, "checkpoint_path", None) if args is not None else None
    if force:
        src = "forced override"
    elif read_checkpoint_meta(ckpt).get("text_tokenizer_name"):
        src = f"checkpoint {META_FILENAME} (recorded at training time)"
    elif config is not None and getattr(config, "text_tokenizer_name", None):
        src = "model config (recorded at training time)"
    elif args is not None and getattr(args, "text_encoder_model", None):
        src = "--text_encoder_model"
    elif args is not None and getattr(args, "text_tokenizer", None):
        src = "--text_tokenizer"
    else:
        src = "DEFAULT -- checkpoint predates text_tokenizer_name and no flag was given"
    return f"tokenizer: {name}  (from {src})"
