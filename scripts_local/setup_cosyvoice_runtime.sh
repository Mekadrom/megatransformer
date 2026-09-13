#!/usr/bin/env bash
# Build a CosyVoice 2 runtime checkout for the world-voice decode path.
#
# This is NOT pip-installable and deliberately never enters .venv: the code puts it on
# sys.path instead. cv_extra/ holds --no-deps installs that would otherwise fight `uv sync`
# (which is why they live in their own directory rather than the project venv).
#
# What the DECODE path needs and does not need: flow + HiFT are ordinary torch modules, so
# decode needs neither the ONNX speech tokenizer (that builds caches, already done) nor the
# text frontend (we feed unit ids, not text) nor the Qwen LLM (the world model replaces it).
# Several cv_extra entries exist only to satisfy module-level imports that are never called.
#
#   ./scripts_local/setup_cosyvoice_runtime.sh [target_dir]
#
# then:  --voice_cosyvoice2_runtime_dir <target_dir>   (or export COSYVOICE_RUNTIME=<dir>)
set -euo pipefail

DIR="${1:-$HOME/dev/projects/cosyvoice-runtime}"
PY="${PYTHON:-python3}"
echo "==> target: $DIR"
mkdir -p "$DIR"

# 1. upstream repo. --recursive is REQUIRED: third_party/Matcha-TTS provides
#    matcha.utils.audio.mel_spectrogram, which is the prompt-mel extractor.
if [ -d "$DIR/CosyVoice/.git" ]; then
  echo "==> CosyVoice already present, skipping clone"
else
  git clone --recursive https://github.com/FunAudioLLM/CosyVoice "$DIR/CosyVoice"
fi
if [ ! -d "$DIR/CosyVoice/third_party/Matcha-TTS" ]; then
  echo "==> fetching Matcha-TTS submodule"
  git -C "$DIR/CosyVoice" submodule update --init --recursive
fi

# 2. cv_extra: --no-deps so none of this leaks into the project venv.
#    Pins that matter:
#      antlr4-python3-runtime==4.9.3  -- 4.13 raises "Could not deserialize ATN with version 3"
#      setuptools<81                  -- 81+ removed pkg_resources, which pyworld imports
echo "==> building cv_extra (--no-deps)"
"$PY" -m pip install --no-deps --upgrade --target "$DIR/cv_extra" \
  onnxruntime conformer hydra-core "antlr4-python3-runtime==4.9.3" \
  lightning gdown beautifulsoup4 soupsieve wget pyworld "setuptools==80.9.0"

# 3. cv_shims: CosyVoice imports modelscope.snapshot_download at module top. We always pass a
#    local model_dir, so a stub that raises if called is enough and avoids the real dependency.
echo "==> writing cv_shims/modelscope.py"
mkdir -p "$DIR/cv_shims"
cat > "$DIR/cv_shims/modelscope.py" <<'SHIM'
# Minimal stub: CosyVoice only imports snapshot_download at module top; we always
# pass a local model_dir so it is never actually invoked.
def snapshot_download(*args, **kwargs):
    raise RuntimeError("modelscope stub: pass a local model_dir; download not supported")
SHIM

# 4. prove it imports the way the decoder will, rather than trusting the file listing.
echo "==> verifying"
COSYVOICE_RUNTIME="$DIR" "$PY" - <<'CHECK'
import os, sys
d = os.environ["COSYVOICE_RUNTIME"]
for p in (os.path.join(d, "cv_extra"), os.path.join(d, "cv_shims"),
          os.path.join(d, "CosyVoice"), os.path.join(d, "CosyVoice", "third_party", "Matcha-TTS")):
    assert os.path.isdir(p), f"MISSING {p}"
    sys.path.insert(0, p)
# Import exactly what DECODE imports. Deliberately NOT matcha.utils.audio: it does
# `from librosa.filters import mel` -> numba, which caps NumPy at 2.4 and fails on a newer
# venv -- and nothing on the decode path needs it (the yaml's feat_extractor is stripped and
# the prompt mel is reimplemented on torchaudio). Checking it here would reject a runtime
# that works fine.
from cosyvoice.flow.flow import CausalMaskedDiffWithXvec   # noqa: F401
from cosyvoice.flow.flow_matching import CausalConditionalCFM  # noqa: F401  (pulls matcha)
from cosyvoice.utils.common import ras_sampling            # noqa: F401
from cosyvoice.hifigan.generator import HiFTGenerator      # noqa: F401
print("OK: flow, flow_matching (via matcha), hifigan and utils.common all import")
CHECK

echo
echo "==> done. Use it with:"
echo "    --voice_cosyvoice2_runtime_dir $DIR"
echo "  or: export COSYVOICE_RUNTIME=$DIR"
echo
echo "NOTE: model weights are NOT here -- they stay in the HF cache and are passed via"
echo "      --voice_cosyvoice2_model_dir <CosyVoice2-0.5B snapshot>."
