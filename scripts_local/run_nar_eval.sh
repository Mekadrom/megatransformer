#!/usr/bin/env bash
# NAR-relevant eval suite for world_tts_cosyvoice2_smollm2_nar_0.
#
# The AR bars do not transfer unchanged:
#   - early_text_delta is meaningless (a masked model has no left-to-right history), and
#   - the text horizon is flat by construction (no sequential error accumulation),
# so the primary diagnostic is the MASK-RATIO CURVE. r=1.0 is the inference starting
# condition (text alone) and the NAR analogue of AR's early frames; r->0 is the
# late-refinement condition where context can substitute for text.
#
#   text_delta high at r=1.0, falling as r->0   => healthy: text drives content
#   text_delta ~0 at every r                    => text ignored
#   acc climbs steeply as r->0, text_delta ~0   => inpainting from neighbours (the crutch,
#                                                  NAR edition -- the 2026-08-13 failure)
#
# Usage: bash scripts_local/run_nar_eval.sh <gpu> [step]
set -u
GPU="${1:?usage: run_nar_eval.sh <gpu> [step]}"
STEP="${2:-23000}"
DEV="cuda:${GPU}"

CACHE=./cached_datasets/Mekadrom/libritts_r_cosyvoice2_smollm2
CB=$CACHE/val/cosyvoice2_codebook.pt
CV2=/mnt/nasbro/cache/hf_home/hub/models--FunAudioLLM--CosyVoice2-0.5B/snapshots/eec1ae6c79877dbd9379285cf8789c9e0879293d
RUN=runs/world/world_tts_cosyvoice2_smollm2_nar_0
CKPT=$RUN/checkpoint-$STEP
OUT=scripts_local/eval_output/world_tts_nar_${STEP}
LOGS=$OUT/logs
mkdir -p "$LOGS"

COMMON="--cache_dir $CACHE --codebook $CB --config small_sum \
  --text_encoder_model HuggingFaceTB/SmolLM2-135M --voice_max_frames 250 --device $DEV \
  --checkpoint_path $CKPT --step $STEP --mrope_scale_side text"
BASE="--ngram_ceiling 0.0545 --asymptote 0.2261 --repeat 0.0314"

run () { local n="$1"; shift; echo "=== [$(date +%H:%M:%S)] $n"
  ( time "$@" ) > "$LOGS/$n.log" 2>&1; echo "    exit=$? -> $LOGS/$n.log"; }

# 1. THE MASK-RATIO CURVE (teacher-forced, no generation). Four points so the shape is
#    readable, not just a single number.
for r in 1.0 0.75 0.5 0.25; do
  run "ratio_$r" uv run python scripts_local/world_voice_ar_diagnostics.py \
    $COMMON $BASE --no_voice_predict_f0 --n 1024 --nar_mask_ratio $r --skip_generation \
    --out_dir $OUT/ratio_$r
done

# 2. Free-running degeneration through the MaskGIT sampler (K=16), against GT stats.
run freerun uv run python scripts_local/world_voice_ar_diagnostics.py \
  $COMMON $BASE --no_voice_predict_f0 --n 64 --nar_mask_ratio 1.0 --gen_n 32 \
  --nar_rounds 16 --out_dir $OUT/freerun

# 3. REFINEMENT GAIN. K=1 samples independent per-position marginals; if it is not clearly
#    worse than K=16, the head is not modelling the joint and the iteration is decorative.
run wer_K1 uv run python scripts_local/cosyvoice_wer_eval.py \
  $COMMON --cosyvoice_dir $CV2 --n 64 --nar_rounds 1 --out_dir $OUT/wer_K1
run wer_K16 uv run python scripts_local/cosyvoice_wer_eval.py \
  $COMMON --cosyvoice_dir $CV2 --n 64 --nar_rounds 16 --out_dir $OUT/wer_K16

# 4. Audio, for the ear. The AR arm's bar at 44k was truncated LCS 0.1104.
run renders uv run python scripts_local/render_distill_audio.py \
  $COMMON --cosyvoice_dir $CV2 --n 8 --nar_rounds 16 --out_dir $OUT/renders

echo "=== [$(date +%H:%M:%S)] NAR eval done -> $OUT"
