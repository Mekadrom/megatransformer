#!/usr/bin/env bash
# TRIMMED continuation of run_mrope44k_suite.sh, picking up after wer_mrope_text_ras.
#
# DROPPED from the original queue, deliberately:
#   - the six Stage-2 scale_side re-runs (ar/wer text+voice @20306, ar text+voice @28000).
#     They existed to re-check "text wins" under correct geometry, but the voice arm is dead,
#     the decision is made, and the paired bootstrap (scripts_local/wer_arm_compare.py) showed
#     arm-level differences of this size are not resolvable at n=64 -- so they would re-litigate
#     a settled choice with an instrument known to be underpowered. ~6.5 GPU-hours.
#   - render_mrope_text: already produced at T=0.6 for BOTH arms on GPU1
#     (render_t06_mrope_text / render_t06_distill). Pure duplicate.
#
# KEPT, with the reason each still earns its GPU time:
#   wer_distill_ras     -- second RAS data point; RAS is the only intervention off the
#                          entropy axis and its P(better) was 0.84, short of conclusive.
#   ar_mrope_text_ras   -- free-running degeneration under RAS at the operating point.
#   ar_mrope_off        -- how much the M-RoPE eval bug distorted things; decides which
#                          memory entries get rewritten vs deleted.
#   horizon_sweep       -- horizontal-vs-vertical: is the text horizon widening over training,
#                          or is the onset bucket just getting taller?
#   per_head_*          -- real-model replication for the alignment-head-ablation study, now
#                          that the original per-head measurement is known void.
#
# Usage: bash scripts_local/run_mrope44k_suite_trimmed.sh <gpu_index>
set -u
GPU="${1:?usage: run_mrope44k_suite_trimmed.sh <gpu_index>}"
DEV="cuda:${GPU}"

CACHE=./cached_datasets/Mekadrom/libritts_r_cosyvoice2_smollm2
CB=$CACHE/val/cosyvoice2_codebook.pt
CV2=/mnt/nasbro/cache/hf_home/hub/models--FunAudioLLM--CosyVoice2-0.5B/snapshots/eec1ae6c79877dbd9379285cf8789c9e0879293d
TEXT_RUN=runs/world/world_tts_cosyvoice2_smollm2_mrope_scale_text_0
DISTILL_RUN=runs/world/world_tts_cosyvoice2_smollm2_distill_0
OUT=scripts_local/eval_output/world_tts_mrope44k
LOGS=$OUT/logs
mkdir -p "$LOGS"

BASE="--ngram_ceiling 0.0545 --asymptote 0.2261 --repeat 0.0314"
COMMON="--cache_dir $CACHE --codebook $CB --config small_sum \
  --text_encoder_model HuggingFaceTB/SmolLM2-135M --voice_max_frames 250 --device $DEV"

run () {
  local name="$1"; shift
  echo "=== [$(date +%H:%M:%S)] $name"
  ( time "$@" ) > "$LOGS/$name.log" 2>&1
  echo "    exit=$? -> $LOGS/$name.log"
}

run wer_distill_ras uv run python scripts_local/cosyvoice_wer_eval.py \
  --checkpoint_path $DISTILL_RUN/checkpoint-44000 --step 44000 --cosyvoice_dir $CV2 \
  --n 64 --ras_win 10 $COMMON --out_dir $OUT/wer_distill_ras

run ar_mrope_text_ras uv run python scripts_local/world_voice_ar_diagnostics.py \
  --checkpoint_path $TEXT_RUN/checkpoint-44000 --step 44000 --mrope_scale_side text \
  --no_voice_predict_f0 --gen_n 32 --voice_ras_win 10 --n 64 $COMMON $BASE \
  --out_dir $OUT/ar_mrope_text_ras

run ar_mrope_off uv run python scripts_local/world_voice_ar_diagnostics.py \
  --checkpoint_path $TEXT_RUN/checkpoint-44000 --step 44000 --mrope_scale_side off \
  --skip_generation --no_voice_predict_f0 --n 1024 $COMMON $BASE \
  --out_dir $OUT/ar_mrope_off

run horizon_sweep uv run python scripts_local/voice_text_horizon_sweep.py \
  --run_dir $TEXT_RUN --steps 20000,32000,44000 --mrope_scale_side text \
  --n 1024 $COMMON --out_dir $OUT/horizon_sweep

run per_head_mrope_text uv run python scripts_local/voice_text_attention_map.py \
  --checkpoint_path $TEXT_RUN/checkpoint-44000 --step 44000 --mrope_scale_side text \
  --per_head --n 256 $COMMON --out_dir $OUT/per_head_mrope_text

run per_head_distill uv run python scripts_local/voice_text_attention_map.py \
  --checkpoint_path $DISTILL_RUN/checkpoint-44000 --step 44000 \
  --per_head --n 256 $COMMON --out_dir $OUT/per_head_distill

echo "=== [$(date +%H:%M:%S)] trimmed suite done -> $OUT"
