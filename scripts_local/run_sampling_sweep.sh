#!/usr/bin/env bash
# Sampling sweep on ONE checkpoint (mrope_scale_text@44000): what decode regime does this
# model actually want? T=0.6 (the training-viz default nobody chose deliberately) mode-collapses
# into repetition loops; T=1.0 looked healthy only because tail noise masked the collapse.
# Greedy has never been tried on this arm at all.
#
# TF sections are temperature-independent, so --n is small here on purpose: only section 3
# (free-running degeneration) is meaningful in these reports.
#
# Usage: bash scripts_local/run_sampling_sweep.sh <gpu_index>
set -u
GPU="${1:?usage: run_sampling_sweep.sh <gpu_index>}"
DEV="cuda:${GPU}"

CACHE=./cached_datasets/Mekadrom/libritts_r_cosyvoice2_smollm2
CB=$CACHE/val/cosyvoice2_codebook.pt
CV2=/mnt/nasbro/cache/hf_home/hub/models--FunAudioLLM--CosyVoice2-0.5B/snapshots/eec1ae6c79877dbd9379285cf8789c9e0879293d
CKPT=runs/world/world_tts_cosyvoice2_smollm2_mrope_scale_text_0/checkpoint-44000
OUT=eval_output/world_tts_mrope44k/sampling_sweep
LOGS=$OUT/logs
mkdir -p "$LOGS"

COMMON="--cache_dir $CACHE --codebook $CB --config small_sum \
  --text_encoder_model HuggingFaceTB/SmolLM2-135M --voice_max_frames 250 --device $DEV \
  --checkpoint_path $CKPT --step 44000 --mrope_scale_side text"
BASE="--ngram_ceiling 0.0545 --asymptote 0.2261 --repeat 0.0314"

sweep () {  # sweep <name> <extra flags...>
  local name="$1"; shift
  echo "=== [$(date +%H:%M:%S)] sweep:$name"
  ( time uv run python scripts_local/world_voice_ar_diagnostics.py \
      $COMMON $BASE --no_voice_predict_f0 --n 64 --gen_n 32 \
      --out_dir "$OUT/$name" "$@" ) > "$LOGS/$name.log" 2>&1
  echo "    exit=$? -> $LOGS/$name.log"
}

sweep greedy          --voice_temperature 0.0
sweep t0.8            --voice_temperature 0.8
sweep t0.8_topp0.9    --voice_temperature 0.8 --voice_top_p 0.9
sweep t0.6_topp0.9    --voice_temperature 0.6 --voice_top_p 0.9
sweep t1.0_topk50     --voice_temperature 1.0 --voice_top_k 50
# t0.6 and t1.0 (untruncated) already exist as ar_mrope_text / ar_mrope_text_t1.0.

# Audio for the two regimes with no renders yet: greedy, and a truncated middle ground.
render () {  # render <name> <extra flags...>
  local name="$1"; shift
  echo "=== [$(date +%H:%M:%S)] render:$name"
  ( time uv run python scripts_local/render_distill_audio.py \
      $COMMON --cosyvoice_dir $CV2 --n 6 --out_dir "$OUT/render_$name" "$@" \
    ) > "$LOGS/render_$name.log" 2>&1
  echo "    exit=$? -> $LOGS/render_$name.log"
}

render greedy       --voice_temperature 0.0
render t0.8_topp0.9 --voice_temperature 0.8 --voice_top_p 0.9

echo "=== [$(date +%H:%M:%S)] sweep done -> $OUT"
