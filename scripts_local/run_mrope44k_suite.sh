#!/usr/bin/env bash
# World-TTS eval suite, run serially on one GPU.
#
#   STAGE 1 — the ask: M-RoPE scale_side=text @44000 vs the distill baseline @44000
#             (matched step; distill is the best pre-M-RoPE model and the only run with a
#             44k checkpoint -- the voice-scaled arm died at 28k).
#   STAGE 2 — re-run of every diagnostic invalidated by the M-RoPE eval bug (load_world_model
#             never re-enabled M-RoPE, so the trunk ran single-axis RoPE with sequential
#             positions). Redone at the SAME steps the published numbers were taken at.
#
# Usage: bash scripts_local/run_mrope44k_suite.sh <gpu_index>
set -u
GPU="${1:?usage: run_mrope44k_suite.sh <gpu_index>}"
DEV="cuda:${GPU}"

CACHE=./cached_datasets/Mekadrom/libritts_r_cosyvoice2_smollm2
CB=$CACHE/val/cosyvoice2_codebook.pt
CV2=/mnt/nasbro/cache/hf_home/hub/models--FunAudioLLM--CosyVoice2-0.5B/snapshots/eec1ae6c79877dbd9379285cf8789c9e0879293d
TEXT_RUN=runs/world/world_tts_cosyvoice2_smollm2_mrope_scale_text_0
VOICE_RUN=runs/world/world_tts_cosyvoice2_smollm2_mrope_0
DISTILL_RUN=runs/world/world_tts_cosyvoice2_smollm2_distill_0
OUT=scripts_local/eval_output/world_tts_mrope44k
LOGS=$OUT/logs
mkdir -p "$LOGS"

# CosyVoice2 text-free baselines (from runs/world/world_tts_cosyvoice2_smollm2_0__baseline_*),
# NOT the Mimi-era script defaults (0.211/0.229/0.117) the scripts still ship.
BASE="--ngram_ceiling 0.0545 --asymptote 0.2261 --repeat 0.0314"
# --no_voice_predict_f0 exists ONLY in world_voice_ar_diagnostics; the others already default
# to no-F0, so they must not be passed it.
COMMON="--cache_dir $CACHE --codebook $CB --config small_sum \
  --text_encoder_model HuggingFaceTB/SmolLM2-135M --voice_max_frames 250 --device $DEV"

run () {  # run <name> <cmd...>
  local name="$1"; shift
  echo "=== [$(date +%H:%M:%S)] $name"
  ( time "$@" ) > "$LOGS/$name.log" 2>&1
  echo "    exit=$? -> $LOGS/$name.log"
}

ar () {  # ar <name> <ckpt> <step> [extra...]
  local name="$1" ckpt="$2" step="$3"; shift 3
  run "$name" uv run python scripts_local/world_voice_ar_diagnostics.py \
    --checkpoint_path "$ckpt" --step "$step" --no_voice_predict_f0 \
    --n 1024 $COMMON $BASE --out_dir "$OUT/$name" "$@"
}

wer () {  # wer <name> <ckpt> <step> [extra...]
  local name="$1" ckpt="$2" step="$3"; shift 3
  run "$name" uv run python scripts_local/cosyvoice_wer_eval.py \
    --checkpoint_path "$ckpt" --step "$step" --cosyvoice_dir $CV2 \
    --n 64 $COMMON --out_dir "$OUT/$name" "$@"
}

echo "### STAGE 1 — matched-step 44000: M-RoPE(text) vs distill"

ar  ar_mrope_text  $TEXT_RUN/checkpoint-44000    44000 --gen_n 32 --mrope_scale_side text
ar  ar_distill     $DISTILL_RUN/checkpoint-44000 44000 --gen_n 32
wer wer_mrope_text $TEXT_RUN/checkpoint-44000    44000 --mrope_scale_side text
wer wer_distill    $DISTILL_RUN/checkpoint-44000 44000

# RAS arm. "The text-scaled arm does not need RAS" rested on adj_repeat 0.0257 -- a number
# measured with M-RoPE OFF, so it is one of the invalidated ones. Re-test at matched step:
# WER/LCS with RAS on both arms, plus the unit-level degeneration stats on the text arm.
wer wer_mrope_text_ras $TEXT_RUN/checkpoint-44000    44000 --mrope_scale_side text --ras_win 10
wer wer_distill_ras    $DISTILL_RUN/checkpoint-44000 44000 --ras_win 10
# RAS does not touch teacher forcing, so this run's TF section is a cheaper duplicate of
# ar_mrope_text (n=64, ignore it) -- what is wanted here is the free-running block.
ar  ar_mrope_text_ras  $TEXT_RUN/checkpoint-44000    44000 --gen_n 32 --mrope_scale_side text \
    --voice_ras_win 10 --n 64   # trailing --n overrides the helper's 1024

# Audio to actually listen to (the ear is the arbiter): gen-RAS / gen-plain / GT.
run render_mrope_text uv run python scripts_local/render_distill_audio.py \
  --checkpoint_path $TEXT_RUN/checkpoint-44000 --step 44000 --mrope_scale_side text \
  --cosyvoice_dir $CV2 --n 8 $COMMON --out_dir $OUT/render_mrope_text

# What the pre-fix protocol cost: same weights, M-RoPE deliberately disengaged.
ar  ar_mrope_off   $TEXT_RUN/checkpoint-44000    44000 --skip_generation --mrope_scale_side off

# Horizontal vs vertical: is the text horizon moving RIGHT, or just the onset getting taller?
run horizon_sweep uv run python scripts_local/voice_text_horizon_sweep.py \
  --run_dir $TEXT_RUN --steps 20000,32000,44000 --mrope_scale_side text \
  --n 1024 $COMMON --out_dir $OUT/horizon_sweep

echo "### STAGE 2 — re-run of the M-RoPE-invalidated diagnostics, correct geometry"

# 2a. The scale_side ablation, redone at 20306 -- the exact step the published verdict
#     ("text wins") was taken at, so the new table is directly comparable to the old one.
ar  ar_text_20306   $TEXT_RUN/checkpoint-20306  20306 --gen_n 32 --mrope_scale_side text
ar  ar_voice_20306  $VOICE_RUN/checkpoint-20306 20306 --gen_n 32 --mrope_scale_side voice
wer wer_text_20306  $TEXT_RUN/checkpoint-20306  20306 --mrope_scale_side text
wer wer_voice_20306 $VOICE_RUN/checkpoint-20306 20306 --mrope_scale_side voice

# 2b. Same ablation at 28000 = the latest step BOTH arms reached (the voice arm was killed
#     there), as a second, better-trained read on the same question.
ar  ar_text_28000   $TEXT_RUN/checkpoint-28000  28000 --gen_n 32 --mrope_scale_side text
ar  ar_voice_28000  $VOICE_RUN/checkpoint-28000 28000 --gen_n 32 --mrope_scale_side voice

# 2c. Per-head attention: "no head is a sharp aligner under M-RoPE" was measured with M-RoPE
#     OFF, which would scramble exactly this statistic. Compare against distill@44k's
#     documented sharpest head (entropy 0.207 = 9% of uniform, corr +0.85).
run per_head_mrope_text uv run python scripts_local/voice_text_attention_map.py \
  --checkpoint_path $TEXT_RUN/checkpoint-44000 --step 44000 --mrope_scale_side text \
  --per_head --n 256 $COMMON --out_dir $OUT/per_head_mrope_text

run per_head_distill uv run python scripts_local/voice_text_attention_map.py \
  --checkpoint_path $DISTILL_RUN/checkpoint-44000 --step 44000 \
  --per_head --n 256 $COMMON --out_dir $OUT/per_head_distill

echo "=== [$(date +%H:%M:%S)] suite done -> $OUT"
# NOT automated: head_ablation_probe.py needs --block/--head, which come from the per-head
# result above. Run it once the sharpest head is known.
