#!/usr/bin/env bash
# Temperature sweep, unistream decode, EMA weights, for the bistream and unistream
# world-voice runs at a matched step.
#
# SHARDABLE + RESUMABLE per the GPU tenancy rules: --shard i/n takes every n-th job,
# --resume skips any job whose report json already exists. Yielding a GPU mid-sweep
# costs at most the in-flight arm.
#
#   CUDA_VISIBLE_DEVICES=3 ./scripts_local/world_voice_temp_sweep.sh --shard 0/2 --resume
#   CUDA_VISIBLE_DEVICES=0 ./scripts_local/world_voice_temp_sweep.sh --shard 1/2 --resume
set -u
SHARD=0; NSHARD=1; RESUME=0; N=12
while [ $# -gt 0 ]; do
  case "$1" in
    --shard) SHARD="${2%%/*}"; NSHARD="${2##*/}"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --n) N="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

O=eval_output/world_voice_temp_sweep
CACHE=./cached_datasets/Mekadrom/libriheavy_cosyvoice2_smollm2
CV2=/mnt/nasbro/cache/hf_home/hub/models--FunAudioLLM--CosyVoice2-0.5B/snapshots/eec1ae6c79877dbd9379285cf8789c9e0879293d
UNI=runs/world_voice/cosyvoice2_smollm2_libriheavy_ar_flat_lr_ema_mrope75_0/checkpoint-68000-ema
BI=runs/world_voice/cosyvoice2_smollm2_libriheavy_bistream_scratch_0/checkpoint-68000-ema
mkdir -p "$O/logs"

TEMPS="0.0 0.2 0.4 0.5 0.6 0.7 0.8 1.0"
i=0
for MODEL in uni bi; do
  for T in $TEMPS; do
    if [ $((i % NSHARD)) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
    i=$((i+1))
    case "$MODEL" in uni) CK="$UNI" ;; bi) CK="$BI" ;; esac
    LABEL="${MODEL}_t$(echo "$T" | tr -d '.')"
    if [ "$RESUME" = "1" ] && [ -f "$O/$LABEL/wer_step68000_ras10.json" ]; then
      echo "[$(date '+%H:%M:%S')] SKIP $LABEL (resume)"; continue
    fi
    echo "[$(date '+%H:%M:%S')] START $LABEL (cuda:${CUDA_VISIBLE_DEVICES:-?})"
    python scripts_local/cosyvoice_wer_eval.py \
      --checkpoint_path "$CK" --step 68000 \
      --cache_dir "$CACHE" --codebook "$CACHE/val/cosyvoice2_codebook.pt" \
      --cosyvoice_dir "$CV2" \
      --config small_sum --text_encoder_model HuggingFaceTB/SmolLM2-135M \
      --voice_max_frames 250 \
      --mrope_voice_rate 7.5 --mrope_scale_side text \
      --ras_win 10 --ras_tau 0.1 --voice_temperature "$T" \
      --n "$N" --device cuda:0 \
      --out_dir "$O/$LABEL" --save_audio "$O/$LABEL/wav" > "$O/logs/$LABEL.log" 2>&1
    echo "[$(date '+%H:%M:%S')] $LABEL exit=$?"
  done
done
echo "[$(date '+%H:%M:%S')] SHARD $SHARD/$NSHARD DONE"
