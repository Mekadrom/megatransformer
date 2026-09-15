#!/usr/bin/env bash
# RAS tau sweep on the bistream checkpoint. tau sets the repeat threshold:
# RAS fires when the chosen unit occurred >= win*tau times in the last `win`.
# At win=10 tau=0.1 that is ONE repeat -- ~8x stricter than GT's own adjacent-repeat
# rate of 0.0792, which is why every RAS arm under-repeats (0.0096-0.0127).
#
# ALL arms pass --skip_ceiling so the RNG stream matches across arms: the CV2 flow
# decoder draws randn per call, so a ceiling decode on sample i shifts sample i+1's
# generation. Mixing the flag across arms confounds the comparison.
#
#   CUDA_VISIBLE_DEVICES=3 ./scripts_local/world_voice_tau_sweep.sh --shard 0/2 --resume
#   CUDA_VISIBLE_DEVICES=0 ./scripts_local/world_voice_tau_sweep.sh --shard 1/2 --resume
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
O=eval_output/world_voice_tau_sweep
CACHE=./cached_datasets/Mekadrom/libriheavy_cosyvoice2_smollm2
CV2=/mnt/nasbro/cache/hf_home/hub/models--FunAudioLLM--CosyVoice2-0.5B/snapshots/eec1ae6c79877dbd9379285cf8789c9e0879293d
BI=runs/world_voice/cosyvoice2_smollm2_libriheavy_bistream_scratch_0/checkpoint-68000-ema
mkdir -p "$O/logs"
i=0
# regime: greedy (T=0, resample inherits an effective 1.0) and T=0.6 with the
# resample pinned to 1.0 -- the two configurations that terminate cleanly.
for REGIME in "0.0 none" "0.6 1.0"; do
  set -- $REGIME; T="$1"; RT="$2"
  for TAU in 0.0 0.1 0.2 0.3 0.5; do
    if [ $((i % NSHARD)) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
    i=$((i+1))
    LABEL="bi_t${T/./}_tau${TAU/./}_rt${RT/./}"
    EXTRA=""
    [ "$RT" != "none" ] && EXTRA="--ras_temperature $RT"
    if [ "$RESUME" = "1" ] && [ -f "$O/$LABEL/wer_step68000_ras10.json" ]; then
      echo "[$(date '+%H:%M:%S')] SKIP $LABEL"; continue
    fi
    echo "[$(date '+%H:%M:%S')] START $LABEL"
    python scripts_local/cosyvoice_wer_eval.py \
      --checkpoint_path "$BI" --step 68000 \
      --cache_dir "$CACHE" --codebook "$CACHE/val/cosyvoice2_codebook.pt" \
      --cosyvoice_dir "$CV2" \
      --config small_sum --text_encoder_model HuggingFaceTB/SmolLM2-135M \
      --voice_max_frames 250 \
      --mrope_voice_rate 7.5 --mrope_scale_side text \
      --ras_win 10 --ras_tau "$TAU" --voice_temperature "$T" \
      --n "$N" --device cuda:0 --skip_ceiling \
      --out_dir "$O/$LABEL" --save_audio "$O/$LABEL/wav" $EXTRA \
      > "$O/logs/$LABEL.log" 2>&1
    echo "[$(date '+%H:%M:%S')] $LABEL exit=$?"
  done
done
echo "[$(date '+%H:%M:%S')] SHARD $SHARD/$NSHARD DONE"
