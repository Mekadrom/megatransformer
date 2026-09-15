#!/usr/bin/env bash
# DECISION RUN: greedy vs T=0.6 + ras_temperature 1.0, at n=48, WITH seed replicates
# so the comparison carries its own noise estimate instead of borrowing the n=12 band.
#
# Every arm passes --skip_ceiling: the CV2 flow decoder draws randn per decode, so a
# ceiling pass on sample i shifts sample i+1's generation. Mixing the flag across arms
# confounds them (learned the hard way 2026-09-15).
#
#   CUDA_VISIBLE_DEVICES=3 ./scripts_local/world_voice_n48_decision.sh --shard 0/2 --resume
#   CUDA_VISIBLE_DEVICES=0 ./scripts_local/world_voice_n48_decision.sh --shard 1/2 --resume
set -u
SHARD=0; NSHARD=1; RESUME=0; N=48
while [ $# -gt 0 ]; do
  case "$1" in
    --shard) SHARD="${2%%/*}"; NSHARD="${2##*/}"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --n) N="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
O=eval_output/world_voice_n48_decision
CACHE=./cached_datasets/Mekadrom/libriheavy_cosyvoice2_smollm2
CV2=/mnt/nasbro/cache/hf_home/hub/models--FunAudioLLM--CosyVoice2-0.5B/snapshots/eec1ae6c79877dbd9379285cf8789c9e0879293d
BI=runs/world_voice/cosyvoice2_smollm2_libriheavy_bistream_scratch_0/checkpoint-68000-ema
UNI=runs/world_voice/cosyvoice2_smollm2_libriheavy_ar_flat_lr_ema_mrope75_0/checkpoint-68000-ema
mkdir -p "$O/logs"
# label | ckpt | temp | tau | ras_temp | seed
JOBS="
A_bi_greedy_tau01_s1|$BI|0.0|0.1|none|1
B_bi_t06rt10_tau01_s1|$BI|0.6|0.1|1.0|1
C_bi_t06rt10_tau03_s1|$BI|0.6|0.3|1.0|1
D_uni_greedy_tau01_s1|$UNI|0.0|0.1|none|1
A2_bi_greedy_tau01_s2|$BI|0.0|0.1|none|2
B2_bi_t06rt10_tau01_s2|$BI|0.6|0.1|1.0|2
"
i=0
for J in $JOBS; do
  [ -z "$J" ] && continue
  if [ $((i % NSHARD)) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
  i=$((i+1))
  IFS='|' read -r LABEL CK T TAU RT SEED <<< "$J"
  if [ "$RESUME" = "1" ] && [ -f "$O/$LABEL/wer_step68000_ras10.json" ]; then
    echo "[$(date '+%H:%M:%S')] SKIP $LABEL"; continue
  fi
  EXTRA=""
  [ "$RT" != "none" ] && EXTRA="--ras_temperature $RT"
  echo "[$(date '+%H:%M:%S')] START $LABEL"
  python scripts_local/cosyvoice_wer_eval.py \
    --checkpoint_path "$CK" --step 68000 \
    --cache_dir "$CACHE" --codebook "$CACHE/val/cosyvoice2_codebook.pt" \
    --cosyvoice_dir "$CV2" \
    --config small_sum --text_encoder_model HuggingFaceTB/SmolLM2-135M \
    --voice_max_frames 250 \
    --mrope_voice_rate 7.5 --mrope_scale_side text \
    --ras_win 10 --ras_tau "$TAU" --voice_temperature "$T" --seed "$SEED" \
    --n "$N" --device cuda:0 --skip_ceiling \
    --out_dir "$O/$LABEL" $EXTRA > "$O/logs/$LABEL.log" 2>&1
  echo "[$(date '+%H:%M:%S')] $LABEL exit=$?"
done
echo "[$(date '+%H:%M:%S')] SHARD $SHARD/$NSHARD DONE"
