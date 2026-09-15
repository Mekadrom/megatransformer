#!/usr/bin/env bash
# Sampling sweep for the UNISTREAM baseline at its eval-loss minimum (ck90000).
# 3 configs x 2 seeds, n=48, audio saved. Runs CONCURRENCY arms at once: these are
# batch-1 autoregressive jobs that leave a 4090 ~80% idle on their own.
# All arms --skip_ceiling so the RNG streams match across arms.
set -u
CONC=${CONC:-3}
O=eval_output/world_voice_ck90k
CACHE=./cached_datasets/Mekadrom/libriheavy_cosyvoice2_smollm2
CV2=/mnt/nasbro/cache/hf_home/hub/models--FunAudioLLM--CosyVoice2-0.5B/snapshots/eec1ae6c79877dbd9379285cf8789c9e0879293d
CK=runs/world_voice/cosyvoice2_smollm2_libriheavy_ar_flat_lr_ema_mrope75_0/checkpoint-90000-ema
mkdir -p "$O/logs"
# label|temp|ras_temp
JOBS="greedy|0.0|none t07|0.7|none t06rt10|0.6|1.0"
for SEED in 1 2; do
  for J in $JOBS; do
    IFS='|' read -r NAME T RT <<< "$J"
    LABEL="${NAME}_s${SEED}"
    [ -f "$O/$LABEL/wer_step90000_ras10.json" ] && { echo "SKIP $LABEL"; continue; }
    EXTRA=""; [ "$RT" != "none" ] && EXTRA="--ras_temperature $RT"
    while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 10; done
    echo "[$(date '+%H:%M:%S')] START $LABEL (T=$T rt=$RT)"
    ( python scripts_local/cosyvoice_wer_eval.py \
        --checkpoint_path "$CK" --step 90000 \
        --cache_dir "$CACHE" --codebook "$CACHE/val/cosyvoice2_codebook.pt" \
        --cosyvoice_dir "$CV2" \
        --config small_sum --text_encoder_model HuggingFaceTB/SmolLM2-135M \
        --voice_max_frames 250 --mrope_voice_rate 7.5 --mrope_scale_side text \
        --ras_win 10 --ras_tau 0.1 --voice_temperature "$T" --seed "$SEED" \
        --n 48 --device cuda:0 --skip_ceiling \
        --out_dir "$O/$LABEL" --save_audio "$O/$LABEL/wav" $EXTRA \
        > "$O/logs/$LABEL.log" 2>&1
      echo "[$(date '+%H:%M:%S')] $LABEL exit=$?" ) &
  done
done
wait
echo "[$(date '+%H:%M:%S')] CK90K SWEEP DONE"
