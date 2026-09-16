#!/usr/bin/env bash
# Does a low iteration cap fail SELECTIVELY on hard material?
# The ear reports 3 iterations sounding fine; the hypothesis is that hard words and fast
# speech are where shallow trunk compute would break down. A flat LCS across caps would
# mean the trunk barely uses its depth; a gap that widens on the hard subset would confirm it.
set -u
trap 'pkill -P $$ 2>/dev/null' EXIT INT TERM
CONC=${CONC:-3}
BASE=eval_output/world_voice
CACHE=./cached_datasets/Mekadrom/libriheavy_cosyvoice2_smollm2
CV2=/mnt/nasbro/cache/hf_home/hub/models--FunAudioLLM--CosyVoice2-0.5B/snapshots/eec1ae6c79877dbd9379285cf8789c9e0879293d
CK=runs/world_voice/cosyvoice2_smollm2_libriheavy_ar_flat_lr_ema_mrope75_0/checkpoint-90000-ema
mkdir -p "$BASE/logs" "$BASE/reports"
for CAP in 1 2 3 4 6 8 16 32; do
  for SEED in 1 2; do
    LABEL="cap${CAP}_s${SEED}"
    [ -f "$BASE/reports/$LABEL/wer_step90000_ras10.json" ] && { echo "SKIP $LABEL"; continue; }
    while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 10; done
    echo "[$(date '+%H:%M:%S')] START $LABEL"
    ( python scripts_local/cosyvoice_wer_eval.py \
        --checkpoint_path "$CK" --step 90000 \
        --cache_dir "$CACHE" --codebook "$CACHE/val/cosyvoice2_codebook.pt" \
        --cosyvoice_dir "$CV2" \
        --config small_sum --text_encoder_model HuggingFaceTB/SmolLM2-135M \
        --voice_max_frames 250 --mrope_voice_rate 7.5 --mrope_scale_side text \
        --ras_win 10 --ras_tau 0.1 --voice_temperature 0.0 --seed "$SEED" \
        --exit_criteria logit_kl --exit_criteria_threshold 1e-4 --trunk_iters "$CAP" \
        --n 48 --device cuda:0 --skip_ceiling \
        --out_dir "$BASE/reports/$LABEL" > "$BASE/logs/$LABEL.log" 2>&1
      echo "[$(date '+%H:%M:%S')] $LABEL exit=$?" ) &
  done
done
wait
echo "[$(date '+%H:%M:%S')] ITERCAP SWEEP DONE"
