#!/usr/bin/env bash
# Where does the logit_kl compute/quality curve bend?
# Known: 5e-4 -> depth 19.50, LCS 0.8849;  none -> depth 32, LCS 0.9103.
# TIGHTER thresholds exit LATER, so depth should rise toward 32 and LCS toward 0.9103.
# The question is whether there is a knee -- a threshold that recovers most of the 0.0255
# while still saving meaningful trunk compute.
set -u
trap 'pkill -P $$ 2>/dev/null' EXIT INT TERM
CONC=${CONC:-3}
BASE=eval_output/world_voice
CACHE=./cached_datasets/Mekadrom/libriheavy_cosyvoice2_smollm2
CV2=/mnt/nasbro/cache/hf_home/hub/models--FunAudioLLM--CosyVoice2-0.5B/snapshots/eec1ae6c79877dbd9379285cf8789c9e0879293d
CK=runs/world_voice/cosyvoice2_smollm2_libriheavy_ar_flat_lr_ema_mrope75_0/checkpoint-90000-ema
mkdir -p "$BASE/logs" "$BASE/reports" "$BASE/audio"
# nickname:threshold   (5e-4 already measured as crit_logitkl_*)
for PAIR in 1e-3:1e-3 1e-4:1e-4 5e-5:5e-5 1e-5:1e-5; do
  NICK="${PAIR%%:*}"; THR="${PAIR##*:}"
  for SEED in 1 2; do
    LABEL="thr_logitkl_${NICK}_s${SEED}"
    [ -f "$BASE/reports/$LABEL/wer_step90000_ras10.json" ] && { echo "SKIP $LABEL"; continue; }
    while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 10; done
    echo "[$(date '+%H:%M:%S')] START $LABEL (threshold $THR)"
    ( python scripts_local/cosyvoice_wer_eval.py \
        --checkpoint_path "$CK" --step 90000 \
        --cache_dir "$CACHE" --codebook "$CACHE/val/cosyvoice2_codebook.pt" \
        --cosyvoice_dir "$CV2" \
        --config small_sum --text_encoder_model HuggingFaceTB/SmolLM2-135M \
        --voice_max_frames 250 --mrope_voice_rate 7.5 --mrope_scale_side text \
        --ras_win 10 --ras_tau 0.1 --voice_temperature 0.0 --seed "$SEED" \
        --n 48 --device cuda:0 --skip_ceiling \
        --exit_criteria logit_kl --exit_criteria_threshold "$THR" \
        --out_dir "$BASE/reports/$LABEL" --save_audio "$BASE/audio/$LABEL" \
        > "$BASE/logs/$LABEL.log" 2>&1
      echo "[$(date '+%H:%M:%S')] $LABEL exit=$?" ) &
  done
done
wait
echo "[$(date '+%H:%M:%S')] THRESHOLD SWEEP DONE"
