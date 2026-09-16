#!/usr/bin/env bash
# A/B the EOV confidence guard at the best known config (unistream ck90000-ema, greedy).
# Output layout: eval_output/world_voice/<type>/<nickname>/
set -u
# Kill our own children if the driver is killed. Without this, `kill <driver>` orphans the
# python arms, which keep holding GPU memory and OOM the relaunch. `pkill -P $$` targets
# THIS shell's children by parent pid -- no name pattern, so it cannot match our own shell.
trap 'pkill -P $$ 2>/dev/null' EXIT INT TERM
CONC=${CONC:-2}
BASE=eval_output/world_voice
CACHE=./cached_datasets/Mekadrom/libriheavy_cosyvoice2_smollm2
CV2=/mnt/nasbro/cache/hf_home/hub/models--FunAudioLLM--CosyVoice2-0.5B/snapshots/eec1ae6c79877dbd9379285cf8789c9e0879293d
CK=runs/world_voice/cosyvoice2_smollm2_libriheavy_ar_flat_lr_ema_mrope75_0/checkpoint-90000-ema
mkdir -p "$BASE/logs" "$BASE/reports" "$BASE/audio"
# Nicknames only; flags resolved by case. A space-separated job list would word-split
# "--eov_min_prob 0.10 --eov_max_entropy 3.0" into separate loop iterations.
for SEED in 1 2; do
  for NICK in baseline eov_guard; do
    case "$NICK" in
      baseline)  EXTRA="" ;;
      eov_guard) EXTRA="--eov_min_prob 0.10 --eov_max_entropy 3.0" ;;
    esac
    LABEL="${NICK}_s${SEED}"
    [ -f "$BASE/reports/$LABEL/wer_step90000_ras10.json" ] && { echo "SKIP $LABEL"; continue; }
    while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 10; done
    echo "[$(date '+%H:%M:%S')] START $LABEL ${EXTRA:-(no guard)}"
    ( python scripts_local/cosyvoice_wer_eval.py \
        --checkpoint_path "$CK" --step 90000 \
        --cache_dir "$CACHE" --codebook "$CACHE/val/cosyvoice2_codebook.pt" \
        --cosyvoice_dir "$CV2" \
        --config small_sum --text_encoder_model HuggingFaceTB/SmolLM2-135M \
        --voice_max_frames 250 --mrope_voice_rate 7.5 --mrope_scale_side text \
        --ras_win 10 --ras_tau 0.1 --voice_temperature 0.0 --seed "$SEED" \
        --n 48 --device cuda:0 --skip_ceiling \
        --out_dir "$BASE/reports/$LABEL" --save_audio "$BASE/audio/$LABEL" $EXTRA \
        > "$BASE/logs/$LABEL.log" 2>&1
      echo "[$(date '+%H:%M:%S')] $LABEL exit=$?" ) &
  done
done
wait
echo "[$(date '+%H:%M:%S')] EOV GUARD AB DONE"
