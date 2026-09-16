#!/usr/bin/env bash
# Third point on the depth curve: NoOp = the full 32-iteration budget the model was
# TRAINED at. With legacy <10 and logit_kl ~18, this says whether quality keeps rising
# with depth (criterion is just a compute knob) or peaks in between.
set -u
trap 'pkill -P $$ 2>/dev/null' EXIT INT TERM
BASE=eval_output/world_voice
CACHE=./cached_datasets/Mekadrom/libriheavy_cosyvoice2_smollm2
CV2=/mnt/nasbro/cache/hf_home/hub/models--FunAudioLLM--CosyVoice2-0.5B/snapshots/eec1ae6c79877dbd9379285cf8789c9e0879293d
CK=runs/world_voice/cosyvoice2_smollm2_libriheavy_ar_flat_lr_ema_mrope75_0/checkpoint-90000-ema
while ! grep -q "EXITCRIT AB DONE" "$BASE/logs/_driver_exitcrit.log"; do sleep 30; done
for SEED in 1 2; do
  LABEL="crit_none_s${SEED}"
  [ -f "$BASE/reports/$LABEL/wer_step90000_ras10.json" ] && { echo "SKIP $LABEL"; continue; }
  echo "[$(date '+%H:%M:%S')] START $LABEL (full 32 iters)"
  ( python scripts_local/cosyvoice_wer_eval.py \
      --checkpoint_path "$CK" --step 90000 \
      --cache_dir "$CACHE" --codebook "$CACHE/val/cosyvoice2_codebook.pt" \
      --cosyvoice_dir "$CV2" \
      --config small_sum --text_encoder_model HuggingFaceTB/SmolLM2-135M \
      --voice_max_frames 250 --mrope_voice_rate 7.5 --mrope_scale_side text \
      --ras_win 10 --ras_tau 0.1 --voice_temperature 0.0 --seed "$SEED" \
      --n 48 --device cuda:0 --skip_ceiling \
      --out_dir "$BASE/reports/$LABEL" --save_audio "$BASE/audio/$LABEL" \
      --exit_criteria none > "$BASE/logs/$LABEL.log" 2>&1
    echo "[$(date '+%H:%M:%S')] $LABEL exit=$?" ) &
done
wait
echo "[$(date '+%H:%M:%S')] FULLDEPTH DONE"
