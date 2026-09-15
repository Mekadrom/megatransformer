#!/usr/bin/env bash
# Second seed for both UNISTREAM cells, completing a 2-seed 2x2 with A/A2/B/B2.
# Without these the unistream row is single-seed, and its greedy-vs-decoupled gap
# (0.054) is SMALLER than the measured seed spread (0.065) -- i.e. currently unreadable.
set -u
O=eval_output/world_voice_n48_decision
CACHE=./cached_datasets/Mekadrom/libriheavy_cosyvoice2_smollm2
CV2=/mnt/nasbro/cache/hf_home/hub/models--FunAudioLLM--CosyVoice2-0.5B/snapshots/eec1ae6c79877dbd9379285cf8789c9e0879293d
UNI=runs/world_voice/cosyvoice2_smollm2_libriheavy_ar_flat_lr_ema_mrope75_0/checkpoint-68000-ema
while ! grep -q "FOLLOWUP DONE" eval_output/world_voice_n48_followup.log; do sleep 30; done
run () {
  LABEL="$1"; T="$2"; RT="$3"
  [ -f "$O/$LABEL/wer_step68000_ras10.json" ] && { echo "SKIP $LABEL"; return; }
  EXTRA=""; [ "$RT" != "none" ] && EXTRA="--ras_temperature $RT"
  echo "[$(date '+%H:%M:%S')] START $LABEL"
  python scripts_local/cosyvoice_wer_eval.py \
    --checkpoint_path "$UNI" --step 68000 \
    --cache_dir "$CACHE" --codebook "$CACHE/val/cosyvoice2_codebook.pt" \
    --cosyvoice_dir "$CV2" \
    --config small_sum --text_encoder_model HuggingFaceTB/SmolLM2-135M \
    --voice_max_frames 250 \
    --mrope_voice_rate 7.5 --mrope_scale_side text \
    --ras_win 10 --ras_tau 0.1 --voice_temperature "$T" --seed 2 \
    --n 48 --device cuda:0 --skip_ceiling \
    --out_dir "$O/$LABEL" $EXTRA > "$O/logs/$LABEL.log" 2>&1
  echo "[$(date '+%H:%M:%S')] $LABEL exit=$?"
}
run D2_uni_greedy_tau01_s2 0.0 none
run E2_uni_t06rt10_tau01_s2 0.6 1.0
echo "[$(date '+%H:%M:%S')] UNI SEEDS DONE"
