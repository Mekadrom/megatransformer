#!/usr/bin/env bash
# Follow-ups to the n=48 decision run. Gated on BOTH decision shards finishing.
#   E: uni at T=0.6+rt1.0 -> completes a {uni,bi} x {greedy, t06rt10} 2x2 with A/B/D
#   G: bi greedy on RAW (non-EMA) weights -> pairs with A to settle EMA-vs-raw,
#      which produced OPPOSITE signs at different temperatures on n=12 and was
#      left unresolved in the findings.
set -u
O=eval_output/world_voice_n48_decision
CACHE=./cached_datasets/Mekadrom/libriheavy_cosyvoice2_smollm2
CV2=/mnt/nasbro/cache/hf_home/hub/models--FunAudioLLM--CosyVoice2-0.5B/snapshots/eec1ae6c79877dbd9379285cf8789c9e0879293d
UNI_EMA=runs/world_voice/cosyvoice2_smollm2_libriheavy_ar_flat_lr_ema_mrope75_0/checkpoint-68000-ema
BI_RAW=runs/world_voice/cosyvoice2_smollm2_libriheavy_bistream_scratch_0/checkpoint-68000
# Gate on the shard log FILES, never a process pattern this shell itself contains.
while ! grep -q "DONE" eval_output/world_voice_n48_shard0.log \
   || ! grep -q "DONE" eval_output/world_voice_n48_shard1.log; do sleep 30; done
run () {
  LABEL="$1"; CK="$2"; T="$3"; TAU="$4"; RT="$5"
  [ -f "$O/$LABEL/wer_step68000_ras10.json" ] && { echo "SKIP $LABEL"; return; }
  EXTRA=""; [ "$RT" != "none" ] && EXTRA="--ras_temperature $RT"
  echo "[$(date '+%H:%M:%S')] START $LABEL"
  python scripts_local/cosyvoice_wer_eval.py \
    --checkpoint_path "$CK" --step 68000 \
    --cache_dir "$CACHE" --codebook "$CACHE/val/cosyvoice2_codebook.pt" \
    --cosyvoice_dir "$CV2" \
    --config small_sum --text_encoder_model HuggingFaceTB/SmolLM2-135M \
    --voice_max_frames 250 \
    --mrope_voice_rate 7.5 --mrope_scale_side text \
    --ras_win 10 --ras_tau "$TAU" --voice_temperature "$T" --seed 1 \
    --n 48 --device cuda:0 --skip_ceiling \
    --out_dir "$O/$LABEL" $EXTRA > "$O/logs/$LABEL.log" 2>&1
  echo "[$(date '+%H:%M:%S')] $LABEL exit=$?"
}
run E_uni_t06rt10_tau01_s1 "$UNI_EMA" 0.6 0.1 1.0
run G_biRAW_greedy_tau01_s1 "$BI_RAW" 0.0 0.1 none
echo "[$(date '+%H:%M:%S')] FOLLOWUP DONE"
