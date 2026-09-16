#!/usr/bin/env bash
# AUTONOMOUS RE-MEASUREMENT of 2026-09-15's world-voice findings under a CORRECT recurrent
# exit criterion. Everything measured earlier that day used `kl_divergence`, which compares
# post-norm activations with F.kl_div and therefore exits whenever a SIGNED quantity goes
# negative -- freezing arbitrary positions mid-computation rather than converged ones.
#
# Stages (each 2 seeds, n=48, audio saved):
#   crit_*   criterion shootout at the best known config
#   samp_*   sampling config, under logit_kl
#   ckpt_*   checkpoint ranking, under logit_kl
#   guard_*  is the EOV confidence guard still worth anything under logit_kl
#
# Resumable: any arm whose report json exists is skipped.
set -u
trap 'pkill -P $$ 2>/dev/null' EXIT INT TERM
CONC=${CONC:-2}
BASE=eval_output/world_voice
CACHE=./cached_datasets/Mekadrom/libriheavy_cosyvoice2_smollm2
CV2=/mnt/nasbro/cache/hf_home/hub/models--FunAudioLLM--CosyVoice2-0.5B/snapshots/eec1ae6c79877dbd9379285cf8789c9e0879293d
UNI90=runs/world_voice/cosyvoice2_smollm2_libriheavy_ar_flat_lr_ema_mrope75_0/checkpoint-90000-ema
UNI68=runs/world_voice/cosyvoice2_smollm2_libriheavy_ar_flat_lr_ema_mrope75_0/checkpoint-68000-ema
BI68=runs/world_voice/cosyvoice2_smollm2_libriheavy_bistream_scratch_0/checkpoint-68000-ema
mkdir -p "$BASE/logs" "$BASE/reports" "$BASE/audio"

# label | checkpoint | step | extra flags
run_arm () {
  LABEL="$1"; CK="$2"; STEP="$3"; shift 3; EXTRA="$*"
  [ -f "$BASE/reports/$LABEL/wer_step${STEP}_ras10.json" ] && { echo "[$(date '+%H:%M:%S')] SKIP $LABEL"; return; }
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 10; done
  echo "[$(date '+%H:%M:%S')] START $LABEL :: $EXTRA"
  ( python scripts_local/cosyvoice_wer_eval.py \
      --checkpoint_path "$CK" --step "$STEP" \
      --cache_dir "$CACHE" --codebook "$CACHE/val/cosyvoice2_codebook.pt" \
      --cosyvoice_dir "$CV2" \
      --config small_sum --text_encoder_model HuggingFaceTB/SmolLM2-135M \
      --voice_max_frames 250 --mrope_voice_rate 7.5 --mrope_scale_side text \
      --ras_win 10 --ras_tau 0.1 --n 48 --device cuda:0 --skip_ceiling \
      --out_dir "$BASE/reports/$LABEL" --save_audio "$BASE/audio/$LABEL" $EXTRA \
      > "$BASE/logs/$LABEL.log" 2>&1
    echo "[$(date '+%H:%M:%S')] $LABEL exit=$?" ) &
}

LKL="--exit_criteria logit_kl --exit_criteria_threshold 5e-4"
for S in 1 2; do
  # --- stage crit: readout-free latent_diff at two thresholds (logit_kl / legacy / none
  #     are already covered by the crit_* arms from the earlier drivers)
  run_arm "crit_latentdiff03_s${S}" "$UNI90" 90000 --voice_temperature 0.0 --seed $S --exit_criteria latent_diff --exit_criteria_threshold 0.03
  run_arm "crit_latentdiff01_s${S}" "$UNI90" 90000 --voice_temperature 0.0 --seed $S --exit_criteria latent_diff --exit_criteria_threshold 0.01
done
for S in 1 2; do
  # --- stage samp: sampling config under logit_kl (greedy == crit_logitkl_s*)
  run_arm "samp_t07_logitkl_s${S}"     "$UNI90" 90000 --voice_temperature 0.7 --seed $S $LKL
  run_arm "samp_t06rt10_logitkl_s${S}" "$UNI90" 90000 --voice_temperature 0.6 --ras_temperature 1.0 --seed $S $LKL
done
for S in 1 2; do
  # --- stage ckpt: checkpoint ranking under logit_kl (uni90k == crit_logitkl_s*)
  run_arm "ckpt_uni68k_logitkl_s${S}" "$UNI68" 68000 --voice_temperature 0.0 --seed $S $LKL
  run_arm "ckpt_bi68k_logitkl_s${S}"  "$BI68"  68000 --voice_temperature 0.0 --seed $S $LKL
done
for S in 1 2; do
  # --- stage guard: is the EOV guard still worth anything (guard_off == crit_logitkl_s*)
  run_arm "guard_on_logitkl_s${S}" "$UNI90" 90000 --voice_temperature 0.0 --seed $S $LKL --eov_min_prob 0.10 --eov_max_entropy 3.0
done
wait
echo "[$(date '+%H:%M:%S')] RECHECK CAMPAIGN DONE"
