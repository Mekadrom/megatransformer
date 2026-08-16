#!/usr/bin/env bash
# Data-parallel TTS-intelligibility WER: launch N tts_intelligibility workers on one GPU, each
# taking a disjoint stripe of the same seed-selected index set, then merge. AR generation is
# batch-1 latency-bound so one process idles the GPU; N in parallel fill it (~N x throughput).
#
# Usage: wer_sharded.sh <GPU> <NSHARDS> <OUT_DIR> -- <all tts_intelligibility args except --output_dir/--num_shards/--shard_id>
set -euo pipefail
cd /home/zadar/dev/projects/megatransformer-refactor
GPU="$1"; N="$2"; OUT="$3"; shift 3; [ "$1" = "--" ] && shift
mkdir -p "$OUT"
pids=()
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES="$GPU" uv run python -m megatransformer.scripts.eval.world.tts_intelligibility \
    "$@" --num_shards "$N" --shard_id "$i" --output_dir "$OUT/shard_$i" > "$OUT/shard_$i.log" 2>&1 &
  pids+=($!)
done
echo "launched $N workers on GPU $GPU (pids: ${pids[*]})"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" = 1 ] && { echo "a worker failed; see $OUT/shard_*.log"; tail -5 "$OUT"/shard_*.log; exit 1; }
uv run python scripts_local/wer_merge.py "$OUT"
