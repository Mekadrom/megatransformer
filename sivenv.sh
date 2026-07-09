#!/usr/bin/env bash
# Diagnostics env helper — one place for the checks I otherwise re-type every session.
# Usage:
#   ./sivenv.sh gpu                  # GPU index/used/total/util (one line each)
#   ./sivenv.sh runs [substr]        # run dirs matching substr, with min/max checkpoint step
#   ./sivenv.sh ck <substr> [step]   # step given -> exists? (prints path or NO); else min/max/last-5 steps
#   ./sivenv.sh path <substr>        # resolve substr -> full run dir path(s)
#   ./sivenv.sh eval [substr]        # eval_output dirs matching substr
#   ./sivenv.sh status               # GPU + latest checkpoint per run (default)
# substr is matched against the run DIR name (e.g. batchnorm, gnf->groupnormfrontend, mhasp, ecapa, covreg).
ROOT=/home/zadar/dev/projects/megatransformer-refactor
cd "$ROOT" || exit 1
_steps(){ ls -d "$1"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n; }
cmd=${1:-status}
case "$cmd" in
  gpu)
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader ;;
  runs)
    sub=${2:-}
    for d in runs/*/*/; do d=${d%/}; [ -d "$d" ] || continue; b=$(basename "$d")
      [[ -n "$sub" && "$b" != *"$sub"* ]] && continue
      s=$(_steps "$d"); printf '%-92s min=%-7s max=%s\n' "$b" "$(echo "$s"|head -1)" "$(echo "$s"|tail -1)"; done ;;
  ck)
    sub=$2; step=$3
    for d in runs/*/*"$sub"*/; do d=${d%/}; [ -d "$d" ] || continue; b=$(basename "$d")
      if [ -n "$step" ]; then
        p="$d/checkpoint-$step"; [ -d "$p" ] && echo "$b -> $p" || echo "$b -> NO checkpoint-$step"
      else
        s=$(_steps "$d"); printf '%s\n  min=%s max=%s last=[%s]\n' "$b" "$(echo "$s"|head -1)" "$(echo "$s"|tail -1)" "$(echo "$s"|tail -5|tr '\n' ' ')"
      fi; done ;;
  path)
    for d in runs/*/*"$2"*/; do d=${d%/}; [ -d "$d" ] && echo "$d"; done ;;
  eval)
    find eval_output -maxdepth 2 -type d 2>/dev/null | grep -i "${2:-.}" | sort ;;
  status)
    echo "== GPU =="; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
    echo "== runs (max ckpt) =="
    for d in runs/*/*/; do d=${d%/}; [ -d "$d" ] || continue; printf '  %-90s %s\n' "$(basename "$d")" "$(_steps "$d"|tail -1)"; done ;;
  *) echo "usage: ./sivenv.sh {gpu | runs [substr] | ck <substr> [step] | path <substr> | eval [substr] | status}" ;;
esac
