"""Tabulate the world-voice temperature sweep as arms complete."""
import glob, json, os, re, sys

root = sys.argv[1] if len(sys.argv) > 1 else "eval_output/world_voice_temp_sweep"
rows = []
for d in sorted(glob.glob(os.path.join(root, "*_t*"))):
    js = glob.glob(os.path.join(d, "wer_step*.json"))
    if not js:
        continue
    j = json.load(open(js[0]))
    lab = os.path.basename(d)
    m = re.match(r"(uni|bi)_t(\d+)", lab)
    if not m:
        continue
    model, traw = m.group(1), m.group(2)
    temp = float(traw[0] + "." + traw[1:]) if len(traw) > 1 else float(traw)
    rows.append((model, temp, j))

if not rows:
    print("no completed arms yet")
    sys.exit(0)


def g(j, *path, default=float("nan")):
    cur = j
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


keys = None
for model in ("uni", "bi"):
    sub = sorted([r for r in rows if r[0] == model], key=lambda r: r[1])
    if not sub:
        continue
    print(f"\n=== {model} ===")
    if keys is None:
        keys = sorted(sub[0][2].keys())
        print(f"(json keys: {keys})\n")
    hdr = f"{'T':>5}  {'WER':>7} {'CER':>7} {'LCS':>7} {'hyp/ref':>8} {'cadence':>8}"
    print(hdr)
    for _, t, j in sub:
        wer = g(j, "generated", "wer", default=g(j, "wer"))
        cer = g(j, "generated", "cer", default=g(j, "cer"))
        lcs = g(j, "generated", "lcs_recall", default=g(j, "lcs_recall"))
        hr = g(j, "generated", "hyp_ref_word_ratio", default=g(j, "hyp_ref_word_ratio"))
        cad = g(j, "cadence_ratio", default=g(j, "generated", "cadence_ratio"))
        print(f"{t:>5.1f}  {wer:>7.4f} {cer:>7.4f} {lcs:>7.4f} {hr:>8.3f} {cad:>8.3f}")
