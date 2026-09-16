"""Does the model fail on RARE words specifically, regardless of trunk depth?

The ear reports that 'archaeologist', 'meticulous' and 'sixteenth' fail at 3, 32 AND 64
iterations. If true, the failure is capability, not compute, and it should show as per-word
recall falling with word rarity -- flat across depth.

Per-word recall: for each reference word, is it present in the ASR transcript of the render?
Bucketed by corpus frequency (computed from the val cache's own transcripts, so 'rare' means
rare in the training distribution, not in English at large).
"""
import collections, glob, json, math, os, statistics, sys

FREQ = collections.Counter()
for f in sorted(glob.glob("cached_datasets/Mekadrom/libriheavy_cosyvoice2_smollm2/val/shard_*.pt")):
    import torch
    d = torch.load(f, map_location="cpu", weights_only=False)
    for t in d["text"]:
        FREQ.update("".join(c.lower() if c.isalnum() or c.isspace() else " " for c in str(t)).split())
TOT = sum(FREQ.values())
print(f"frequency table: {len(FREQ)} types / {TOT} tokens from the val transcripts\n")


def norm(s):
    return "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in str(s)).split()


def bucket(w):
    c = FREQ.get(w, 0)
    if c == 0:   return "0 unseen"
    if c <= 2:   return "1 very rare (<=2)"
    if c <= 10:  return "2 rare (3-10)"
    if c <= 50:  return "3 uncommon (11-50)"
    if c <= 500: return "4 common (51-500)"
    return "5 very common (>500)"


def analyse(path, label):
    _j = json.load(open(path))
    rows = _j["rows"] if isinstance(_j, dict) else _j
    per = collections.defaultdict(lambda: [0, 0])   # bucket -> [hit, total]
    for r in rows:
        ref, hyp = r.get("ref"), r.get("hyp")
        if not ref or hyp is None:
            continue
        H = set(norm(hyp))
        for w in norm(ref):
            b = bucket(w)
            per[b][1] += 1
            if w in H:
                per[b][0] += 1
    if not per:
        return None
    print(f"=== {label}")
    print(f"  {'frequency bucket':24s} {'recall':>8} {'n':>7}")
    for b in sorted(per):
        h, t = per[b]
        print(f"  {b:24s} {h/t:>8.3f} {t:>7}")
    return per


for cap, lab in ((3, "cap 3"), (8, "cap 8"), (32, "cap 32 (logit_kl 1e-4)")):
    fs = glob.glob(f"eval_output/world_voice/reports/cap{cap}_s1/wer_step*.json")
    if not fs:
        fs = glob.glob("eval_output/world_voice/reports/crit_logitkl_s1/wer_step*.json") if cap == 32 else []
    if fs:
        analyse(fs[0], lab)
        print()
fs = glob.glob("eval_output/world_voice/reports/crit_none_s1/wer_step*.json")
if fs:
    analyse(fs[0], "full depth (none)")
