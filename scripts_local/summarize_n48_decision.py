"""Paired per-utterance comparison for the n=48 decision run.

Comparing two aggregate LCS numbers throws away the pairing: all arms score the SAME
utterances, so the per-utterance difference has far less variance than either arm's
marginal spread. This computes LCS per utterance, then the paired delta with a
bootstrap CI and a sign test, and reports the seed-replicate delta as the floor any
real effect must clear.
"""
import glob, json, os, sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "eval_output/world_voice_n48_decision"


def norm(s):
    return "".join(c for c in str(s).lower() if c.isalnum() or c.isspace()).split()


def lcs_recall(ref, hyp):
    n, m = len(ref), len(hyp)
    if n == 0:
        return None
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        for j in range(1, m + 1):
            cur[j] = prev[j - 1] + 1 if ref[i - 1] == hyp[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[m] / n


def load(label):
    fs = glob.glob(os.path.join(ROOT, label, "wer_step*.json"))
    if not fs:
        return None
    rows = json.load(open(fs[0]))["rows"]
    out = {}
    for r in rows:
        v = lcs_recall(norm(r.get("ref", "")), norm(r.get("hyp", "")))
        if v is not None:
            out[r["idx"]] = v
    return out


def paired(a, b, la, lb):
    if a is None or b is None:
        print(f"  {la} vs {lb}: one or both arms pending")
        return
    keys = sorted(set(a) & set(b))
    d = [b[k] - a[k] for k in keys]
    n = len(d)
    mean = sum(d) / n
    wins = sum(1 for x in d if x > 1e-9)
    losses = sum(1 for x in d if x < -1e-9)
    # deterministic bootstrap CI (fixed LCG; Math.random equivalents are unavailable
    # in some runners and a fixed seed keeps this reproducible anyway)
    seed, boot = 12345, []
    for _ in range(2000):
        s = 0.0
        for _ in range(n):
            seed = (1103515245 * seed + 12345) % (1 << 31)
            s += d[seed % n]
        boot.append(s / n)
    boot.sort()
    lo, hi = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]
    sig = "CI EXCLUDES 0" if (lo > 0 or hi < 0) else "CI INCLUDES 0 -- not resolved"
    print(f"  {lb} - {la}:  mean paired delta {mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"({sig})")
    print(f"     per-utterance: {wins} better / {losses} worse / {n - wins - losses} tied  (n={n})")


A  = load("A_bi_greedy_tau01_s1")
A2 = load("A2_bi_greedy_tau01_s2")
B  = load("B_bi_t06rt10_tau01_s1")
B2 = load("B2_bi_t06rt10_tau01_s2")
C  = load("C_bi_t06rt10_tau03_s1")
D  = load("D_uni_greedy_tau01_s1")
E  = load("E_uni_t06rt10_tau01_s1")
G  = load("G_biRAW_greedy_tau01_s1")

print("=== NOISE FLOOR (same config, different seed) ===")
paired(A, A2, "A(greedy s1)", "A2(greedy s2)")
paired(B, B2, "B(t06rt10 s1)", "B2(t06rt10 s2)")
print("\n=== THE DECISION: greedy vs T=0.6 + rt1.0 ===")
paired(A, B, "A(greedy)", "B(t06rt10)")
print("\n=== tau 0.1 vs 0.3 within the rt1.0 regime ===")
paired(B, C, "B(tau0.1)", "C(tau0.3)")
print("\n=== bistream vs unistream ===")
paired(D, A, "D(uni greedy)", "A(bi greedy)")
paired(E, B, "E(uni t06rt10)", "B(bi t06rt10)")
print("\n=== EMA vs raw (bistream, greedy) ===")
paired(G, A, "G(raw)", "A(EMA)")
