"""CosyVoice 2 A/B: early_text_delta for the curric vs nocurric arms at matched steps.

Runs world_voice_ar_diagnostics.py on each arm's checkpoints with the CosyVoice 2 shape
(512-dim codebook, no F0 head, 250-frame budget) and — critically — at each arm's OWN
training alpha.

WHY THE ALPHA MATTERS: the curric arm trains with voice->voice attention scaled by a
schedule (alpha=floor for mask_steps, then a floor->cap ramp over ramp_steps, then cap).
Probing it at alpha=1 before the ramp finishes is OUT OF DISTRIBUTION -- the model never
saw voice history at full strength, accuracy collapses, and the number is not that model's
real conditioning. The nocurric arm has no schedule and is always alpha=1.

CONSEQUENCE FOR READING THE RESULTS: before the ramp completes (step >= mask+ramp, i.e.
30000 with the launched settings) the two arms are in DIFFERENT REGIMES and their numbers
are not apples-to-apples. The clean comparison starts at 30k. Before that, only one
asymmetric read is valid: if the CURRIC arm leads while handicapped, that is meaningful
(a model denied history beating one with full access); the reverse is expected by
construction and means nothing.

  uv run python scripts_local/world_tts_ab_probe.py --step 35000 --device cuda:3
  uv run python scripts_local/world_tts_ab_probe.py --latest --device cuda:3     # newest common step
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

CACHE = "cached_datasets/Mekadrom/libritts_r_cosyvoice2_smollm2"
CODEBOOK = f"{CACHE}/val/cosyvoice2_codebook.pt"
ARMS = {
    # name: (run_dir, curriculum settings as LAUNCHED; None = no curriculum, always alpha=1)
    "curric":   ("runs/world/world_tts_cosyvoice2_smollm2_0",
                 {"mask": 10000, "ramp": 20000, "floor": 0.0, "cap": 1.0, "power": 1.0}),
    "nocurric": ("runs/world/world_tts_cosyvoice2_smollm2_nocurric_0", None),
}


def alpha_at(step, cur):
    """Mirror of CommonTrainer._voice_attn_alpha (training.py). None => no curriculum."""
    if cur is None:
        return None
    if step < cur["mask"]:
        return cur["floor"]
    t = step - cur["mask"]
    if t >= cur["ramp"]:
        return cur["cap"]
    return cur["floor"] + (cur["cap"] - cur["floor"]) * (t / cur["ramp"]) ** cur["power"]


def checkpoints(run_dir):
    out = []
    for d in glob.glob(os.path.join(run_dir, "checkpoint-*")):
        m = re.search(r"checkpoint-(\d+)$", d)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def nearest(steps, target, tol):
    c = [s for s in steps if abs(s - target) <= tol]
    return min(c, key=lambda s: abs(s - target)) if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=None, help="probe both arms at ~this step")
    ap.add_argument("--latest", action="store_true", help="use the newest step both arms reached")
    ap.add_argument("--tol", type=int, default=1200, help="checkpoint-matching tolerance")
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--bs", type=int, default=8, help="keep small — shares a GPU with training")
    ap.add_argument("--out_dir", default="eval_output/world_tts_cosy_ab")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    have = {k: checkpoints(v[0]) for k, v in ARMS.items()}
    for k, v in have.items():
        print(f"{k:9s}: {len(v)} checkpoints, latest={v[-1] if v else None}")
    if a.latest:
        if not all(have.values()):
            sys.exit("Both arms need at least one checkpoint.")
        a.step = min(max(v) for v in have.values())
        print(f"--latest -> probing at ~{a.step}")
    if a.step is None:
        sys.exit("Pass --step or --latest.")

    os.makedirs(a.out_dir, exist_ok=True)
    results = {}
    for name, (run_dir, cur) in ARMS.items():
        s = nearest(have[name], a.step, a.tol)
        if s is None:
            print(f"[skip] {name}: no checkpoint within {a.tol} of {a.step}")
            continue
        alpha = alpha_at(s, cur)
        ckpt = os.path.join(run_dir, f"checkpoint-{s}")
        out = os.path.join(a.out_dir, f"{name}_step{s}")
        cmd = [
            "uv", "run", "python", "scripts_local/world_voice_ar_diagnostics.py",
            "--checkpoint_path", ckpt, "--step", str(s),
            "--cache_dir", CACHE, "--codebook", CODEBOOK,
            "--config", "small_sum",
            "--text_encoder_model", "HuggingFaceTB/SmolLM2-135M",
            "--voice_max_frames", "250", "--no_voice_predict_f0",
            "--n", str(a.n), "--bs", str(a.bs), "--device", a.device,
            "--repeat", "0.0314", "--ngram_ceiling", "0.0545", "--asymptote", "0.2261",
            "--skip_generation", "--out_dir", out,
        ]
        if alpha is not None:
            cmd += ["--voice_attn_alpha", f"{alpha:.4f}"]
        print(f"\n=== {name} @ {s}  alpha={alpha if alpha is not None else '1.0 (no curriculum)'} ===")
        print(" ".join(cmd))
        if a.dry_run:
            continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[FAIL] {name}: {r.stderr[-600:]}")
            continue
        etd = re.search(r"early_text_delta \(clean text signal\)\*\* \| \*\*([+-]?[\d.]+)", r.stdout)
        acc = re.search(r"acc_real \(content top-1\) \| ([\d.]+)", r.stdout)
        results[name] = {
            "step": s, "alpha": alpha,
            "early_text_delta": float(etd.group(1)) if etd else None,
            "acc_real": float(acc.group(1)) if acc else None,
            "report": os.path.join(out, "report.md"),
        }
        print(f"  early_text_delta={results[name]['early_text_delta']}  acc_real={results[name]['acc_real']}")

    if results and not a.dry_run:
        p = os.path.join(a.out_dir, "results.jsonl")
        with open(p, "a") as f:
            f.write(json.dumps({"target_step": a.step, "arms": results}) + "\n")
        print(f"\nappended -> {p}")
        if len(results) == 2:
            c, n = results.get("curric"), results.get("nocurric")
            ramp_end = ARMS["curric"][1]["mask"] + ARMS["curric"][1]["ramp"]
            comparable = min(c["step"], n["step"]) >= ramp_end
            print(f"\ncurric {c['early_text_delta']}  vs  nocurric {n['early_text_delta']}")
            print("COMPARABLE (both post-ramp, alpha=1 for each)" if comparable else
                  f"NOT apples-to-apples yet: curric is at alpha={c['alpha']:.3f}, ramp ends at "
                  f"{ramp_end}. Only a CURRIC lead is meaningful here.")


if __name__ == "__main__":
    main()
