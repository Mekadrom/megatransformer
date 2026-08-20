"""Does the AR conditioning head sample a MULTIMODAL conditional as evenly as the parallel one?

WHY. The only reason T3/T4 exist instead of a point regressor is that p(Qwen3 conditioning |
caption) is multimodal: an MSE optimum averages the modes into an off-manifold blur, which is
exactly the collapse the naive run showed. A generative head is supposed to COMMIT to one mode
per draw and to visit the modes in proportion to their true weight. A build-time smoke test on
the AR head (ar_cond_flow_head.py) hinted it commits but SKEWS -- a ~3/13 split on a 2-mode toy
where the parallel head (cond_flow_head.py) managed ~13/11 -- on 16 draws of an ~800-step head,
which is far too little to call. This script is that test done properly: matched budget, many
draws, several training seeds, a step sweep to separate undertraining from a real preference,
and w=1 vs w=3 because guidance trades diversity for fidelity and w=3 is the operating point.

WHY IT MATTERS. A head with a mode preference has less per-draw diversity than it should, and
per-draw diversity is precisely where T3's headroom lived (w=3 mean 0.344 -> best-of-2 0.362).
A skew also rhymes with the original failure: a head that prefers one mode has a default.

THE TASK. C synthetic captions. Caption c owns a context ctx_c (M x adapter_dim) and two target
sequences A_c, B_c (L x seq_dim); a training draw picks mode A with probability --p_mode_a and
adds --sigma noise. The head must learn a genuinely bimodal conditional. Memorisation is fine
and is the point -- we are testing REPRESENTATION and SAMPLING of a bimodal conditional, not
generalisation.

METRICS (per head, per training step, per guidance w)
  minority_frac   mean over captions of min(nA,nB)/N.   0.5 = both modes equally; 0.0 = locked.
                  THE headline number -- label-free, and what "3/13" was measuring (0.19).
  frac_a          pooled fraction assigned to mode A. Compare to --p_mode_a: tests whether the
                  head reproduces the MIXTURE WEIGHT or distorts it.
  commitment      mean |t|, where t projects a draw onto the A->B axis (-1 at A, +1 at B, 0 at
                  the midpoint). ~1 = on a mode; ~0 = the conditional-mean blur a point head gives.
  dispersion      sample std / true-conditional std. ~1 = right spread (both heads passed this
                  at build time: 1.621 vs 1.630); <1 = under-dispersed.
  on_mode_frac    fraction of draws landing within 0.25 mode-gaps of a mode. READ THIS FIRST:
  dist_near       mean nearest-mode distance in mode-gap units. Together they answer "has this
                  head learned the task at all", which commitment/dispersion CANNOT -- an
                  untrained head reads commitment ~0.8 and dispersion ~1.0 purely because random
                  noise projected onto the mode axis has |t| ~ 0.8. A mode-split number from a
                  head with on_mode ~0 is meaningless; check this column before reading any other.
  tok0_lock_in    P(full-sequence mode == the mode token 0 alone landed in). Tests the mechanism
                  hypothesis for an AR skew: sample() feeds token 0 forward, so every later token
                  is conditioned on it and one draw gates the whole sequence. The parallel head
                  integrates all slots jointly, so it should NOT show the same lock-in.

TWO CONFOUNDS THIS CONTROLS FOR (both found while validating the harness, not hypothetical):
  * TIMESTEP DENSITY. CondFlowHead.loss samples ONE t per batch row and broadcasts it across all
    K slots; ARCondFlowHead.loss samples tau per (row, token). At matched steps the AR head gets
    L x more timestep coverage, so a step-matched gap can be supervision density rather than
    factorisation. --nar_time_per_token equalises it.
  * SAMPLER COARSENESS. Both live arms run flow_steps=8. --eval_flow_steps sweeps the Euler step
    count at inference with no retraining, so "commits at 32 but not at 8" is separable from
    "cannot represent the conditional".
  Also note the heads condition differently by construction: the AR per-token flow takes context
  through AdaLN modulation (c_in(z) + timestep), while the parallel head sees context ONLY via
  cross-attention. At small widths that makes the parallel head much harder to fit -- run this at
  the REAL dims (the defaults), not at toy dims, or the comparison is about width, not shape.

CAVEAT. This is a 2-mode toy at the real dims, not Qwen3 space. It can show that one head skews
where the other does not; it cannot say how much CLIPScore that costs. Judge the real arms by
rendered CLIPScore, always.

Example (single line, pick the GPU explicitly):
  CUDA_VISIBLE_DEVICES=3 uv run python scripts_local/flow_head_mode_split_probe.py --device cuda --out_dir eval_output/flow_head_mode_split/toy_2mode
"""
import argparse
import json
import math
import os
import time

import torch

from megatransformer.model.image.ar_cond_flow_head import ARCondFlowHead
from megatransformer.model.image.cond_flow_head import CondFlowHead


# ── task ──────────────────────────────────────────────────────────────────────────────
def build_task(args, device):
    """Fixed contexts + two target modes per caption. Deterministic in --task_seed."""
    g = torch.Generator(device="cpu").manual_seed(args.task_seed)
    ctx = torch.randn(args.captions, args.ctx_len, args.ctx_dim, generator=g)
    modes = torch.randn(args.captions, 2, args.seq_len, args.seq_dim, generator=g)
    # Separate the modes so "which mode did this draw land in" is unambiguous: push them apart
    # along their own difference until the gap is --mode_sep in units of the noise floor.
    d = modes[:, 1] - modes[:, 0]
    # The gap is set in units of the NOISE BALL RADIUS, sigma*sqrt(numel) -- not sigma. In
    # 20x2560 dims the noise ball has radius ~22.6 at sigma=0.1, so a "20*sigma" gap would be
    # 11x SMALLER than the within-mode noise and the modes would be unidentifiable.
    radius = args.sigma * math.sqrt(args.seq_len * args.seq_dim)
    scale = args.mode_sep * radius / d.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-6)
    mid = modes.mean(1)
    modes = torch.stack([mid - 0.5 * scale * d, mid + 0.5 * scale * d], dim=1)
    return ctx.to(device), modes.to(device)


def draw_targets(modes, idx, args, gen):
    """Sample (mode, noise) targets for the caption indices `idx`."""
    a = torch.rand(idx.shape[0], device=modes.device, generator=gen) < args.p_mode_a
    m = modes[idx, (~a).long()]                       # mode 0 = A, mode 1 = B
    return m + args.sigma * torch.randn(m.shape, device=modes.device, generator=gen)


def true_dispersion(modes, args, gen, n=1024, chunk=64):
    """Empirical std of the true conditional, averaged over captions.

    Chunked: at the real dims a single n x L x D allocation is ~840MB.
    """
    out = []
    for c in range(modes.shape[0]):
        tot = sq = cnt = 0.0
        for start in range(0, n, chunk):
            m = min(chunk, n - start)
            idx = torch.full((m,), c, device=modes.device, dtype=torch.long)
            x = draw_targets(modes, idx, args, gen).float()
            tot += x.sum().item()
            sq += (x * x).sum().item()
            cnt += x.numel()
        out.append(math.sqrt(max(sq / cnt - (tot / cnt) ** 2, 0.0)))
    return sum(out) / len(out)


# ── heads ─────────────────────────────────────────────────────────────────────────────
def make_heads(args, device):
    common = dict(seq_dim=args.seq_dim, ctx_dim=args.ctx_dim, steps=args.flow_steps,
                  time_sampling=args.time_sampling, cfg_dropout=args.cfg_dropout)
    nar = CondFlowHead(dim=args.flow_dim, n_heads=args.flow_heads,
                       n_layers=args.flow_layers, **common).to(device)
    ar = ARCondFlowHead(dim=args.ar_dim, n_heads=args.ar_heads, n_layers=args.ar_layers,
                        flow_layers=args.ar_flow_layers, max_len=max(args.ar_max_len, args.seq_len),
                        **common).to(device)
    return {"nar": nar, "ar": ar}


def nar_loss_per_token_time(head, target, ctx):
    """CondFlowHead.loss, but with a timestep per SLOT instead of one per row.

    The shipped CondFlowHead samples one t per batch row and broadcasts it over all K slots;
    ARCondFlowHead samples tau per (row, token). At matched steps that is L x more timestep
    coverage for the AR head, which would show up as an architectural difference when it is
    really supervision density. This variant removes that asymmetry.
    """
    b, k, _ = target.shape
    if head.cfg_dropout > 0 and head.training:
        drop = (torch.rand(b, device=ctx.device) < head.cfg_dropout).view(-1, 1, 1)
        ctx = torch.where(drop, head._null(ctx), ctx)
    t = head._sample_t(b * k, target.device, torch.float32).to(target.dtype).view(b, k)
    noise = torch.randn_like(target)
    tv = t.unsqueeze(-1)
    x_t = (1 - tv) * noise + tv * target
    # velocity() takes a (B,) timestep for its AdaLN modulation, so fold the slot axis into the
    # batch axis and give each slot its own single-slot sequence.
    v = head.velocity(x_t.reshape(b * k, 1, -1), t.reshape(b * k),
                      ctx.repeat_interleave(k, dim=0)).reshape(b, k, -1)
    return torch.nn.functional.mse_loss(v, target - noise)


def head_loss(name, head, target, ctx, args):
    if name == "ar":
        return head.loss(target, ctx, mask=None)
    if args.nar_time_per_token:
        return nar_loss_per_token_time(head, target, ctx)
    return head.loss(target, ctx)


def head_sample(name, head, ctx, args, gen, w, steps):
    return head.sample(ctx, args.seq_len, steps=steps, generator=gen, guidance=w)


# ── evaluation ────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(name, head, ctx, modes, args, w, steps, device, seed):
    """Draw --draws samples per caption at guidance w and score the mode structure."""
    head.eval()
    C, L, D = args.captions, args.seq_len, args.seq_dim
    A, B = modes[:, 0], modes[:, 1]
    diff = (B - A).reshape(C, -1)
    denom = diff.norm(dim=1).clamp_min(1e-6)
    unit = diff / denom.unsqueeze(1)
    mid = ((A + B) * 0.5).reshape(C, -1)

    assign = torch.zeros(C, args.draws, dtype=torch.long, device=device)
    tok0 = torch.zeros(C, args.draws, dtype=torch.long, device=device)
    tproj, sample_var, dnear, dpos = [], [], [], []
    gen = torch.Generator(device=device).manual_seed(seed)
    for c in range(C):
        got = 0
        parts = []
        while got < args.draws:
            n = min(args.sample_chunk, args.draws - got)
            x = head_sample(name, head, ctx[c].unsqueeze(0).expand(n, -1, -1).contiguous(),
                            args, gen, w, steps)
            parts.append(x.float())
            got += n
        x = torch.cat(parts, 0)                                    # (draws, L, D)
        flat = x.reshape(x.shape[0], -1)
        da = (flat - A[c].reshape(1, -1)).norm(dim=1)
        db = (flat - B[c].reshape(1, -1)).norm(dim=1)
        assign[c] = (db < da).long()                               # 0 = A, 1 = B
        # Distance to the nearest mode in units of the mode gap. ~sigma*sqrt(numel)/gap for a
        # head that has learned the task; ~0.5+ for one still emitting noise. This is what
        # separates "committed to a mode" from "not trained yet" -- |t| below cannot.
        dnear.append(torch.minimum(da, db) / denom[c])
        # token 0 only -- the AR head's first draw, which every later token conditions on
        d0a = (x[:, 0] - A[c, 0].unsqueeze(0)).norm(dim=1)
        d0b = (x[:, 0] - B[c, 0].unsqueeze(0)).norm(dim=1)
        tok0[c] = (d0b < d0a).long()
        # position on the A->B axis: -1 at A, +1 at B, 0 at the conditional-mean blur
        tproj.append(((flat - mid[c].unsqueeze(0)) @ unit[c]) / (0.5 * denom[c]))
        # per-position distance to the assigned mode, normalised by that position's mode gap
        tgt = torch.where(assign[c].view(-1, 1, 1).bool(), B[c].unsqueeze(0), A[c].unsqueeze(0))
        gap_pos = (B[c] - A[c]).norm(dim=-1).clamp_min(1e-6)          # (L,)
        dpos.append(((x - tgt).norm(dim=-1) / gap_pos.unsqueeze(0)).mean(0))
        sample_var.append(x.var(unbiased=False).item())

    tproj = torch.stack(tproj)
    dnear = torch.stack(dnear)
    n_b = assign.sum(1)
    n_a = args.draws - n_b
    minority = torch.minimum(n_a, n_b).float() / args.draws
    return {
        "guidance": w,
        "flow_steps": steps,
        "minority_frac": minority.mean().item(),
        "minority_frac_sd": minority.std(unbiased=False).item(),
        "frac_a": (n_a.sum().item() / (C * args.draws)),
        "commitment": tproj.abs().mean().item(),
        "dist_near": dnear.mean().item(),
        "on_mode_frac": (dnear < 0.25).float().mean().item(),
        "dist_near_by_pos": torch.stack(dpos).mean(0).tolist(),
        # >1 means the last quarter of the sequence sits further from its mode than the first
        # quarter -- the signature of AR error accumulation.
        "pos_drift": (torch.stack(dpos).mean(0)[-max(1, L // 4):].mean()
                      / torch.stack(dpos).mean(0)[:max(1, L // 4)].mean().clamp_min(1e-9)).item(),
        "dispersion_raw": math.sqrt(sum(sample_var) / len(sample_var)),
        "tok0_lock_in": (tok0 == assign).float().mean().item(),
        "tok0_minority_frac": (torch.minimum(args.draws - tok0.sum(1), tok0.sum(1)).float()
                               / args.draws).mean().item(),
        "per_caption_counts": [[int(a), int(b)] for a, b in zip(n_a[:args.show_captions].tolist(),
                                                                n_b[:args.show_captions].tolist())],
    }


def train_and_probe(name, head, ctx, modes, args, device, seed, disp_true):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    checkpoints = sorted(args.eval_at)
    rows, t0 = [], time.time()
    for step in range(1, checkpoints[-1] + 1):
        head.train()
        idx = torch.randint(0, args.captions, (args.batch_size,), device=device, generator=gen)
        target = draw_targets(modes, idx, args, gen)
        loss = head_loss(name, head, target, ctx[idx], args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
        # Constant LR after a short warmup: a decaying schedule would confound the step sweep.
        for pg in opt.param_groups:
            pg["lr"] = args.lr * min(1.0, step / max(1, args.warmup_steps))
        opt.step()
        if step in checkpoints:
            for w in args.guidance:
              for fs in args.eval_flow_steps:
                r = evaluate(name, head, ctx, modes, args, w, fs, device, seed + 2)
                r["dispersion"] = r.pop("dispersion_raw") / disp_true
                r.update(head=name, step=step, seed=seed, train_loss=loss.item())
                rows.append(r)
                print(f"  [{name} seed={seed} step={step} w={w} fs={fs}] "
                      f"minority={r['minority_frac']:.3f} frac_a={r['frac_a']:.3f} "
                      f"on_mode={r['on_mode_frac']:.3f} d_near={r['dist_near']:.3f} "
                      f"commit={r['commitment']:.3f} disp={r['dispersion']:.3f} "
                      f"tok0_lock={r['tok0_lock_in']:.3f} drift={r['pos_drift']:.3f} "
                      f"loss={loss.item():.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    return rows


# ── reporting ─────────────────────────────────────────────────────────────────────────
def binom_p(k, n, p):
    """Two-sided binomial p-value, no scipy.

    Exact for small n; normal approximation with a continuity correction above that -- the exact
    path builds a pmf over n+1 terms, and n here is captions*draws*seeds (~12k), where that is
    both ruinously slow and underflows p**i to 0.
    """
    if n == 0:
        return 1.0
    if n <= 1000:
        pmf = [math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
        return min(1.0, sum(v for v in pmf if v <= pmf[k] * (1 + 1e-9)))
    sd = math.sqrt(n * p * (1 - p))
    if sd <= 0:
        return 1.0 if k == n * p else 0.0
    z = (abs(k - n * p) - 0.5) / sd
    return min(1.0, math.erfc(max(z, 0.0) / math.sqrt(2)))


def aggregate(rows, keys=("minority_frac", "frac_a", "on_mode_frac", "dist_near", "commitment",
            "dispersion", "tok0_lock_in", "pos_drift")):
    out = {}
    for r in rows:
        out.setdefault((r["head"], r["step"], r["guidance"], r["flow_steps"]), []).append(r)
    agg = []
    for (head, step, w, fs), rs in sorted(out.items()):
        e = {"head": head, "step": step, "guidance": w, "flow_steps": fs,
             "n_seeds": len(rs)}
        for k in keys:
            v = [r[k] for r in rs]
            e[k] = sum(v) / len(v)
            e[k + "_sd"] = (sum((x - e[k]) ** 2 for x in v) / len(v)) ** 0.5
        agg.append(e)
    return agg


def render_table(agg, args):
    lines = ["| head | step | w | sampler_steps | minority_frac | frac_a | on_mode | d_near "
             "| commitment | dispersion | tok0_lock_in | pos_drift |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in agg:
        lines.append(
            f"| {e['head']} | {e['step']} | {e['guidance']} | {e['flow_steps']} | "
            f"{e['minority_frac']:.3f} ± {e['minority_frac_sd']:.3f} | "
            f"{e['frac_a']:.3f} | {e['on_mode_frac']:.3f} | {e['dist_near']:.3f} | "
            f"{e['commitment']:.3f} | {e['dispersion']:.3f} | "
            f"{e['tok0_lock_in']:.3f} | {e['pos_drift']:.3f} |")
    lines += ["", f"ideal: minority_frac 0.5 | frac_a {args.p_mode_a} | on_mode ~1 | "
                  f"d_near ~0 | commitment ~1 | dispersion ~1 | tok0_lock_in ~0.5 for a head "
                  f"with no first-token gating",
              "NOTE: commitment ~0.8 and dispersion ~1 are ALSO what an UNTRAINED head reads "
              "(random noise projected on the mode axis). Trust on_mode/d_near for 'has it "
              "learned the task at all'."]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda", help="pick the GPU EXPLICITLY (CUDA_VISIBLE_DEVICES)")
    p.add_argument("--out_dir", default="eval_output/flow_head_mode_split/toy_2mode")
    p.add_argument("--heads", default="nar,ar", help="comma list: nar (T3 parallel), ar (T4)")
    # task
    p.add_argument("--captions", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=20, help="native Qwen3 caption length is ~18-23")
    p.add_argument("--seq_dim", type=int, default=2560, help="real Qwen3 hidden size")
    p.add_argument("--ctx_dim", type=int, default=768, help="adapter_dim -- the Q-Former width")
    p.add_argument("--ctx_len", type=int, default=64, help="adapter seq_len -- Q-Former slots")
    p.add_argument("--p_mode_a", type=float, default=0.5, help="true mixture weight of mode A")
    p.add_argument("--sigma", type=float, default=0.1, help="within-mode noise")
    p.add_argument("--mode_sep", type=float, default=20.0, help="mode gap in units of sigma")
    p.add_argument("--task_seed", type=int, default=1234)
    # heads (defaults mirror small_sum_zimage_t4_ar / _t3)
    p.add_argument("--flow_dim", type=int, default=512)
    p.add_argument("--flow_heads", type=int, default=8)
    p.add_argument("--flow_layers", type=int, default=4)
    p.add_argument("--ar_dim", type=int, default=512)
    p.add_argument("--ar_heads", type=int, default=8)
    p.add_argument("--ar_layers", type=int, default=4)
    p.add_argument("--ar_flow_layers", type=int, default=3)
    p.add_argument("--ar_max_len", type=int, default=128)
    p.add_argument("--flow_steps", type=int, default=8)
    p.add_argument("--time_sampling", default="logit_normal")
    p.add_argument("--cfg_dropout", type=float, default=0.1, help="matches the live runs")
    p.add_argument("--nar_time_per_token", action="store_true",
                   help="give the parallel head a timestep per SLOT (the AR head already gets "
                        "one per token). Off = faithful to the shipped losses; on = isolates "
                        "the factorisation from timestep-supervision density.")
    # training
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4, help="matches --lr_flow in the live run")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--eval_at", default="500,1000,2000,4000,8000,16000",
                   help="step sweep -- separates undertraining from a real preference")
    p.add_argument("--seeds", default="0,1,2", help="independent training seeds")
    # evaluation
    p.add_argument("--draws", type=int, default=128, help="samples per caption per checkpoint")
    p.add_argument("--guidance", default="1.0,3.0", help="w=3 is the live operating point")
    p.add_argument("--eval_flow_steps", default="8,32",
                   help="sampler Euler steps to sweep at eval time (no retraining). 8 = what "
                        "both live arms use; a head that commits at 32 but not 8 is telling you "
                        "the LIVE sampler is too coarse for a multimodal conditional.")
    p.add_argument("--sample_chunk", type=int, default=256)
    p.add_argument("--show_captions", type=int, default=8, help="raw per-caption counts to record")
    args = p.parse_args()
    args.eval_at = [int(x) for x in args.eval_at.split(",") if x]
    args.guidance = [float(x) for x in args.guidance.split(",") if x]
    args.eval_flow_steps = [int(x) for x in args.eval_flow_steps.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    heads_wanted = [h.strip() for h in args.heads.split(",") if h.strip()]

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    ctx, modes = build_task(args, device)
    disp_true = true_dispersion(modes, args, torch.Generator(device=device).manual_seed(7))
    print(f"task: {args.captions} captions x 2 modes, seq {args.seq_len}x{args.seq_dim}, "
          f"ctx {args.ctx_len}x{args.ctx_dim}, true conditional std {disp_true:.4f}", flush=True)

    rows = []
    for seed in seeds:
        torch.manual_seed(seed)
        heads = make_heads(args, device)
        for name in heads_wanted:
            n_par = sum(q.numel() for q in heads[name].parameters())
            print(f"[{name}] seed {seed}: {n_par/1e6:.2f}M params ({n_par:,})", flush=True)
            rows += train_and_probe(name, heads[name], ctx, modes, args, device, seed, disp_true)

    agg = aggregate(rows)
    # Is the pooled split distinguishable from the true mixture weight?
    for e in agg:
        n = args.captions * args.draws * e["n_seeds"]
        e["frac_a_p_value"] = binom_p(int(round(e["frac_a"] * n)), n, args.p_mode_a)
    table = render_table(agg, args)
    print("\n" + table, flush=True)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({"args": vars(args), "true_dispersion": disp_true,
                   "aggregate": agg, "runs": rows}, f, indent=2)
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write(table + "\n")
    print(f"\nwrote {args.out_dir}/results.json and summary.md")


if __name__ == "__main__":
    main()
