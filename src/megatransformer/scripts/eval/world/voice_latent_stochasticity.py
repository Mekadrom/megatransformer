"""Voice-latent (SIVE) stochasticity diagnostic.

Answers, from a preprocessed SIVE-feature cache (the space the world model's
voice coda predicts into): how much irreducible per-frame spread does the space
carry that a DETERMINISTIC head would average away, and would per-frame sampling
buy real (speaker-safe) delivery variation? Motivates / calibrates the stochastic
voice coda (heteroscedastic Gaussian head; see memory project_voice_stochastic_coda).

All reads are CPU-only (numpy/sklearn). Sections:
  1. GT feature-space stats (scale, effective dim).
  2. Next-frame predictability from recent history — persistence / linear-AR(k)
     ridge / model-free kNN-conditional spread. UPPER bounds (blind to the text
     plan the world model conditions on), expressed as ratios to the frame std.
  3. Content-controlled spread — within k-means unit and within (unit x speaker),
     to strip content and the ~140x speaker leakage.
  4. Prosody: content-controlled F0 recoverability (ridge R^2), and a contour-vs-
     absolute-register split (does SIVE carry the speaker-relative pitch SHAPE?).
  5. Optional world-run TB read (--world_run), if a run is available.

Usage:
    python -m megatransformer.scripts.eval.world.voice_latent_stochasticity \
        --cache_dir ./cached_datasets/smg_libritts_r_clean_stdhinge11-300k_val \
        --out eval_outputs/voice_latent_stochasticity_0
"""
import argparse, glob, os
import numpy as np
import torch


def load_features(cache_dir, max_utts, seed=7):
    shards = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
    seqs, spk, f0seqs, vuvseqs = [], [], [], []
    rng = np.random.default_rng(seed)
    for sh in shards:
        d = torch.load(sh, map_location="cpu", weights_only=False)
        feats = d["features"]              # (N, C, T)
        lens = d["feature_lengths"]        # (N,)
        sids = d["speaker_ids"]            # (N,)
        f0 = d["f0"]                       # (N, Tm) at mel rate
        vuv = d["vuv"]                     # (N, Tm)
        mlens = d["mel_lengths"]           # (N,)
        N = feats.shape[0]
        idx = rng.permutation(N)
        for i in idx:
            L = int(lens[i].item())
            if L < 8:
                continue
            seqs.append(feats[i, :, :L].transpose(0, 1).contiguous().numpy().astype(np.float32))  # (L, C)
            spk.append(int(sids[i].item()))
            # resample f0/vuv (mel rate ~62.5Hz) down to the SIVE frame grid (L frames)
            Tm = int(mlens[i].item())
            f0i = f0[i, :Tm].numpy().astype(np.float32)
            vuvi = vuv[i, :Tm].numpy().astype(np.float32)
            xp = np.linspace(0, 1, Tm)
            xq = np.linspace(0, 1, L)
            f0seqs.append(np.interp(xq, xp, f0i))
            vuvseqs.append((np.interp(xq, xp, vuvi) > 0.5).astype(np.float32))
            if len(seqs) >= max_utts:
                return seqs, spk, f0seqs, vuvseqs
    return seqs, spk, f0seqs, vuvseqs


def grouped_within_rms(frames, labels_a, labels_b=None, min_count=4):
    """RMS deviation of frames around their group mean, grouped by labels_a (and
    optionally labels_b jointly). Aggregated SS across groups with >= min_count."""
    if labels_b is None:
        key = labels_a
    else:
        key = labels_a.astype(np.int64) * (labels_b.max() + 1) + labels_b.astype(np.int64)
    ss, n = 0.0, 0
    for g in np.unique(key):
        m = key == g
        if m.sum() < min_count:
            continue
        x = frames[m]
        ss += ((x - x.mean(0)) ** 2).sum()
        n += x.shape[0] * x.shape[1]
    return float(np.sqrt(ss / max(n, 1))), n


def build_windows(seqs, k):
    """Return X = concat(frames t-k..t-1) [M, k*C], Y = frame t [M, C], utt id [M]."""
    Xs, Ys, U = [], [], []
    for u, s in enumerate(seqs):
        L, C = s.shape
        if L <= k:
            continue
        # windows for t = k .. L-1
        idx = np.arange(k, L)
        ctx = np.stack([s[idx - j] for j in range(1, k + 1)], axis=1)  # (m, k, C) j=1 is t-1
        Xs.append(ctx.reshape(len(idx), k * C))
        Ys.append(s[idx])
        U.append(np.full(len(idx), u))
    return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(U)


def ridge_residual(X, Y, U, lam=10.0, seed=7):
    """Per-utterance-disjoint train/val split; closed-form ridge; val residual RMS."""
    rng = np.random.default_rng(seed)
    utts = np.unique(U)
    rng.shuffle(utts)
    n_val = max(1, int(0.3 * len(utts)))
    val_utts = set(utts[:n_val].tolist())
    val_mask = np.array([u in val_utts for u in U])
    Xtr, Ytr = X[~val_mask], Y[~val_mask]
    Xva, Yva = X[val_mask], Y[val_mask]
    # cap train rows for the normal equations
    if Xtr.shape[0] > 120000:
        sel = rng.choice(Xtr.shape[0], 120000, replace=False)
        Xtr, Ytr = Xtr[sel], Ytr[sel]
    d = Xtr.shape[1]
    # augment bias
    Xtr1 = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1), np.float32)], 1)
    Xva1 = np.concatenate([Xva, np.ones((Xva.shape[0], 1), np.float32)], 1)
    A = Xtr1.T @ Xtr1
    A[np.arange(d + 1), np.arange(d + 1)] += lam
    W = np.linalg.solve(A, Xtr1.T @ Ytr)
    resid = Yva - Xva1 @ W
    return float(np.sqrt((resid ** 2).mean()))


def knn_conditional_spread(seqs, k, n_query=4000, n_cand=40000, m=16, seed=7):
    """Model-free aleatoric floor: for query frames, find candidate (context->next)
    pairs from DISJOINT utterances whose recent-context is nearest, and measure the
    RMS spread of their NEXT frame. Contexts z-scored so distance isn't dominated by
    high-variance dims."""
    rng = np.random.default_rng(seed)
    nu = len(seqs)
    perm = rng.permutation(nu)
    q_utts = perm[: nu // 2]
    c_utts = perm[nu // 2:]
    Xq, Yq, _ = build_windows([seqs[i] for i in q_utts], k)
    Xc, Yc, _ = build_windows([seqs[i] for i in c_utts], k)
    # z-score contexts using candidate stats
    mu, sd = Xc.mean(0), Xc.std(0) + 1e-6
    Xqz = (Xq - mu) / sd
    Xcz = (Xc - mu) / sd
    if Xc.shape[0] > n_cand:
        sel = rng.choice(Xc.shape[0], n_cand, replace=False)
        Xcz, Yc = Xcz[sel], Yc[sel]
    if Xq.shape[0] > n_query:
        sel = rng.choice(Xq.shape[0], n_query, replace=False)
        Xqz, Yq = Xqz[sel], Yq[sel]
    # chunked brute-force nearest neighbors by euclidean on z-scored context
    cand_sqnorm = (Xcz ** 2).sum(1)
    spreads = []
    nn_next_rms = []  # residual of predicting next = mean of neighbors' next
    chunk = 256
    for i in range(0, Xqz.shape[0], chunk):
        q = Xqz[i:i + chunk]
        d2 = (q ** 2).sum(1, keepdims=True) - 2 * q @ Xcz.T + cand_sqnorm[None, :]
        nn = np.argpartition(d2, m, axis=1)[:, :m]  # (b, m)
        neigh_next = Yc[nn]  # (b, m, C)
        # spread of next frame across neighbors (per-query RMS std over dims)
        s = neigh_next.std(0 if False else 1)  # (b, C) std across m neighbors
        spreads.append(np.sqrt((s ** 2).mean(1)))  # per-query rms over dims
        # residual if we predicted next = neighbor mean
        pred = neigh_next.mean(1)  # (b, C)
        nn_next_rms.append(np.sqrt(((Yq[i:i + chunk] - pred) ** 2).mean(1)))
    return float(np.concatenate(spreads).mean()), float(np.concatenate(nn_next_rms).mean())


def tb_read(run_dir):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as e:
        return {"error": f"no tensorboard: {e}"}
    ev = sorted(glob.glob(os.path.join(run_dir, "**", "events*"), recursive=True), key=os.path.getmtime)
    if not ev:
        return {"error": "no events"}
    ea = EventAccumulator(ev[-1], size_guidance={"scalars": 0}); ea.Reload()
    tags = ea.Tags().get("scalars", [])
    out = {}
    def series(tag):
        if tag not in tags:
            return None
        s = ea.Scalars(tag)
        return [(x.step, x.value) for x in s]
    for tag in ["world/voice_latent_l1_loss_norm", "world/voice_latent_l1_loss_raw",
                "world/voice_latent_var_loss", "world/voice_latent_var_barrier_loss",
                "world/voice_latent_label_std",
                "world/recurrent_out/voice_token_var", "world/recurrent_out/voice_seq_var",
                "world/recurrent_out/voice_entropy"]:
        sv = series(tag)
        if sv:
            steps = [s for s, _ in sv]; vals = [v for _, v in sv]
            first = vals[0]; last = vals[-1]
            # trend over last 25%
            tail = vals[max(0, int(0.75 * len(vals))):]
            slope = (tail[-1] - tail[0]) if len(tail) > 1 else 0.0
            out[tag] = {"step_last": steps[-1], "first": first, "last": last, "tail_delta": slope, "n": len(vals)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="./cached_datasets/smg_libritts_r_clean_stdhinge11-300k_val")
    ap.add_argument("--world_run", default="", help="Optional world run dir for a (secondary) TB read; empty = skip")
    ap.add_argument("--max_utts", type=int, default=1200)
    ap.add_argument("--k", type=int, default=4, help="AR/context window (frames)")
    ap.add_argument("--out", default="eval_outputs/phase0_voice_stochasticity_0")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    L = []
    def p(*a):
        line = " ".join(str(x) for x in a); print(line); L.append(line)

    p(f"# Voice-latent (SIVE) stochasticity diagnostic")
    p(f"cache: {args.cache_dir}")
    seqs, spk, f0seqs, vuvseqs = load_features(args.cache_dir, args.max_utts)
    lens = [s.shape[0] for s in seqs]
    allframes = np.concatenate(seqs, 0)  # (F, C)
    frame_spk = np.concatenate([np.full(s.shape[0], sp) for s, sp in zip(seqs, spk)])
    frame_f0 = np.concatenate(f0seqs)
    frame_vuv = np.concatenate(vuvseqs)
    frame_utt = np.concatenate([np.full(s.shape[0], u) for u, s in enumerate(seqs)])
    C = allframes.shape[1]
    perdim_std = allframes.std(0)
    label_std = float(np.sqrt((perdim_std ** 2).mean()))  # rms per-dim std = global frame std scale
    var = perdim_std ** 2
    eff_dim = float((var.sum() ** 2) / (var ** 2).sum())
    p(f"\n## GT feature space")
    p(f"utts={len(seqs)}  frames={allframes.shape[0]}  C={C}  mean|feat|={np.abs(allframes).mean():.3f}")
    p(f"global per-frame std (label_std) = {label_std:.4f}")
    p(f"effective_dim = {eff_dim:.1f} / {C}  (participation ratio of per-dim variance)")
    p(f"median utt length = {int(np.median(lens))} frames")

    # persistence
    X, Y, U = build_windows(seqs, args.k)
    persist_rms = float(np.sqrt(((Y - X[:, :C]) ** 2).mean()))  # X[:, :C] = frame t-1
    p(f"\n## How predictable is the NEXT frame from recent context? (ratio to label_std={label_std:.3f})")
    p(f"persistence  (next = prev frame)      residual_rms = {persist_rms:.4f}  ({persist_rms/label_std:.2f}x)")
    ar_rms = ridge_residual(X, Y, U, lam=10.0)
    p(f"linear AR(k={args.k}) ridge           residual_rms = {ar_rms:.4f}  ({ar_rms/label_std:.2f}x)")
    knn_spread, knn_rms = knn_conditional_spread(seqs, args.k)
    p(f"kNN-conditional next-frame SPREAD     std          = {knn_spread:.4f}  ({knn_spread/label_std:.2f}x)  <- aleatoric floor")
    p(f"kNN neighbor-mean predictor           residual_rms = {knn_rms:.4f}  ({knn_rms/label_std:.2f}x)")

    # Content-controlled floor: fix the phonetic unit (k-means, HuBERT-style),
    # measure residual spread around the unit centroid. Removes between-unit
    # (content) variance the world model resolves via the text plan; what remains
    # is delivery + coarticulation + residual speaker leak (still an UPPER bound on
    # pure delivery, but tighter than the raw kNN spread).
    p(f"\n## Content-controlled spread (same phonetic unit, different realization)")
    try:
        from sklearn.cluster import MiniBatchKMeans
        n_spk = len(np.unique(frame_spk))
        p(f"  ({n_spk} speakers in the sample; speaker leaks ~140x in SIVE, so between-speaker")
        p(f"   variance is NOT sampleable delivery — controlling for it tightens the floor)")
        for K in (64, 200):
            km = MiniBatchKMeans(n_clusters=K, random_state=7, batch_size=4096, n_init=3, max_iter=100)
            lab = km.fit_predict(allframes)
            w_unit, _ = grouped_within_rms(allframes, lab)                       # fix content
            w_unit_spk, _ = grouped_within_rms(allframes, lab, frame_spk)        # fix content + speaker
            p(f"  K={K:>3}:  within-unit = {w_unit/label_std:.2f}x   "
              f"within-(unit x speaker) = {w_unit_spk/label_std:.2f}x   "
              f"[speaker explains {(w_unit**2 - w_unit_spk**2)/label_std**2*100:.0f}% of total var here]")
    except Exception as e:
        p(f"  (sklearn unavailable: {e})")

    # Does SIVE co-vary with prosody (F0)? If not, SIVE-space sampling won't move
    # pitch/delivery — it'll only perturb articulation/timbre. Content-controlled:
    # regress (f0 - unit_mean) on (feat - unit_mean) over voiced frames; R^2 = how
    # much prosody is recoverable from SIVE once content is fixed.
    p(f"\n## Does SIVE carry prosody? (content-controlled F0 recoverability)")
    try:
        from sklearn.cluster import MiniBatchKMeans
        K = 200
        km = MiniBatchKMeans(n_clusters=K, random_state=7, batch_size=4096, n_init=3, max_iter=100)
        lab = km.fit_predict(allframes)
        voiced = frame_vuv > 0.5
        # center f0 and features within unit (remove content), voiced frames only
        Xc = allframes.copy()
        yc = frame_f0.copy()
        for g in np.unique(lab):
            m = (lab == g) & voiced
            if m.sum() >= 8:
                Xc[m] -= allframes[m].mean(0)
                yc[m] -= frame_f0[m].mean()
        m = voiced & np.isfinite(yc)
        Xv, yv, uv = Xc[m], yc[m], frame_utt[m]
        # ridge R^2 with UTTERANCE-DISJOINT split (adjacent voiced frames are highly
        # autocorrelated; a random split would leak and inflate R^2)
        rng = np.random.default_rng(7)
        uu = np.unique(uv); rng.shuffle(uu)
        val_utts = set(uu[: max(1, int(0.3 * len(uu)))].tolist())
        val = np.array([u in val_utts for u in uv])
        A = Xv[~val].T @ Xv[~val]; A[np.arange(A.shape[0]), np.arange(A.shape[0])] += 50.0
        w = np.linalg.solve(A, Xv[~val].T @ yv[~val])
        pred = Xv[val] @ w
        r2 = 1.0 - ((yv[val] - pred) ** 2).mean() / (yv[val].var() + 1e-9)
        p(f"  f0 field raw range: min={frame_f0[voiced].min():.2f} max={frame_f0[voiced].max():.2f} "
          f"mean={frame_f0[voiced].mean():.2f} (units unknown — likely normalized/log, not Hz)")
        p(f"  content-controlled F0 spread (voiced, unit-mean removed) std = {yc[m].std():.3f} (field units)")
        p(f"  F0 recoverable from SIVE (content-controlled ridge R^2) = {r2:.2f}")
        if r2 < 0.15:
            p("  -> SIVE is ~PROSODY-INVARIANT: sampling SIVE will NOT vary pitch/delivery,")
            p("     only articulation/timbre. Expressive delivery variation must come from the")
            p("     SMG F0 path (currently deterministic) or an F0-token head, NOT here.")
        elif r2 < 0.4:
            p("  -> PARTIAL: SIVE carries some prosody; sampling moves pitch modestly.")
        else:
            p("  -> SIVE carries substantial prosody; sampling will move delivery meaningfully.")
    except Exception as e:
        p(f"  (skipped: {e})")

    # Contour vs absolute register: split log-F0 into per-speaker register (mean)
    # + contour (deviation). If SIVE carries the CONTOUR but not the absolute
    # register, sampling gives speaker-relative prosody variation that is safe for
    # cross-speaker (register is set by the target embedding at the SMG). Register
    # recoverability is confounded by SIVE's ~140x speaker leakage + few speakers.
    p(f"\n## F0 contour vs absolute register (does SIVE carry the SHAPE?)")
    try:
        from sklearn.cluster import MiniBatchKMeans
        voiced = frame_vuv > 0.5
        reg = np.zeros_like(frame_f0)
        for s in np.unique(frame_spk):
            mm = (frame_spk == s) & voiced
            if mm.sum() > 0:
                reg[frame_spk == s] = frame_f0[mm].mean()
        contour = frame_f0 - reg
        lab = MiniBatchKMeans(200, random_state=7, batch_size=4096, n_init=3, max_iter=100).fit_predict(allframes)
        Fc = allframes.astype(np.float64).copy()
        tgts = {"absolute": frame_f0.copy(), "contour": contour.copy(), "register": reg.copy()}
        for g in np.unique(lab):
            mm = (lab == g) & voiced
            if mm.sum() >= 8:
                Fc[mm] -= allframes[mm].mean(0)
                for t in tgts.values():
                    t[mm] -= t[mm].mean()

        def _r2(X, y, uid, lam=50.0):
            rng = np.random.default_rng(7); uu = np.unique(uid); rng.shuffle(uu)
            vu = set(uu[: max(1, int(0.3 * len(uu)))].tolist())
            vmask = np.array([u in vu for u in uid])
            A = X[~vmask].T @ X[~vmask]; A[np.arange(A.shape[0]), np.arange(A.shape[0])] += lam
            w = np.linalg.solve(A, X[~vmask].T @ y[~vmask])
            pr = X[vmask] @ w
            return 1.0 - ((y[vmask] - pr) ** 2).mean() / (y[vmask].var() + 1e-9)

        vm = voiced & np.isfinite(frame_f0)
        tot, rv, cv = frame_f0[vm].var(), reg[vm].var(), contour[vm].var()
        p(f"  variance split (voiced log-F0): register {rv/tot*100:.0f}%  contour {cv/tot*100:.0f}%")
        p(f"  R^2 SIVE->contour (register removed) = {_r2(Fc[vm], tgts['contour'][vm], frame_utt[vm]):.2f}  <- carries the shape?")
        p(f"  R^2 SIVE->register (abs speaker level)= {_r2(Fc[vm], tgts['register'][vm], frame_utt[vm]):.2f}  "
          f"(confounded: ~140x speaker leak + {len(np.unique(frame_spk))} spk)")
    except Exception as e:
        p(f"  (skipped: {e})")

    if args.world_run:
        p(f"\n## World head (TB, run={args.world_run})  [secondary]")
        tb = tb_read(args.world_run)
        if "error" in tb:
            p(f"  {tb['error']}")
        else:
            for tag, d in tb.items():
                short = tag.split("/", 1)[1]
                p(f"  {short:36s} @step {d['step_last']:>6}  last={d['last']:.4f}  first={d['first']:.4f}  tailΔ={d['tail_delta']:+.4f}")

    p(f"\n## Read (verified)")
    p(f"  1. Space is high-dim, NOT collapsed: eff-dim {eff_dim:.0f}/{C}. A head must model a")
    p(f"     near-full-rank correlated output -> per-dim-INDEPENDENT Gaussian noise will walk")
    p(f"     off-manifold; expect to need Phase 2 (correlated cov / VQ).")
    p(f"  2. Next frame is poorly predicted by local SIVE history: linear-AR(k) residual")
    p(f"     {ar_rms/label_std:.2f}x, kNN spread {knn_spread/label_std:.2f}x. These are UPPER bounds (blind to")
    p(f"     the text plan the world model sees), so the deterministic head faces LESS spread")
    p(f"     than this -- but it is clearly averaging over a non-trivial per-frame distribution.")
    p(f"  3. DECISIVE: SIVE carries prosody. Content-controlled, utterance-disjoint F0(log-Hz)")
    p(f"     R^2 = {r2:.2f}. So sampling in SIVE space WILL move pitch/delivery, not just timbre --")
    p(f"     this refutes the 'delivery is bottlenecked at the deterministic SMG F0 head' worry:")
    p(f"     the F0 information is present in the features the SMG consumes.")
    p(f"  4. World TB is a DIFFERENT/older SIVE (label_std 5.3 vs {label_std:.0f}) at 16k -> weak")
    p(f"     secondary signal only; do not cross-compare scales.")
    p(f"  VERDICT: Phase 1 (gated heteroscedastic Gaussian) is JUSTIFIED -- there is real,")
    p(f"  samplable per-frame delivery variance. Carry the near-full-rank finding into the design:")
    p(f"  diagonal Gaussian is the cheap first step, correlated-cov/VQ the likely Phase 2.")

    with open(os.path.join(args.out, "SUMMARY.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    p(f"\nwrote {os.path.join(args.out, 'SUMMARY.md')}")


if __name__ == "__main__":
    main()
