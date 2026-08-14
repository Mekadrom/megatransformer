"""How much PROSODY (F0 contour) does each encoder's content feature carry? The SMG
pitch head reads the content feature as the "contour" in f0_predictor_input='features'
mode; if a feature is prosody-poor the pitch path must fall back to an externally
supplied contour ('contour' mode). This measures features->log-F0 predictability (R2 on
held-out voiced frames), matched to the 1000-utt dumps' waveforms (raw_subset.pt).

Calibration: SIVE should read ~0.60 (the known SIVE F0 R2). Then Mimi/ContentVec relative
to that says whether Mimi can drive 'features' mode or needs 'contour'.

  CUDA_VISIBLE_DEVICES=3 uv run python -m scripts_local.content_encoder_eval.f0_probe \
      --dumps sive mimi contentvec --n 400
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcrepe

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def crepe_f0(wav16k, n_frames):
    """torchcrepe log-F0 + voiced mask, resampled to n_frames."""
    wav = wav16k.float().reshape(1, -1).to(DEVICE)
    hop = 160  # 100 Hz
    pitch, periodicity = torchcrepe.predict(
        wav, 16000, hop_length=hop, fmin=50, fmax=550, model="tiny",
        return_periodicity=True, device=DEVICE, pad=True)
    pitch = pitch[0]           # [M]
    per = periodicity[0]
    logf0 = torch.log(pitch.clamp(min=1e-3))
    # resample to n_frames
    logf0 = F.interpolate(logf0[None, None], size=n_frames, mode="linear", align_corners=False)[0, 0]
    per = F.interpolate(per[None, None], size=n_frames, mode="linear", align_corners=False)[0, 0]
    voiced = per > 0.5
    return logf0.cpu(), voiced.cpu()


class MLP(nn.Module):
    def __init__(self, d, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--dump_dir", default="./eval_output/content_encoder_eval")
    ap.add_argument("--raw", default="./eval_output/content_encoder_eval/raw_subset.pt")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    raw = torch.load(a.raw, weights_only=False)["records"]
    rng = np.random.RandomState(a.seed)

    print(f"{'encoder':<14} {'F0 R2':>7} {'vuv_acc':>8} {'n_utt':>6} {'voiced%':>8}")
    print("-" * 50)
    for name in a.dumps:
        d = torch.load(f"{a.dump_dir}/{name}.pt", weights_only=False)
        recs = d["records"]
        nmax = min(a.n, len(recs), len(raw))
        X, Y, V = [], [], []  # per-frame feature, logf0, voiced
        utt_id = []
        for i in range(nmax):
            f = recs[i]["feats"].float()                 # [L,D]
            wav = raw[i]["waveform"].float()
            lf0, voiced = crepe_f0(wav, f.shape[0])
            X.append(f); Y.append(lf0); V.append(voiced)
            utt_id.append(np.full(f.shape[0], i))
        X = torch.cat(X); Y = torch.cat(Y); V = torch.cat(V)
        utt_id = np.concatenate(utt_id)
        # CONTOUR metric: remove per-utterance voiced mean (the speaker OFFSET the SMG
        # gets from ECAPA), leaving the within-utterance F0 variation the features must
        # carry. Matches the config's "% of within-utterance F0 variation" framing.
        for i in range(nmax):
            m = (utt_id == i) & V.numpy()
            if m.sum() > 1:
                mt = torch.from_numpy(m)
                Y[mt] = Y[mt] - Y[mt].mean()
        # utt-level split
        uids = np.arange(nmax); rng.shuffle(uids)
        te_u = set(uids[:max(2, nmax // 5)].tolist())
        te_mask = np.isin(utt_id, list(te_u))
        tr = torch.from_numpy(~te_mask & V.numpy())
        te = torch.from_numpy(te_mask & V.numpy())
        # standardize features on train
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xn = (X - mu) / sd
        # standardize target on train (for stable regression; R2 is scale-invariant)
        ym, ys = Y[tr].mean(), Y[tr].std() + 1e-6

        probe = MLP(X.shape[1]).to(DEVICE)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        Xtr, Ytr = Xn[tr].to(DEVICE), ((Y[tr] - ym) / ys).to(DEVICE)
        probe.train()
        ntr = Xtr.shape[0]
        for step in range(a.steps):
            bi = torch.randint(0, ntr, (4096,), device=DEVICE)
            pred = probe(Xtr[bi])
            loss = F.mse_loss(pred, Ytr[bi])
            opt.zero_grad(); loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            pv = probe(Xn[te].to(DEVICE)).cpu() * ys + ym
        yte = Y[te]
        ss_res = ((yte - pv) ** 2).sum()
        ss_tot = ((yte - yte.mean()) ** 2).sum()
        r2 = float(1 - ss_res / ss_tot)
        voiced_pct = float(V.float().mean())
        print(f"{name:<14} {r2:>7.3f} {'-':>8} {nmax:>6} {voiced_pct:>7.1%}")


if __name__ == "__main__":
    main()
