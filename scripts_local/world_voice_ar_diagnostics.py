"""World-TTS voice-AR diagnostics: is the SIVE-VQ token stream autoregressable WITHOUT
n-gram degeneration, and is the AR actually reading the text?

Three probes on a world-model checkpoint (held-out val), written as a markdown report so a
later checkpoint re-run is a clean before/after:

  1. TEACHER-FORCED unit prediction (real text): top-1 accuracy + perplexity on content
     units, vs the text-free n-gram ceiling/asymptote (from ngram_unit_baseline.py). Above
     the ceiling => the model uses more than local unit history.
  2. TEXT ABLATION: same batch with transcripts SHUFFLED across the batch (each voice paired
     with a wrong text). Delta accuracy = the text's causal contribution. ~0 => text ignored
     (which is what "coherent bursts unrelated to the prompt" looks like).
  3. FREE-RUNNING GENERATION degeneration: generate from text prompts and measure length +
     EOV behavior, adjacent-repeat rate, longest run, codebook coverage, unit entropy, and
     distinct bi/tri-gram ratios -- all vs the ground-truth val distribution. Catches
     collapse-to-repetition / looping vs healthy diverse-but-unconditioned output.

Reuses visualize.py's model/dataset loading. Needs a GPU.

Usage:
  python scripts_local/world_voice_ar_diagnostics.py \
      --checkpoint_path runs/world/<run>/checkpoint-7000 --step 7000 \
      --cache_dir <voice_base_dir> --codebook <codebook.pt> \
      [--config small_sum] [--n 256] [--gen_n 32] [--device cuda:0] \
      [--ngram_ceiling 0.211 --asymptote 0.229 --repeat 0.117]
"""
import argparse
import math
import os
from argparse import Namespace
from collections import Counter

import torch
import torch.nn.functional as F

from megatransformer.scripts.eval.world.visualize import load_world_model, load_dataset
from megatransformer.scripts.data.world.data_collator import MultimodalDataCollator
from megatransformer.utils.codebook import load_codebook
from megatransformer.utils import constants


def bootstrap_delta_ci(per_utt, iters=10000, seed=0, alpha=0.05):
    """95% CI on (real-shuf)/n, resampling UTTERANCES. per_utt: [(real_hits, shuf_hits, n)].

    At n=256 the CI on early_text_delta is ~+-0.012 — wider than the step-to-step moves we
    were reading as trends. Report the interval so that stops happening.
    """
    import random as _rnd
    rng = _rnd.Random(seed)
    m = len(per_utt)
    if m == 0:
        return float("nan"), float("nan")
    samples = []
    for _ in range(iters):
        r = sh = c = 0
        for _ in range(m):
            x, y, k = per_utt[rng.randrange(m)]
            r += x; sh += y; c += k
        samples.append((r - sh) / max(c, 1))
    samples.sort()
    return samples[int(alpha / 2 * iters)], samples[int((1 - alpha / 2) * iters)]


def text_horizon(buckets, threshold=0.01):
    """Frame index where text_delta decays below `threshold` — the CORRECT HORIZON.

    The ear reports a good onset that "falls off quickly", and the buckets show text_delta
    roughly HALVING every 8 frames. A taller first bucket is not the same as a wider one:
    improving onset conditioning raises delta at frames 0-8, while actually getting better at
    speech means holding delta FURTHER IN. This collapses the decay curve to one trackable
    number so that distinction is visible across checkpoints.

    Linear interpolation between bucket midpoints; returns inf if delta never drops below
    threshold, 0.0 if it starts below it.
    """
    pts = [((b["lo"] + min(b["hi"], 256)) / 2.0, b["delta"]) for b in buckets]
    if not pts:
        return float("nan")
    if pts[0][1] < threshold:
        return 0.0
    for (x0, d0), (x1, d1) in zip(pts, pts[1:]):
        if d1 < threshold <= d0:
            if d0 == d1:
                return x1
            return x0 + (x1 - x0) * (d0 - threshold) / (d0 - d1)
    return float("inf")


def build_args(a, feature_channels):
    """A Namespace satisfying visualize.load_world_model + load_dataset for this run shape.

    feature_channels is DERIVED from the codebook's dim (the dataset builds voice features as
    centroids[unit_ids], so they are always the codebook width): 256 for Mimi cb0, 512 for
    CosyVoice 2. Never hardcode it -- a mismatch silently state-mismatches the load."""
    return Namespace(
        config=a.config, checkpoint_path=a.checkpoint_path,
        voice_codebook_path=a.codebook, voice_feature_channels=feature_channels,
        voice_predict_f0=a.voice_predict_f0,
        include_modes="voice", include_tasks="voice_synthesis",
        cache_dir=None, text_cache_dir=None, audio_cache_dir=None,
        voice_cache_dir=a.cache_dir, image_cache_dir=None,
        use_memorization_dataset=False, max_samples=None, num_eval_samples=a.n,
        tie_word_embeddings=False, share_block_weights=False,
        gen_query_mode=None, n_image_gen_positions=None, iteration_norm=None,
        text_encoder_model=getattr(a, "text_encoder_model", None),
        # M-RoPE: use_mrope is auto-detected from the checkpoint by load_world_model, but
        # scale_side/rate carry no weights, so they must be passed through (a wrong side
        # evaluates the trunk under the wrong coordinate system). See add_mrope_args.
        mrope_scale_side=getattr(a, "mrope_scale_side", None),
        mrope_voice_rate=getattr(a, "mrope_voice_rate", None),
    )


def add_mrope_args(ap):
    """M-RoPE eval flags. Required for an M-RoPE checkpoint; ignored for any other."""
    ap.add_argument("--mrope_scale_side", default=None, choices=["voice", "text", "off"],
                    help="REQUIRED for an M-RoPE checkpoint (it has no weights to detect it "
                         "from): which stream absorbs the frame rate, matching the training CLI. "
                         "'off' DELIBERATELY evaluates an M-RoPE checkpoint under single-axis "
                         "RoPE -- the protocol every pre-fix diagnostic accidentally used; only "
                         "useful for measuring how much that mismatch cost.")
    ap.add_argument("--mrope_voice_rate", type=float, default=None,
                    help="M-RoPE frame rate (default 6.0, matching training)")
    return ap


def make_collator(K, max_frames, special_token_base=constants.SPECIAL_TOKEN_BASE):
    return MultimodalDataCollator(
        max_seq_len=1024, max_waveforms=160000, max_mel_spec_frames=625,
        max_sive_feature_frames=max_frames, voice_eov_id=K,
        special_token_base=special_token_base,
    )


@torch.no_grad()
def tf_unit_stats(model, logits, tgt, K):
    """Content-only top-1 accuracy + CE/ppl. tgt: (B,T) with -100 pad, K = EOV id."""
    B, T, V = logits.shape
    tgt = tgt[:, :T]
    if tgt.shape[1] < T:
        tgt = F.pad(tgt, (0, T - tgt.shape[1]), value=-100)
    pred = logits.argmax(-1)
    content = (tgt >= 0) & (tgt < K)           # exclude EOV(=K) and pad(-100)
    incl_eov = tgt != -100
    acc_c = (pred[content] == tgt[content]).float().mean().item() if content.any() else float("nan")
    acc_e = (pred[incl_eov] == tgt[incl_eov]).float().mean().item() if incl_eov.any() else float("nan")
    ce = F.cross_entropy(logits.reshape(B * T, V), tgt.reshape(B * T), ignore_index=-100)
    return acc_c, acc_e, ce.item(), math.exp(ce.item())


@torch.no_grad()
def run_tf_and_ablation(model, dataset, collator, device, n, K, bs=16, early_k=8,
                        voice_attn_alpha=None, nar_mask_ratio=None, nar_seed=0):
    """Teacher-forced forward with real vs batch-shuffled text; aggregate content accuracy.

    early_k: also track accuracy on the FIRST early_k voice frames separately. There the
    shifted-TF input carries little/no voice history, so text is the dominant (at frame 0,
    the ONLY) predictive signal -- the cleanest place to see whether text is used, since it
    isn't confounded by voice-history redundancy the way the all-position delta is.

    nar_mask_ratio: REQUIRED for a masked-parallel (NAR) checkpoint, meaningless otherwise.
    Without it this probe hands the model the FULL ground-truth voice as input, which for a
    NAR model is mask-ratio 0 -- a condition it can satisfy by copying its input. Accuracy
    then approaches 1.0 and text_delta collapses toward 0, which reads exactly like "text
    conditioning died" when it only means the probe asked a degenerate question. With a ratio
    set, that fraction of frames is replaced by the model's learned MASK feature and EVERY
    metric below is computed on masked positions only -- the training condition, and the one
    inference actually visits.

    The ratio is the NAR analogue of AR's early/late split: r=1.0 (no voice context, text is
    the only signal) corresponds to AR's early frames, r->0 (context available) to its late
    ones. So text_delta at r=1.0 is the clean text signal here, and early_text_delta -- which
    assumes a left-to-right history that a masked model does not have -- is not meaningful."""
    # Guard here rather than in one caller, so head_ablation_probe / voice_text_horizon_sweep
    # and anything else built on this function inherit it automatically.
    if getattr(model, "voice_mask_feature", None) is not None and nar_mask_ratio is None:
        raise SystemExit(
            "NAR (masked-parallel) checkpoint probed without nar_mask_ratio.\n"
            "  At mask-ratio 0 the model is handed the full ground-truth voice and can satisfy\n"
            "  the task by COPYING: accuracy -> ~1.0, text_delta -> ~0, which reads as collapsed\n"
            "  conditioning but is a degenerate question. Pass --nar_mask_ratio (1.0 = text alone,\n"
            "  0.5 = mid-refinement).")
    idxs = list(range(min(n, len(dataset))))
    tot = {"real_hits": 0, "shuf_hits": 0, "content": 0, "ce_real": 0.0, "ce_n": 0,
           "eov_hits": 0, "eov_tot": 0,
           "early_real": 0, "early_shuf": 0, "early_content": 0,
           # Top-k membership (real text): is the TRUE unit in the model's top-k? Far less
           # entropy-sensitive than top-1 -- disentangles "conditioning is weak" from "the target
           # is one-to-many so top-1 is capped but the right unit is right there in the top few".
           "real_top5": 0, "real_top10": 0, "early_top5": 0, "early_top10": 0}
    per_utt_all, per_utt_early = [], []
    # POSITION-RESOLVED decay. The ear reports "onset is very good, falls off quickly" — a
    # correct horizon of a few words that then collapses. early_text_delta only sees frames
    # 0-8 and everything past that is averaged into one number, so neither can show the decay
    # curve or whether the horizon GROWS with training. Buckets are frame ranges at 25Hz
    # (~6 frames/word), so bucket edges are roughly 1.3, 2.6, 5, 10, 21, 42 words in.
    BUCKETS = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 10**9)]
    bstats = {b: {"real": 0, "shuf": 0, "n": 0} for b in BUCKETS}
    for s in range(0, len(idxs), bs):
        samples = [dataset[i] for i in idxs[s:s + bs]]
        samples = [x for x in samples if any(k.startswith("voice_") for k in x)]
        if len(samples) < 2:
            continue
        collator.force_direction = "synthesis"
        b = collator(samples)
        text = b["text_token_ids"].to(device)
        vin = b["voice_features"].unsqueeze(1).to(device)
        nar_masked = None
        if nar_mask_ratio is not None:
            _mf = getattr(model, "voice_mask_feature", None)
            if _mf is None:
                raise RuntimeError(
                    "--nar_mask_ratio given but the model has no voice_mask_feature; this is "
                    "not a NAR checkpoint (or it was loaded without voice_nar).")
            _g = torch.Generator(device="cpu").manual_seed(nar_seed + s)
            _bb, _nn, _cc, _tt = vin.shape
            # Same mask for the real-text and shuffled-text forwards, so the ablation is
            # paired: any difference is the text, not a different corruption.
            nar_masked = (torch.rand(_bb, _nn, 1, _tt, generator=_g) < float(nar_mask_ratio)
                          ).to(device)
            vin = torch.where(nar_masked, _mf.view(1, 1, _cc, 1).to(vin.dtype), vin)
            nar_masked = nar_masked.squeeze(2).reshape(_bb * _nn, _tt)
        vlen = b["voice_feature_lengths"].unsqueeze(1).to(device)
        vlbl = b["voice_features"].to(device)
        tgt = b["voice_unit_ids"].to(device)
        syn = b["is_synthesis"].to(device)

        def fwd(text_ids):
            out = model(text_input_ids=text_ids, voice_inputs=vin, voice_lengths=vlen,
                        voice_latent_labels=vlbl, is_synthesis=syn, decode_outputs=False,
                        voice_attn_alpha=voice_attn_alpha)
            return out["voice_unit_logits"]

        lr = fwd(text)                                   # real text
        ls = fwd(torch.roll(text, 1, dims=0))            # each voice + WRONG (rolled) text
        Bc, Tc, V = lr.shape
        t = tgt[:, :Tc]
        if t.shape[1] < Tc:
            t = F.pad(t, (0, Tc - t.shape[1]), value=-100)
        content = (t >= 0) & (t < K)
        if nar_masked is not None:
            # Score ONLY masked positions: a revealed frame is an input the model was handed,
            # and counting it measures copying rather than prediction.
            _m = nar_masked[:, :Tc]
            if _m.shape[0] == content.shape[0]:
                content = content & _m.to(content.device)
        pr, ps = lr.argmax(-1), ls.argmax(-1)
        tot["real_hits"] += (pr[content] == t[content]).sum().item()
        tot["shuf_hits"] += (ps[content] == t[content]).sum().item()
        tot["content"] += int(content.sum().item())
        # Top-k membership on real text (compute top-10 once, derive top-5 from it).
        top10 = lr.topk(10, dim=-1).indices                  # (Bc, Tc, 10)
        tmatch = (top10 == t.unsqueeze(-1))
        hit10 = tmatch.any(-1)
        hit5 = tmatch[..., :5].any(-1)
        tot["real_top5"] += int(hit5[content].sum().item())
        tot["real_top10"] += int(hit10[content].sum().item())
        # Early-frame (text-dominant) region: first early_k voice positions.
        # per-utterance tallies for the bootstrap CI
        hit_r = (pr == t) & content
        hit_s = (ps == t) & content
        early = content.clone()
        early[:, early_k:] = False
        for _b in range(Bc):
            _n = int(content[_b].sum().item())
            if _n:
                per_utt_all.append((int(hit_r[_b].sum().item()), int(hit_s[_b].sum().item()), _n))
            _e = int(early[_b].sum().item())
            if _e:
                per_utt_early.append((int((hit_r[_b] & early[_b]).sum().item()),
                                      int((hit_s[_b] & early[_b]).sum().item()), _e))
        if early.any():
            tot["early_real"] += (pr[early] == t[early]).sum().item()
            tot["early_shuf"] += (ps[early] == t[early]).sum().item()
            tot["early_content"] += int(early.sum().item())
            tot["early_top5"] += int(hit5[early].sum().item())
            tot["early_top10"] += int(hit10[early].sum().item())
        pos_ix = torch.arange(Tc, device=t.device).unsqueeze(0).expand(Bc, Tc)
        for lo, hi in BUCKETS:
            m = content & (pos_ix >= lo) & (pos_ix < hi)
            if m.any():
                bstats[(lo, hi)]["real"] += int((pr[m] == t[m]).sum().item())
                bstats[(lo, hi)]["shuf"] += int((ps[m] == t[m]).sum().item())
                bstats[(lo, hi)]["n"] += int(m.sum().item())
        ce = F.cross_entropy(lr.reshape(Bc * Tc, V), t.reshape(Bc * Tc), ignore_index=-100)
        tot["ce_real"] += ce.item() * Bc
        tot["ce_n"] += Bc
        eov = t == K
        if eov.any():
            tot["eov_hits"] += (pr[eov] == t[eov]).sum().item()
            tot["eov_tot"] += int(eov.sum().item())
    acc_real = tot["real_hits"] / max(tot["content"], 1)
    acc_shuf = tot["shuf_hits"] / max(tot["content"], 1)
    ce_real = tot["ce_real"] / max(tot["ce_n"], 1)
    e_real = tot["early_real"] / max(tot["early_content"], 1)
    e_shuf = tot["early_shuf"] / max(tot["early_content"], 1)
    cc = max(tot["content"], 1)
    ec = max(tot["early_content"], 1)
    all_lo, all_hi = bootstrap_delta_ci(per_utt_all)
    early_lo, early_hi = bootstrap_delta_ci(per_utt_early)
    return {
        "n_utt": len(per_utt_all),
        "text_delta_ci": (all_lo, all_hi), "early_text_delta_ci": (early_lo, early_hi),
        "acc_real": acc_real, "acc_shuf": acc_shuf, "text_delta": acc_real - acc_shuf,
        "ce_real": ce_real, "ppl_real": math.exp(ce_real),
        "eov_acc": tot["eov_hits"] / max(tot["eov_tot"], 1), "content_positions": tot["content"],
        "early_acc_real": e_real, "early_acc_shuf": e_shuf, "early_text_delta": e_real - e_shuf,
        "buckets": [{"lo": lo, "hi": hi, "n": v["n"],
                     "acc_real": v["real"] / max(v["n"], 1),
                     "acc_shuf": v["shuf"] / max(v["n"], 1),
                     "delta": (v["real"] - v["shuf"]) / max(v["n"], 1)}
                    for (lo, hi), v in bstats.items() if v["n"] > 0],
        "top5_real": tot["real_top5"] / cc, "top10_real": tot["real_top10"] / cc,
        "early_top5_real": tot["early_top5"] / ec, "early_top10_real": tot["early_top10"] / ec,
    }


def seq_degeneration(seqs, K):
    """Aggregate degeneration stats over a list of unit-id sequences (content units, no EOV)."""
    if not seqs:
        return {}
    lens = [len(s) for s in seqs]
    rep = run = tot_pairs = 0
    uni, bi, tri = Counter(), set(), set()
    n_bi = n_tri = 0
    for s in seqs:
        for u in s:
            uni[u] += 1
        for t in range(1, len(s)):
            tot_pairs += 1
            if s[t] == s[t - 1]:
                rep += 1
        # longest run of a repeated unit
        cur = 1
        for t in range(1, len(s)):
            cur = cur + 1 if s[t] == s[t - 1] else 1
            run = max(run, cur)
        for t in range(1, len(s)):
            bi.add((s[t - 1], s[t])); n_bi += 1
        for t in range(2, len(s)):
            tri.add((s[t - 2], s[t - 1], s[t])); n_tri += 1
    # POSITION-RESOLVED REPETITION. adj_repeat pooled over a whole utterance cannot tell
    # "repetitive from the first frame" (a conditioning/decoder failure) from "drifts into a
    # loop" (self-conditioning, an exposure-bias failure). Those want different fixes, so
    # bucket the adjacent-repeat rate by position. GT is measured the same way, so the
    # comparison is like-for-like rather than against a single pooled number.
    RB = [(1, 32), (32, 64), (64, 128), (128, 10 ** 9)]
    pos_rep = {b: [0, 0] for b in RB}
    for s in seqs:
        for t in range(1, len(s)):
            for b in RB:
                if b[0] <= t < b[1]:
                    pos_rep[b][1] += 1
                    if s[t] == s[t - 1]:
                        pos_rep[b][0] += 1
                    break
    total = sum(uni.values())
    probs = [c / total for c in uni.values()]
    ent = -sum(p * math.log2(p) for p in probs)
    return {
        "adj_repeat_by_pos": {f"{a}-{'+' if b > 10**8 else b}": (r / n if n else float('nan'))
                              for (a, b), (r, n) in pos_rep.items()},
        "adj_repeat_by_pos_n": {f"{a}-{'+' if b > 10**8 else b}": n
                                for (a, b), (r, n) in pos_rep.items()},
        "n_seqs": len(seqs),
        "len_mean": sum(lens) / len(lens), "len_min": min(lens), "len_max": max(lens),
        "adj_repeat_rate": rep / max(tot_pairs, 1),
        "longest_run": run,
        "distinct_units": len(uni), "coverage": len(uni) / K,
        "unit_entropy_bits": ent, "max_entropy_bits": math.log2(K),
        "distinct_bigram_ratio": len(bi) / max(n_bi, 1),
        "distinct_trigram_ratio": len(tri) / max(n_tri, 1),
    }


@torch.no_grad()
def run_generation(model, dataset, collator, device, gen_n, K, budget,
                   bov_id=constants.BOV_TOKEN_ID, ras_win=0, ras_tau=0.1,
                   voice_temp=0.6, top_k=None, top_p=None, nar_rounds=16,
                   nar_choice_temp=1.0):
    """Free-running generation from text prompts; return generated + GT unit sequences + EOV info.

    Also collects per-sample prompt TEXT length (tokens before BOV) so the caller can correlate
    it with generated length -- the clean 'does text drive duration' signal (short prompt ->
    short utterance) independent of whether the CONTENT aligns."""
    gen_seqs, gt_seqs, prompt_lens, ent_traces = [], [], [], []
    eov_fired = budget_hit = 0
    n_utts = []          # disjoint utterances per prompt; >1 is a termination failure
    collator.force_direction = "synthesis"
    count = 0
    for i in range(len(dataset)):
        s = dataset[i]
        if not any(k.startswith("voice_") for k in s):
            continue
        b = collator([s])
        text = b["text_token_ids"][0]
        bov = (text == bov_id).nonzero(as_tuple=True)[0]
        if len(bov) == 0:
            continue
        prompt_lens.append(int(bov[0].item()))   # text tokens before BOV = transcript length proxy
        prompt = text[:bov[0].item() + 1].unsqueeze(0).to(device)
        if getattr(model, "voice_mask_feature", None) is not None:
            # NAR: masked-parallel decode. The AR loop would step a bidirectional head one
            # frame at a time, which measures a procedure the model is never used under.
            _ids, _ = model.generate_voice_nar_from_prompt(
                prompt, n_rounds=nar_rounds, temperature=voice_temp,
                fallback_frames=budget, choice_temperature=nar_choice_temp)
            out = {"voice_unit_id_trace": [_ids]}
        else:
            out = model.generate(text_input_ids=prompt, max_new_tokens=512,
                                 voice_token_budget=budget, voice_temperature=voice_temp,
                                 voice_top_k=top_k, voice_top_p=top_p,
                                 voice_ras_win=ras_win, voice_ras_tau=ras_tau,
                                 decode_outputs=False)
        _ent = out.get("voice_unit_id_entropy_trace") or out.get("voice_unit_entropy_trace")
        if _ent and _ent[0]:
            ent_traces.append([float(x) for x in _ent[0]])
        elif getattr(model, "last_nar_round_entropy", None):
            ent_traces.append([float(x) for x in model.last_nar_round_entropy])
        # FIRST UTTERANCE, not the flat trace. `voice_unit_id_trace` spans EVERY voice block
        # the call produced, and a model that ends one utterance and starts another therefore
        # reports a length above the per-block budget (measured 2026-08-24: len_mean 373.9 and
        # len_max 500 against a 250 budget, which is arithmetically impossible for one block).
        # That inflates every length statistic, makes `len(trace) >= budget` fire for any
        # multi-block generation, and reduces "EOV fired" to "the LAST block ended with EOV".
        # A render is one utterance, so the statistics must be one utterance too.
        segs = out.get("voice_unit_id_segments")
        if segs and segs[0]:
            n_utts.append(len([sg for sg in segs[0] if len(sg) > 0]))
            trace = list(segs[0][0])
        else:
            n_utts.append(1)
            trace = out.get("voice_unit_id_trace", [[]])[0]
        if trace and trace[-1] == K:            # EOV fired -> strip it
            eov_fired += 1
            trace = trace[:-1]
        elif len(trace) >= budget:
            budget_hit += 1
        gen_seqs.append([int(x) for x in trace])
        # GT content units for the same utterance
        gt = b["voice_unit_ids"][0]
        gt = gt[(gt >= 0) & (gt < K)].tolist()
        gt_seqs.append(gt)
        count += 1
        if count >= gen_n:
            break
    return gen_seqs, gt_seqs, eov_fired, budget_hit, prompt_lens, ent_traces, n_utts


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx > 0 and vy > 0 else float("nan")


def fmt(d, keys=None):
    keys = keys or d.keys()
    return "\n".join(f"| {k} | {d[k]:.4f} |" if isinstance(d[k], float) else f"| {k} | {d[k]} |"
                     for k in keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--cache_dir", required=True, help="voice base dir (has /val)")
    ap.add_argument("--codebook", required=True)
    ap.add_argument("--config", default="small_sum")
    ap.add_argument("--voice_max_frames", type=int, default=209,
                    help="Voice frame budget = the run's voice_max_frames (collator cap AND the "
                         "generation budget). Mimi cb0 @12.5Hz = 209; CosyVoice 2 @25Hz = 250 "
                         "(=voice_max_seconds*sr//voice_hop_length). Must match the training run.")
    ap.add_argument("--voice_predict_f0", dest="voice_predict_f0", action="store_true", default=True,
                    help="Checkpoint has the F0 prediction head (Mimi runs). Default on.")
    ap.add_argument("--no_voice_predict_f0", dest="voice_predict_f0", action="store_false",
                    help="Checkpoint has NO F0 head (CosyVoice 2 runs — prosody lives in the token "
                         "+ frozen decoder, so the run trains without --voice_predict_f0).")
    ap.add_argument("--text_encoder_model", default=None,
                    help="Pretrained-LLM text encoder id (e.g. HuggingFaceTB/SmolLM2-135M) if the "
                         "checkpoint was trained with one; must match, else the load state-mismatches.")
    ap.add_argument("--n", type=int, default=1024, help="TF/ablation val utterances (n=256 gives a ~+-0.012 CI on early_text_delta — too wide; do not go lower)")
    ap.add_argument("--bs", type=int, default=16, help="TF/ablation batch size (lower to avoid OOM "
                    "when sharing a GPU with a training run)")
    ap.add_argument("--gen_n", type=int, default=128,
                    help="free-running generations. 128, not 32/48: the measured noise floor at "
                         "gen_n=48 (n=4 identical ck60000 runs, 2026-09-06) is duration r sd "
                         "0.0271 and length-hit-rate sd 3.39 points, which cannot resolve the "
                         "differences these comparisons turn on. Noise falls as 1/sqrt(n), so "
                         "128 buys ~1.63x precision (r sd ~0.017, hit rate ~2.1 pts) at ~2.7x "
                         "the wall clock, since batch-1 AR generation dominates the runtime. "
                         "A different gen_n estimates the SAME quantity with different "
                         "precision, so old numbers stay valid — but state the gen_n when "
                         "mixing them in one comparison, since the error bars differ.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ngram_ceiling", type=float, default=0.211)
    ap.add_argument("--asymptote", type=float, default=0.229)
    ap.add_argument("--repeat", type=float, default=0.117)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--voice_attn_alpha", type=float, default=None,
                    help="Voice->voice attention scale to run the TF/ablation forward at, for a "
                         "1:1 test with the training regime. Pass the curriculum alpha AT THIS "
                         "STEP (0.0 during the NAR phase, the ramp value during 10k-30k, 1.0 "
                         "post-ramp). Default None = 1.0 (full attention, the inference regime) — "
                         "but during NAR that is OOD (the model never trained with history) and "
                         "acc collapses, so it is NOT the model's real conditioning. Generation "
                         "(section 3) always runs at alpha=1 (generate() has no alpha hook).")
    ap.add_argument("--nar_choice_temperature", type=float, default=1.0,
                    help="Gumbel noise on MaskGIT confidence (annealed). 0 = greedy "
                         "reveal, which self-reinforces the repetition mode.")
    ap.add_argument("--nar_rounds", type=int, default=16,
                    help="MaskGIT refinement rounds for the free-running section on a NAR "
                         "checkpoint (ignored for AR).")
    ap.add_argument("--nar_mask_ratio", type=float, default=None,
                    help="REQUIRED for a masked-parallel (NAR) checkpoint. Fraction of voice "
                         "frames replaced by the learned MASK feature; all TF/ablation metrics "
                         "are then computed on masked positions ONLY. Without it the probe hands "
                         "the model the full ground-truth voice, which a NAR model can satisfy by "
                         "copying -- accuracy goes to ~1.0 and text_delta to ~0, which looks like "
                         "collapsed conditioning but is a degenerate question. 1.0 = generate from "
                         "text alone (the inference starting condition, and the NAR analogue of "
                         "AR's early frames); 0.5 = mid-refinement.")
    ap.add_argument("--voice_temperature", type=float, default=0.6,
                    help="voice unit sampling temperature. Default 0.6 MATCHES THE TRAINING-TIME VIZ (train.py: viz_voice_temperature=0.6), i.e. the TensorBoard renders the ear has been judging. These scripts previously HARDCODED 1.0, which samples far into the 6561-way tail and is audibly less coherent than the model's actual operating point -- so every free-running number they produced described the wrong regime.")
    ap.add_argument("--voice_top_k", type=int, default=None, help="top-k truncation for voice unit sampling (0/None = off). Only active when --voice_temperature > 0.")
    ap.add_argument("--voice_top_p", type=float, default=None, help="top-p / nucleus truncation for voice unit sampling (0/None = off). Only active when --voice_temperature > 0. The natural middle ground: T=0.6 mode-collapses into repetition loops, T=1.0 draws tail noise -- nucleus cuts the tail without sharpening into a loop.")
    ap.add_argument("--voice_ras_win", type=int, default=0,
                    help="Repetition-aware sampling window (CosyVoice 2 uses 10). If the sampled "
                         "unit occurred >= win*tau times in the last `win` emitted units, ban it and "
                         "resample. 0 = off. EOV is exempt from the ban.")
    ap.add_argument("--voice_ras_tau", type=float, default=0.1,
                    help="RAS repetition threshold (CosyVoice 2 uses 0.1 => any repeat within the window)")
    ap.add_argument("--skip_generation", action="store_true",
                    help="Skip the free-running generation section (always alpha=1, slow). Use for "
                         "an alpha-sweep where only the TF/ablation numbers vary with alpha.")
    add_mrope_args(ap)
    a = ap.parse_args()

    device = a.device
    codebook = load_codebook(a.codebook)
    K, D = int(codebook.shape[0]), int(codebook.shape[1])
    args = build_args(a, D)

    print(f"Loading {a.checkpoint_path} ...", flush=True)
    model = load_world_model(args, device)
    model.set_voice_codebook(codebook)
    model.to(device).eval()

    # Control-token base from the loaded model (32000 default / native LLM vocab in pretrained
    # mode). The collator must inject the placeholder/BOV ids at the SAME base the model detects.
    sp_base = getattr(model.config, "special_token_base", constants.SPECIAL_TOKEN_BASE)
    sp = constants.special_token_ids(sp_base)
    eval_dataset = load_dataset(args, "val")
    collator = make_collator(K, a.voice_max_frames, special_token_base=sp_base)
    print(f"val: {len(eval_dataset)} | K={K} | feature_channels={D} | "
          f"voice_max_frames={a.voice_max_frames} | predict_f0={a.voice_predict_f0} | "
          f"special_token_base={sp_base}", flush=True)

    alpha_label = ("1.0 (inference regime, full voice attention)" if a.voice_attn_alpha is None
                   else f"{a.voice_attn_alpha} (1:1 with training at this step)")
    print(f"1/3 teacher-forced + text ablation (voice_attn_alpha={alpha_label}) ...", flush=True)
    # Guard, not a default: probing a NAR checkpoint at mask-ratio 0 asks it to copy its own
    # input and yields ~1.0 accuracy with ~0 text_delta -- a confident, wrong "conditioning
    # collapsed" verdict. Same discipline as --mrope_scale_side: fail loudly instead.
    _is_nar = getattr(model, "voice_mask_feature", None) is not None
    if _is_nar and a.nar_mask_ratio is None:
        raise SystemExit(
            "This is a NAR (masked-parallel) checkpoint, so --nar_mask_ratio is required.\n"
            "  Without it the probe hands the model the full ground-truth voice, which it can\n"
            "  satisfy by copying -- the numbers look catastrophic for reasons that have nothing\n"
            "  to do with the model. Use 1.0 (text alone; the inference starting condition) and\n"
            "  0.5 (mid-refinement); early_text_delta is NOT meaningful for a masked model.")
    if a.nar_mask_ratio is not None and not _is_nar:
        raise SystemExit("--nar_mask_ratio given but this checkpoint has no MASK feature "
                         "(not a NAR model, or loaded without voice_nar).")
    tf = run_tf_and_ablation(model, eval_dataset, collator, device, a.n, K, bs=a.bs,
                             voice_attn_alpha=a.voice_attn_alpha,
                             nar_mask_ratio=a.nar_mask_ratio)
    if not a.skip_generation:
        print("2/3 free-running generation ...", flush=True)
        gen, gt, eov_fired, budget_hit, prompt_lens, ent_traces, n_utts = run_generation(
            model, eval_dataset, collator, device, a.gen_n, K,
            budget=a.voice_max_frames, bov_id=sp.BOV,
            ras_win=a.voice_ras_win, ras_tau=a.voice_ras_tau,
            voice_temp=a.voice_temperature, top_k=a.voice_top_k, top_p=a.voice_top_p,
            nar_rounds=a.nar_rounds, nar_choice_temp=a.nar_choice_temperature)
        if ent_traces:
            import statistics as _st
            _all = [e for tr in ent_traces for e in tr]
            _q = [[] for _ in range(4)]
            for tr in ent_traces:
                for i, e in enumerate(tr):
                    _q[min(3, int(4 * i / max(1, len(tr))))].append(e)
            _qs = [(_st.mean(x) if x else float("nan")) for x in _q]
            print(f"  per-step predictive entropy (raw, bits): mean {_st.mean(_all):.3f} | "
                  f"by quartile of the utterance {_qs[0]:.2f} {_qs[1]:.2f} {_qs[2]:.2f} "
                  f"{_qs[3]:.2f}", flush=True)
            tf["freerun_entropy_mean"] = _st.mean(_all)
            tf["freerun_entropy_quartiles"] = _qs
        print("3/3 degeneration stats ...", flush=True)
        gen_deg = seq_degeneration(gen, K)
        gt_deg = seq_degeneration(gt, K)
    else:
        print("(skipping generation — --skip_generation)", flush=True)

    out_dir = a.out_dir or f"eval_output/world_ar_diag/step_{a.step}"
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    lines.append(f"# World voice-AR diagnostics — step {a.step}\n")
    lines.append(f"checkpoint: `{a.checkpoint_path}`  |  val n(TF)={a.n}  gen_n={a.gen_n}  K={K}\n")
    lines.append(f"TF/ablation voice_attn_alpha = **{alpha_label}**  |  generation always alpha=1.\n")
    if a.nar_mask_ratio is not None:
        lines.append(f"**NAR probe: mask_ratio={a.nar_mask_ratio}** -- every TF/ablation number "
                     f"below is on MASKED positions only. early_* rows are not meaningful for a "
                     f"masked model (no left-to-right history); read text_delta at ratio 1.0 as "
                     f"the clean text signal instead.\n")
    lines.append(f"generation sampling: **voice_temperature={a.voice_temperature}**"
                 f"{' (matches the training-time viz)' if a.voice_temperature == 0.6 else ''}"
                 f"  |  RAS win={a.voice_ras_win}. Section 3 ONLY -- sections 1/1b/2 are "
                 f"teacher-forced and temperature-independent.\n")

    lines.append("## 1. Teacher-forced unit prediction (held-out) vs n-gram ceiling\n")
    lines.append("| metric | value |\n|---|---|")
    lines.append(f"| acc_real (content top-1) | {tf['acc_real']:.4f} |")
    lines.append(f"| top-5 (content, real) | {tf['top5_real']:.4f} |")
    lines.append(f"| top-10 (content, real) | {tf['top10_real']:.4f} |")
    lines.append(f"| early top-1 (real) | {tf['early_acc_real']:.4f} |")
    lines.append(f"| early top-5 (real) | {tf['early_top5_real']:.4f} |")
    lines.append(f"| early top-10 (real) | {tf['early_top10_real']:.4f} |")
    lines.append(f"| ppl_real | {tf['ppl_real']:.2f} |")
    lines.append(f"| ce_real (nats) | {tf['ce_real']:.4f} |")
    lines.append(f"| eov_position_acc | {tf['eov_acc']:.4f} |")
    lines.append(f"| — n-gram ceiling (ref) | {a.ngram_ceiling:.4f} |")
    lines.append(f"| — n-gram asymptote (ref) | {a.asymptote:.4f} |")
    lines.append(f"| — repeat crutch (ref) | {a.repeat:.4f} |")
    # Top-k vs top-1: a large top-1->top-k lift on real text means the TRUE unit is in the
    # model's top few -- the top-1 ceiling is target entropy (one-to-many), not a conditioning wall.
    lines.append(f"\n**top-1 {tf['acc_real']:.3f} -> top-5 {tf['top5_real']:.3f} -> top-10 "
                 f"{tf['top10_real']:.3f}** (all-position). A big lift => the top-1 ceiling is "
                 f"target multimodality, not a conditioning/capability wall.\n")
    verdict = ("ABOVE asymptote — uses more than local statistics" if tf['acc_real'] > a.asymptote
               else "ABOVE ceiling — likely using text/long-range" if tf['acc_real'] > a.ngram_ceiling
               else "between crutch and ceiling — local-statistics regime" if tf['acc_real'] > a.repeat
               else "at/below repeat crutch — not yet learned")
    lines.append(f"\n**acc_real vs baselines: {verdict}.**\n")

    lines.append("## 1b. Position-resolved decay (does the correct horizon grow?)\n")
    lines.append("| frames | ~words in | n | acc_real | acc_shuf | text_delta |\n|---|---|---|---|---|---|")
    for b in tf.get("buckets", []):
        hi = "+" if b["hi"] > 10**8 else str(b["hi"])
        lines.append(f"| {b['lo']}-{hi} | {b['lo']/6:.0f}-{'' if hi=='+' else f'{int(hi)/6:.0f}'} | "
                     f"{b['n']} | {b['acc_real']:.4f} | {b['acc_shuf']:.4f} | {b['delta']:+.4f} |")
    _h = text_horizon(tf.get("buckets", []))
    _hw = "never drops" if _h == float("inf") else f"{_h:.1f} frames (~{_h/6:.1f} words)"
    lines.append(f"\n**TEXT HORIZON (delta < 0.01): {_hw}** — the single trackable number. "
                 f"A rising early_text_delta with a FLAT horizon means the onset is getting "
                 f"taller, not the model getting better at speech.\n")
    lines.append("\nThe ear reports a correct ONSET that falls off. If the model is improving, the "
                 "delta should hold further into the utterance over training — a GROWING horizon — "
                 "rather than the onset bucket simply getting taller.\n")

    lines.append("## 2. Text ablation (real vs shuffled transcript)\n")
    lines.append("| metric | value |\n|---|---|")
    lines.append(f"| acc_real (all positions) | {tf['acc_real']:.4f} |")
    lines.append(f"| acc_shuffled_text (all) | {tf['acc_shuf']:.4f} |")
    lines.append(f"| text_delta (all positions) | {tf['text_delta']:+.4f} "
                 f"[{tf['text_delta_ci'][0]:+.4f}, {tf['text_delta_ci'][1]:+.4f}] |")
    lines.append(f"| early_acc_real (first frames) | {tf['early_acc_real']:.4f} |")
    lines.append(f"| early_acc_shuffled | {tf['early_acc_shuf']:.4f} |")
    lines.append(f"| **early_text_delta (clean text signal)** | **{tf['early_text_delta']:+.4f}** "
                 f"95% CI [{tf['early_text_delta_ci'][0]:+.4f}, {tf['early_text_delta_ci'][1]:+.4f}] |")
    # CosyVoice2 teacher on this same cache (n=1024): the CEILING for this metric.
    lines.append(f"| — TEACHER early_text_delta (ceiling) | +0.0538 [+0.0480, +0.0597] |")
    lines.append(f"| — TEACHER text_delta all-pos (ceiling) | +0.0594 [+0.0576, +0.0612] |")
    lines.append(f"| — TEACHER acc_real (ceiling) | 0.1283 |")
    _tr = tf['text_delta'] / max(tf['acc_real'], 1e-9)
    lines.append(f"| **text-attributed fraction** (delta/acc_real) | **{_tr:.3f}** vs teacher 0.463 |")
    lines.append("\nAll-position delta is confounded: teacher-forced voice history is redundant "
                 "with the text, so it understates text's role. The **early_text_delta** (first "
                 f"frames, little/no voice history) is the clean signal.\n")
    etd = tf['early_text_delta']
    tv = ("text IS driving the early frames — conditioning works" if etd > 0.03
          else "text near-ignored even where it's the ONLY signal — conditioning not engaged "
               "(explains prompt-unrelated output)" if etd < 0.01
          else "weak/partial text signal at utterance onset")
    lines.append(f"**early_text_delta read: {tv}.**\n")

    if a.skip_generation:
        lines.append("## 3. Free-running generation — SKIPPED (--skip_generation)\n")
        report = "\n".join(lines)
        path = os.path.join(out_dir, "report.md")
        with open(path, "w") as f:
            f.write(report)
        print("\n" + report)
        return

    lines.append("## 3. Free-running generation — degeneration vs ground truth\n")
    lines.append(f"EOV fired: {eov_fired}/{gen_deg.get('n_seqs',0)}  |  budget-capped: {budget_hit}\n")
    if n_utts:
        _multi = sum(1 for x in n_utts if x > 1)
        lines.append(
            f"**Disjoint utterances per prompt: mean {sum(n_utts)/len(n_utts):.2f}, "
            f"max {max(n_utts)}; {_multi}/{len(n_utts)} prompts produced MORE THAN ONE.** "
            f"Every statistic below is the FIRST utterance only, which is what a render is. "
            f"A count above 1 means the model ended an utterance and began another unprompted "
            f"— a termination failure that is invisible if the blocks are concatenated.\n")
    # Text-length -> generated-length correlation: does the model read the text to decide HOW
    # LONG to speak? A high r means text drives DURATION (structural conditioning) even if
    # content isn't aligned. GT r is the ceiling (how well real speech length tracks text length).
    gen_lens = [len(s) for s in gen]
    gt_lens = [len(s) for s in gt]
    r_gen = pearson(prompt_lens, gen_lens)
    r_gt = pearson(prompt_lens, gt_lens)
    lines.append(f"**text-length → generated-length correlation: r={r_gen:+.3f}** "
                 f"(GT ceiling r={r_gt:+.3f}). High r = text drives DURATION (structural "
                 f"conditioning), independent of content alignment.\n")
    # PER-UTTERANCE length agreement. len_mean is a BAD summary when the length distribution
    # is BIMODAL: near-zero collapses and overruns cancel, so the mean can land exactly on GT
    # while most utterances are wrong in opposite directions. Measured 2026-09-02 at ck50000
    # under RAS: len_mean 179.06 vs GT 179.67 (looks perfect) while 3/8 rendered utterances
    # collapsed to 4-22 frames. These four numbers are what that mean was hiding.
    _pairs = [(g, t) for g, t in zip(gen_lens, gt_lens) if t > 0]
    if _pairs:
        _rel = [g / t for g, t in _pairs]
        _hit = sum(1 for r in _rel if 0.7 <= r <= 1.3) / len(_rel)
        _trunc = sum(1 for r in _rel if r < 0.5) / len(_rel)
        _over = sum(1 for r in _rel if r > 1.5) / len(_rel)
        _med = sorted(abs(r - 1.0) for r in _rel)[len(_rel) // 2]
        lines.append(
            f"**Per-utterance length agreement** (n={len(_rel)}): "
            f"**hit rate {_hit:.1%}** within +-30% of GT  |  collapsed (<50% of GT) "
            f"{_trunc:.1%}  |  overrun (>150%) {_over:.1%}  |  median |len/GT - 1| = {_med:.2f}. "
            f"Read this INSTEAD of len_mean — a bimodal split of collapses and overruns averages "
            f"to a mean that looks correct.\n")

    lines.append("| metric | generated | ground-truth |\n|---|---|---|")
    for k in ["len_mean", "len_min", "len_max", "adj_repeat_rate", "longest_run",
              "distinct_units", "coverage", "unit_entropy_bits", "distinct_bigram_ratio",
              "distinct_trigram_ratio"]:
        g = gen_deg.get(k, float('nan')); r = gt_deg.get(k, float('nan'))
        gs = f"{g:.4f}" if isinstance(g, float) else str(g)
        rs = f"{r:.4f}" if isinstance(r, float) else str(r)
        lines.append(f"| {k} | {gs} | {rs} |")
    lines.append(f"\nmax unit entropy = {gen_deg.get('max_entropy_bits', 0):.2f} bits (log2 K).")
    _gp = gen_deg.get("adj_repeat_by_pos") or {}
    _tp = gt_deg.get("adj_repeat_by_pos") or {}
    if _gp:
        lines.append("\n**Adjacent-repeat by POSITION** — flat = repetitive from the start "
                     "(conditioning/decoder); rising = drifts into a loop (self-conditioning, "
                     "exposure bias). These want different fixes.\n")
        lines.append("| frames | n pairs | generated | ground-truth |\n|---|---|---|---|")
        for k in _gp:
            _n = (gen_deg.get("adj_repeat_by_pos_n") or {}).get(k, 0)
            lines.append(f"| {k} | {_n} | {_gp[k]:.4f} | {_tp.get(k, float('nan')):.4f} |")
        lines.append("")
    lines.append("Degeneration flags: adj_repeat_rate >> GT, longest_run large, low coverage,")
    lines.append("entropy << GT, or distinct-bigram ratio << GT all indicate collapse/looping.\n")

    report = "\n".join(lines)
    path = os.path.join(out_dir, "report.md")
    with open(path, "w") as f:
        f.write(report)
    print("\n" + report)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
