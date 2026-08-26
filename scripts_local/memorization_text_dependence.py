"""Did the memorization run actually READ the transcript, or key on the sequence layout?

`memorize32_0` reaches train unit-accuracy 1.0 by ~step 1400 on 32 utterances at
mask-ratio 1.0 (no voice context at all -- every unit predicted from text + duration alone).
That looks like proof the text is being read. It is not, because of a confound:

    over those 32 samples, (voice_frame_len, text_len) is a UNIQUE index.
    30 of 32 exact frame lengths are distinct; the one collision (3 samples at 173 frames)
    is broken by text length. All 32 are the SAME speaker, so speaker is not the key either.

So the model can reproduce all 32 exactly while ignoring every text TOKEN, keyed on two
integers it gets for free from the sequence layout. A shuffled-transcript TRAINING control
cannot detect this: shuffling pairs each voice with a fixed wrong transcript, which is still
a unique key, so that arm memorizes too and the wrong conclusion ("text is decorative") comes
out looking confirmed.

This probe asks the question on the ALREADY-TRAINED checkpoint instead, by editing the
transcript at the SAMPLE level (before collation, so every control token, placeholder
position and duration bucket stays exactly where training put it) and re-measuring accuracy:

  real      unchanged -- the memorized condition
  roll      sample i gets sample i-1's transcript. Content AND text_len change: this is the
            standard ablation, and the only one comparable to published text_delta numbers.
  matched   sample i gets sample i-1's transcript, cycled/truncated to sample i's OWN text
            length. Content changes, the (frame_len, text_len) key is PRESERVED. This is the
            decisive arm.
  within    sample i's own tokens, randomly permuted. Length, token multiset and every
            marginal statistic preserved; only ORDER is destroyed.
  constant  every sample gets sample 0's transcript, cycled to its own length. Removes all
            cross-sample text information while preserving the layout key.

Reading it:
  matched accuracy stays ~1.0   => the transcript is decorative; the run memorized the
                                   layout. The bistream prerequisite as written cannot be
                                   answered by a memorization run on this sample set.
  matched accuracy collapses    => text content is genuinely read.
  within >> matched             => the model uses the token multiset (a bag), not the order.

Usage:
  python scripts_local/memorization_text_dependence.py \
      --checkpoint_path runs/world_voice/memorize32_0/checkpoint-20000 --step 20000 \
      --cache_dir ./cached_datasets/Mekadrom/libritts_r_cosyvoice2_smollm2 \
      --codebook .../cosyvoice2_codebook.pt --config small_sum \
      --text_encoder_model HuggingFaceTB/SmolLM2-135M --voice_max_frames 250 \
      --duration_token --mrope_scale_side text --device cuda:0 \
      --out_dir eval_output/world_voice_memorization/text_dependence
"""
import argparse
import os
import random

import torch
import torch.nn.functional as F

from megatransformer.scripts.eval.world.visualize import load_world_model
from megatransformer.scripts.data.world.data_collator import MultimodalDataCollator
from megatransformer.scripts.data.world.memorization_dataset import MultimodalMemorizationDataset
from megatransformer.utils.codebook import load_codebook
from megatransformer.utils import constants
from megatransformer.scripts.eval.world.visualize import resolve_shard_dir

try:                      # run as `python scripts_local/memorization_text_dependence.py`
    import world_voice_ar_diagnostics as diag          # build_args / add_mrope_args
except ImportError:       # run as a module from the repo root
    import scripts_local.world_voice_ar_diagnostics as diag


def _cycle_to(tokens: torch.Tensor, n: int) -> torch.Tensor:
    """Repeat/truncate `tokens` to exactly n entries. Cycling (not padding) keeps the stream
    made of real word tokens, so a drop cannot be blamed on an out-of-distribution pad run."""
    if tokens.numel() == 0:
        return tokens.new_zeros(n)
    reps = (n + tokens.numel() - 1) // tokens.numel()
    return tokens.repeat(reps)[:n]


def make_variant(samples, kind, seed=0):
    """Return a NEW list of sample dicts with text_token_ids replaced per `kind`."""
    rng = random.Random(seed)
    out = []
    n = len(samples)
    for i, s in enumerate(samples):
        s = dict(s)
        own = s["text_token_ids"]
        own_len = int(s["text_text_length"])
        if kind == "real":
            pass
        elif kind == "roll":
            src = samples[(i - 1) % n]
            s["text_token_ids"] = src["text_token_ids"]
            s["text_text_length"] = src["text_text_length"]
        elif kind == "matched":
            src = samples[(i - 1) % n]
            src_len = int(src["text_text_length"])
            s["text_token_ids"] = _cycle_to(src["text_token_ids"][:src_len], own_len)
            s["text_text_length"] = own_len
        elif kind == "within":
            toks = own[:own_len].tolist()
            rng.shuffle(toks)
            s["text_token_ids"] = torch.tensor(toks, dtype=own.dtype)
            s["text_text_length"] = own_len
        elif kind == "constant":
            src = samples[0]
            src_len = int(src["text_text_length"])
            s["text_token_ids"] = _cycle_to(src["text_token_ids"][:src_len], own_len)
            s["text_text_length"] = own_len
        else:
            raise ValueError(kind)
        out.append(s)
    return out


@torch.no_grad()
def score(model, samples, collator, device, K, mask_ratio, seed=0):
    """Per-sample masked-position unit accuracy at the given NAR mask ratio."""
    collator.force_direction = "synthesis"
    b = collator(samples)
    text = b["text_token_ids"].to(device)
    vin = b["voice_features"].unsqueeze(1).to(device)
    mf = getattr(model, "voice_mask_feature", None)
    if mf is None:
        raise SystemExit("not a NAR checkpoint (no voice_mask_feature)")
    bb, nn, cc, tt = vin.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    masked = (torch.rand(bb, nn, 1, tt, generator=g) < float(mask_ratio)).to(device)
    vin = torch.where(masked, mf.view(1, 1, cc, 1).to(vin.dtype), vin)
    masked = masked.squeeze(2).reshape(bb * nn, tt)

    out = model(text_input_ids=text,
                voice_inputs=vin,
                voice_lengths=b["voice_feature_lengths"].unsqueeze(1).to(device),
                voice_latent_labels=b["voice_features"].to(device),
                is_synthesis=b["is_synthesis"].to(device),
                decode_outputs=False)
    logits = out["voice_unit_logits"]
    tgt = b["voice_unit_ids"].to(device)
    T = logits.shape[1]
    tgt = tgt[:, :T]
    if tgt.shape[1] < T:
        tgt = F.pad(tgt, (0, T - tgt.shape[1]), value=-100)
    m = masked[:, :T]
    pred = logits.argmax(-1)
    content = (tgt >= 0) & (tgt < K) & m
    per = []
    for i in range(tgt.shape[0]):
        c = content[i]
        per.append(((pred[i][c] == tgt[i][c]).float().mean().item() if c.any() else float("nan"),
                    int(c.sum())))
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                         torch.where(content, tgt, torch.full_like(tgt, -100)).reshape(-1),
                         ignore_index=-100).item()
    acc = (pred[content] == tgt[content]).float().mean().item()
    return acc, ce, per, pred.detach().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--codebook", required=True)
    ap.add_argument("--config", default="small_sum")
    ap.add_argument("--text_encoder_model", default=None)
    ap.add_argument("--voice_max_frames", type=int, default=250)
    ap.add_argument("--max_samples", type=int, default=32)
    ap.add_argument("--split", default="train",
                    help="MUST be the split the memorization run trained on (train).")
    ap.add_argument("--duration_token", action="store_true",
                    help="Run trained with --voice_nar_duration_token: the collator must emit "
                         "the same [BOV][DUR_k][PH][EOV] block or every position shifts.")
    ap.add_argument("--nar_mask_ratio", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=8, help="must divide max_samples; the batch is "
                    "only a compute grouping here, every sample is scored independently")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--voice_predict_f0", dest="voice_predict_f0", action="store_true", default=False)
    diag.add_mrope_args(ap)
    a = ap.parse_args()

    codebook = load_codebook(a.codebook)
    K, D = int(codebook.shape[0]), int(codebook.shape[1])
    a.n = a.max_samples            # build_args reads a.n
    args = diag.build_args(a, D)

    print(f"Loading {a.checkpoint_path} ...", flush=True)
    model = load_world_model(args, a.device)
    model.set_voice_codebook(codebook)
    model.to(a.device).eval()

    sp_base = getattr(model.config, "special_token_base", constants.SPECIAL_TOKEN_BASE)
    sp = constants.special_token_ids(sp_base)
    voice_dir = resolve_shard_dir(a.cache_dir, a.split)
    # NOTE: visualize.load_dataset builds the memorization dataset WITHOUT the codebook, which
    # on the discrete-unit path yields samples with no `features` at all. Construct it here.
    ds = MultimodalMemorizationDataset(voice_shard_dir=voice_dir, max_samples=a.max_samples,
                                       voice_codebook=a.codebook)
    collator = MultimodalDataCollator(
        max_seq_len=1024, max_waveforms=160000, max_mel_spec_frames=625,
        max_sive_feature_frames=a.voice_max_frames, voice_eov_id=K,
        special_token_base=sp_base, emit_duration_token=a.duration_token)

    samples = [ds[i] for i in range(len(ds))]
    print(f"{len(samples)} samples | K={K} | D={D} | mask_ratio={a.nar_mask_ratio} | "
          f"duration_token={a.duration_token}", flush=True)

    KINDS = ["real", "roll", "matched", "within", "constant"]
    results = {}
    for kind in KINDS:
        var = make_variant(samples, kind, seed=a.seed)
        accs, ces, pers, preds = [], [], [], []
        for s in range(0, len(var), a.bs):
            chunk = var[s:s + a.bs]
            acc, ce, per, pr = score(model, chunk, collator, a.device, K,
                                     a.nar_mask_ratio, seed=a.seed + s)
            accs.append((acc, len(chunk))); ces.append(ce); pers.extend(per)
            preds.append(pr)
        tot_n = sum(n for _, n in accs)
        agg = sum(acc * n for acc, n in accs) / max(tot_n, 1)
        results[kind] = {"acc": agg, "ce": sum(ces) / len(ces), "per": pers, "preds": preds}
        print(f"  {kind:9s} acc={agg:.4f}  ce={results[kind]['ce']:.4f}", flush=True)

    # RETRIEVAL vs READING. `roll` hands sample i the whole of sample i-1's transcript. If the
    # model is a lookup table keyed on text, it should now emit sample i-1's UNITS. If it is
    # reading the text, the output should match neither i's stored units (it doesn't -- that is
    # the `roll` row) nor i-1's. Scored at overlapping positions only, against i-1's targets.
    retr_hits = retr_tot = 0
    off = 0
    for chunk_pred in results["roll"]["preds"]:
        for r in range(chunk_pred.shape[0]):
            i = off + r
            src = samples[(i - 1) % len(samples)]["voice_unit_ids"]
            n = min(int(samples[(i - 1) % len(samples)]["voice_feature_length"]),
                    int(samples[i]["voice_feature_length"]), chunk_pred.shape[1])
            if n <= 0:
                continue
            retr_hits += int((chunk_pred[r, :n] == src[:n]).sum())
            retr_tot += n
        off += chunk_pred.shape[0]
    retr = retr_hits / max(retr_tot, 1)
    print(f"  retrieval  roll-pred vs SOURCE units acc={retr:.4f}  (n={retr_tot})", flush=True)

    lines = [f"# Memorization text-dependence probe — step {a.step}", "",
             f"`{a.checkpoint_path}`, {len(samples)} train samples, NAR mask ratio "
             f"{a.nar_mask_ratio} (all positions predicted from text + duration alone).", "",
             "Text is edited at the SAMPLE level, so control tokens, placeholder positions and "
             "the duration bucket are identical across every arm.", "",
             "| arm | what changes | unit acc | CE |", "|---|---|---|---|"]
    WHAT = {"real": "nothing (memorized condition)",
            "roll": "content + text_len (standard ablation)",
            "matched": "content only — **layout key preserved**",
            "within": "token ORDER only (same multiset)",
            "constant": "all cross-sample text info removed"}
    for k in KINDS:
        lines.append(f"| {k} | {WHAT[k]} | {results[k]['acc']:.4f} | {results[k]['ce']:.4f} |")
    lines += ["", f"**matched delta vs real: {results['matched']['acc'] - results['real']['acc']:+.4f}**",
              "",
              f"**Retrieval check**: under `roll`, do the predictions match the SOURCE "
              f"utterance's units (i.e. is the model a lookup table keyed on text)? "
              f"**{retr:.4f}** over {retr_tot} overlapping positions. Near 1.0 = pure "
              f"retrieval; near chance = the text is being read, not looked up.",
              "", "Per-sample accuracy (real / matched):", "",
              "| idx | frames | text_len | real | matched | within |", "|---|---|---|---|---|---|"]
    for i, s in enumerate(samples):
        fl = int(s["voice_feature_length"]); tl = int(s["text_text_length"])
        lines.append(f"| {i} | {fl} | {tl} | {results['real']['per'][i][0]:.3f} | "
                     f"{results['matched']['per'][i][0]:.3f} | {results['within']['per'][i][0]:.3f} |")
    report = "\n".join(lines) + "\n"
    print()
    print(report)
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        p = os.path.join(a.out_dir, f"text_dependence_{a.step}.md")
        with open(p, "w") as f:
            f.write(report)
        print(f"-> {p}")


if __name__ == "__main__":
    main()
