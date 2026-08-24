"""Why is a render N seconds long? Audit free-running voice generation, per utterance.

Written because a training-viz render was reported at ~20 s when `--voice_token_budget 250`
at 25 Hz caps ONE voice block at 10 s. Three different things produce a too-long render and
they need different fixes, so measure which:

  1. The model never emits EOV, and one block runs to the frame budget.
  2. The model emits EOV, then starts ANOTHER voice block (BOV sampled again), so the call
     returns several utterances. The viz renders utterance 0 -- but padded to the longest,
     which the frozen decoder turns into babble (fixed 2026-08-24; this audit still reports
     the padding so the fix can be confirmed).
  3. Something upstream of both: the budget not arriving, or a sample-rate mismatch in the
     render itself, in which case none of the numbers below will look wrong at all.

Reports, per prompt: how many utterances came back, each one's REAL frame count, the padded
width the viz used to see, whether EOV or the budget ended each block, and the seconds of
audio that implies at the token rate.

Usage:
  python scripts_local/voice_render_length_audit.py --checkpoint_path <ckpt> --step N \
      --cache_dir <voice base> --codebook <codebook.pt> [--device cuda:3] [--n 8] \
      [--mrope_scale_side text] [--voice_temperature 0.6] [--out_dir eval_output/...]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from megatransformer.scripts.eval.world.visualize import load_world_model, load_dataset
from megatransformer.scripts.data.world.data_collator import MultimodalDataCollator
from megatransformer.utils.codebook import load_codebook
from megatransformer.utils import constants
import world_voice_ar_diagnostics as diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--codebook", required=True)
    ap.add_argument("--config", default="small_sum")
    ap.add_argument("--text_encoder_model", default=None)
    ap.add_argument("--voice_max_frames", type=int, default=250,
                    help="MUST match the run's --voice_token_budget; it is the generation cap.")
    ap.add_argument("--token_rate", type=float, default=25.0, help="CosyVoice 2 = 25 Hz")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--voice_temperature", type=float, default=0.6,
                    help="Match the training viz (train.py viz_voice_temperature=0.6), or the "
                         "numbers describe a regime nobody listens at.")
    ap.add_argument("--voice_predict_f0", dest="voice_predict_f0", action="store_true", default=False)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--suppress_media_tokens", dest="suppress", action="store_true", default=True,
                    help="Ban BO*/placeholder ids from the text sampler (the eval default).")
    ap.add_argument("--no_suppress_media_tokens", dest="suppress", action="store_false",
                    help="Let the model sample BO*. Run BOTH to A/B whether the ban actually "
                         "prevents a second voice block -- the live eval produced up to SIX "
                         "utterances with the ban nominally on, which either means it is not "
                         "reaching generate() or that blocks start by some path other than a "
                         "sampled BOV.")
    ap.add_argument("--out_dir", default=None)
    diag.add_mrope_args(ap)
    a = ap.parse_args()

    cb = load_codebook(a.codebook)
    K, D = int(cb.shape[0]), int(cb.shape[1])
    a.n = a.n
    args = diag.build_args(a, D)
    model = load_world_model(args, a.device)
    model.set_voice_codebook(cb)
    model.to(a.device).eval()

    sp_base = getattr(model.config, "special_token_base", constants.SPECIAL_TOKEN_BASE)
    sp = constants.special_token_ids(sp_base)
    ds = load_dataset(args, "val")
    col = diag.make_collator(K, a.voice_max_frames, special_token_base=sp_base)
    col.force_direction = "synthesis"

    rows = []
    for i in range(min(a.n, len(ds))):
        s = ds[i]
        b = col([s])
        tids = b["text_token_ids"][0]
        bov = (tids == sp.BOV).nonzero(as_tuple=True)[0]
        if len(bov) == 0:
            continue
        prompt = tids[:int(bov[0]) + 1].unsqueeze(0).to(a.device)
        with torch.no_grad():
            out = model.generate(text_input_ids=prompt, max_new_tokens=a.max_new_tokens,
                                 temperature=0.8, voice_temperature=a.voice_temperature,
                                 voice_token_budget=a.voice_max_frames,
                                 suppress_media_control_tokens=a.suppress)
        vp = out.get("voice_latent_preds")
        if vp is None or vp.numel() == 0:
            rows.append((i, 0, [], 0, 0.0, 0.0))
            continue
        cnt = int(out["voice_counts"][0])
        lens = [int(x) for x in out["voice_lengths"][0][:cnt]]
        padded = int(vp.shape[-1])
        # What the viz used to render (utterance 0 at the PADDED width) vs its real length.
        rows.append((i, cnt, lens, padded, lens[0] / a.token_rate, padded / a.token_rate))

    tgt = a.voice_max_frames
    lines = [f"# Render length audit — step {a.step}", "",
             f"`{a.checkpoint_path}`, n={len(rows)}, budget {tgt} frames "
             f"({tgt / a.token_rate:.1f}s at {a.token_rate:g} Hz), "
             f"voice_temperature {a.voice_temperature}, "
             f"media-token suppression {'ON' if a.suppress else 'OFF'}.", "",
             "| prompt | utterances | real frames each | padded width | utt0 real s | utt0 padded s | ended by |",
             "|---|---|---|---|---|---|---|"]
    n_budget = n_eov = 0
    for (i, cnt, lens, padded, s_real, s_pad) in rows:
        if not lens:
            lines.append(f"| {i} | 0 | — | — | — | — | no voice emitted |")
            continue
        # A block that stopped exactly at the cap was ended by the BUDGET, not by EOV.
        ends = ["budget" if L >= tgt else "EOV" for L in lens]
        n_budget += sum(1 for e in ends if e == "budget")
        n_eov += sum(1 for e in ends if e == "EOV")
        lines.append(f"| {i} | {cnt} | {lens} | {padded} | {s_real:.1f} | {s_pad:.1f} | "
                     f"{', '.join(ends)} |")
    tot = n_budget + n_eov
    lines += ["", f"**Blocks ended by the frame budget: {n_budget}/{tot}** "
                  f"(EOV fired for {n_eov}).",
              "", "Reading it: several utterances per prompt means the model emitted EOV and "
                  "then started speaking again, so the call returns a padded stack; a padded "
                  "width well above utt0's real length is what the frozen decoder used to "
                  "render as babble. All blocks at exactly the budget means EOV never fires "
                  "and the length is a termination failure, not a padding artifact."]
    rep = "\n".join(lines) + "\n"
    print(rep)
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        p = os.path.join(a.out_dir, f"render_length_audit_{a.step}.md")
        open(p, "w").write(rep)
        print(f"-> {p}")


if __name__ == "__main__":
    main()
