"""ASR-WER on world-model speech decoded by the frozen CosyVoice 2 decoder.

WHY THIS EXISTS: our frame-level metrics (acc_real, early_text_delta) compare position t to
position t, so they are ALIGNMENT-SENSITIVE — emitting the right phonemes one or two frames
early/late scores as wrong on every subsequent frame while sounding BETTER. M-RoPE changed
the coordinate system governing timing, so exactly that failure mode is in play: at 20.3k
its acc_real (0.0653) is below the distill run's at 44k (0.0940) while the audio is audibly
more intelligible. Whisper does not care about frame offsets, so WER adjudicates.

Reports THREE numbers per checkpoint:
  gen   — WER of the model's free-running speech
  ceil  — WER of GROUND-TRUTH units through the SAME frozen decoder (the pipeline ceiling;
          anything the decoder+ASR lose is not the world model's fault)
  gap   — gen - ceil, the world model's own contribution to unintelligibility

Also reports hyp/ref length ratio: past runs were DELETION-dominated (hyp 44-67% of ref),
so a WER near 1.0 with a low ratio means "said almost nothing", not "said wrong things".

  uv run python scripts_local/cosyvoice_wer_eval.py --checkpoint_path <ckpt> --step N \
      --cache_dir <cache> --codebook <cb> --cosyvoice_dir <cv2> --n 32 --device cuda:3
"""
import argparse, glob, json, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_voice_ar_diagnostics import build_args, make_collator, add_mrope_args
from megatransformer.scripts.eval.world.visualize import load_world_model, load_dataset
from megatransformer.scripts.eval.world.tts_intelligibility import normalize_text
from megatransformer.utils.codebook import load_codebook
from megatransformer.utils import constants
from megatransformer.model.voice.cosyvoice2_decoder import CosyVoice2Decoder

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint_path", required=True)
ap.add_argument("--step", type=int, required=True)
ap.add_argument("--cache_dir", required=True)
ap.add_argument("--codebook", required=True)
ap.add_argument("--cosyvoice_dir", required=True)
ap.add_argument("--config", default="small_sum")
ap.add_argument("--text_encoder_model", default="HuggingFaceTB/SmolLM2-135M")
ap.add_argument("--voice_max_frames", type=int, default=250)
ap.add_argument("--voice_predict_f0", dest="voice_predict_f0", action="store_true", default=False)
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--ras_win", type=int, default=0, help="0 = off (match the training-time renders)")
ap.add_argument("--ras_tau", type=float, default=0.1)
ap.add_argument("--whisper_model", default="base")
ap.add_argument("--skip_ceiling", action="store_true", help="skip the GT-units ceiling pass")
ap.add_argument("--device", default="cuda:3")
ap.add_argument("--out_dir", default="eval_output/world_tts_wer")
ap.add_argument("--voice_temperature", type=float, default=0.6,
                help="voice unit sampling temperature. Default 0.6 MATCHES THE TRAINING-TIME VIZ (train.py: viz_voice_temperature=0.6), i.e. the TensorBoard renders the ear has been judging. These scripts previously HARDCODED 1.0, which samples far into the 6561-way tail and is audibly less coherent than the model's actual operating point -- so every free-running number they produced described the wrong regime.")
ap.add_argument("--nar_reveal", type=str, default="confidence",
                choices=["confidence", "sequential", "random"],
                help="Which positions to commit each round. 'confidence' is MaskGIT's and "
                     "assumes confidence tracks correctness; here it tracks repetition.")
ap.add_argument("--seed", "--nar_seed", dest="nar_seed", type=int, default=None,
                help="Seed the RNG before generation. NOT NAR-specific despite the legacy "
                     "--nar_seed spelling: it seeds the global torch RNG after the decoder "
                     "loads, so it governs AR sampling too. Decoding is stochastic, so two runs of the "
                     "SAME config differ -- and the paired bootstrap over utterances does NOT "
                     "capture that, because it treats each run's generations as fixed. Run one "
                     "config at several seeds to get the run-to-run floor before believing any "
                     "difference between configs.")
ap.add_argument("--duration_shuffle", action="store_true", default=False,
                help="NAR causal probe: overwrite each utterance's predicted duration bucket "
                     "with ANOTHER utterance's. If generated length follows the substituted "
                     "bucket, the token controls length; if it does not, length is being driven "
                     "by something else and the correlation is incidental.")
ap.add_argument("--nar_choice_temperature", type=float, default=1.0,
                help="Gumbel noise on MaskGIT confidence, annealed to 0 over the "
                     "rounds. 0 = pure greedy reveal, which self-reinforces the "
                     "repetition mode (measured 2.7x worse at 16 rounds than at 1).")
ap.add_argument("--nar_rounds", type=int, default=16,
                help="MaskGIT refinement rounds for a NAR checkpoint (ignored for AR).")
ap.add_argument("--voice_top_k", type=int, default=None, help="top-k truncation for voice unit sampling (0/None = off). Only active when --voice_temperature > 0.")
ap.add_argument("--voice_top_p", type=float, default=None, help="top-p / nucleus truncation for voice unit sampling (0/None = off). Only active when --voice_temperature > 0. The natural middle ground: T=0.6 mode-collapses into repetition loops, T=1.0 draws tail noise -- nucleus cuts the tail without sharpening into a loop.")
add_mrope_args(ap)
a = ap.parse_args()
voice_temp = a.voice_temperature

cb = load_codebook(a.codebook); K, D = int(cb.shape[0]), int(cb.shape[1])
args = build_args(a, D)
model = load_world_model(args, a.device); model.set_voice_codebook(cb)
model.to(a.device).eval()
sp_base = getattr(model.config, "special_token_base", constants.SPECIAL_TOKEN_BASE)
sp = constants.special_token_ids(sp_base)
_IS_NAR = getattr(model, "voice_mask_feature", None) is not None
print(f"decode path: {'NAR masked-parallel' if _IS_NAR else 'AR'}", flush=True)
ds = load_dataset(args, "val")
coll = make_collator(K, a.voice_max_frames, special_token_base=sp_base); coll.force_direction = "synthesis"
dec = CosyVoice2Decoder.from_pretrained(a.cosyvoice_dir, device=a.device)
sr = dec.sample_rate
# Seed AFTER the decoder is constructed, not before: loading it perturbs (and apparently
# fixes) the global RNG, so an earlier torch.manual_seed was being overwritten -- three
# different --nar_seed values produced byte-identical output, which looked like "decoding is
# deterministic" when it actually meant "the seed never took".
if a.nar_seed is not None:
    torch.manual_seed(a.nar_seed)
    print(f"seeded generation with {a.nar_seed} (after decoder load)", flush=True)
os.makedirs(a.out_dir, exist_ok=True)

import whisper, librosa, numpy as np
from jiwer import wer as jiwer_wer, cer as jiwer_cer
asr = whisper.load_model(a.whisper_model, device=a.device)
print(f"decoder sr={sr}, whisper={a.whisper_model}, ras_win={a.ras_win}", flush=True)


def transcribe(wav_24k):
    # Whisper is a 16 kHz model; the CosyVoice 2 decoder emits 24 kHz. Resampling is not
    # optional here -- feeding 24 kHz straight in mis-times everything.
    x = wav_24k.numpy().astype(np.float32)
    x16 = librosa.resample(x, orig_sr=sr, target_sr=16000)
    # fp16 on CUDA: Whisper defaults to fp32 and warns otherwise. Transcription is a
    # small share of the wall clock here (AR generation at batch 1 dominates), so this
    # is a minor win -- taken because it is free and changes no output materially.
    r = asr.transcribe(x16, language="en", fp16=str(a.device).startswith("cuda"))
    return r.get("text", "").strip()


rows, done = [], 0
for i in range(len(ds)):
    s = ds[i]
    if not any(k.startswith("voice_") for k in s):
        continue
    b = coll([s])
    text = b["text_token_ids"][0]
    bov = (text == sp.BOV).nonzero(as_tuple=True)[0]
    ref = str(s.get("voice_voice_text", "")).strip()
    spk = s.get("voice_speaker_embedding")
    if len(bov) == 0 or not ref or spk is None:
        continue
    prompt = text[:bov[0].item() + 1].unsqueeze(0).to(a.device)
    with torch.no_grad():
        if _IS_NAR:
            # A NAR checkpoint decodes by masked refinement; running the AR loop on it would
            # step a bidirectional head one frame at a time and measure nothing real.
            _fb = None
            if a.duration_shuffle:
                from megatransformer.utils import constants as _C
                # Deterministic derangement-ish: use the bucket of the utterance 7 ahead.
                _other = (int(rows[-7]["ref_frames"]) if len(rows) >= 7
                          else int(s["voice_feature_length"]))
                _fb = _C.duration_bucket(_other)
                row_forced = _fb
            _ids, _ = model.generate_voice_nar_from_prompt(
                prompt, n_rounds=a.nar_rounds, temperature=voice_temp, sp=sp,
                choice_temperature=a.nar_choice_temperature, reveal=a.nar_reveal,
                fallback_frames=a.voice_max_frames, force_bucket=_fb)
            out = {"voice_unit_id_trace": [_ids]}
        else:
            out = model.generate(text_input_ids=prompt, max_new_tokens=512,
                                 voice_token_budget=a.voice_max_frames, voice_temperature=voice_temp,
                                 voice_top_k=a.voice_top_k, voice_top_p=a.voice_top_p,
                                 voice_ras_win=a.ras_win, voice_ras_tau=a.ras_tau,
                                 decode_outputs=False)
    tr = [int(x) for x in out.get("voice_unit_id_trace", [[]])[0] if 0 <= int(x) < K]
    L_ref = int(s["voice_feature_length"])
    row = {"idx": i, "ref": ref, "gen_frames": len(tr), "ref_frames": L_ref}
    if a.duration_shuffle and _IS_NAR:
        from megatransformer.utils import constants as _C2
        row["forced_bucket"] = row_forced
        row["forced_frames"] = _C2.duration_bucket_frames(row_forced)
        row["true_bucket"] = _C2.duration_bucket(L_ref)
    if tr:
        w = dec.decode(torch.tensor(tr), spk)
        if w is not None:
            row["hyp"] = transcribe(w)
        # TRUNCATED-TO-REFERENCE pass. Both arms over-run (2x+ GT length) and Whisper decodes
        # WITH CONTEXT, so a long garbage tail can corrupt the transcription of a correct
        # opening -- observed: a clip heard by ear as "What do you make of it, grey" came back
        # from Whisper as "What do you want to do to J.J. Blades? I'm not going to...".
        # Scoring the first L_ref frames separates CONTENT quality from the TERMINATION
        # failure, which the untruncated number conflates.
        if len(tr) > L_ref > 0:
            wt = dec.decode(torch.tensor(tr[:L_ref]), spk)
            if wt is not None:
                row["hyp_trunc"] = transcribe(wt)
        else:
            row["hyp_trunc"] = row.get("hyp")
    if not a.skip_ceiling:
        L = int(s["voice_feature_length"])
        w = dec.decode(s["voice_unit_ids"][:L], spk)
        if w is not None:
            row["hyp_ceiling"] = transcribe(w)
    rows.append(row); done += 1
    print(f"  [{done}] ref: {ref[:52]}\n        gen: {row.get('hyp','')[:52]}", flush=True)
    if done >= a.n:
        break


def lcs_recall(ref_words, hyp_words):
    """Fraction of REFERENCE words appearing in order in the hypothesis (LCS / len(ref)).

    WER is contaminated here from BOTH directions: the model runs 2.11x GT length, so a
    correct opening followed by rambling scores as massive INSERTIONS, while other utterances
    generate almost nothing and score as DELETIONS. Both land near WER 1.0 for opposite
    reasons. LCS recall ignores insertions entirely, so it answers the narrower question the
    ear is actually asking: did the reference words get said, in order?
    """
    n, m = len(ref_words), len(hyp_words)
    if n == 0:
        return float("nan")
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        for j in range(1, m + 1):
            cur[j] = prev[j - 1] + 1 if ref_words[i - 1] == hyp_words[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[m] / n


def score(key):
    pairs = [(normalize_text(r["ref"]), normalize_text(r.get(key, ""))) for r in rows if key in r]
    pairs = [(x, y) for x, y in pairs if x]
    if not pairs:
        return None
    refs, hyps = [p[0] for p in pairs], [p[1] for p in pairs]
    ratio = sum(len(h.split()) for h in hyps) / max(sum(len(r.split()) for r in refs), 1)
    rec = [lcs_recall(r.split(), h.split()) for r, h in pairs]
    return {"n": len(pairs), "wer": jiwer_wer(refs, hyps), "cer": jiwer_cer(refs, hyps),
            "hyp_ref_word_ratio": ratio,
            "lcs_recall": sum(rec) / len(rec)}


g, t, c = score("hyp"), score("hyp_trunc"), score("hyp_ceiling")
print(f"\n=== step {a.step}  (n={len(rows)}, ras_win={a.ras_win}) ===")
if g: print(f"  GENERATED   WER {g['wer']:.4f}  CER {g['cer']:.4f}  "
            f"LCS-recall {g['lcs_recall']:.4f}  hyp/ref words {g['hyp_ref_word_ratio']:.2f}")
if t: print(f"  TRUNC->REF  WER {t['wer']:.4f}  CER {t['cer']:.4f}  "
            f"LCS-recall {t['lcs_recall']:.4f}  hyp/ref words {t['hyp_ref_word_ratio']:.2f}")
if c: print(f"  CEILING(GT) WER {c['wer']:.4f}  CER {c['cer']:.4f}  "
            f"LCS-recall {c['lcs_recall']:.4f}  hyp/ref words {c['hyp_ref_word_ratio']:.2f}")
if g and c:
    print(f"  GAP (model's own cost)  WER {g['wer']-c['wer']:+.4f}  "
          f"LCS-recall {g['lcs_recall']-c['lcs_recall']:+.4f}")
    print("  ^ LCS-recall is the length-robust one: WER here is contaminated by BOTH "
          "insertions (2.1x over-length) and deletions (near-empty generations).")
print("\nA low hyp/ref ratio with high WER = DELETION-dominated (said too little), not wrong words.")
json.dump({"step": a.step, "ras_win": a.ras_win, "generated": g, "truncated": t,
           "ceiling": c, "rows": rows},
          open(os.path.join(a.out_dir, f"wer_step{a.step}_ras{a.ras_win}.json"), "w"), indent=2)
print(f"wrote {a.out_dir}/wer_step{a.step}_ras{a.ras_win}.json")
