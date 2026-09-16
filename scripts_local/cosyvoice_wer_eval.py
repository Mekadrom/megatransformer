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
import torchaudio

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
ap.add_argument("--ras_win", type=int, default=0,
                help="Repetition-aware sampling window. 0 = OFF. NOTE: the training-time "
                     "viz default flipped to 10 at commit 6af9337, so 0 no longer matches "
                     "the renders -- it reproduces the 1.63x-drawl arm. Pass 10 to match.")
ap.add_argument("--ras_tau", type=float, default=0.1)
ap.add_argument("--exit_criteria", default=None,
                choices=["kl_divergence","logit_kl","latent_diff","none"],
                help="Override the recurrent exit criterion AFTER loading. Default None "
                     "keeps the checkpoint config (kl_divergence on every run to date, "
                     "which is the numerically broken one). logit_kl is Huginn's real "
                     "criterion and needs a readout -- for voice that is wired in "
                     "generate() via _trunk_readout_voice.")
ap.add_argument("--trunk_iters", type=int, default=None,
                help="Hard CAP on recurrent iterations (sets recurrent_block.mean_thinking_steps, "
                     "which at eval is returned verbatim by n_k_steps as the cap -- the Poisson "
                     "sampling it feeds is training-only). The exit criterion may still stop "
                     "earlier. None = model default.")
ap.add_argument("--exit_criteria_threshold", type=float, default=None,
                help="Threshold for --exit_criteria. logit_kl: 5e-4 (paper) / 1e-3 (ref).")
ap.add_argument("--eov_min_prob", type=float, default=None,
                help="EOV confidence guard: ban EOV when p(EOV) is below this AND entropy is "
                     "above --eov_max_entropy. Needs BOTH to be set. Targets premature "
                     "collapse, which measures as a loss of coherence (flat distribution), "
                     "not a confident stop. Suggested 0.10 / 3.0.")
ap.add_argument("--eov_max_entropy", type=float, default=None,
                help="Entropy (nats) above which a low-probability EOV is rejected. Pairs "
                     "with --eov_min_prob; both must be set or the guard is inert.")
ap.add_argument("--ras_temperature", type=float, default=None,
                help="Temperature for the RAS RESAMPLE only, decoupled from the pick. Default "
                     "None = inherit the pick's scaling, which is DISCONTINUOUS at T=0 "
                     "(greedy resamples at an effective 1.0, T=0.2 resamples 5x sharper than "
                     "any pick). Set this to hold the resample fixed while sweeping "
                     "--voice_temperature.")
ap.add_argument("--whisper_model", default="base")
ap.add_argument("--skip_ceiling", action="store_true", help="skip the GT-units ceiling pass")
ap.add_argument("--device", default="cuda:3")
ap.add_argument("--out_dir", default="eval_output/world_tts_wer")
ap.add_argument("--save_audio", default=None,
                help="Directory to write decoded .wav files: gen_<i>.wav (full generation), "
                     "gen_trunc_<i>.wav (truncated to the reference length, only when the "
                     "generation over-ran), and ceiling_<i>.wav (GROUND-TRUTH units through "
                     "the SAME frozen decoder). Listen to gen vs ceiling: whatever the "
                     "ceiling also gets wrong is the decoder/pipeline, not the world model. "
                     "Writes manifest.tsv mapping index -> reference and transcripts.")
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
ap.add_argument("--bistream_text_chunk", type=int, default=0,
                help="k text tokens per chunk. >0 collates AND decodes bistream. A bistream "
                     "checkpoint decoded WITHOUT this stops at its first fill_token and "
                     "scores as catastrophic under-speaking -- it is the wrong protocol for "
                     "a bistream run, not a result.")
ap.add_argument("--bistream_voice_chunk", type=int, default=0,
                help="s voice frames per chunk (pairs with --bistream_text_chunk).")
ap.add_argument("--bistream_prob", type=float, default=1.0,
                help="Fraction of ELIGIBLE samples collated bistream. 1.0 here (not the "
                     "training 0.5) so the arm measures the bistream path, not a mixture.")
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
if a.exit_criteria is not None:
    from megatransformer.model import recurrent_criteria as _rc
    _thr = a.exit_criteria_threshold
    _blk = model.recurrent_block
    if a.exit_criteria == "none":
        _blk.exit_criteria = _rc.NoOpCriteria()
    elif a.exit_criteria == "logit_kl":
        _blk.exit_criteria = _rc.LogitKLCriteria(_thr if _thr is not None else 5e-4)
    elif a.exit_criteria == "latent_diff":
        _blk.exit_criteria = _rc.LatentDiffCriteria(_thr if _thr is not None else 0.03)
    else:
        _blk.exit_criteria = _rc.KLDivergenceCriteria(_thr if _thr is not None else 1e-4)
    print(f"exit criterion overridden -> {a.exit_criteria} "
          f"(threshold {getattr(_blk.exit_criteria, 'threshold', None)})", flush=True)

if a.trunk_iters is not None and a.trunk_iters > 0:
    model.recurrent_block.mean_thinking_steps = int(a.trunk_iters)
    print(f"trunk iteration cap -> {a.trunk_iters}", flush=True)

ds = load_dataset(args, "val")
coll = make_collator(K, a.voice_max_frames, special_token_base=sp_base,
                     bistream_text_chunk=a.bistream_text_chunk,
                     bistream_voice_chunk=a.bistream_voice_chunk,
                     bistream_prob=a.bistream_prob); coll.force_direction = "synthesis"
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
if a.save_audio:
    os.makedirs(a.save_audio, exist_ok=True)


def _save(wav, name):
    """Write one decoded waveform. No-op unless --save_audio was passed.

    dec.decode() returns a 1-D float tensor at the decoder's own 24 kHz; torchaudio wants
    (channels, samples), and the tensor must be on CPU and detached.
    """
    if not a.save_audio or wav is None:
        return
    x = wav.detach().to("cpu").float()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    torchaudio.save(os.path.join(a.save_audio, name), x, sr)

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
            # BISTREAM continuation -- see world_voice_ar_diagnostics for the full rationale.
            # The collated bistream prompt is [text 0:k][BOV], so bov's position IS how many
            # transcript tokens the prompt consumed; the rest is fed back as the model emits
            # fill_token. Without these a bistream checkpoint ends after one ~s-frame chunk.
            _bi_kwargs = {}
            if a.bistream_text_chunk > 0:
                _full = s.get("text_token_ids")
                _tl = s.get("text_text_length")
                if _full is not None:
                    if _tl is not None:
                        _full = _full[:int(_tl)]
                    _bi_kwargs = {
                        "voice_bistream_text": _full.reshape(1, -1).to(a.device),
                        "voice_bistream_text_chunk": a.bistream_text_chunk,
                        "voice_bistream_text_offset": int(bov[0].item()),
                    }
            out = model.generate(text_input_ids=prompt, max_new_tokens=512,
                                 voice_token_budget=a.voice_max_frames, voice_temperature=voice_temp,
                                 voice_top_k=a.voice_top_k, voice_top_p=a.voice_top_p,
                                 voice_ras_win=a.ras_win, voice_ras_tau=a.ras_tau,
                                 voice_ras_temperature=a.ras_temperature,
                                 voice_eov_min_prob=a.eov_min_prob,
                                 voice_eov_max_entropy=a.eov_max_entropy,
                                 decode_outputs=False, **_bi_kwargs)
    # FIRST UTTERANCE, not the flat trace. `voice_unit_id_trace` spans EVERY voice block the
    # call produced, so a model that ends one utterance and starts another (i.e. anything
    # WITHOUT --unmask_eos_in_synthesis) gets its blocks concatenated here -- inflating length
    # and handing Whisper a long tail that corrupts the transcription of the correct part.
    # A render is one utterance, so the measurement must be one utterance.
    #
    # ⚠️ This changes NOTHING for runs that emit a single utterance (measured: ar_flat_lr_1 is
    # 1.00 utterances/prompt at every checkpoint), so the 2026-08-25 trend numbers stand. It
    # DOES mean pre-EOS-fix runs were scored on concatenated output and are understated --
    # re-measure before comparing across that boundary.
    _segs = out.get("voice_unit_id_segments")
    _raw = (_segs[0][0] if _segs and _segs[0] and len(_segs[0][0]) > 0
            else out.get("voice_unit_id_trace", [[]])[0])
    tr = [int(x) for x in _raw if 0 <= int(x) < K]
    L_ref = int(s["voice_feature_length"])
    row = {"idx": i, "ref": ref, "gen_frames": len(tr), "ref_frames": L_ref}
    _its = out.get("recurrent_iteration_counts") or []
    if _its:
        row["mean_iters"] = sum(int(x) for x in _its) / len(_its)
    if a.duration_shuffle and _IS_NAR:
        from megatransformer.utils import constants as _C2
        row["forced_bucket"] = row_forced
        row["forced_frames"] = _C2.duration_bucket_frames(row_forced)
        row["true_bucket"] = _C2.duration_bucket(L_ref)
    if tr:
        w = dec.decode(torch.tensor(tr), spk)
        if w is not None:
            row["hyp"] = transcribe(w)
            _save(w, f"gen_{i:04d}.wav")
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
                _save(wt, f"gen_trunc_{i:04d}.wav")
        else:
            row["hyp_trunc"] = row.get("hyp")
    if not a.skip_ceiling:
        L = int(s["voice_feature_length"])
        # NOTE: `w` is deliberately a fresh binding -- the generated waveform above has already
        # been transcribed and saved, so shadowing it here is harmless.
        w = dec.decode(s["voice_unit_ids"][:L], spk)
        if w is not None:
            row["hyp_ceiling"] = transcribe(w)
            _save(w, f"ceiling_{i:04d}.wav")
    rows.append(row); done += 1
    print(f"  [{done}] ref: {ref[:52]}\n        gen: {row.get('hyp','')[:52]}", flush=True)
    if done >= a.n:
        break

if a.save_audio:
    # Without this the wav files are anonymous indices. Whisper's transcripts are what the
    # report scores, so pairing them with the file makes a disagreement between the ear and
    # the number traceable to a specific clip instead of a vague "some of them are wrong".
    with open(os.path.join(a.save_audio, "manifest.tsv"), "w") as _f:
        _f.write("idx\tgen_frames\tref_frames\treference\thyp_gen\thyp_ceiling\n")
        for _r in rows:
            _f.write("\t".join(str(_r.get(_k, "")).replace("\t", " ")
                                for _k in ("idx", "gen_frames", "ref_frames",
                                           "ref", "hyp", "hyp_ceiling")) + "\n")
    print(f"wrote {a.save_audio}/manifest.tsv", flush=True)


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


# Frames actually rendered for each scored variant. CADENCE needs this: hyp_ref_word_ratio
# counts WORDS and len_mean counts FRAMES, and neither can see speaking rate. A model that
# spends near-GT duration on FEWER words is slow per spoken word while looking correct — or
# even short — on both existing metrics. frames_per_hyp_word divides the two, and comparing
# it against the CEILING row (GT units through the same decoder) isolates cadence from
# content loss, since both then describe the same words-per-frame question.
_FRAMES_FOR = {
    "hyp":         lambda r: r["gen_frames"],
    "hyp_trunc":   lambda r: min(r["gen_frames"], r["ref_frames"]),
    "hyp_ceiling": lambda r: r["ref_frames"],
}


def score(key):
    sel = [r for r in rows if key in r and normalize_text(r["ref"])]
    pairs = [(normalize_text(r["ref"]), normalize_text(r.get(key, ""))) for r in sel]
    if not pairs:
        return None
    refs, hyps = [p[0] for p in pairs], [p[1] for p in pairs]
    ratio = sum(len(h.split()) for h in hyps) / max(sum(len(r.split()) for r in refs), 1)
    rec = [lcs_recall(r.split(), h.split()) for r, h in pairs]
    out = {"n": len(pairs), "wer": jiwer_wer(refs, hyps), "cer": jiwer_cer(refs, hyps),
           "hyp_ref_word_ratio": ratio,
           "lcs_recall": sum(rec) / len(rec)}
    fget = _FRAMES_FOR.get(key)
    if fget is not None:
        tot_f = sum(fget(r) for r in sel)
        tot_w = sum(len(h.split()) for h in hyps)
        if tot_w > 0:
            out["frames_per_hyp_word"] = tot_f / tot_w
            out["sec_per_hyp_word"] = tot_f / tot_w / 25.0   # CosyVoice 2 units are 25 Hz
    return out


g, t, c = score("hyp"), score("hyp_trunc"), score("hyp_ceiling")
print(f"\n=== step {a.step}  (n={len(rows)}, ras_win={a.ras_win}) ===")
if g: print(f"  GENERATED   WER {g['wer']:.4f}  CER {g['cer']:.4f}  "
            f"LCS-recall {g['lcs_recall']:.4f}  hyp/ref words {g['hyp_ref_word_ratio']:.2f}")
if t: print(f"  TRUNC->REF  WER {t['wer']:.4f}  CER {t['cer']:.4f}  "
            f"LCS-recall {t['lcs_recall']:.4f}  hyp/ref words {t['hyp_ref_word_ratio']:.2f}")
if c: print(f"  CEILING(GT) WER {c['wer']:.4f}  CER {c['cer']:.4f}  "
            f"LCS-recall {c['lcs_recall']:.4f}  hyp/ref words {c['hyp_ref_word_ratio']:.2f}")
if g and c and "frames_per_hyp_word" in g and "frames_per_hyp_word" in c:
    gf, cf = g["frames_per_hyp_word"], c["frames_per_hyp_word"]
    print(f"  CADENCE  generated {gf:.1f} frames/word ({g['sec_per_hyp_word']:.3f}s)  vs  "
          f"ceiling {cf:.1f} ({c['sec_per_hyp_word']:.3f}s)  =  {gf/cf:.2f}x")
    print("  ^ >1 means SLOW per spoken word. Independent of hyp/ref word count and of "
          "len_mean, either of which can look correct while cadence is wrong.")
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
