"""Speaker-identity diagnostics for world-voice, decomposed into trunk vs decoder.

The ASR metrics are saturated (the model beats ground-truth audio through the same
decoder), so LCS/WER can no longer rank this direction. What the ear still reports is
speaker identity wandering: conditioned on one reference voice, renders vary.

This measures that, and attributes it. Every number is a campplus-192 cosine, the same
speaker space the decoder is conditioned on, so "similarity" means the thing the decoder
was actually told to match.

Four renders per utterance, all through the SAME frozen CosyVoice 2 decoder:

  ceiling_self   GT units   + target embedding   -> cos vs target
  model_self     GEN units  + target embedding   -> cos vs target
  ceiling_cross  GT units   + OTHER embedding    -> cos vs other, and vs target
  model_cross    GEN units  + OTHER embedding    -> cos vs other, and vs target

From these:
  IDENTITY GAP  = ceiling_self - model_self.  The decoder's own ceiling is subtracted, so
                  what remains is the TRUNK's contribution -- units that do not render
                  cleanly under the requested voice.
  CONTROLLABILITY = cos(render under X, X) - cos(render under X, other).  How much the
                  rendered identity follows the EMBEDDING rather than whatever the units
                  imply. Measured for GT units too, so the decoder's own controllability
                  is the reference rather than an assumption.
  WITHIN-SPEAKER SPREAD = std of model_self cosine across utterances of ONE speaker. This
                  is the "liberally chooses variations" complaint as a number.
"""
import argparse, glob, json, os, statistics, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from world_voice_ar_diagnostics import build_args, make_collator, add_mrope_args
from megatransformer.scripts.eval.world.visualize import load_world_model, load_dataset
from megatransformer.utils.codebook import load_codebook
from megatransformer.utils import constants
from megatransformer.model.voice.cosyvoice2_decoder import CosyVoice2Decoder
from megatransformer.utils.cosyvoice2_encoders import CampplusBatchProcessor

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint_path", required=True)
ap.add_argument("--step", type=int, required=True)
ap.add_argument("--cache_dir", required=True)
ap.add_argument("--codebook", required=True)
ap.add_argument("--cosyvoice_dir", required=True)
ap.add_argument("--config", default="small_sum")
ap.add_argument("--text_encoder_model", default="HuggingFaceTB/SmolLM2-135M")
ap.add_argument("--voice_max_frames", type=int, default=250)
ap.add_argument("--voice_predict_f0", action="store_true", default=False)
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--per_speaker", type=int, default=1,
                help="Utterances per speaker. >1 makes the WITHIN-SPEAKER spread measurable "
                     "-- sequential sampling of the val set yields almost no repeat speakers, "
                     "so the 'identity wanders across renders' complaint cannot be scored.")
ap.add_argument("--scan", type=int, default=1500,
                help="How many val samples to scan when grouping by speaker.")
ap.add_argument("--voice_temperature", type=float, default=0.0)
ap.add_argument("--ras_win", type=int, default=10)
ap.add_argument("--ras_tau", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--exit_criteria", default=None,
                choices=["kl_divergence", "logit_kl", "latent_diff", "none"])
ap.add_argument("--exit_criteria_threshold", type=float, default=None)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out_dir", default="eval_output/world_voice/reports/spk_identity")
ap.add_argument("--save_audio", default=None)
add_mrope_args(ap)
a = ap.parse_args()

cb = load_codebook(a.codebook); K, D = int(cb.shape[0]), int(cb.shape[1])
args = build_args(a, D)
model = load_world_model(args, a.device); model.set_voice_codebook(cb)
model.to(a.device).eval()
sp_base = getattr(model.config, "special_token_base", constants.SPECIAL_TOKEN_BASE)
sp = constants.special_token_ids(sp_base)

if a.exit_criteria is not None:
    from megatransformer.model import recurrent_criteria as _rc
    _t = a.exit_criteria_threshold
    _b = model.recurrent_block
    _b.exit_criteria = {
        "none": lambda: _rc.NoOpCriteria(),
        "logit_kl": lambda: _rc.LogitKLCriteria(_t if _t is not None else 5e-4),
        "latent_diff": lambda: _rc.LatentDiffCriteria(_t if _t is not None else 0.03),
        "kl_divergence": lambda: _rc.KLDivergenceCriteria(_t if _t is not None else 1e-4),
    }[a.exit_criteria]()
    print(f"exit criterion -> {a.exit_criteria} "
          f"({getattr(_b.exit_criteria,'threshold',None)})", flush=True)

ds = load_dataset(args, "val")
coll = make_collator(K, a.voice_max_frames, special_token_base=sp_base)
coll.force_direction = "synthesis"
dec = CosyVoice2Decoder.from_pretrained(a.cosyvoice_dir, device=a.device)
sr = dec.sample_rate
spk_enc = CampplusBatchProcessor(a.cosyvoice_dir, source_sr=sr)
torch.manual_seed(a.seed)
os.makedirs(a.out_dir, exist_ok=True)
if a.save_audio:
    os.makedirs(a.save_audio, exist_ok=True)


def embed(wav):
    """campplus-192 of a decoded waveform (24 kHz in, resampled internally)."""
    if wav is None or wav.numel() < sr // 4:
        return None
    out = spk_enc.process_batch([wav.reshape(-1).cpu()],
                                torch.tensor([wav.numel()], dtype=torch.long))
    e = (out["speaker_embeddings"] if isinstance(out, dict) else out)[0]
    return e.reshape(-1).float()


def cos(a_, b_):
    if a_ is None or b_ is None:
        return None
    return float(torch.nn.functional.cosine_similarity(a_.unsqueeze(0), b_.unsqueeze(0)))


# Collect candidates first so a cross-speaker partner is always available.
# With --per_speaker > 1, group by speaker and take the most-represented ones, so the
# within-speaker spread has real n behind it.
_pool = []
for i in range(min(len(ds), a.scan)):
    s = ds[i]
    if not any(k.startswith("voice_") for k in s):
        continue
    if s.get("voice_speaker_embedding") is None:
        continue
    _pool.append((i, s))
    if a.per_speaker <= 1 and len(_pool) >= a.n:
        break

if a.per_speaker > 1:
    by = collections.defaultdict(list)
    for i, s in _pool:
        by[int(s.get("voice_speaker_id", -1))].append((i, s))
    ranked = sorted(by.items(), key=lambda kv: -len(kv[1]))
    cands = []
    for sid, items in ranked:
        if len(items) < a.per_speaker:
            continue
        cands.extend(items[:a.per_speaker])
        if len(cands) >= a.n:
            break
    cands = cands[:a.n]
    print(f"grouped: {len(cands)} utterances over "
          f"{len({int(s.get('voice_speaker_id',-1)) for _, s in cands})} speakers "
          f"({a.per_speaker} each)", flush=True)
else:
    cands = _pool[:a.n]

rows = []
for n_done, (i, s) in enumerate(cands):
    spk_t = s["voice_speaker_embedding"].reshape(-1).float()
    # cross partner: the NEXT candidate with a different speaker id
    sid = int(s.get("voice_speaker_id", -1)) if s.get("voice_speaker_id") is not None else -1
    other = next((o for _, o in cands
                  if int(o.get("voice_speaker_id", -2)) != sid
                  and o.get("voice_speaker_embedding") is not None), None)
    spk_o = other["voice_speaker_embedding"].reshape(-1).float() if other is not None else None

    b = coll([s])
    text = b["text_token_ids"][0]
    bov = (text == sp.BOV).nonzero(as_tuple=True)[0]
    if len(bov) == 0:
        continue
    prompt = text[:bov[0].item() + 1].unsqueeze(0).to(a.device)
    with torch.no_grad():
        out = model.generate(text_input_ids=prompt, max_new_tokens=512,
                             voice_token_budget=a.voice_max_frames,
                             voice_temperature=a.voice_temperature,
                             voice_ras_win=a.ras_win, voice_ras_tau=a.ras_tau,
                             decode_outputs=False)
    segs = out.get("voice_unit_id_segments")
    raw = (segs[0][0] if segs and segs[0] and len(segs[0][0]) > 0
           else out.get("voice_unit_id_trace", [[]])[0])
    gen = [int(x) for x in raw if 0 <= int(x) < K]
    L = int(s["voice_feature_length"])
    gt = s["voice_unit_ids"][:L]
    if not gen:
        continue

    with torch.no_grad():
        w_c_self = dec.decode(gt, spk_t)
        w_m_self = dec.decode(torch.tensor(gen), spk_t)
        w_c_cross = dec.decode(gt, spk_o) if spk_o is not None else None
        w_m_cross = dec.decode(torch.tensor(gen), spk_o) if spk_o is not None else None

    e_c_self, e_m_self = embed(w_c_self), embed(w_m_self)
    e_c_cross, e_m_cross = embed(w_c_cross), embed(w_m_cross)
    row = {
        "idx": i, "speaker_id": sid, "gen_frames": len(gen), "ref_frames": L,
        "ceiling_self": cos(e_c_self, spk_t),
        "model_self": cos(e_m_self, spk_t),
        "ceiling_cross_to_other": cos(e_c_cross, spk_o),
        "ceiling_cross_to_target": cos(e_c_cross, spk_t),
        "model_cross_to_other": cos(e_m_cross, spk_o),
        "model_cross_to_target": cos(e_m_cross, spk_t),
        "model_vs_ceiling_render": cos(e_m_self, e_c_self),
    }
    rows.append(row)
    if a.save_audio:
        import torchaudio
        for tag, w in (("ceiling_self", w_c_self), ("model_self", w_m_self)):
            if w is not None:
                torchaudio.save(os.path.join(a.save_audio, f"{tag}_{i:04d}.wav"),
                                w.reshape(1, -1).cpu(), sr)
    print(f"  [{len(rows)}/{len(cands)}] spk {sid:>5}  ceiling {row['ceiling_self']:.3f}  "
          f"model {row['model_self']:.3f}", flush=True)

json.dump(rows, open(os.path.join(a.out_dir, f"spk_identity_step{a.step}.json"), "w"), indent=1)


def st(name, vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        print(f"  {name:38s} (none)"); return None
    m = statistics.mean(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    print(f"  {name:38s} n={len(vals):>3}  mean {m:.4f}  sd {sd:.4f}  "
          f"min {min(vals):.4f}  max {max(vals):.4f}")
    return m


print(f"\n=== SPEAKER IDENTITY — step {a.step}, n={len(rows)}, "
      f"T={a.voice_temperature}, criterion={a.exit_criteria or 'checkpoint default'} ===\n")
print("COSINE TO THE REQUESTED SPEAKER (campplus-192):")
c_self = st("ceiling  (GT units + target spk)", [r["ceiling_self"] for r in rows])
m_self = st("model    (GEN units + target spk)", [r["model_self"] for r in rows])
if c_self and m_self:
    print(f"\n  IDENTITY GAP (ceiling - model) = {c_self - m_self:+.4f}   <- the TRUNK's share")
    print(f"  the decoder's own ceiling is {c_self:.4f}; anything below that is not its fault")

print("\nCONTROLLABILITY — does the render follow the EMBEDDING or the UNITS?")
for pre in ("ceiling", "model"):
    to_o = [r[f"{pre}_cross_to_other"] for r in rows]
    to_t = [r[f"{pre}_cross_to_target"] for r in rows]
    pairs = [(o, t) for o, t in zip(to_o, to_t) if o is not None and t is not None]
    if pairs:
        sep = statistics.mean([o - t for o, t in pairs])
        print(f"  {pre:8s}: rendered under OTHER spk -> cos(other) {statistics.mean([o for o,_ in pairs]):.4f} "
              f"vs cos(target) {statistics.mean([t for _,t in pairs]):.4f}   separation {sep:+.4f}")
print("  (large positive separation = identity tracks the embedding; near zero = the UNITS"
      "\n   carry speaker information that fights the conditioning)")

print("\nWITHIN-SPEAKER CONSISTENCY (the 'wanders across renders' complaint):")
by_spk = collections.defaultdict(list)
for r in rows:
    if r["model_self"] is not None:
        by_spk[r["speaker_id"]].append(r["model_self"])
multi = {k: v for k, v in by_spk.items() if len(v) > 1}
if multi:
    sds = [statistics.pstdev(v) for v in multi.values()]
    print(f"  speakers with >1 utterance: {len(multi)}   mean within-speaker sd {statistics.mean(sds):.4f}")
    for k, v in sorted(multi.items(), key=lambda x: -len(x[1]))[:5]:
        print(f"    spk {k:>5}  n={len(v)}  mean {statistics.mean(v):.4f}  sd {statistics.pstdev(v):.4f}")
else:
    print("  (no speaker had >1 utterance in this sample; raise --n)")
st("model render vs ceiling render", [r["model_vs_ceiling_render"] for r in rows])
