"""Does bistream extrapolate to LONGER text than unistream?

WHY THIS EXISTS: every bistream-vs-unistream comparison to date (including the n=48 two-seed
decision run that established the content NULL) ran on <=10 s val utterances AND decoded the
bistream checkpoint in UNISTREAM mode. Chunk-interleaving's claim is that text->speech
alignment becomes LOCAL and MONOTONIC, which is a claim about what happens when the text is
longer than anything trained on. That regime was never measured; findings/world-voice.md
line 158 and 1905 both flag the chunked arm as pending.

PROTOCOL: a length LADDER built by concatenating k consecutive val transcripts (k = 1,2,4,8).
Concatenation keeps the content distribution fixed and varies only length, and it gives a real
GT ceiling for free (concatenate the source unit_ids and decode those through CosyVoice 2).
One fixed speaker embedding across every arm and rung, so speaker variance cannot leak in.

ARMS
  uni_uni   unistream ckpt, unistream decode   (the control)
  bi_uni    bistream ckpt,  unistream decode   (what every prior comparison measured)
  bi_bi     bistream ckpt,  CHUNKED decode     (never measured -- the actual test)
  ceiling   GT units through the same decoder

The headline is RECALL vs k: unistream is expected to truncate (the 10 s EOV is learned, not
a budget -- see the budget 250/500 byte-identical result), so what matters is whether
bistream's recall decays more SLOWLY, not whether it is higher at k=1.
"""
import argparse, json, os, sys

import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_voice_ar_diagnostics import build_args, add_mrope_args
from megatransformer.scripts.eval.world.visualize import load_world_model, load_dataset
from megatransformer.scripts.eval.world.tts_intelligibility import normalize_text
from megatransformer.scripts.data.world.data_collator import MultimodalDataCollator
from megatransformer.utils.codebook import load_codebook
from megatransformer.utils import constants
from megatransformer.model.voice.cosyvoice2_decoder import CosyVoice2Decoder

ap = argparse.ArgumentParser()
ap.add_argument("--uni_checkpoint", required=True)
ap.add_argument("--bi_checkpoint", required=True)
ap.add_argument("--cache_dir", required=True)
ap.add_argument("--codebook", required=True)
ap.add_argument("--cosyvoice_dir", required=True)
ap.add_argument("--config", default="small_sum")
ap.add_argument("--text_encoder_model", default="HuggingFaceTB/SmolLM2-135M")
ap.add_argument("--bistream_text_chunk", type=int, default=4,
                help="MUST match the training CLI of --bi_checkpoint. The live run used 4.")
ap.add_argument("--bistream_voice_chunk", type=int, default=30)
ap.add_argument("--rungs", default="1,2,4,8", help="k values: how many transcripts to concatenate")
ap.add_argument("--n", type=int, default=6, help="distinct text groups per rung")
ap.add_argument("--voice_max_frames", type=int, default=1500,
                help="Generous cap so TRUNCATION is the model's choice, not the budget.")
ap.add_argument("--arms", default="ceiling,uni_uni,bi_uni,bi_bi")
ap.add_argument("--ras_win", type=int, default=10)
ap.add_argument("--ras_tau", type=float, default=0.1)
ap.add_argument("--voice_temperature", type=float, default=0.0)
ap.add_argument("--exit_criteria", default="logit_kl")
ap.add_argument("--exit_criteria_threshold", type=float, default=1e-4)
ap.add_argument("--trunk_iters", type=int, default=8)
ap.add_argument("--whisper_model", default="small.en")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out_dir", required=True)
ap.add_argument("--save_audio", default=None)
ap.add_argument("--resume", action="store_true",
                help="Skip (arm, rung, group) rows already present in rows.jsonl.")
ap.add_argument("--voice_predict_f0", action="store_true", default=False)
add_mrope_args(ap)
a = ap.parse_args()

os.makedirs(a.out_dir, exist_ok=True)
if a.save_audio:
    os.makedirs(a.save_audio, exist_ok=True)
ROWS_PATH = os.path.join(a.out_dir, "rows.jsonl")

done_keys = set()
if a.resume and os.path.exists(ROWS_PATH):
    for line in open(ROWS_PATH):
        try:
            r = json.loads(line)
            done_keys.add((r["arm"], r["k"], r["group"]))
        except Exception:
            pass
    print(f"[resume] {len(done_keys)} rows already done", flush=True)

RUNGS = [int(x) for x in a.rungs.split(",") if x.strip()]
ARMS = [x.strip() for x in a.arms.split(",") if x.strip()]

centroids = load_codebook(a.codebook)
K = centroids.shape[0]
FEATC = centroids.shape[1]
sp = None  # built from model.config once a checkpoint is loaded
print(f"codebook K={K} dim={FEATC}", flush=True)

# max_seq_len must hold the longest rung: ~8 x 185 frames + text tokens.
#
# ⚠️ The collators MUST be built with the MODEL's special_token_base, not
# constants.SPECIAL_TOKEN_BASE. With --text_encoder_model SmolLM2 the base follows that
# tokenizer's vocab, while the module constant is 32000 (Mistral-era). Building the collator
# on the constant emits BOV at the wrong id, so (text == sp.BOV) finds NOTHING and every
# generation is silently skipped -- which is exactly how this script failed the first time.
coll_uni = coll_bi = None


def build_collators(sp_base):
    global coll_uni, coll_bi
    coll_uni = MultimodalDataCollator(
        max_seq_len=2048, max_waveforms=160000, max_mel_spec_frames=625,
        max_sive_feature_frames=a.voice_max_frames, voice_eov_id=K,
        special_token_base=sp_base,
        bistream_text_chunk=0, bistream_voice_chunk=0, bistream_prob=0.0, voice_fill_id=None)
    coll_uni.force_direction = "synthesis"
    coll_bi = MultimodalDataCollator(
        max_seq_len=2048, max_waveforms=160000, max_mel_spec_frames=625,
        max_sive_feature_frames=a.voice_max_frames, voice_eov_id=K,
        special_token_base=sp_base,
        bistream_text_chunk=a.bistream_text_chunk,
        bistream_voice_chunk=a.bistream_voice_chunk,
        bistream_prob=1.0, voice_fill_id=K + 1)
    coll_bi.force_direction = "synthesis"

a.checkpoint_path = a.uni_checkpoint  # build_args wants one; swapped per-arm below
margs = build_args(a, FEATC)
ds = load_dataset(margs, "val")
print(f"dataset: {len(ds)} samples", flush=True)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.text_encoder_model)

import whisper, librosa, numpy as np
asr = whisper.load_model(a.whisper_model, device=a.device)
dec = CosyVoice2Decoder.from_pretrained(a.cosyvoice_dir, device=a.device)
SR = dec.sample_rate


def transcribe(wav_24k):
    x = wav_24k.detach().to("cpu").float().reshape(-1).numpy()
    x16 = librosa.resample(x, orig_sr=SR, target_sr=16000)
    return asr.transcribe(x16, language="en",
                          fp16=str(a.device).startswith("cuda")).get("text", "").strip()


def save_wav(wav, name):
    if not a.save_audio or wav is None:
        return
    x = wav.detach().to("cpu").float()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    torchaudio.save(os.path.join(a.save_audio, name), x, SR)


def lcs_recall(ref_words, hyp_words):
    if not ref_words:
        return float("nan")
    prev = [0] * (len(hyp_words) + 1)
    for rw in ref_words:
        cur = [0]
        for j, hw in enumerate(hyp_words):
            cur.append(prev[j] + 1 if rw == hw else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1] / len(ref_words)


# ---- build the length ladder -------------------------------------------------
# Collect usable val samples once, then group consecutive ones. A group of k is used at EVERY
# rung <= its size, so rung k=1 and k=8 share a prefix and the comparison is nested.
usable = []
for i in range(len(ds)):
    s = ds[i]
    if not any(key.startswith("voice_") for key in s):
        continue
    txt = str(s.get("voice_voice_text", "")).strip()
    if not txt or s.get("voice_speaker_embedding") is None:
        continue
    usable.append(s)
    if len(usable) >= max(RUNGS) * a.n + 8:
        break
print(f"usable val samples: {len(usable)}", flush=True)

FIXED_SPK = usable[0]["voice_speaker_embedding"]


def make_group(group_idx, k):
    """Concatenate k consecutive transcripts + their GT units into one synthetic sample."""
    base = group_idx * max(RUNGS)
    parts = usable[base:base + k]
    if len(parts) < k:
        return None
    text = " ".join(str(p["voice_voice_text"]).strip() for p in parts)
    units, total = [], 0
    for p in parts:
        L = int(p["voice_feature_length"])
        units.append(p["voice_unit_ids"][:L])
        total += L
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    s = dict(parts[0])
    s["text_token_ids"] = ids
    s["text_text_length"] = int(ids.shape[0])
    s["voice_voice_text"] = text
    s["voice_unit_ids"] = torch.cat(units, dim=0)
    s["voice_feature_length"] = min(total, a.voice_max_frames)
    s["voice_speaker_embedding"] = FIXED_SPK
    return s


@torch.no_grad()
def run_arm(model, arm, s, tag):
    global sp
    coll = coll_bi if arm == "bi_bi" else coll_uni
    b = coll([s])
    text = b["text_token_ids"][0]
    bov = (text == sp.BOV).nonzero(as_tuple=True)[0]
    if len(bov) == 0:
        raise RuntimeError(
            f"no BOV ({sp.BOV}) in the collated prompt for {tag} -- the collator's "
            f"special_token_base does not match the model's")
    prompt = text[:bov[0].item() + 1].unsqueeze(0).to(a.device)
    kw = {}
    if arm == "bi_bi":
        full = s["text_token_ids"].reshape(1, -1).to(a.device)
        kw = {"voice_bistream_text": full,
              "voice_bistream_text_chunk": a.bistream_text_chunk,
              "voice_bistream_text_offset": int(bov[0].item())}
    out = model.generate(text_input_ids=prompt, max_new_tokens=512,
                         voice_token_budget=a.voice_max_frames,
                         voice_temperature=a.voice_temperature,
                         voice_ras_win=a.ras_win, voice_ras_tau=a.ras_tau,
                         decode_outputs=False, **kw)
    segs = out.get("voice_unit_id_segments")
    raw = (segs[0][0] if segs and segs[0] and len(segs[0][0]) > 0
           else out.get("voice_unit_id_trace", [[]])[0])
    tr = [int(x) for x in raw if 0 <= int(x) < K]
    if not tr:
        return {"gen_frames": 0, "hyp": ""}
    w = dec.decode(torch.tensor(tr), FIXED_SPK)
    if w is None:
        return {"gen_frames": len(tr), "hyp": ""}
    save_wav(w, f"{tag}.wav")
    return {"gen_frames": len(tr), "hyp": transcribe(w)}


def emit(row):
    with open(ROWS_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")


def set_exit(model):
    rb = model.recurrent_block
    if a.exit_criteria:
        rb.exit_criteria = a.exit_criteria
        rb.exit_criteria_threshold = a.exit_criteria_threshold
    if a.trunk_iters:
        rb.mean_thinking_steps = a.trunk_iters


# ---- ceiling arm (no model needed) -------------------------------------------
if "ceiling" in ARMS:
    print("=== ceiling ===", flush=True)
    for k in RUNGS:
        for g in range(a.n):
            if ("ceiling", k, g) in done_keys:
                continue
            s = make_group(g, k)
            if s is None:
                continue
            ref = s["voice_voice_text"]
            L = int(s["voice_feature_length"])
            w = dec.decode(s["voice_unit_ids"][:L], FIXED_SPK)
            hyp = transcribe(w) if w is not None else ""
            save_wav(w, f"ceiling_k{k}_g{g}.wav")
            emit({"arm": "ceiling", "k": k, "group": g, "ref": ref,
                  "ref_words": len(normalize_text(ref).split()),
                  "hyp": hyp, "gen_frames": L, "ref_frames": L,
                  "recall": lcs_recall(normalize_text(ref).split(), normalize_text(hyp).split())})
            print(f"  ceiling k={k} g={g} frames={L}", flush=True)

# ---- model arms --------------------------------------------------------------
for ckpt_key, ckpt_path, arms_here in (
        ("uni", a.uni_checkpoint, [x for x in ARMS if x == "uni_uni"]),
        ("bi", a.bi_checkpoint, [x for x in ARMS if x in ("bi_uni", "bi_bi")])):
    if not arms_here:
        continue
    margs.checkpoint_path = ckpt_path
    model = load_world_model(margs, a.device)
    model.set_voice_codebook(centroids)
    model.to(a.device).eval()
    sp_base = getattr(model.config, "special_token_base", constants.SPECIAL_TOKEN_BASE)
    sp = constants.special_token_ids(sp_base)
    build_collators(sp_base)
    print(f"  special_token_base={sp_base} BOV={sp.BOV}", flush=True)
    set_exit(model)
    print(f"=== loaded {ckpt_key}: {ckpt_path} ===", flush=True)
    print(f"  arms_here={arms_here} RUNGS={RUNGS} n={a.n}", flush=True)
    for arm in arms_here:
        for k in RUNGS:
            for g in range(a.n):
                if (arm, k, g) in done_keys:
                    continue
                s = make_group(g, k)
                if s is None:
                    print(f"  SKIP {arm} k={k} g={g}: make_group None", flush=True)
                    continue
                ref = s["voice_voice_text"]
                r = run_arm(model, arm, s, f"{arm}_k{k}_g{g}")
                if r is None:
                    continue
                emit({"arm": arm, "k": k, "group": g, "ref": ref,
                      "ref_words": len(normalize_text(ref).split()),
                      "hyp": r["hyp"], "gen_frames": r["gen_frames"],
                      "ref_frames": int(s["voice_feature_length"]),
                      "recall": lcs_recall(normalize_text(ref).split(),
                                           normalize_text(r["hyp"]).split())})
                print(f"  {arm} k={k} g={g} frames={r['gen_frames']}/"
                      f"{int(s['voice_feature_length'])} :: {r['hyp'][:60]}", flush=True)
    del model
    torch.cuda.empty_cache()

print(f"\nwrote {ROWS_PATH}", flush=True)
