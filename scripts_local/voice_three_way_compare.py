"""Three-way, prompt-conditioned: world model vs GT-units ceiling vs CosyVoice 2's own LM.

All three render through the SAME frozen CosyVoice 2 flow+HiFT decoder, with the SAME
zero-shot prompt (a different utterance by the same speaker), so the only thing that varies
is WHERE THE UNITS CAME FROM:

  ceiling  GT units             -- the real speaker's own phonetics
  model    world model units    -- 127.5M trunk trained on LibriHeavy
  cv2      CosyVoice 2 LM units -- 505.8M, the module the world model replaces

Scored on content (WER/LCS/CER via Whisper) and identity (campplus cosine to the TARGET
audio's embedding -- not the one used to condition, which would be circular).

⚠️ Restricted to DIGIT-FREE text. CosyVoice 2's text_normalize is the identity on such text
(spell_out_number only rewrites digit runs; split_paragraph needs >80 tokens and our max is
56; en_tn_model needs WeTextProcessing, absent here, so CV2 would skip it too). That gives
the CV2 arm exactly the text its own frontend would produce, with no dependency installs and
no approximation.
"""
import argparse, glob, json, os, re, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint_path", required=True)
ap.add_argument("--step", type=int, required=True)
ap.add_argument("--prompt_cache", required=True)
ap.add_argument("--codebook", required=True)
ap.add_argument("--cosyvoice_dir", required=True)
ap.add_argument("--config", default="small_sum")
ap.add_argument("--text_encoder_model", default="HuggingFaceTB/SmolLM2-135M")
ap.add_argument("--voice_max_frames", type=int, default=250)
ap.add_argument("--voice_predict_f0", action="store_true", default=False)
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--voice_temperature", type=float, default=0.0)
ap.add_argument("--ras_win", type=int, default=10)
ap.add_argument("--ras_tau", type=float, default=0.1)
ap.add_argument("--exit_criteria", default="none")
ap.add_argument("--exit_criteria_threshold", type=float, default=None)
ap.add_argument("--whisper_model", default="base")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out_dir", required=True)
ap.add_argument("--save_audio", default=None)
from world_voice_ar_diagnostics import build_args, add_mrope_args
add_mrope_args(ap)
a = ap.parse_args()

from megatransformer.scripts.eval.world.visualize import load_world_model
from megatransformer.utils.codebook import load_codebook
from megatransformer.utils import constants
from megatransformer.model.voice.cosyvoice2_decoder import CosyVoice2Decoder
from megatransformer.utils.cosyvoice2_encoders import CampplusBatchProcessor
from cosyvoice2_llm_baseline import load_cosyvoice2_llm, qwen_tokenizer, cv2_generate_units

cb = load_codebook(a.codebook); K, D = int(cb.shape[0]), int(cb.shape[1])
a.cache_dir = a.prompt_cache; a.codebook = a.codebook
args = build_args(a, D)
model = load_world_model(args, a.device); model.set_voice_codebook(cb)
model.to(a.device).eval()
sp_base = getattr(model.config, "special_token_base", constants.SPECIAL_TOKEN_BASE)
sp = constants.special_token_ids(sp_base)
if a.exit_criteria:
    from megatransformer.model import recurrent_criteria as _rc
    _t = a.exit_criteria_threshold
    model.recurrent_block.exit_criteria = {
        "none": lambda: _rc.NoOpCriteria(),
        "logit_kl": lambda: _rc.LogitKLCriteria(_t if _t is not None else 5e-4),
        "latent_diff": lambda: _rc.LatentDiffCriteria(_t if _t is not None else 0.03),
        "kl_divergence": lambda: _rc.KLDivergenceCriteria(_t if _t is not None else 1e-4),
    }[a.exit_criteria]()

dec = CosyVoice2Decoder.from_pretrained(a.cosyvoice_dir, device=a.device)
sr = dec.sample_rate
spk_enc = CampplusBatchProcessor(a.cosyvoice_dir, source_sr=sr)
cv2_llm, _ = load_cosyvoice2_llm(a.cosyvoice_dir, device=a.device)
qtok = qwen_tokenizer(a.cosyvoice_dir)
import whisper, librosa, numpy as np
from jiwer import wer as jwer, cer as jcer
asr = whisper.load_model(a.whisper_model, device=a.device)
torch.manual_seed(a.seed)
os.makedirs(a.out_dir, exist_ok=True)
if a.save_audio:
    os.makedirs(a.save_audio, exist_ok=True)

shard = torch.load(sorted(glob.glob(os.path.join(a.prompt_cache, "shard_*.pt")))[0],
                   map_location="cpu", weights_only=False)


def transcribe(w):
    x = librosa.resample(w.numpy().astype(np.float32), orig_sr=sr, target_sr=16000)
    return asr.transcribe(x, language="en", fp16=str(a.device).startswith("cuda")).get("text", "").strip()


def norm(s):
    return "".join(c for c in str(s).lower() if c.isalnum() or c.isspace()).split()


def lcs(r, h):
    n, m = len(r), len(h)
    if n == 0: return float("nan")
    prev = [0]*(m+1)
    for i in range(1, n+1):
        cur = [0]*(m+1)
        for j in range(1, m+1):
            cur[j] = prev[j-1]+1 if r[i-1] == h[j-1] else max(prev[j], cur[j-1])
        prev = cur
    return prev[m]/n


def embed(w):
    o = spk_enc.process_batch([w.reshape(-1).cpu()], torch.tensor([w.numel()]))
    return ((o["speaker_embeddings"] if isinstance(o, dict) else o)[0]).reshape(-1).float()


def cos(x, y):
    return float(torch.nn.functional.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0)))


rows, used = [], 0
for i in range(shard["num_samples"]):
    text = shard["text"][i]
    if re.search(r"\d", text):      # CV2 normalize would NOT be identity here
        continue
    tl = int(shard["text_lengths"][i]); fl = int(shard["feature_lengths"][i])
    pl = int(shard["prompt_unit_lengths"][i]); ml = int(shard["prompt_mel_lengths"][i])
    if tl < 3 or fl < 20 or pl < 20:
        continue
    p_ids = shard["prompt_unit_ids"][i, :pl]
    p_mel = shard["prompt_mel"][i, :ml].float()
    spk_prompt = shard["prompt_speaker_embeddings"][i]
    spk_target = shard["speaker_embeddings"][i]

    prompt = torch.cat([shard["token_ids"][i, :tl],
                        torch.tensor([sp.BOV])]).unsqueeze(0).to(a.device)
    with torch.no_grad():
        o = model.generate(text_input_ids=prompt, max_new_tokens=512,
                           voice_token_budget=a.voice_max_frames,
                           voice_temperature=a.voice_temperature,
                           voice_ras_win=a.ras_win, voice_ras_tau=a.ras_tau,
                           decode_outputs=False)
    segs = o.get("voice_unit_id_segments")
    raw = (segs[0][0] if segs and segs[0] and len(segs[0][0]) > 0
           else o.get("voice_unit_id_trace", [[]])[0])
    m_units = [int(x) for x in raw if 0 <= int(x) < K]
    if not m_units:
        continue
    c_units = cv2_generate_units(cv2_llm, qtok, text, shard["prompt_text"][i],
                                 p_ids, spk_prompt, device=a.device)

    arms = {"ceiling": shard["unit_ids"][i, :fl],
            "model": torch.tensor(m_units),
            "cv2": torch.tensor(c_units)}
    row = {"idx": i, "speaker_id": int(shard["speaker_ids"][i]), "ref": text,
           "ref_frames": fl}
    for name, u in arms.items():
        if u.numel() == 0:
            continue
        with torch.no_grad():
            w = dec.decode(u, spk_prompt, prompt_ids=p_ids, prompt_feat=p_mel)
        if w is None:
            continue
        h = transcribe(w)
        row[f"{name}_frames"] = int(u.numel())
        row[f"{name}_hyp"] = h
        row[f"{name}_lcs"] = lcs(norm(text), norm(h))
        row[f"{name}_wer"] = jwer(" ".join(norm(text)), " ".join(norm(h))) if norm(h) else 1.0
        row[f"{name}_cer"] = jcer(" ".join(norm(text)), " ".join(norm(h))) if norm(h) else 1.0
        row[f"{name}_spk"] = cos(embed(w), spk_target)
        if a.save_audio:
            import torchaudio
            torchaudio.save(os.path.join(a.save_audio, f"{name}_{i:04d}.wav"),
                            w.reshape(1, -1).cpu(), sr)
    rows.append(row); used += 1
    print(f"  [{used}/{a.n}] idx {i}  ceil {row.get('ceiling_lcs',float('nan')):.2f}  "
          f"model {row.get('model_lcs',float('nan')):.2f}  cv2 {row.get('cv2_lcs',float('nan')):.2f}",
          flush=True)
    if used >= a.n:
        break

json.dump(rows, open(os.path.join(a.out_dir, f"three_way_step{a.step}.json"), "w"), indent=1)

HUMAN = 0.7005
print(f"\n=== THREE-WAY, prompt-conditioned — step {a.step}, n={len(rows)} ===")
print(f"{'arm':10s} {'LCS':>7} {'WER':>7} {'CER':>7} {'spk cos':>9} {'spk sd':>7} "
      f"{'<0.70':>6} {'len/GT':>7}")
for name in ("ceiling", "model", "cv2"):
    L = [r[f"{name}_lcs"] for r in rows if f"{name}_lcs" in r]
    W = [r[f"{name}_wer"] for r in rows if f"{name}_wer" in r]
    C = [r[f"{name}_cer"] for r in rows if f"{name}_cer" in r]
    S = [r[f"{name}_spk"] for r in rows if f"{name}_spk" in r]
    F = [r[f"{name}_frames"]/max(1, r["ref_frames"]) for r in rows if f"{name}_frames" in r]
    if not L:
        print(f"{name:10s} (none)"); continue
    print(f"{name:10s} {statistics.mean(L):>7.4f} {statistics.mean(W):>7.4f} "
          f"{statistics.mean(C):>7.4f} {statistics.mean(S):>9.4f} "
          f"{statistics.pstdev(S):>7.4f} {sum(1 for x in S if x < HUMAN):>6} "
          f"{statistics.mean(F):>7.2f}")
print(f"\n  speaker cosine reference: two REAL recordings of the same speaker = {HUMAN:.4f}")
print("  '<0.70' counts renders below that human baseline.")
print("\n  NOTE campplus is accent-INVARIANT by design: a render can score high here and "
      "still\n  carry the wrong accent. Listen to the audio triplets for that.")
