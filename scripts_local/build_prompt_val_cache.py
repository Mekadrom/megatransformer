"""Build a val-style cache that carries a ZERO-SHOT PROMPT for every utterance.

Why: every world-voice eval to date renders with the campplus embedding ALONE. CosyVoice 2's
actual interface is zero-shot from a reference CLIP -- prompt speech tokens plus a 24 kHz
prompt mel concatenated ahead of the target inside the flow. Embedding-only rendering is
measurably different (it rendered a male reference as female, 2026-09-15), so both the
"ceiling" and any speaker-similarity number computed without a prompt are against a
handicapped decoder.

The existing val cache stores no waveforms, so the prompt cannot be reconstructed from it.
This rebuilds from the LibriHeavy source parquets.

⚠️ The prompt is ALWAYS A DIFFERENT UTTERANCE BY THE SAME SPEAKER, never the target itself --
otherwise the decoder is handed the answer and every metric is meaningless.

Fields per sample (target):    unit_ids, feature_lengths, speaker_embeddings, speaker_ids,
                               token_ids, text_lengths, text
Fields per sample (prompt):    prompt_unit_ids, prompt_unit_lengths, prompt_mel,
                               prompt_mel_lengths, prompt_speaker_embeddings, prompt_text
  - speaker_embeddings      campplus of the TARGET audio -- the identity to MATCH
  - prompt_speaker_embeddings  campplus of the PROMPT clip -- what CosyVoice 2 conditions on
That split is deliberate: condition on one clip, score against the other.
"""
import argparse, io, json, os, sys, collections
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--parquet_glob", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--cosyvoice_dir", required=True)
ap.add_argument("--tokenizer_name", default="HuggingFaceTB/SmolLM2-135M")
ap.add_argument("--limit", type=int, default=2000)
ap.add_argument("--shard_size", type=int, default=2000)
ap.add_argument("--voice_max_seconds", type=float, default=10.0)
ap.add_argument("--prompt_max_seconds", type=float, default=6.0)
ap.add_argument("--prompt_min_seconds", type=float, default=2.0)
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()

if os.path.isdir(a.out_dir) and os.listdir(a.out_dir):
    raise SystemExit(f"refusing to write into non-empty {a.out_dir}")
os.makedirs(a.out_dir, exist_ok=True)

import glob as _glob
import pyarrow.parquet as pq
import torchaudio
from transformers import AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from megatransformer.utils.cosyvoice2_encoders import (
    CosyVoice2BatchProcessor, CampplusBatchProcessor)
from megatransformer.model.voice.cosyvoice2_decoder import CosyVoice2Decoder, _matcha_mel

files = sorted(_glob.glob(a.parquet_glob))
if not files:
    raise SystemExit(f"no parquet matched {a.parquet_glob}")
print(f"source: {len(files)} parquet file(s)", flush=True)

tok = AutoTokenizer.from_pretrained(a.tokenizer_name)
unit_enc = CosyVoice2BatchProcessor(a.cosyvoice_dir, voice_max_frames=4096,
                                    mel_frame_rate=50.0, device=a.device, source_sr=16000)
spk_enc = CampplusBatchProcessor(a.cosyvoice_dir, source_sr=16000)
PROMPT_MEL = CosyVoice2Decoder.PROMPT_MEL


def decode_audio(blob):
    wav, sr = torchaudio.load(io.BytesIO(blob))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    return wav, sr


def rs(wav, sr, target):
    return wav if sr == target else torchaudio.functional.resample(wav, sr, target)


# ---- pass 1: index rows by speaker so a prompt partner always exists -------------------
by_spk = collections.defaultdict(list)
rows_raw = []
for f in files:
    t = pq.read_table(f, columns=["audio", "speaker_id", "text_original", "audio_duration"])
    d = t.to_pylist()
    for r in d:
        dur = float(r.get("audio_duration") or 0)
        if dur <= 0 or dur > a.voice_max_seconds:
            continue
        idx = len(rows_raw)
        rows_raw.append(r)
        by_spk[str(r["speaker_id"])].append(idx)
    print(f"  indexed {f.split('/')[-1]}: {len(rows_raw)} usable rows so far", flush=True)
    if len(rows_raw) >= a.limit * 3:
        break

usable = [(i, r) for i, r in enumerate(rows_raw)
          if len(by_spk[str(r["speaker_id"])]) >= 2][:a.limit]
print(f"{len(usable)} targets over {len({str(r['speaker_id']) for _, r in usable})} speakers "
      f"(speakers with >=2 utterances only)", flush=True)

# ---- pass 2: encode ------------------------------------------------------------------
buf = collections.defaultdict(list)
shard_files, shard_offsets, written = [], [], 0


def flush(force=False):
    global buf, written
    n = len(buf["unit_ids"])
    if n == 0 or (n < a.shard_size and not force):
        return
    def pad(seqs, dtype=torch.long):
        m = max(int(s.shape[0]) for s in seqs)
        out = torch.zeros(len(seqs), m, dtype=dtype)
        for i, s in enumerate(seqs):
            out[i, :s.shape[0]] = s.to(dtype)
        return out
    mels = buf["prompt_mel"]
    mm = max(int(x.shape[0]) for x in mels)
    mel_t = torch.zeros(len(mels), mm, mels[0].shape[1], dtype=torch.float16)
    for i, x in enumerate(mels):
        mel_t[i, :x.shape[0]] = x.half()
    shard = {
        "unit_ids": pad(buf["unit_ids"]),
        "feature_lengths": torch.tensor(buf["feature_lengths"], dtype=torch.long),
        "speaker_embeddings": torch.stack(buf["speaker_embeddings"]),
        "speaker_ids": torch.tensor(buf["speaker_ids"], dtype=torch.long),
        "token_ids": pad(buf["token_ids"]),
        "text_lengths": torch.tensor(buf["text_lengths"], dtype=torch.long),
        "text": list(buf["text"]),
        "prompt_unit_ids": pad(buf["prompt_unit_ids"]),
        "prompt_unit_lengths": torch.tensor(buf["prompt_unit_lengths"], dtype=torch.long),
        "prompt_mel": mel_t,
        "prompt_mel_lengths": torch.tensor(buf["prompt_mel_lengths"], dtype=torch.long),
        "prompt_speaker_embeddings": torch.stack(buf["prompt_speaker_embeddings"]),
        "prompt_text": list(buf["prompt_text"]),
        "num_samples": n,
    }
    name = f"shard_{len(shard_files):06d}.pt"
    torch.save(shard, os.path.join(a.out_dir, name))
    shard_files.append(name); shard_offsets.append(written); written += n
    print(f"  wrote {name} ({n} samples)", flush=True)
    buf = collections.defaultdict(list)


spk_str2int = {}
for n_done, (i, r) in enumerate(usable):
    sid_s = str(r["speaker_id"])
    partner = next((j for j in by_spk[sid_s] if j != i), None)
    if partner is None:
        continue
    try:
        wav_t, sr_t = decode_audio(r["audio"]["bytes"])
        wav_p, sr_p = decode_audio(rows_raw[partner]["audio"]["bytes"])
    except Exception as e:
        print(f"  skip {i}: audio decode {type(e).__name__}", flush=True); continue

    t16 = rs(wav_t, sr_t, 16000)
    p16 = rs(wav_p, sr_p, 16000)
    # trim the PROMPT only -- the target must stay whole
    lo, hi = int(a.prompt_min_seconds * 16000), int(a.prompt_max_seconds * 16000)
    if p16.shape[-1] < lo:
        continue
    p16 = p16[..., :hi]
    p24 = rs(wav_p, sr_p, 24000)[..., :int(a.prompt_max_seconds * 24000)]

    u_t = unit_enc.process_batch([t16.reshape(-1)], torch.tensor([t16.shape[-1]]))
    u_p = unit_enc.process_batch([p16.reshape(-1)], torch.tensor([p16.shape[-1]]))
    n_t = int(u_t["feature_lengths"][0]); n_p = int(u_p["feature_lengths"][0])
    e_t = spk_enc.process_batch([t16.reshape(-1)], torch.tensor([t16.shape[-1]]))
    e_p = spk_enc.process_batch([p16.reshape(-1)], torch.tensor([p16.shape[-1]]))
    e_t = (e_t["speaker_embeddings"] if isinstance(e_t, dict) else e_t)[0].reshape(-1).clone()
    e_p = (e_p["speaker_embeddings"] if isinstance(e_p, dict) else e_p)[0].reshape(-1).clone()
    mel = _matcha_mel(p24.reshape(1, -1).float(), **PROMPT_MEL).squeeze(0).transpose(0, 1)

    txt = str(r.get("text_original") or "")
    ptxt = str(rows_raw[partner].get("text_original") or "")
    ids = tok(txt, add_special_tokens=False)["input_ids"]
    sid_i = spk_str2int.setdefault(sid_s, len(spk_str2int))

    buf["unit_ids"].append(u_t["unit_ids"][0, :n_t].clone())
    buf["feature_lengths"].append(n_t)
    buf["speaker_embeddings"].append(e_t)
    buf["speaker_ids"].append(sid_i)
    buf["token_ids"].append(torch.tensor(ids, dtype=torch.long))
    buf["text_lengths"].append(len(ids))
    buf["text"].append(txt)
    buf["prompt_unit_ids"].append(u_p["unit_ids"][0, :n_p].clone())
    buf["prompt_unit_lengths"].append(n_p)
    buf["prompt_mel"].append(mel.clone())
    buf["prompt_mel_lengths"].append(int(mel.shape[0]))
    buf["prompt_speaker_embeddings"].append(e_p)
    buf["prompt_text"].append(ptxt)
    if len(buf["unit_ids"]) >= a.shard_size:
        flush()
    if (n_done + 1) % 50 == 0:
        print(f"  [{n_done+1}/{len(usable)}]", flush=True)

flush(force=True)
json.dump({"shard_files": shard_files, "shard_offsets": shard_offsets,
           "total_samples": written, "dataset_type": "stat-shards",
           "num_speakers": len(spk_str2int)},
          open(os.path.join(a.out_dir, "shard_index.json"), "w"), indent=1)
json.dump({"content_encoder": "cosyvoice2", "encoder_dim": 512, "sample_rate": 16000,
           "voice_max_seconds": a.voice_max_seconds,
           "prompt_max_seconds": a.prompt_max_seconds,
           "prompt_from": "different utterance, same speaker",
           "prompt_mel": PROMPT_MEL, "source": a.parquet_glob,
           "tokenizer": a.tokenizer_name, "num_speakers": len(spk_str2int)},
          open(os.path.join(a.out_dir, "config.json"), "w"), indent=1)
print(f"\nDONE: {written} samples, {len(shard_files)} shard(s), "
      f"{len(spk_str2int)} speakers -> {a.out_dir}")
