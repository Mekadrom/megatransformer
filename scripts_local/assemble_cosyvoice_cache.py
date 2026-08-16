"""Assemble a world-model voice cache from the CosyVoice 2 extraction (Stage 2).

Input: cached_datasets/cosyvoice2_extract/{train,val}_shard_*.pt — each a list of record dicts
  {utt_id, split, speaker_id, transcript (RAW), cosyvoice2_tokens (list[int], 25Hz), n_tokens,
   duration_sec, campplus_embedding (192,)} — plus cosyvoice2_codebook.pt (6561, D).

Output: a MultimodalShardedDataset-format voice cache mirroring the Mimi cache's DISCRETE path
(the dataset's `elif "unit_ids" in shard` branch): stores unit_ids + feature_lengths + SmolLM2
text + campplus speaker, NO mel/f0/vuv (prosody lives in the CosyVoice2 token+decoder; predict_f0
is off for this run). Text is de-corrupted + normalized + SmolLM2-tokenized exactly like
retokenize_shards.py so it lines up with the smollm2 world runs.

  uv run python scripts_local/assemble_cosyvoice_cache.py \
      --extract_dir cached_datasets/cosyvoice2_extract \
      --out_dir cached_datasets/Mekadrom/libritts_r_cosyvoice2_smollm2 \
      --shard_size 2000
"""
import argparse, glob, json, os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig

from megatransformer.scripts.data.voice.preprocess import normalize_transcript

MAX_SEQ_LEN = 256


def build_shard(records, tokenizer, pad_id):
    """records: list of dicts -> a columnar shard dict (padded tensors)."""
    texts, tok_rows, unit_rows, flens, spk_emb, spk_ids = [], [], [], [], [], []
    for r in records:
        text = normalize_transcript(r["transcript"])
        ids = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN, padding=False,
                        return_attention_mask=False)["input_ids"]
        texts.append(text)
        tok_rows.append(torch.tensor(ids, dtype=torch.long))
        u = torch.as_tensor(r["cosyvoice2_tokens"], dtype=torch.long).reshape(-1)
        unit_rows.append(u)
        flens.append(int(r.get("n_tokens", u.numel())))
        emb = torch.as_tensor(r["campplus_embedding"], dtype=torch.float32).reshape(-1)
        spk_emb.append(emb)
        spk_ids.append(int(r["speaker_id"]) if str(r["speaker_id"]).isdigit() else abs(hash(r["speaker_id"])) % (2**31))

    def pad_stack(rows, pad):
        m = max(int(x.shape[-1]) for x in rows)
        return torch.stack([F.pad(x, (0, m - x.shape[-1]), value=pad) if x.shape[-1] < m else x for x in rows], 0)

    return {
        "unit_ids": pad_stack(unit_rows, 0),                      # (N, T) — pad 0, masked by feature_lengths
        "feature_lengths": torch.tensor(flens, dtype=torch.long),
        "token_ids": pad_stack(tok_rows, pad_id),                 # (N, L)
        "text_lengths": torch.tensor([int(t.shape[-1]) for t in tok_rows], dtype=torch.long),
        "text": texts,
        "speaker_embeddings": torch.stack(spk_emb, 0),            # (N, 192) campplus
        "speaker_ids": torch.tensor(spk_ids, dtype=torch.long),
        "num_samples": len(records),
    }


def write_split(records, out_split, tokenizer, pad_id, shard_size, codebook_path, extra_cfg):
    os.makedirs(out_split, exist_ok=True)
    shard_files, offsets, total, off = [], [], 0, 0
    speakers = set()
    for si, start in enumerate(range(0, len(records), shard_size)):
        chunk = records[start:start + shard_size]
        shard = build_shard(chunk, tokenizer, pad_id)
        speakers.update(shard["speaker_ids"].tolist())
        fn = f"shard_{si:06d}.pt"
        torch.save(shard, os.path.join(out_split, fn))
        shard_files.append(fn); offsets.append(off); off += shard["num_samples"]; total += shard["num_samples"]
        print(f"  {os.path.basename(out_split)}/{fn}: {shard['num_samples']} rows, "
              f"units {tuple(shard['unit_ids'].shape)}, text {tuple(shard['token_ids'].shape)}")
    json.dump({"shard_files": shard_files, "shard_offsets": offsets, "total_samples": total,
               "dataset_type": "stat-shards", "num_speakers": len(speakers)},
              open(os.path.join(out_split, "shard_index.json"), "w"))
    cfg = {"content_encoder": "cosyvoice2", "frame_rate_hz": 25.0, "tokenizer_name": "HuggingFaceTB/SmolLM2-135M",
           "unit_vocab_size": extra_cfg["unit_vocab"], "feature_channels": extra_cfg["feature_channels"],
           "predict_f0": False, "codebook": "cosyvoice2_codebook.pt", **extra_cfg}
    json.dump(cfg, open(os.path.join(out_split, "config.json"), "w"), indent=2)
    # codebook lives next to the shards so --voice_codebook_path can point here
    if codebook_path:
        import shutil
        shutil.copy2(codebook_path, os.path.join(out_split, "cosyvoice2_codebook.pt"))
    return total, len(speakers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shard_size", type=int, default=2000)
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    pad_id = tok.pad_token_id or 0
    llm_vocab = int(AutoConfig.from_pretrained(a.tokenizer).vocab_size)
    cb_path = os.path.join(a.extract_dir, "cosyvoice2_codebook.pt")
    codebook = torch.load(cb_path, map_location="cpu", weights_only=False)
    K, D = int(codebook.shape[0]), int(codebook.shape[1])
    print(f"codebook: ({K}, {D}) -> unit_vocab={K+1}, feature_channels={D}")
    extra = {"unit_vocab": K + 1, "feature_channels": D, "special_token_base": llm_vocab}

    for split in ["train", "val"]:
        files = sorted(glob.glob(os.path.join(a.extract_dir, f"{split}_shard_*.pt")))
        if not files:
            print(f"[skip] no {split} shards"); continue
        records = []
        for f in files:
            records.extend(torch.load(f, map_location="cpu", weights_only=False))
        print(f"=== {split}: {len(records)} records ===")
        tot, nsp = write_split(records, os.path.join(a.out_dir, split), tok, pad_id, a.shard_size, cb_path, extra)
        print(f"  -> {tot} samples, {nsp} speakers")
    print(f"\nDone. CosyVoice2 world cache at {a.out_dir}  (feature_channels={D}, unit_vocab={K+1})")


if __name__ == "__main__":
    main()
