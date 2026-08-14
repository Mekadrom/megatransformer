"""ISOLATED-venv extractor for the GLM-4-Voice tokenizer (Whisper-large enc +
4x avg-pool -> 12.5 Hz + single VQ 16384). Runs under the pinned glmvenv
(transformers==4.44.1, torch==2.3.0) so it never touches the training .venv.

Imports ONLY torch + the vendored GLM speech_tokenizer code + a raw-waveform dump
(dump_raw_subset.py) — no megatransformer. Emits glm.pt in the SAME record format
as extract.py so compare.py consumes it identically.

Run (from repo root, with the vendored code + raw dump already present):
  GLM_SRC=<...>/scratchpad/GLM-4-Voice \
  <glmvenv>/bin/python scripts_local/content_encoder_eval/extract_glm.py \
      --raw eval_output/content_encoder_eval/raw_subset.pt \
      --out eval_output/content_encoder_eval/glm.pt
"""
import argparse
import os
import sys

import torch


def build_model(glm_src, device):
    """The vendored load_quantize_encoder filters keys by a 'model.encoder.' prefix
    that this checkpoint does NOT use (its keys are top-level: codebook.weight,
    conv1.weight, layers.N...). So build the WhisperVQEncoder and load the raw keys
    directly (encoder-only build already has exactly these)."""
    import glob
    sys.path.insert(0, glm_src)
    from huggingface_hub import snapshot_download
    from speech_tokenizer.configuration_whisper import WhisperVQConfig
    from speech_tokenizer.modeling_whisper import WhisperVQEncoder
    from transformers import WhisperFeatureExtractor
    import safetensors

    local = snapshot_download("THUDM/glm-4-voice-tokenizer")
    config = WhisperVQConfig.from_pretrained(local)
    config.quantize_encoder_only = True
    model = WhisperVQEncoder(config)
    state_dict = {}
    for path in glob.glob(os.path.join(local, "model*.safetensors")):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("layer_norm"):  # encoder final LN absent in enc-only build
                    continue
                state_dict[key] = f.get_tensor(key)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"GLM load: {len(state_dict)} keys, missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing sample:", missing[:5])
    model.eval().to(device)
    fe = WhisperFeatureExtractor.from_pretrained(local)
    return model, fe


@torch.no_grad()
def encode_utt(model, fe, wav_16k, device):
    """Replicates speech_tokenizer.utils.extract_speech_token for ONE utterance,
    additionally returning the dequantized codebook embedding. 30s chunking."""
    stride = (model.conv1.stride[0] * model.conv2.stride[0]
              * (model.config.pooling_kernel_size or 1) * fe.hop_length)
    audio = wav_16k.float().cpu().numpy()
    all_ids = []
    t = 0
    while t * 16000 < audio.shape[0]:
        seg = audio[t * 16000:(t + 30) * 16000]
        feats = fe([seg], sampling_rate=16000, return_attention_mask=True,
                   return_tensors="pt", padding="longest", pad_to_multiple_of=stride)
        feats = feats.to(device)
        out = model(**feats)
        ids = out.quantized_token_ids  # [1, T_pool]
        am = feats.attention_mask[:, ::model.conv1.stride[0] * model.conv2.stride[0]]
        am = am[:, ::model.config.pooling_kernel_size]
        ids = ids[0][am[0].bool()]
        all_ids.append(ids)
        t += 30
    ids = torch.cat(all_ids)                         # [L]
    emb = model.codebook(ids)                        # [L, d_model] dequantized
    return emb.float().cpu(), ids.long().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--glm_src", default=os.environ.get("GLM_SRC", ""))
    ap.add_argument("--pooled_only", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    model, fe = build_model(a.glm_src, a.device)
    raw = torch.load(a.raw, weights_only=False)
    recs_in = raw["records"]

    records = []
    dim = 1280
    for j, s in enumerate(recs_in):
        wav = s["waveform"].float().reshape(-1)
        feats, ids = encode_utt(model, fe, wav, a.device)
        if feats.shape[0] < 2:
            continue
        dim = feats.shape[1]
        if a.pooled_only:
            records.append({"pooled": feats.mean(0).float().clone(),
                            "speaker_id": int(s["speaker_id"])})
        else:
            records.append({
                "feats": feats.half().clone(),
                "codes": ids.clone(),
                "pooled": feats.mean(0).float().clone(),
                "ctc_tokens": s["ctc_tokens"].long().clone(),
                "speaker_id": int(s["speaker_id"]),
            })
        if (j + 1) % 500 == 0:
            print(f"  {j + 1}/{len(recs_in)}")

    meta = {"encoder": "glm", "dim": dim, "frame_rate": 12.5,
            "is_discrete": True, "n": len(records), "pooled_only": a.pooled_only}
    torch.save({"meta": meta, "records": records}, a.out)
    print(f"saved {len(records)} -> {a.out}  meta={meta}")


if __name__ == "__main__":
    main()
