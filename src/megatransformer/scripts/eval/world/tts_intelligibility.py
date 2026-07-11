"""TTS intelligibility probe for the world model — WER of Whisper-ASR'd generated speech.

The direct "did it actually say the words" metric, decoupled from latent-similarity
proxies (which were confounded by speaker-entangled SIVE). Pipeline per utterance:
  transcript -> world model.generate (text->voice latent) -> SMG.decode -> vocoder -> wav
  -> Whisper ASR -> WER/CER vs the transcript.

Uses the target speaker embedding (--static_speaker_embedding_path, the deployed TTS voice)
for the SMG, falling back to each utterance's own embedding. Voice sampling is deterministic
(mu) by default so WER measures the model's best guess; raise --voice_temperature to probe
stochastic intelligibility.

  python -m megatransformer.scripts.eval.world.tts_intelligibility \
    --checkpoint_path runs/world/world_tts_sive_0/checkpoint-N --config small_sum --include_modes text,voice --tie_word_embeddings --bf16 \
    --voice_cache_dir ../cached_datasets/world_voice_libritts_val \
    --voice_smg_checkpoint_path <SMG_CKPT> --voice_smg_config medium_decoder_only_1d_3x --voice_smg_sive_encoder_dim 256 \
    --vocoder_config hifigan --static_speaker_embedding_path saved_embeddings/real_spk93.pt \
    --whisper_model base --max_samples 200
"""
import argparse, json, os, re
import numpy as np
import torch
from torch.amp import autocast

from transformers import AutoTokenizer
from megatransformer.utils import model_loading_utils
from megatransformer.utils.constants import BOV_TOKEN_ID
from megatransformer.scripts.data.voice.dataset import VoiceShardedDataset
from megatransformer.scripts.eval.world.eval_voice_synthesis import (
    load_world_model, decode_sive_to_mel, mel_to_waveform, encode_static_prompt,
)


def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — standard WER normalization."""
    s = (s or "").lower().strip()
    s = re.sub(r"[^\w\s']", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_args():
    p = argparse.ArgumentParser(description="World-model TTS intelligibility (WER via Whisper)")
    # world model
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--include_modes", default="text,voice")
    p.add_argument("--tie_word_embeddings", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--device", default=None)
    # data: the voice_synthesis val cache (has transcripts + content features + speaker)
    p.add_argument("--voice_cache_dir", required=True)
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    # SMG + vocoder + target voice
    p.add_argument("--voice_smg_checkpoint_path", required=True)
    p.add_argument("--voice_smg_config", default="medium_decoder_only_1d_3x")
    p.add_argument("--voice_smg_sive_encoder_dim", type=int, default=256)
    p.add_argument("--vocoder_config", default="hifigan")
    p.add_argument("--vocoder_checkpoint_path", default=None)
    p.add_argument("--mel_hop_length", type=int, default=256,
                   help="Hop the SMG's mel is at (256 for SIVE/62.5Hz, 320 for ContentVec/50Hz). "
                        "Resampled to the vocoder's rate before synthesis so WER isn't measured on sped-up audio.")
    p.add_argument("--static_speaker_embedding_path", default=None)
    # generation
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8, help="Text sampling temperature")
    p.add_argument("--voice_temperature", type=float, default=0.0, help="Voice latent sampling (0=deterministic mu)")
    p.add_argument("--voice_variance_floor", type=float, default=0.0)
    # ASR / WER
    p.add_argument("--whisper_model", default="base")
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--output_dir", default="eval_outputs/tts_intelligibility_0")
    p.add_argument("--save_worst", type=int, default=8, help="Save the K worst-WER wavs for inspection")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    print(f"Loading world model {args.config} @ {args.checkpoint_path} ...")
    model = load_world_model(args, device).to(device).eval()

    smg_overrides = {"sive_encoder_dim": args.voice_smg_sive_encoder_dim} if args.voice_smg_sive_encoder_dim else {}
    from megatransformer.model.smg.smg import SMG
    smg = model_loading_utils.load_model(SMG, args.voice_smg_config, checkpoint_path=args.voice_smg_checkpoint_path,
                                         strict=False, overrides=smg_overrides).to(device).eval()
    from megatransformer.utils.audio_utils import SharedWindowBuffer
    vocoder = model_loading_utils.load_vocoder(args.vocoder_checkpoint_path, args.vocoder_config, SharedWindowBuffer()).to(device).eval()
    from megatransformer.utils.visualization import render_vocoder_audio
    _voc_hop = getattr(getattr(vocoder, "config", None), "hop_length", args.mel_hop_length)

    static_spk = None
    if args.static_speaker_embedding_path:
        static_spk = torch.load(args.static_speaker_embedding_path, map_location="cpu", weights_only=True).float().reshape(-1)

    import whisper
    asr = whisper.load_model(args.whisper_model)
    from jiwer import wer as jiwer_wer, cer as jiwer_cer

    ds = VoiceShardedDataset(args.voice_cache_dir, columns=["features", "speaker_embeddings", "text", "token_ids", "text_lengths"])
    idxs = sorted(np.random.RandomState(args.seed).choice(len(ds), min(args.max_samples, len(ds)), replace=False).tolist())

    rows, n_no_voice = [], 0
    for k, i in enumerate(idxs):
        s = ds[i]
        ref = s.get("text")
        if isinstance(ref, (list, tuple)):
            ref = ref[0] if ref else ""
        ref = str(ref or "").strip()
        if not ref and s.get("token_ids") is not None:  # fall back to detokenizing
            tl = int(s.get("text_length", len(s["token_ids"])))
            ref = tokenizer.decode([t for t in s["token_ids"][:tl].tolist() if 0 < t < 32000], skip_special_tokens=True)
        if not ref:
            continue

        prompt = encode_static_prompt(ref[:500], [BOV_TOKEN_ID], tokenizer, args.max_new_tokens, 1024, device)
        with autocast(device, dtype=dtype, enabled=args.bf16):
            outputs = model.generate(text_input_ids=prompt, max_new_tokens=args.max_new_tokens,
                                     temperature=args.temperature, voice_temperature=args.voice_temperature,
                                     voice_variance_floor=args.voice_variance_floor)
        vp = outputs.get("voice_latent_preds")
        if vp is None or vp.numel() == 0:
            n_no_voice += 1
            rows.append({"idx": int(i), "ref": ref, "hyp": "", "wer": 1.0, "cer": 1.0, "no_voice": True})
            continue

        spk = static_spk if static_spk is not None else s.get("speaker_embedding")
        if spk is None:
            continue
        mel = decode_sive_to_mel(smg, vp[0, 0], spk)  # (n_mels, T)
        wav = np.asarray(render_vocoder_audio(vocoder, mel.unsqueeze(0),
                         mel_hop_length=args.mel_hop_length, vocoder_hop_length=_voc_hop),
                         dtype=np.float32).reshape(-1)
        hyp = asr.transcribe(wav, language="en", fp16=False).get("text", "")

        r, h = normalize_text(ref), normalize_text(hyp)
        w = float(jiwer_wer(r, h)) if r else float("nan")
        c = float(jiwer_cer(r, h)) if r else float("nan")
        rows.append({"idx": int(i), "ref": ref, "hyp": hyp.strip(), "wer": w, "cer": c, "no_voice": False, "wav": wav})
        if (k + 1) % 10 == 0:
            done = [x for x in rows if not x["no_voice"]]
            print(f"  {k+1}/{len(idxs)}  running WER={np.mean([x['wer'] for x in done]):.3f}  no_voice={n_no_voice}")

    gen = [x for x in rows if not x["no_voice"]]
    wers = np.array([x["wer"] for x in gen])
    cers = np.array([x["cer"] for x in gen])
    coverage = len(gen) / max(1, len(rows))
    print(f"\n=== TTS intelligibility @ {os.path.basename(args.checkpoint_path)} (n={len(rows)}) ===")
    print(f"  coverage (produced voice) : {coverage*100:.1f}%  ({n_no_voice} produced no voice)")
    if len(gen):
        print(f"  WER  mean={wers.mean():.3f}  median={np.median(wers):.3f}  (lower better; 0=perfect)")
        print(f"  CER  mean={cers.mean():.3f}  median={np.median(cers):.3f}")
        print(f"  %% samples WER<=0.2 (intelligible): {(wers <= 0.2).mean()*100:.1f}%")

    # save worst K wavs + report
    import torchaudio
    rd = os.path.join(args.output_dir, "worst"); os.makedirs(rd, exist_ok=True)
    worst = sorted([x for x in gen if "wav" in x], key=lambda x: -x["wer"])[:args.save_worst]
    for j, x in enumerate(worst):
        torchaudio.save(os.path.join(rd, f"worst{j}_wer{x['wer']:.2f}_idx{x['idx']}.wav"),
                        torch.from_numpy(x["wav"]).reshape(1, -1), args.sample_rate)
    report = {"checkpoint": args.checkpoint_path, "n": len(rows), "coverage": coverage,
              "wer_mean": float(wers.mean()) if len(gen) else None, "wer_median": float(np.median(wers)) if len(gen) else None,
              "cer_mean": float(cers.mean()) if len(gen) else None, "voice_temperature": args.voice_temperature,
              "samples": [{k: v for k, v in x.items() if k != "wav"} for x in rows]}
    with open(os.path.join(args.output_dir, "tts_intelligibility.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWorst {len(worst)} wavs + report -> {args.output_dir}")
    if worst:
        print("Worst examples (ref | hyp):")
        for x in worst[:5]:
            print(f"  WER {x['wer']:.2f}: '{x['ref'][:60]}' | '{x['hyp'][:60]}'")


if __name__ == "__main__":
    main()
