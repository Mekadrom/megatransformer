"""One-off: generate voice from short TEXT prompts across a temperature x seed grid, to test
whether the world-TTS model is really CONTENT-conditioning (a "Yes." prompt -> a "yes" clip)
or just duration-conditioning + prior.

The discriminating design: alongside the "yes" variants, include CONTRASTING short words
(No./Okay./Hello.). If each prompt yields its OWN word -> content conditioning. If everything
collapses to one common word -> prior + duration, not content.

Reuses the (Mimi-ready) loaders from eval_voice_synthesis. Needs a GPU.

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python3 scripts_local/probe_word_prompt.py \
    --checkpoint_path runs/world/<run>/checkpoint-<step> \
    --voice_cache_dir ./cached_datasets/Mekadrom/libritts_r_clean_mimi_wavlm \
    --voice_codebook_path ./cached_datasets/Mekadrom/libritts_r_clean_mimi_wavlm/val/mimi_semantic_codebook.pt \
    --voice_smg_checkpoint_path runs/smg/<smg>/checkpoint-<n> \
    --voice_smg_config medium_decoder_only_1d_15x2_mimicontour_vocos \
    --voice_smg_speaker_embedding_dim 768 --vocoder_config vocos \
    --out_dir eval_output/word_probe
"""
import argparse
import os
from argparse import Namespace

import torch
import torchaudio
from transformers import AutoTokenizer

from megatransformer.scripts.eval.world.eval_voice_synthesis import (
    load_world_model, load_dataset, decode_sive_to_mel, mel_to_waveform, encode_static_prompt,
)
from megatransformer.model.smg.smg import SMG
from megatransformer.utils import model_loading_utils
from megatransformer.utils.constants import BOV_TOKEN_ID


# Verbatim (the exact prompt that worked) + normalized variants + CONTRASTING words.
DEFAULT_PROMPTS = ['"Yes.".', 'Yes.', 'yes', 'No.', 'Okay.', 'Hello.']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--config", default="small_sum")
    p.add_argument("--voice_cache_dir", required=True, help="voice base dir (for a speaker embedding)")
    p.add_argument("--voice_codebook_path", required=True)
    p.add_argument("--voice_feature_channels", type=int, default=256)
    # SMG + vocoder
    p.add_argument("--voice_smg_checkpoint_path", required=True)
    p.add_argument("--voice_smg_config", default="medium_decoder_only_1d_15x2_mimicontour_vocos")
    p.add_argument("--voice_smg_sive_encoder_dim", type=int, default=256)
    p.add_argument("--voice_smg_speaker_embedding_dim", type=int, default=768)
    p.add_argument("--vocoder_config", default="vocos")
    p.add_argument("--vocoder_checkpoint_path", default=None)
    p.add_argument("--sample_rate", type=int, default=24000)
    p.add_argument("--mel_hop_length", type=int, default=256)
    p.add_argument("--voice_token_budget", type=int, default=125)
    p.add_argument("--voice_prenet_dropout", type=float, default=None,
                   help="Apply prenet dropout at INFERENCE (Tacotron-style, kept on). REQUIRED to "
                        "correctly infer a prenet-curriculum world checkpoint: set it to the value "
                        "the model TRAINED with at this step (train/voice_prenet_dropout in TB). "
                        "None/0 = off (correct only for pre-prenet checkpoints).")
    # grid
    p.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    p.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 0.3, 0.6, 0.8])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out_dir", default="eval_output/word_probe")
    p.add_argument("--device", default="cuda")
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()


def world_args(a):
    return Namespace(
        config=a.config, checkpoint_path=a.checkpoint_path, include_modes="voice",
        tie_word_embeddings=False,
        voice_feature_channels=a.voice_feature_channels, voice_predict_f0=True,
        voice_codebook_path=a.voice_codebook_path,
        cache_dir=None, text_cache_dir=None, voice_cache_dir=a.voice_cache_dir,
        use_memorization_dataset=False, max_samples=64,
    )


def load_smg(a, device):
    smg_overrides = {
        "sive_encoder_dim": a.voice_smg_sive_encoder_dim,
        "hop_length": a.mel_hop_length,
        "sample_rate": a.sample_rate,
        "speaker_embedding_dim": a.voice_smg_speaker_embedding_dim,
        "code_embed_init": "learned_random",
    }
    smg = model_loading_utils.load_model(
        SMG, a.voice_smg_config, checkpoint_path=a.voice_smg_checkpoint_path,
        strict=False, overrides=smg_overrides, device=device,
    )
    smg.eval()
    return smg


def safe_name(prompt):
    return "".join(c if c.isalnum() else "_" for c in prompt)[:24] or "empty"


def main():
    a = parse_args()
    device = a.device
    dtype = torch.bfloat16 if a.bf16 else torch.float32
    os.makedirs(a.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    wargs = world_args(a)
    print(f"Loading world model {a.checkpoint_path} ...", flush=True)
    model = load_world_model(wargs, device)
    model.eval()
    # Prenet-curriculum checkpoints trained with prenet dropout on the AR input; per Tacotron it
    # must stay ON at inference or the clean input is OOD (-> immediate-EOV collapse). load builds
    # from small_sum (prenet_dropout=0), so set it post-load to the training value for this step.
    if a.voice_prenet_dropout is not None:
        model.voice_feature_extractor.config.prenet_dropout = a.voice_prenet_dropout
        print(f"inference prenet_dropout set to {a.voice_prenet_dropout}", flush=True)

    smg = load_smg(a, device)
    vocoder = model_loading_utils.load_vocoder(a.vocoder_checkpoint_path, a.vocoder_config, None)

    # A single fixed speaker embedding (real WavLM 768) so every clip is the SAME voice --
    # we are testing CONTENT, not speaker. Grab the first val sample that has one.
    ds = load_dataset(wargs, split="val")
    spk = None
    for i in range(min(64, len(ds))):
        v = ds[i].get("voice_speaker_embedding")
        if v is not None:
            spk = v.to(device=device, dtype=dtype)
            break
    if spk is None:
        raise SystemExit("No voice_speaker_embedding found in val cache.")

    print(f"grid: {len(a.prompts)} prompts x {len(a.temperatures)} temps x {len(a.seeds)} seeds", flush=True)
    rows = []
    for prompt in a.prompts:
        for temp in a.temperatures:
            # greedy (temp 0) is deterministic -> one seed only
            seeds = [a.seeds[0]] if temp == 0.0 else a.seeds
            for seed in seeds:
                torch.manual_seed(seed)
                prompt_ids = encode_static_prompt(prompt, [BOV_TOKEN_ID], tokenizer, 512, 1024, device)
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=dtype, enabled=a.bf16):
                        out = model.generate(
                            text_input_ids=prompt_ids, max_new_tokens=512,
                            voice_temperature=temp, voice_token_budget=a.voice_token_budget,
                        )
                vp = out.get("voice_latent_preds")
                trace = out.get("voice_unit_id_trace", [[]])[0]
                n_frames = len(trace)
                K = model.voice_codebook.shape[0]
                eov = bool(trace and int(trace[-1]) == K)
                if vp is None or vp.numel() == 0:
                    rows.append((prompt, temp, seed, n_frames, eov, "EMPTY")); continue
                lat = vp[0, 0]
                contour = None
                f0 = out.get("voice_f0_preds")
                if f0 is not None and f0.numel() > 0:
                    contour = f0[0, 0]
                mel = decode_sive_to_mel(smg, lat, spk, f0_contour=contour)
                wav = mel_to_waveform(vocoder, mel, mel_hop_length=a.mel_hop_length)
                if not isinstance(wav, torch.Tensor):
                    wav = torch.as_tensor(wav)
                wav = wav.float().cpu()
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
                dur = wav.shape[-1] / a.sample_rate
                fn = f"{safe_name(prompt)}__t{temp}__s{seed}.wav"
                torchaudio.save(os.path.join(a.out_dir, fn), wav, a.sample_rate)
                rows.append((prompt, temp, seed, n_frames, eov, f"{dur:.2f}s {fn}"))
                print(f"  {prompt!r:14} t={temp} s={seed}  frames={n_frames} eov={eov} {dur:.2f}s", flush=True)

    # summary
    print("\n=== SUMMARY (prompt | temp | seed | frames | eov | dur/file) ===")
    for r in rows:
        print("  " + " | ".join(str(x) for x in r))
    with open(os.path.join(a.out_dir, "summary.txt"), "w") as f:
        f.write("prompt | temp | seed | frames | eov | dur/file\n")
        for r in rows:
            f.write(" | ".join(str(x) for x in r) + "\n")
    print(f"\nWrote WAVs + summary to {a.out_dir}")


if __name__ == "__main__":
    main()
