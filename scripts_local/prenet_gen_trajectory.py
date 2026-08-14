"""Diagnose the world-21000 free-running greedy collapse: is it prenet-caused, progressive,
or greedy-specific? Measures GENERATION LENGTH + EOV directly from voice_unit_id_trace -- no
SMG/vocoder needed, so it's fast across many checkpoints.

Controls:
  - no-prenet BASELINE at matched steps (does it collapse too? -> isolates prenet vs stage).
  - prenet checkpoints ACROSS the ramp (progressive vs transient).
  - greedy vs temp 0.6 x seeds (greedy-argmax-EOV pathology vs real break).
"""
import argparse
from argparse import Namespace
import torch
from transformers import AutoTokenizer

from megatransformer.scripts.eval.world.eval_voice_synthesis import load_world_model, load_dataset
from megatransformer.utils.constants import BOV_TOKEN_ID

CB = "./cached_datasets/Mekadrom/libritts_r_clean_mimi_wavlm/val/mimi_semantic_codebook.pt"
VC = "./cached_datasets/Mekadrom/libritts_r_clean_mimi_wavlm/val"
PROMPTS = ['"Yes.".', 'Yes.', 'yes', 'No.', 'Okay.', 'Hello.']

# (label, checkpoint, config) -- LR-volatility test: does the LATE low-cosine-LR baseline
# free-run stably (works) where the MID-schedule higher-LR checkpoint collapsed?
CKPTS = [
    ("base-11730 (works ref)", "runs/world/world_mimi_wavlm_freezef0smg_0/checkpoint-11730", "small_sum"),
    ("base-21000 (collapse ref)", "runs/world/world_mimi_wavlm_freezef0smg_0/checkpoint-21000", "small_sum"),
    ("base-52000 (low LR)", "runs/world/world_mimi_wavlm_freezef0smg_0/checkpoint-52000", "small_sum"),
    ("base-54000 (low LR)", "runs/world/world_mimi_wavlm_freezef0smg_0/checkpoint-54000", "small_sum"),
]


def wargs(ckpt, config):
    return Namespace(config=config, checkpoint_path=ckpt, include_modes="voice",
                     tie_word_embeddings=False, voice_feature_channels=256, voice_predict_f0=True,
                     voice_codebook_path=CB, cache_dir=None, text_cache_dir=None, voice_cache_dir=VC,
                     use_memorization_dataset=False, max_samples=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    def enc(text):
        ids = tok.encode(text, add_special_tokens=True)[:511] + [BOV_TOKEN_ID]
        return torch.tensor([ids], dtype=torch.long, device=dev)

    def gen_len(model, prompt, temp, seed):
        torch.manual_seed(seed)
        with torch.no_grad():
            out = model.generate(text_input_ids=enc(prompt), max_new_tokens=512,
                                 voice_temperature=temp, voice_token_budget=125)
        tr = out.get("voice_unit_id_trace", [[]])[0]
        K = model.voice_codebook.shape[0]
        eov = bool(tr and int(tr[-1]) == K)
        n = len(tr) - (1 if eov else 0)  # content frames (exclude terminal EOV)
        return n, eov

    print(f"{'checkpoint':14} | " + " | ".join(f"{p:8}" for p in PROMPTS) + "  (greedy content-frames; E=EOV@0)")
    print("-" * 100)
    results = {}
    for label, ckpt, cfg in CKPTS:
        model = load_world_model(wargs(ckpt, cfg), dev); model.eval()
        cells = []
        for p in PROMPTS:
            n, eov = gen_len(model, p, 0.0, 0)
            cells.append(f"{n}{'E' if n == 0 else ''}")
        print(f"{label:14} | " + " | ".join(f"{c:8}" for c in cells))
        results[label] = model  # keep last for sampling probe
        if label != "base-54000 (low LR)":
            del model; torch.cuda.empty_cache()

    # greedy-vs-sampling on the latest low-LR checkpoint
    print("\n=== base-54000: greedy vs temp 0.6 ===")
    model = results["base-54000 (low LR)"]
    for p in PROMPTS:
        g, _ = gen_len(model, p, 0.0, 0)
        s = [gen_len(model, p, 0.6, sd)[0] for sd in (0, 1, 2)]
        print(f"  {p:10} greedy={g:3}  temp0.6 seeds={s}")
    print("DONE")


if __name__ == "__main__":
    main()
