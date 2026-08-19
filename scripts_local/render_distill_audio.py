"""Render world-model voice output through the frozen CosyVoice 2 decoder, for LISTENING.

Renders, per utterance: generated-with-RAS, generated-without-RAS, and the GT units decoded
by the same frozen decoder (the ceiling — what perfect units sound like through this vocoder).
The A/B exists because RAS fixed repetition on paper (adj_repeat 0.103 -> 0.0071, longest_run
56 -> 7) but did NOT fix over-length (2.54x -> 2.32x GT), and the ear is the arbiter of
whether that trade lands.
"""
import argparse, os, sys, json
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_voice_ar_diagnostics import build_args, make_collator
from megatransformer.scripts.eval.world.visualize import load_world_model, load_dataset
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
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--ras_win", type=int, default=10)
ap.add_argument("--ras_tau", type=float, default=0.1)
ap.add_argument("--device", default="cuda:2")
ap.add_argument("--out_dir", required=True)
a = ap.parse_args()

a.n_probe = a.n
cb = load_codebook(a.codebook)
K, D = int(cb.shape[0]), int(cb.shape[1])
args = build_args(argparse.Namespace(**{**vars(a), "n": a.n}), D)
model = load_world_model(args, a.device)
model.set_voice_codebook(cb)
model.to(a.device).eval()
sp_base = getattr(model.config, "special_token_base", constants.SPECIAL_TOKEN_BASE)
sp = constants.special_token_ids(sp_base)
ds = load_dataset(args, "val")
coll = make_collator(K, a.voice_max_frames, special_token_base=sp_base)
coll.force_direction = "synthesis"
dec = CosyVoice2Decoder.from_pretrained(a.cosyvoice_dir, device=a.device)
sr = dec.sample_rate
os.makedirs(a.out_dir, exist_ok=True)
print(f"K={K} D={D} sr={sr} -> {a.out_dir}", flush=True)

import soundfile as sf
manifest, done = [], 0
for i in range(len(ds)):
    s = ds[i]
    if not any(k.startswith("voice_") for k in s):
        continue
    b = coll([s])
    text = b["text_token_ids"][0]
    bov = (text == sp.BOV).nonzero(as_tuple=True)[0]
    if len(bov) == 0:
        continue
    prompt = text[:bov[0].item() + 1].unsqueeze(0).to(a.device)
    spk = s.get("voice_speaker_embedding")
    if spk is None:
        continue
    L = int(s["voice_feature_length"])
    gt_ids = s["voice_unit_ids"][:L]
    row = {"idx": i, "text": s.get("voice_voice_text", ""), "gt_frames": L}

    for tag, win in (("ras", a.ras_win), ("plain", 0)):
        with torch.no_grad():
            out = model.generate(text_input_ids=prompt, max_new_tokens=512,
                                 voice_token_budget=a.voice_max_frames, voice_temperature=1.0,
                                 voice_ras_win=win, voice_ras_tau=a.ras_tau, decode_outputs=False)
        tr = out.get("voice_unit_id_trace", [[]])[0]
        tr = [int(x) for x in tr if 0 <= int(x) < K]
        row[f"{tag}_frames"] = len(tr)
        if tr:
            w = dec.decode(torch.tensor(tr), spk)
            if w is not None:
                sf.write(os.path.join(a.out_dir, f"{done:02d}_gen_{tag}.wav"), w.numpy(), sr)
    w = dec.decode(gt_ids, spk)          # ceiling: GT units, same frozen decoder
    if w is not None:
        sf.write(os.path.join(a.out_dir, f"{done:02d}_target.wav"), w.numpy(), sr)
    manifest.append(row)
    print(f"  [{done}] gt={L} ras={row.get('ras_frames')} plain={row.get('plain_frames')} "
          f"| {row['text'][:60]}", flush=True)
    done += 1
    if done >= a.n:
        break

json.dump({"checkpoint": a.checkpoint_path, "step": a.step, "sample_rate": sr,
           "ras_win": a.ras_win, "ras_tau": a.ras_tau, "items": manifest},
          open(os.path.join(a.out_dir, "manifest.json"), "w"), indent=2)
print(f"\nwrote {done*3} wavs + manifest.json to {a.out_dir}")
print("Listen: NN_gen_ras vs NN_gen_plain vs NN_target (GT units, same decoder = the ceiling).")
