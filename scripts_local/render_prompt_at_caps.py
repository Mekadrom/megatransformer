"""Render fixed sentences from a reference clip across several trunk iteration caps.

Reproduces the chat-UI path exactly -- zero-shot prompt (units + 24 kHz mel + campplus from
the SAME clip), greedy + RAS, logit_kl -- while varying only the iteration cap, so the point
where quality breaks down is audible rather than inferred.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torchaudio

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint_path", required=True)
ap.add_argument("--codebook", required=True)
ap.add_argument("--cosyvoice_dir", required=True)
ap.add_argument("--prompt_wav", required=True)
ap.add_argument("--cache_dir", required=True)
ap.add_argument("--config", default="small_sum")
ap.add_argument("--text_encoder_model", default="HuggingFaceTB/SmolLM2-135M")
ap.add_argument("--voice_max_frames", type=int, default=250)
ap.add_argument("--voice_predict_f0", action="store_true", default=False)
ap.add_argument("--n", type=int, default=1)
ap.add_argument("--caps", default="1,2,3,4,8,32")
ap.add_argument("--exit_criteria_threshold", type=float, default=1e-4)
ap.add_argument("--ras_win", type=int, default=10)
ap.add_argument("--ras_tau", type=float, default=0.1)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out_dir", required=True)
from world_voice_ar_diagnostics import build_args, add_mrope_args
add_mrope_args(ap)
a = ap.parse_args()

from transformers import AutoTokenizer
from megatransformer.scripts.eval.world.visualize import load_world_model
from megatransformer.utils.codebook import load_codebook
from megatransformer.utils import constants
from megatransformer.model.voice.cosyvoice2_decoder import CosyVoice2Decoder
from megatransformer.utils.cosyvoice2_encoders import CosyVoice2BatchProcessor, CampplusBatchProcessor
from megatransformer.model import recurrent_criteria as rc

TEXTS = {
    "easy_user": "This is a test. This is only, a test.",
    "hard_words": "The archaeologist's meticulous catalogue distinguished sixteenth-century "
                  "chrysanthemum motifs from later imitations.",
    "fast_clauses": "He ran, he stumbled, he caught himself, he ran again, faster now, "
                    "breathless, certain that something behind him was gaining.",
}

cb = load_codebook(a.codebook); K, D = int(cb.shape[0]), int(cb.shape[1])
args = build_args(a, D)
model = load_world_model(args, a.device); model.set_voice_codebook(cb)
model.to(a.device).eval()
sp = constants.special_token_ids(getattr(model.config, "special_token_base",
                                         constants.SPECIAL_TOKEN_BASE))
tok = AutoTokenizer.from_pretrained(a.text_encoder_model)
dec = CosyVoice2Decoder.from_pretrained(a.cosyvoice_dir, device=a.device)
sr = dec.sample_rate

wav, wsr = torchaudio.load(a.prompt_wav)
if wav.shape[0] > 1:
    wav = wav.mean(0, keepdim=True)
w16 = wav if wsr == 16000 else torchaudio.functional.resample(wav, wsr, 16000)
w24 = wav if wsr == 24000 else torchaudio.functional.resample(wav, wsr, 24000)
ue = CosyVoice2BatchProcessor(a.cosyvoice_dir, voice_max_frames=4096, mel_frame_rate=50.0,
                              device=a.device, source_sr=16000)
se = CampplusBatchProcessor(a.cosyvoice_dir, source_sr=16000)
u = ue.process_batch([w16.reshape(-1)], torch.tensor([w16.shape[-1]]))
p_ids = u["unit_ids"][0, :int(u["feature_lengths"][0])].clone()
e = se.process_batch([w16.reshape(-1)], torch.tensor([w16.shape[-1]]))
spk = ((e["speaker_embeddings"] if isinstance(e, dict) else e)[0]).reshape(-1).float()
p_mel = dec.prompt_mel(w24.reshape(-1), 24000)
print(f"prompt: {a.prompt_wav}  {p_ids.numel()} units, mel {tuple(p_mel.shape)}, "
      f"campplus norm {spk.norm():.2f}", flush=True)

os.makedirs(a.out_dir, exist_ok=True)
base = model.recurrent_block.mean_thinking_steps
for cap in [int(x) for x in a.caps.split(",")]:
    model.recurrent_block.exit_criteria = rc.LogitKLCriteria(a.exit_criteria_threshold)
    model.recurrent_block.mean_thinking_steps = cap
    for name, text in TEXTS.items():
        ids = tok(text, add_special_tokens=False)["input_ids"]
        prompt = torch.cat([torch.tensor(ids), torch.tensor([sp.BOV])]).unsqueeze(0).to(a.device)
        torch.manual_seed(1)
        with torch.no_grad():
            o = model.generate(text_input_ids=prompt, max_new_tokens=512,
                               voice_token_budget=a.voice_max_frames,
                               voice_temperature=0.0, voice_ras_win=a.ras_win,
                               voice_ras_tau=a.ras_tau, decode_outputs=False)
        segs = o.get("voice_unit_id_segments")
        raw = (segs[0][0] if segs and segs[0] and len(segs[0][0]) > 0
               else o.get("voice_unit_id_trace", [[]])[0])
        units = [int(x) for x in raw if 0 <= int(x) < K]
        its = o.get("recurrent_iteration_counts") or []
        if not units:
            print(f"  cap {cap:>2} {name:14s} EMPTY"); continue
        with torch.no_grad():
            w = dec.decode(torch.tensor(units), spk, prompt_ids=p_ids, prompt_feat=p_mel)
        if w is None:
            print(f"  cap {cap:>2} {name:14s} decode failed"); continue
        fn = os.path.join(a.out_dir, f"{name}_cap{cap:02d}.wav")
        torchaudio.save(fn, w.reshape(1, -1).cpu(), sr)
        d = sum(int(x) for x in its)/max(1, len(its)) if its else float("nan")
        print(f"  cap {cap:>2} {name:14s} {len(units):>3} units  {w.numel()/sr:>5.2f}s  "
              f"mean_depth {d:>5.2f}  -> {os.path.basename(fn)}", flush=True)
model.recurrent_block.mean_thinking_steps = base
