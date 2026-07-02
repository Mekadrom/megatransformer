"""
Headless HTTP interaction server for MegaTransformerWorldModel.

Designed for programmatic/agent-driven interrogation of the model (e.g. Claude
asking the model many prompts and analyzing structured responses). Loads the
model stack once at startup, exposes POST /generate, returns JSON with:
- rendered text (with [image N] / [voice N] / [audio N] markers in position)
- raw text (special tokens stripped) for simpler downstream analysis
- absolute file-system paths to decoded image PNGs / voice WAVs
- Whisper transcriptions of generated voice clips (if whisper is installed)
- diagnostics: prompt/completion token counts, latency, stop-logit traces,
  active DiT latent_scale, pixel ranges per generated image

Runs alongside `multimodal_chat.py` (the Gradio UI) — this script is the
programmatic counterpart, it doesn't replace or modify the Gradio interface.
Imports a few decode/render helpers from `multimodal_chat.py` to avoid code
duplication.

Usage:
    python -m megatransformer.scripts.eval.world.chat_headless --checkpoint_path runs/world/my_run/checkpoint-N --config small_sum_dit --include_modes text,voice,image --tie_word_embeddings --bf16 --voice_smg_checkpoint_path ./runs/smg/.../checkpoint-300000 --voice_smg_config medium_decoder_only_1d_3x --voice_smg_sive_encoder_dim 256 --vocoder_config hifigan --image_vae_decoder_config litevae --static_speaker_embedding_path ./logs/speaker_embedding_1.pt --image_latent_channel_scales 0.975,1.027,1.009,1.195,1.259,1.173,1.265,0.932,1.144,0.888,0.522,0.748 --host 127.0.0.1 --port 7861 --session_dir ./runs/headless_sessions --whisper_model base

Client side (curl):
    curl -s -X POST http://127.0.0.1:7861/generate -H 'Content-Type: application/json' -d '{"prompt":"a photo of a red car.","modality_hint":"image","seed":42}' | jq .

Request body schema:
    {
      "prompt": str,                            # required
      "modality_hint": "text"|"image"|"voice"|"audio",  # optional; appends BO*
      "temperature": float,                     # optional; overrides default
      "top_p": float,                           # optional; overrides default
      "seed": int,                              # optional; torch.manual_seed
      "max_new_tokens": int,                    # optional; overrides default
    }

Response body schema (success):
    {
      "request_id": "abc123",
      "session_dir": "/abs/path/to/this/request/outputs",
      "prompt": "...",
      "generation_settings": {...},
      "latency_s": 3.2,
      "text": "rendered text with [image 1] markers...",
      "text_raw": "rendered without markers or special tokens",
      "images": [{"index": 1, "path": "/abs/.../image_1.png", ...}],
      "voice":  [{"index": 1, "path": "/abs/.../voice_1.wav", "transcription": "..."}],
      "audio":  [],
      "diagnostics": {...}
    }
"""

import argparse
import json
import os
import sys
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import numpy as np
import torch
import torchaudio
from torch.amp import autocast

from megatransformer.model.voice.sive.sive import SpeakerInvariantVoiceEncoder
from megatransformer.model.world.world_model import MegaTransformerWorldModel
from megatransformer.utils import model_loading_utils
from megatransformer.utils.audio_utils import SharedWindowBuffer
from megatransformer.utils.constants import (
    BOA_TOKEN_ID, BOV_TOKEN_ID, BOI_TOKEN_ID,
)

# Reuse decode + render helpers from the Gradio script — single source of truth
# for how outputs are post-processed.
from megatransformer.scripts.eval.world.multimodal_chat import (
    decode_image_latent,
    decode_voice_latent,
    render_generated_text,
)


def parse_args():
    p = argparse.ArgumentParser(description="Headless MegaTransformer chat server")
    # World model
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--config", type=str, default="small_sum_dit")
    p.add_argument("--include_modes", type=str, default="text,voice,image")
    p.add_argument("--tie_word_embeddings", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--voice_token_budget", type=int, default=209)
    p.add_argument("--audio_token_budget", type=int, default=209)

    # Voice SMG + vocoder
    p.add_argument("--voice_smg_checkpoint_path", type=str, default=None)
    p.add_argument("--voice_smg_config", type=str, default="medium_decoder_only_1d_3x")
    p.add_argument("--voice_smg_sive_encoder_dim", type=int, default=None)
    p.add_argument("--vocoder_config", type=str, default="hifigan")
    p.add_argument("--vocoder_checkpoint_path", type=str, default=None)
    p.add_argument("--static_speaker_embedding_path", type=str, default=None)
    p.add_argument("--sample_rate", type=int, default=16000)

    # Image VAE
    p.add_argument("--image_vae_decoder_config", type=str, default="litevae")
    p.add_argument("--image_vae_decoder_path", type=str, default=None)
    p.add_argument("--image_latent_scale", type=float, default=None)
    p.add_argument("--image_latent_channel_scales", type=str, default=None)

    # Whisper ASR for voice transcription
    p.add_argument("--whisper_model", type=str, default="base",
                   help="Whisper model size: tiny, base, small, medium, large. "
                        "Set to 'none' to skip transcription entirely.")

    # Server
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--session_dir", type=str, default="./runs/headless_sessions")
    p.add_argument("--device", type=str, default=None)

    return p.parse_args()


class Runner:
    """Owns all loaded models and handles each generation request."""

    def __init__(self, args):
        self.args = args
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if args.bf16 else torch.float32
        self.shared_window_buffer = SharedWindowBuffer()

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

        print(f"[runner] Loading world model from {args.checkpoint_path}...")
        include_modes = [m.strip() for m in args.include_modes.split(",")]
        overrides = {"include_modes": include_modes}
        if args.tie_word_embeddings:
            overrides["tie_word_embeddings"] = True
        self.model = model_loading_utils.load_model(
            MegaTransformerWorldModel, args.config,
            checkpoint_path=args.checkpoint_path,
            overrides=overrides, device=self.device,
        )
        self.model.eval()

        # DiT latent scale override (mirrors multimodal_chat.py)
        from megatransformer.model.image.diffusion_decoder import DiffusionBridgeImageDecoder
        if isinstance(getattr(self.model, "image_generator", None), DiffusionBridgeImageDecoder):
            new_scale = None
            if args.image_latent_channel_scales is not None:
                scales = [float(s) for s in args.image_latent_channel_scales.split(",")]
                n = self.model.image_generator.config.latent_channels
                if len(scales) != n:
                    raise ValueError(f"--image_latent_channel_scales has {len(scales)} values but latent_channels={n}")
                new_scale = torch.tensor(scales, dtype=torch.float)
            elif args.image_latent_scale is not None:
                n = self.model.image_generator.config.latent_channels
                new_scale = torch.full((n,), float(args.image_latent_scale))
            if new_scale is not None:
                buf = self.model.image_generator.latent_scale
                new_scale = new_scale.view(1, -1, 1, 1).to(device=buf.device, dtype=buf.dtype)
                buf.copy_(new_scale)
            active = self.model.image_generator.latent_scale.detach().flatten().cpu().tolist()
            print(f"[runner] DiT latent_scale: {[f'{v:.3f}' for v in active]}")
            self.active_latent_scale = active
        else:
            self.active_latent_scale = None

        # LiteVAE (image decode)
        self.litevae = None
        if args.image_vae_decoder_config == "litevae":
            print("[runner] Loading LiteVAE...")
            from megatransformer.scripts.data.image.vae.preprocess import _load_litevae
            self.litevae = _load_litevae("litevae", device=self.device)
            self.litevae.eval()

        # Voice SMG decoder
        self.smg_decoder = None
        if args.voice_smg_checkpoint_path:
            print(f"[runner] Loading voice SMG ({args.voice_smg_config})...")
            from megatransformer.model.smg.smg import SMG
            smg_overrides = {}
            if args.voice_smg_sive_encoder_dim is not None:
                smg_overrides["sive_encoder_dim"] = args.voice_smg_sive_encoder_dim
            self.smg_decoder = model_loading_utils.load_model(
                SMG, args.voice_smg_config,
                checkpoint_path=args.voice_smg_checkpoint_path,
                device=self.device, strict=False, overrides=smg_overrides,
            )
            self.smg_decoder.eval()

        # Vocoder
        self.vocoder = None
        if args.vocoder_config or args.vocoder_checkpoint_path:
            self.vocoder = model_loading_utils.load_vocoder(
                args.vocoder_checkpoint_path, args.vocoder_config, self.shared_window_buffer,
            )
            if self.vocoder is not None:
                self.vocoder = self.vocoder.to(self.device)

        # Static speaker embedding
        self.speaker_emb = None
        if args.static_speaker_embedding_path:
            self.speaker_emb = torch.load(
                args.static_speaker_embedding_path, map_location="cpu", weights_only=True,
            )

        # Whisper ASR — optional. Graceful if not installed.
        self.whisper_model = None
        if args.whisper_model and args.whisper_model.lower() != "none":
            try:
                import whisper
                print(f"[runner] Loading Whisper ({args.whisper_model})...")
                self.whisper_model = whisper.load_model(args.whisper_model)
            except ImportError:
                print("[runner] whisper not installed (`pip install openai-whisper`); voice transcription disabled.")
            except Exception as e:
                print(f"[runner] Whisper load failed: {e}")

        os.makedirs(args.session_dir, exist_ok=True)
        print("[runner] Ready.")

    def transcribe(self, wav_path: str) -> Optional[str]:
        if self.whisper_model is None:
            return None
        try:
            result = self.whisper_model.transcribe(wav_path, fp16=False)
            return (result.get("text") or "").strip()
        except Exception as e:
            return f"<transcription error: {type(e).__name__}: {e}>"

    def generate(
        self,
        prompt_text: str,
        modality_hint: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> dict:
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        token_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        prompt_len = len(token_ids)
        if modality_hint == "image":
            token_ids.append(BOI_TOKEN_ID)
        elif modality_hint == "voice":
            token_ids.append(BOV_TOKEN_ID)
        elif modality_hint == "audio":
            token_ids.append(BOA_TOKEN_ID)

        prompt = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        t0 = time.time()
        with torch.no_grad():
            with autocast(self.device, dtype=self.dtype, enabled=self.args.bf16):
                outputs = self.model.generate(
                    text_input_ids=prompt,
                    max_new_tokens=max_new_tokens or self.args.max_new_tokens,
                    temperature=temperature if temperature is not None else self.args.temperature,
                    top_p=top_p if top_p is not None else self.args.top_p,
                    audio_token_budget=self.args.audio_token_budget,
                    voice_token_budget=self.args.voice_token_budget,
                    precomputed_latents=True,
                )
        latency_s = time.time() - t0

        request_id = uuid.uuid4().hex[:12]
        session = os.path.join(
            self.args.session_dir,
            f"{int(time.time() * 1000)}_{request_id}",
        )
        os.makedirs(session, exist_ok=True)

        real_img = int(outputs["image_counts"][0].item()) if outputs.get("image_counts") is not None else 0
        real_voice = int(outputs["voice_counts"][0].item()) if outputs.get("voice_counts") is not None else 0
        real_audio = int(outputs["audio_counts"][0].item()) if outputs.get("audio_counts") is not None else 0

        gen_ids = outputs["generated_token_ids"][0].tolist()
        text_marked = render_generated_text(
            gen_ids, self.tokenizer, real_img, real_voice, real_audio,
        )
        text_raw = self.tokenizer.decode(
            [t for t in gen_ids if t < 32000 and t != 0],
            skip_special_tokens=True,
        )

        images_info = []
        if real_img and self.litevae is not None:
            for k in range(real_img):
                latent = outputs["image_latent_preds"][0, k]
                try:
                    img, stats = decode_image_latent(self.litevae, latent, self.device)
                    path = os.path.join(session, f"image_{k + 1}.png")
                    img.save(path)
                    images_info.append({
                        "index": k + 1,
                        "path": os.path.abspath(path),
                        "latent_shape": list(latent.shape),
                        "pixel_range": [stats["pixel_min"], stats["pixel_max"]],
                        "pixel_mean": stats["pixel_mean"],
                        "latent_std_per_channel": stats["latent_std_per_channel"],
                    })
                except Exception as e:
                    images_info.append({
                        "index": k + 1,
                        "error": f"{type(e).__name__}: {e}",
                    })

        voice_info = []
        voice_lens_out = outputs.get("voice_lengths")
        if real_voice:
            if self.smg_decoder is None or self.vocoder is None:
                voice_info.append({"error": "voice generated but SMG/vocoder not loaded"})
            elif self.speaker_emb is None:
                voice_info.append({"error": "voice generated but no --static_speaker_embedding_path"})
            else:
                for k in range(real_voice):
                    latent = outputs["voice_latent_preds"][0, k]
                    if voice_lens_out is not None:
                        T = int(voice_lens_out[0, k].item())
                        latent = latent[:, :T]
                    try:
                        sr, wav_np = decode_voice_latent(
                            latent, self.smg_decoder, self.vocoder,
                            self.speaker_emb, self.args.sample_rate,
                        )
                        path = os.path.join(session, f"voice_{k + 1}.wav")
                        torchaudio.save(path, torch.from_numpy(wav_np).unsqueeze(0), sr)
                        voice_info.append({
                            "index": k + 1,
                            "path": os.path.abspath(path),
                            "sample_rate": sr,
                            "duration_s": float(len(wav_np)) / float(sr),
                            "latent_shape": list(latent.shape),
                            "transcription": self.transcribe(path),
                        })
                    except Exception as e:
                        voice_info.append({
                            "index": k + 1,
                            "error": f"{type(e).__name__}: {e}",
                        })

        from megatransformer.utils.constants import EOI_TOKEN_ID, EOV_TOKEN_ID, EOA_TOKEN_ID
        eoi_count = sum(1 for t in gen_ids if t == EOI_TOKEN_ID)
        eov_count = sum(1 for t in gen_ids if t == EOV_TOKEN_ID)
        eoa_count = sum(1 for t in gen_ids if t == EOA_TOKEN_ID)
        spurious = (eoi_count - real_img) + (eov_count - real_voice) + (eoa_count - real_audio)

        return {
            "request_id": request_id,
            "session_dir": os.path.abspath(session),
            "prompt": prompt_text,
            "generation_settings": {
                "modality_hint": modality_hint,
                "temperature": temperature if temperature is not None else self.args.temperature,
                "top_p": top_p if top_p is not None else self.args.top_p,
                "seed": seed,
                "max_new_tokens": max_new_tokens or self.args.max_new_tokens,
            },
            "latency_s": latency_s,
            "text": text_marked,
            "text_raw": text_raw,
            "images": images_info,
            "voice": voice_info,
            "audio": [],
            "diagnostics": {
                "dit_latent_scale": self.active_latent_scale,
                "prompt_tokens": prompt_len,
                "completion_tokens": len(gen_ids),
                "image_counts_real": real_img,
                "voice_counts_real": real_voice,
                "audio_counts_real": real_audio,
                "eos_tokens_in_stream": {
                    "EOI": eoi_count,
                    "EOV": eov_count,
                    "EOA": eoa_count,
                },
                "spurious_eo_sampled": spurious,
                "voice_stop_logit_trace": outputs.get("voice_stop_logit_trace", [[]])[0],
                "audio_stop_logit_trace": outputs.get("audio_stop_logit_trace", [[]])[0],
                "recurrent_iteration_counts": outputs.get("recurrent_iteration_counts", []),
                "whisper_available": self.whisper_model is not None,
            },
        }


def make_handler(runner: Runner):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/generate":
                return self._respond(404, {"error": "not found"})
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length)) if length else {}
            except Exception as e:
                return self._respond(400, {"error": f"invalid JSON: {e}"})
            if "prompt" not in body:
                return self._respond(400, {"error": "missing 'prompt' field"})
            try:
                result = runner.generate(
                    prompt_text=body["prompt"],
                    modality_hint=body.get("modality_hint"),
                    temperature=body.get("temperature"),
                    top_p=body.get("top_p"),
                    seed=body.get("seed"),
                    max_new_tokens=body.get("max_new_tokens"),
                )
            except Exception as e:
                return self._respond(500, {
                    "error": f"{type(e).__name__}: {e}",
                    "trace": traceback.format_exc(),
                })
            self._respond(200, result)

        def do_GET(self):
            if self.path == "/health":
                return self._respond(200, {
                    "status": "ok",
                    "whisper_available": runner.whisper_model is not None,
                    "device": runner.device,
                })
            self._respond(404, {"error": "not found"})

        def _respond(self, code: int, payload: dict):
            data = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):  # noqa: A002 — http.server API
            sys.stderr.write(
                f"[{self.log_date_time_string()}] {self.address_string()} — {format % args}\n"
            )

    return Handler


def main():
    args = parse_args()
    runner = Runner(args)
    handler = make_handler(runner)
    server = HTTPServer((args.host, args.port), handler)
    print(f"[server] Listening on http://{args.host}:{args.port}")
    print(f"[server] Session outputs → {os.path.abspath(args.session_dir)}")
    print("[server] POST /generate {prompt, modality_hint?, temperature?, top_p?, seed?, max_new_tokens?}")
    print("[server] GET /health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] Shutting down.")


if __name__ == "__main__":
    main()
