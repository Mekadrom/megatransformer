"""
Gradio chat UI for MegaTransformerWorldModel with Copilot-style file @-references.

UX:
- Drag/drop or click to upload image / audio files (a clipboard paste zone is
  provided for images)
- Each uploaded file appears in the attachment list; the reference token
  `@filename.ext` is auto-appended to the prompt and can be cut/pasted to any
  position in the text
- On submit, each `@ref` is expanded inline to `[BO*, *_PH, EO*]` placeholders
  and the corresponding pre-encoded tensor is attached to `generate()` via the
  matching `*_inputs` arg — the `TokenInterleaver` handles positional injection
- Generation target (text / voice / image) controls the trailing BO* token
  appended to the prompt

Voice has TWO backends. Supplying --voice_cosyvoice2_model_dir selects the CosyVoice 2
discrete path (25 Hz FSQ unit ids, frozen flow+HiFT decoder, campplus-192 speaker
conditioning) for both directions; without it the legacy SIVE/SMG/HiFiGAN stack is used.
Text and image paths are identical either way.

Usage (CosyVoice 2 world-voice checkpoint -- current):
    python -m megatransformer.scripts.eval.world.multimodal_chat --checkpoint_path runs/world_voice/<run>/checkpoint-78000 --config small_sum --include_modes text,voice --text_encoder_model HuggingFaceTB/SmolLM2-135M --voice_cosyvoice2_model_dir <CosyVoice2-0.5B snapshot> --voice_codebook_path <cache>/val/cosyvoice2_codebook.pt --mrope_voice_rate 7.5 --mrope_scale_side text --bf16

Usage (legacy SIVE/SMG stack):
    python -m megatransformer.scripts.eval.world.multimodal_chat --checkpoint_path runs/world/my_run/checkpoint-3000 --config small_sum_dit --include_modes text,voice,image --tie_word_embeddings --bf16 --sive_checkpoint_path ./checkpoints/sive --sive_config tiny_deep --voice_smg_checkpoint_path ./checkpoints/smg --voice_smg_config medium_decoder_only_1d_3x --voice_smg_sive_encoder_dim 256 --vocoder_config hifigan --image_vae_decoder_config litevae --static_speaker_embedding_path ./logs/speaker_embedding_1.pt

Speaker conditioning on the CV2 path: uploading a voice file sets the speaker embedding from
THAT clip (campplus of the upload), which is what makes voice cloning work; otherwise
--static_speaker_embedding_path is used and must be a 192-d campplus vector.

⚠️ --mrope_scale_side carries no weights and cannot be detected from the checkpoint. Pass the
value training used, and --mrope_voice_rate to match (7.5 for the LibriHeavy runs, 6.0 for
older ones): a mismatch evaluates under a coordinate system the model never saw.
"""

import argparse
import os
import re
import time
from typing import Optional

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from PIL import Image
from torch.amp import autocast

from megatransformer.utils.tokenizer_resolution import resolve_tokenizer_name
from megatransformer.model.voice.sive.sive import SpeakerInvariantVoiceEncoder
from megatransformer.model.world.world_model import MegaTransformerWorldModel
from megatransformer.utils import model_loading_utils
from megatransformer.utils.audio_utils import SharedWindowBuffer, extract_mels
from megatransformer.utils import constants

IMAGE_EXTS = {"png", "jpg", "jpeg", "webp", "bmp", "gif"}
AUDIO_EXTS = {"wav", "mp3", "flac", "ogg", "m4a", "opus"}


def _placeholder_triplet(sp) -> dict:
    """BO*/PH/EO* id triplets for a resolved SpecialTokenIds (`sp`)."""
    return {
        "voice": (sp.BOV, sp.VOICE_PLACEHOLDER, sp.EOV),
        "audio": (sp.BOA, sp.AUDIO_PLACEHOLDER, sp.EOA),
        "image": (sp.BOI, sp.IMAGE_PLACEHOLDER, sp.EOI),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Multimodal Gradio chat for MegaTransformerWorldModel")
    # World model
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--config", type=str, default="small_sum_dit")
    p.add_argument("--include_modes", type=str, default="text,voice,image")
    p.add_argument("--tie_word_embeddings", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--text_tokenizer", type=str, default=None,
                   help="Corpus tokenizer, as training was given it. Composes with either "
                        "prelude; --text_encoder_model wins when both are set.")
    p.add_argument("--text_encoder_model", type=str, default=None,
                   help="Pretrained text spine (e.g. HuggingFaceTB/SmolLM2-135M) — MUST match how the "
                        "checkpoint was trained, or the prelude weights won't load. Also switches the "
                        "control-token base (special_token_base) to the LLM's native vocab size so "
                        "the interleaver's BO*/PH/EO* ids line up with the trained checkpoint.")

    # SDXL image decoder (for the SDXL-adapter image path). When the loaded
    # model's image_generator is an SDXLConditioningAdapter, generate() returns
    # predicted CLIP conditioning (77x2048 seq + 1280 pooled) rather than a
    # latent, and we render pixels here with a frozen SDXL pipeline.
    p.add_argument("--sdxl_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                   help="SDXL base model for rendering the adapter's predicted conditioning.")
    p.add_argument("--sdxl_gen_steps", type=int, default=30,
                   help="SDXL diffusion steps (adapter path, DPM++ 2M Karras). UI 'image diffusion steps' overrides if >0.")
    p.add_argument("--sdxl_guidance", type=float, default=7.0,
                   help="SDXL classifier-free guidance scale (adapter path).")
    # Z-Image adapter (ZImageConditioningAdapter): renders predicted Qwen3 conditioning
    # (seq, 2560; no pooled) via frozen Z-Image-Turbo. 8 NFEs, guidance 0 (Turbo).
    p.add_argument("--zimage_model", type=str, default="Tongyi-MAI/Z-Image-Turbo",
                   help="Z-Image-Turbo model for rendering the adapter's predicted conditioning.")
    p.add_argument("--zimage_output_gain", type=float, default=1.0,
                   help="Z-Image adapter: inference-only dispersion gain on the whitened prediction. An MSE-trained point estimate is shrunk toward the target mean by 1-R^2, which the DiT renders as washed-out/generic; ~1/alpha undoes it. Measured best ~1.2-1.5 (gain 1.34: CLIPScore 0.287->0.303 vs 0.345 GT). Estimate per-checkpoint with scripts_local/zimage_shrinkage_probe.py. 1.0 = off.")
    # ── generative-head (T3/T4/T5 family) controls ──────────────────────────────────────
    # A flow-head checkpoint SAMPLES its conditioning instead of regressing a point, and
    # essentially all of its quality comes from classifier-free guidance: measured on the same
    # 8-prompt probe, unguided 0.287 vs w=3 0.359 against a 0.368 GT ceiling -- i.e. an unguided
    # flow head only ties the gain-corrected POINT head. Without this flag the demo rendered every
    # T3-family checkpoint at the head's default guidance of 1.0, i.e. unguided.
    p.add_argument("--flow_guidance", type=float, default=None,
                   help="CFG scale for a generative conditioning head (T3 flow_head / T4 ar_flow_head). "
                        "Measured operating point is 3.0; 1.0 = off. Ignored on point-head checkpoints. "
                        "Default None = 3.0 when a flow head is present, else untouched.")
    p.add_argument("--flow_seed", type=int, default=None,
                   help="Seed the conditioning sampler so a generation is reproducible. The head "
                        "emits a DISTRIBUTION, so draws differ; without this a result you liked "
                        "cannot be recovered. None = fresh random draw each time.")
    # Offload is a DISCRETE-GPU optimisation: it keeps the ~20GB Z-Image stack on the host and
    # pages modules to the device on demand so the world model can coexist in limited VRAM. On
    # UNIFIED-MEMORY hardware (Ryzen AI, Apple silicon) host and device are the SAME physical
    # memory, so it buys nothing and costs transfer overhead. It also needs accelerate>=0.17,
    # which used to be declared only in the `training` group -- a demo-only install crashed here.
    p.add_argument("--zimage_offload", choices=["auto", "on", "off"], default="auto",
                   help="Z-Image CPU offload. 'auto' = offload if accelerate is present, else "
                        "load straight to the device. 'off' skips it -- the right choice on "
                        "unified-memory systems. 'on' requires accelerate and errors if missing.")
    p.add_argument("--zimage_gen_steps", type=int, default=8,
                   help="Z-Image diffusion steps (Turbo=8). UI 'image diffusion steps' overrides if >0.")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    # 209 = ceil((10*16000//256)/3) -- a SNAPSHOT of the frame math for SIVE @hop256/stride3.
    # It is wrong for any other frontend: a ContentVec run (@hop320/stride1) needs ~500, and the
    # stale constant truncated renders to 42% of an utterance while training saw the whole thing.
    # train.media_frame_budget() is the canonical derivation; None = derive it from the same
    # hop/stride/seconds args the trainer uses.
    p.add_argument("--voice_token_budget", type=int, default=None,
                   help="Voice generation length cap. Default None = derive via media_frame_budget "
                        "(max_seconds*sample_rate//hop_length, / sive_total_stride).")
    p.add_argument("--audio_token_budget", type=int, default=None,
                   help="Audio generation length cap. Default None = derive (see --voice_token_budget).")
    p.add_argument("--voice_max_seconds", type=float, default=10.0)
    p.add_argument("--voice_sample_rate", type=int, default=16000)
    p.add_argument("--voice_hop_length", type=int, default=256)
    p.add_argument("--audio_max_seconds", type=float, default=10.0)
    p.add_argument("--audio_sample_rate", type=int, default=16000)
    p.add_argument("--audio_hop_length", type=int, default=256)
    p.add_argument("--sive_total_stride", type=int, default=1,
                   help="Content-frontend total stride (SIVE=3, ContentVec/Mimi=1). Feeds the budget derivation.")
    p.add_argument("--image_iteration_override", type=int, default=None,
                   help="Override recurrent iteration count for image gen query positions (eval-only). "
                        "If unset, uses mean_thinking_steps with KL early-exit.")
    p.add_argument("--image_num_inference_steps", type=int, default=None,
                   help="Override DiT sampling steps. If unset, uses the decoder's config default.")
    p.add_argument("--image_sampler", type=str, default="euler",
                   choices=["euler", "heun", "midpoint"],
                   help="Diffusion sampler. heun/midpoint are 2nd-order (2× NFE per step).")

    # SIVE (voice/audio input encoding)
    p.add_argument("--sive_checkpoint_path", type=str, default=None)
    p.add_argument("--sive_config", type=str, default="tiny_deep")
    p.add_argument("--sive_layer", type=int, default=10)
    p.add_argument("--sive_num_speakers", type=int, default=2338)

    # Voice SMG + vocoder (for decoding generated voice)
    p.add_argument("--voice_smg_checkpoint_path", type=str, default=None)
    p.add_argument("--voice_smg_config", type=str, default="medium_decoder_only_1d_3x")
    p.add_argument("--voice_smg_sive_encoder_dim", type=int, default=None)
    p.add_argument("--vocoder_config", type=str, default="hifigan")
    p.add_argument("--vocoder_checkpoint_path", type=str, default=None)
    # CosyVoice 2 discrete voice path. Supplying --voice_cosyvoice2_model_dir switches the
    # VOICE modality (both directions) from the SIVE/SMG/HiFiGAN stack to CV2: uploads are
    # tokenised to 25 Hz FSQ unit ids and fed as centroid vectors, and generated unit ids are
    # rendered by the frozen CV2 flow+HiFT decoder. Text and image paths are untouched.
    p.add_argument("--voice_cosyvoice2_model_dir", type=str, default=None,
                   help="CosyVoice2-0.5B snapshot dir. Enables the discrete voice path.")
    p.add_argument("--voice_codebook_path", type=str, default=None,
                   help="cosyvoice2_codebook.pt (6561x512). REQUIRED with "
                        "--voice_cosyvoice2_model_dir: it defines the unit->feature map the "
                        "prelude consumes AND fixes the EOV id at K.")
    # Name matches the training CLI (training.py:3179) and the error raised by
    # _ensure_importable; --cosyvoice_runtime_dir is kept as an alias.
    p.add_argument("--voice_cosyvoice2_runtime_dir", "--cosyvoice_runtime_dir",
                   dest="voice_cosyvoice2_runtime_dir", type=str, default=None,
                   help="CosyVoice repo checkout (default ~/dev/projects/cosyvoice-runtime, or "
                        "$COSYVOICE_RUNTIME). Set up with scripts_local/setup_cosyvoice_runtime.sh.")
    p.add_argument("--mrope_voice_rate", type=float, default=None,
                   help="M-RoPE voice rate the checkpoint was TRAINED at (7.5 for the LibriHeavy "
                        "runs, 6.0 for older ones). Evaluating under the wrong rate uses a "
                        "coordinate system the model never saw.")
    p.add_argument("--mrope_scale_side", type=str, default=None, choices=["voice", "text", "off"],
                   help="Which stream absorbs the rate. Carries no weights, so it cannot be "
                        "detected from the checkpoint; the loader hard-fails without it.")
    p.add_argument("--voice_temperature", type=float, default=0.0,
                   help="Voice unit sampling temperature. DEFAULT 0.0 (greedy) as of "
                        "2026-09-16: greedy is the best-measured sampler at every checkpoint "
                        "and every exit criterion tested (ck90000 n=48 x2 -- greedy 0.8849 "
                        "LCS vs T=0.6+ras_temp 0.8619 vs T=0.7 0.8384). Greedy degenerates "
                        "(~99%% adjacent repeats) WITHOUT --voice_ras_win > 0, which defaults "
                        "to 10; at ras_tau=0 with no RAS it collapses completely (LCS 0.0, "
                        "empty transcripts). Do not set this to 0 while disabling RAS.")
    p.add_argument("--voice_ras_win", type=int, default=10,
                   help="Repetition-aware sampling window. Default 10 (CosyVoice 2's value): "
                        "the model's free-running argmax is 'repeat the previous unit' almost "
                        "always, so RAS is load-bearing, not a refinement. 0 disables.")
    p.add_argument("--voice_ras_tau", type=float, default=0.1)
    p.add_argument("--voice_prompt_seconds", type=float, default=3.0,
                   help="Seconds of an uploaded voice clip used as ZERO-SHOT prompt conditioning "
                        "(its FSQ units + 24 kHz mel are concatenated ahead of the target inside "
                        "the flow). The 192-d embedding is a global summary; the prompt adds "
                        "per-frame acoustic evidence. Measured +0.014 mean campplus cosine to the "
                        "reference over 6 same-speaker pairs (5/6 positive). 0 disables, leaving "
                        "embedding-only conditioning. Prompt shares flow context with the target, "
                        "so longer is not automatically better.")
    p.add_argument("--voice_min_frame_ratio", type=float, default=0.0,
                   help="Text-proportional EOV floor (CV2's min_len scheme). 0 = off, which is "
                        "the measured-best setting once the resample scaling is correct.")
    p.add_argument("--static_speaker_embedding_path", type=str, default=None,
                   help="Fallback speaker embedding for voice decoding (.pt file)")

    # Image VAE (for encoding input images + decoding generated images)
    p.add_argument("--image_vae_decoder_config", type=str, default="litevae",
                   help="'litevae' to use pretrained LiteVAE for both encode and decode")
    p.add_argument("--image_vae_decoder_path", type=str, default=None,
                   help="Optional internal image decoder checkpoint path")
    # DiT latent scaling — mirrors the training CLI so a checkpoint whose
    # latent_scale buffer wasn't persisted (or whose training command you
    # want to reproduce) can have the correct scales injected at load time.
    # Silently ignored for non-DiffusionBridgeImageDecoder generators.
    p.add_argument("--image_latent_scale", type=float, default=None,
                   help="Global scalar applied to image latents in the diffusion bridge decoder")
    p.add_argument("--image_latent_channel_scales", type=str, default=None,
                   help="Per-channel image latent scales, comma-separated, length must equal latent_channels. Overrides --image_latent_scale if both set.")

    # Audio
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--n_mels", type=int, default=80)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=256)
    p.add_argument("--image_size", type=int, default=256)

    # Runtime
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")

    return p.parse_args()


def classify_media(filename: str) -> Optional[str]:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "voice"
    return None


def safe_ref_name(filename: str, existing: dict) -> str:
    """Strip directory and collisions; return a token-safe @ref suffix.

    Kept restricted to [A-Za-z0-9_.-] so it round-trips through the parser regex.
    """
    base = os.path.basename(filename)
    base = re.sub(r"[^A-Za-z0-9_.\-]+", "_", base)
    base = base.strip("_") or "file"
    if base not in existing:
        return base
    stem, _, ext = base.rpartition(".")
    i = 1
    while True:
        candidate = f"{stem}_{i}.{ext}" if ext else f"{base}_{i}"
        if candidate not in existing:
            return candidate
        i += 1


def encode_image_file(path: str, litevae, image_size: int, device: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - 0.5) / 0.5  # match preprocess normalization (mean=std=0.5)
    with torch.no_grad():
        latent = litevae.encode(tensor.unsqueeze(0).to(device)).mode()
    return latent[0].detach().cpu()  # (C, H', W')


def _audio_load(path: str):
    """(1, T) float32 waveform + sample rate, without depending on torchaudio's backend.

    torchaudio >= 2.9 routes `load` through TorchCodec and raises ImportError when it is
    absent, which broke the Reference voice control on a newer install. soundfile is already
    present transitively (librosa depends on it) and reads wav/flac/ogg directly, so try it
    first and keep torchaudio as the fallback for anything it cannot open.
    """
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)   # (T, C)
        return torch.from_numpy(data).transpose(0, 1).contiguous(), int(sr)
    except Exception:
        import torchaudio as _ta
        w, sr = _ta.load(path)
        return w.float(), int(sr)


def _audio_save(path: str, wav: torch.Tensor, sr: int):
    """Mirror of _audio_load for the same reason."""
    try:
        import soundfile as sf
        sf.write(path, wav.reshape(-1).cpu().numpy(), int(sr))
    except Exception:
        import torchaudio as _ta
        _ta.save(path, wav.reshape(1, -1).cpu(), int(sr))


def encode_voice_file(path: str, sive, shared_window_buffer, args, device: str) -> torch.Tensor:
    waveform, sr = _audio_load(path)
    if sr != args.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, args.sample_rate)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)
    mel = extract_mels(
        shared_window_buffer, waveform,
        sr=args.sample_rate, n_mels=args.n_mels,
        n_fft=args.n_fft, hop_length=args.hop_length,
    )  # (n_mels, T)
    mel = mel.to(device)
    features, _ = sive.extract_features(
        mel.unsqueeze(0),
        lengths=torch.tensor([mel.shape[-1]], device=device),
        layer=args.sive_layer,
    )  # (1, T', encoder_dim)
    features_cf = features.permute(0, 2, 1)[0]  # (encoder_dim, T')
    return features_cf.detach().cpu()


class CosyVoice2Voice:
    """Bundles everything the discrete voice path needs, in BOTH directions.

    Encode (voice -> text): waveform -> FSQ unit ids -> codebook centroids, because
    `voice_inputs` is CENTROID VECTORS upstream of the prelude, never raw ids (see the
    voice_mask_feature comment in world_model.py). Shape contract is identical to the SIVE
    path's (C, T), so nothing downstream changes.

    Decode (text -> voice): generated unit ids -> frozen flow + HiFT -> waveform.

    Speaker conditioning is campplus-192, NOT ECAPA/WavLM: the frozen decoder is conditioned
    on campplus vectors, so any other encoder is off-manifold at decode.
    """

    def __init__(self, args, device: str):
        from megatransformer.model.voice.cosyvoice2_decoder import CosyVoice2Decoder
        from megatransformer.utils.codebook import load_codebook
        if not args.voice_codebook_path:
            raise SystemExit("--voice_cosyvoice2_model_dir requires --voice_codebook_path")
        self.codebook = load_codebook(args.voice_codebook_path)          # (K, D)
        self.K = int(self.codebook.shape[0])
        self.decoder = CosyVoice2Decoder.from_pretrained(
            args.voice_cosyvoice2_model_dir, runtime_dir=args.voice_cosyvoice2_runtime_dir,
            device=device, dtype=torch.float32)
        self.sample_rate = int(getattr(self.decoder, "sample_rate", 24000))
        self._tok = self._spk = None
        self._model_dir = args.voice_cosyvoice2_model_dir
        self._max_frames = int(round(float(args.voice_max_seconds) * 25.0))

    def _lazy_encoders(self):
        # Built on first upload only: the ONNX sessions cost seconds and a TTS-only session
        # never needs them.
        if self._tok is None:
            # Light module on purpose: importing scripts.data.voice.preprocess here pulls
            # datasets/torchcrepe/SIVE and fails on a box that only wants to tokenise audio.
            from megatransformer.utils.cosyvoice2_encoders import (
                CosyVoice2BatchProcessor, CampplusBatchProcessor)
            # CPU on purpose: campplus measured 3.7x SLOWER on GPU, and the FSQ tokenizer is
            # static batch-1, so neither benefits from the device the world model is on.
            self._tok = CosyVoice2BatchProcessor(
                self._model_dir, voice_max_frames=self._max_frames,
                mel_frame_rate=25.0, device="cpu")
            self._spk = CampplusBatchProcessor(self._model_dir)
        return self._tok, self._spk

    def encode(self, waveform_16k: torch.Tensor):
        """(T,) @16 kHz -> (centroids (D, T'), campplus (192,), unit ids (T',))."""
        tok, spk = self._lazy_encoders()
        wl = torch.tensor([waveform_16k.numel()])
        # Both processors return DICTS of padded batch tensors. unit_ids is padded to
        # max_id_frames with zeros, and 0 is a VALID unit id — so the real length must come
        # from feature_lengths, never from a nonzero test.
        tok_out = tok.process_batch([waveform_16k], wl)
        ids = tok_out["unit_ids"][0]
        n = int(tok_out["feature_lengths"][0].item())
        ids = ids[:n].reshape(-1).long()
        emb = spk.process_batch([waveform_16k], wl)["speaker_embeddings"][0]
        ids = ids[(ids >= 0) & (ids < self.K)]
        feats = self.codebook[ids].permute(1, 0).contiguous()            # (D, T')
        return feats, torch.as_tensor(emb).reshape(-1).float(), ids

    def decode(self, unit_ids, speaker_embedding: torch.Tensor,
               prompt_ids=None, prompt_feat=None):
        ids = torch.as_tensor(unit_ids).reshape(-1).long()
        # EOV (id == K) has no codebook row; passing it would index out of bounds.
        ids = ids[(ids >= 0) & (ids < self.K)]
        if ids.numel() == 0:
            return None
        wav = self.decoder.decode(ids, speaker_embedding.reshape(-1).float(),
                                  prompt_ids=prompt_ids, prompt_feat=prompt_feat)
        if wav is None:
            return None
        return self.sample_rate, wav.detach().cpu().numpy()


_PROMPT_MAX_SECONDS = 10.0   # upper bound of the UI slider; encode once, trim per-request


def trim_prompt(prompt_ids, prompt_mel, seconds: float):
    """Cut a stored prompt to `seconds`. Mel is 50 Hz and tokens 25 Hz, so the flow's
    2-mel-frames-per-token contract is preserved by construction."""
    if prompt_ids is None or prompt_mel is None or not seconds or seconds <= 0:
        return None, None
    n_mel = min(int(round(seconds * 50.0)), int(prompt_mel.shape[0]))
    n_tok = min(n_mel // 2, int(prompt_ids.shape[0]))
    if n_tok <= 0:
        return None, None
    return prompt_ids[:n_tok], prompt_mel[:2 * n_tok]


def encode_voice_file_cv2(path: str, cv2: "CosyVoice2Voice", prompt_seconds: float = 0.0):
    """-> (centroids (D,T), campplus (192,), prompt_ids | None, prompt_mel | None)."""
    waveform, sr = _audio_load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # The FSQ tokenizer and campplus read 16 kHz; the flow's prompt mel reads 24 kHz. Keep the
    # original around rather than round-tripping 16k->24k, which would lose the top octave.
    w16 = waveform if sr == 16000 else torchaudio.functional.resample(waveform, sr, 16000)
    feats, spk, ids = cv2.encode(w16.squeeze(0))
    p_ids = p_mel = None
    if prompt_seconds and prompt_seconds > 0:
        n = min(int(prompt_seconds * sr), waveform.shape[-1])
        p_mel = cv2.decoder.prompt_mel(waveform[..., :n], sr)
        # decode() re-trims to 2 mel frames per token; cap here so we never hand it more
        # units than the prompt audio actually covers.
        p_ids = ids[:int(p_mel.shape[0] // 2)]
    return feats, spk, p_ids, p_mel


def render_file_list(state: dict) -> str:
    if not state:
        return "*No files attached — upload or paste images/audio below.*"
    lines = ["**Attached files**"]
    for ref_name, info in state.items():
        mtype = info["type"]
        shape = tuple(info["tensor"].shape)
        lines.append(f"- `@{ref_name}` — {mtype} — {shape}")
    return "\n".join(lines)


def parse_prompt(msg_text: str, state: dict, tokenizer, placeholder_triplet: dict) -> tuple[list[int], list[tuple[str, torch.Tensor]]]:
    """
    Split the message on @ref tokens. For each valid ref, emit the modality
    triplet; otherwise tokenize as text. Returns (token_ids, media_sequence)
    where media_sequence preserves in-order appearance.
    """
    parts = re.split(r"(@[A-Za-z0-9_.\-]+)", msg_text)
    token_ids: list[int] = []
    media_sequence: list[tuple[str, torch.Tensor]] = []
    first_text = True
    for part in parts:
        if part.startswith("@"):
            ref = part[1:]
            # Strip trailing punctuation that isn't part of the filename
            ref = re.sub(r"[.,!?;:]+$", "", ref) if "." not in ref else ref
            if ref in state:
                bo, ph, eo = placeholder_triplet[state[ref]["type"]]
                token_ids.extend([bo, ph, eo])
                media_sequence.append((state[ref]["type"], state[ref]["tensor"]))
                first_text = False  # any prior tokens means we're past the BOS position
                continue
            # Unknown ref — fall through as literal text
        if not part:
            continue
        text_ids = tokenizer.encode(part, add_special_tokens=first_text)
        first_text = False
        token_ids.extend(text_ids)
    return token_ids, media_sequence


def stack_media(media_list: list[torch.Tensor], pad_time_dim: bool, device: str, dtype) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Stack media tensors into (1, N, ...) form. For voice/audio, pad to max
    time and return lengths; for images (fixed shape) return (inputs, None)."""
    if not media_list:
        return None, None
    if not pad_time_dim:
        stacked = torch.stack(media_list, dim=0).unsqueeze(0).to(device=device, dtype=dtype)
        return stacked, None
    max_T = max(t.shape[-1] for t in media_list)
    padded = [F.pad(t, (0, max_T - t.shape[-1])) for t in media_list]
    stacked = torch.stack(padded, dim=0).unsqueeze(0).to(device=device, dtype=dtype)
    lengths = torch.tensor([[t.shape[-1] for t in media_list]], device=device)
    return stacked, lengths


def render_generated_text(
    token_ids: list[int],
    tokenizer,
    sp,
    real_image_count: int = 0,
    real_voice_count: int = 0,
    real_audio_count: int = 0,
) -> str:
    """Decode generated tokens while inserting inline media markers at each EO*.

    The model can sample BO*/EO* tokens directly (they're in-vocab), so the
    token stream may contain spurious EO* tokens that DID NOT correspond to a
    completed media block. The authoritative counts come from the `*_counts`
    tensors. We only emit `[image N]` / `[voice N]` / `[audio N]` markers for
    the first `real_*_count` occurrences of each EO* — later ones are treated
    as noise and stripped entirely.
    """
    emitted = {"image": 0, "voice": 0, "audio": 0}
    caps = {"image": real_image_count, "voice": real_voice_count, "audio": real_audio_count}
    eo_to_label = {sp.EOI: "image", sp.EOV: "voice", sp.EOA: "audio"}
    bo_ids = {sp.BOI, sp.BOV, sp.BOA}
    chunks: list[str] = []
    buf: list[int] = []

    def flush():
        if not buf:
            return
        text_ids = [t for t in buf if t < sp.base and t != 0]
        if text_ids:
            chunks.append(tokenizer.decode(text_ids, skip_special_tokens=True))
        buf.clear()

    for t in token_ids:
        if t in eo_to_label:
            flush()
            label = eo_to_label[t]
            if emitted[label] < caps[label]:
                emitted[label] += 1
                chunks.append(f" [{label} {emitted[label]}] ")
            # else: spurious EO* sampled by text coda — drop it entirely
        elif t in bo_ids:
            flush()  # drop BO* — the model sampled it as a regular token
        else:
            buf.append(t)
    flush()
    return "".join(chunks).strip()


_MODALITY_COLORS = {
    "text": "#4C78A8",
    "voice": "#F58518",
    "audio": "#54A24B",
    "image": "#B279A2",
}


def render_iteration_plot(outputs: dict, cap: Optional[int]):
    """Per-position recurrent iteration counts, coloured by the modality that consumed them.

    The point of the plot is ADAPTIVITY: with a working exit criterion the trunk should
    spend more iterations on positions that need them, so a flat line means the criterion
    is not adapting (which is what the legacy `kl_divergence` produced -- a fixed budget
    with jitter, uncorrelated with difficulty). Positions pinned at the cap are drawn
    hollow: those hit the iteration limit WITHOUT converging, so they were read out
    mid-trajectory and the budget, not the criterion, was the binding constraint.

    Returns None when there is nothing to draw, so the caller can leave the panel empty.
    """
    counts = outputs.get("recurrent_iteration_counts") or []
    mods_per_step = outputs.get("recurrent_step_modalities") or []
    img_iters = (outputs.get("image_recurrent_iterations") or [[]])
    img_iters = img_iters[0] if img_iters else []
    if not counts and not img_iters:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # batch item 0 -- the demo generates one conversation at a time
    mods = [(m[0] if m else "text") for m in mods_per_step]
    if len(mods) < len(counts):
        mods += ["text"] * (len(counts) - len(mods))

    fig, ax = plt.subplots(figsize=(11, 3.4), dpi=110)

    # Token positions, one series per modality so the legend is meaningful.
    for mod in ("text", "voice", "audio"):
        xs = [i for i, m in enumerate(mods) if m == mod]
        if not xs:
            continue
        ys = [counts[i] for i in xs]
        at_cap = [cap is not None and y >= cap for y in ys]
        c = _MODALITY_COLORS[mod]
        # Break the connecting line across gaps. A modality's positions are not contiguous
        # (text resumes after a voice block), and a line drawn straight across the gap reads
        # as a trend through positions this modality never occupied.
        lx, ly = [], []
        for j, (x, y) in enumerate(zip(xs, ys)):
            if j and x != xs[j - 1] + 1:
                lx.append(float("nan")); ly.append(float("nan"))
            lx.append(x); ly.append(y)
        ax.plot(lx, ly, marker="", linewidth=0.8, alpha=0.45, color=c, zorder=1)
        # filled = converged, hollow = hit the cap without converging
        ax.scatter([x for x, a in zip(xs, at_cap) if not a],
                   [y for y, a in zip(ys, at_cap) if not a],
                   s=14, color=c, label=f"{mod} ({len(xs)})", zorder=3)
        ax.scatter([x for x, a in zip(xs, at_cap) if a],
                   [y for y, a in zip(ys, at_cap) if a],
                   s=26, facecolors="none", edgecolors=c, linewidths=1.2, zorder=4)

    # Images are single-shot (one recurrent call over all gen-query positions), so they are
    # not token positions -- draw them past the right edge rather than pretending otherwise.
    if img_iters:
        x0 = max(len(counts), 1) + 1
        xs = [x0 + i * 2 for i in range(len(img_iters))]
        ax.scatter(xs, img_iters, s=46, marker="D", color=_MODALITY_COLORS["image"],
                   label=f"image ({len(img_iters)}, single-shot)", zorder=3)
        for x, y in zip(xs, img_iters):
            ax.annotate(f"{int(y)}", (x, y), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7, color=_MODALITY_COLORS["image"])

    if cap:
        ax.axhline(cap, linestyle="--", linewidth=1.0, color="#888",
                   label=f"cap ({cap})")
        ax.set_ylim(0, cap * 1.12)

    n_cap = sum(1 for y in counts if cap is not None and y >= cap)
    mean = (sum(counts) / len(counts)) if counts else 0.0
    frac = (sum(counts) / (len(counts) * cap)) if (counts and cap) else None
    bits = [f"mean {mean:.1f} iters"]
    if frac is not None:
        bits.append(f"{frac * 100:.0f}% of full budget")
    if cap is not None and counts:
        bits.append(f"{n_cap}/{len(counts)} pinned at cap ({n_cap / len(counts) * 100:.0f}%)")
    ax.set_title("Recurrent iterations per generated position — " + ", ".join(bits), fontsize=10)
    ax.set_xlabel("generated position")
    ax.set_ylabel("iterations")
    ax.grid(alpha=0.25, linewidth=0.6)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    return fig


def decode_image_latent(litevae, latent: torch.Tensor, device: str) -> tuple[Image.Image, dict]:
    """Decode LiteVAE latent → PIL image.

    LiteVAE was trained with preprocessor normalization mean=std=0.5, i.e.
    inputs in [-1, 1]. Its decoder therefore outputs in [-1, 1]. We
    de-normalize `(x + 1) / 2` to reach [0, 1] for display — `clamp(0, 1)`
    alone would silently destroy the entire negative half of the output.

    Also returns raw pixel + latent stats so mis-scaled outputs are visible
    in the status box instead of being papered over by post-decode
    normalization (which is what the training visualization callback does
    via min/max, making OOD outputs look viewable).
    """
    with torch.no_grad():
        z = latent.unsqueeze(0).to(
            device=next(litevae.parameters()).device,
            dtype=next(litevae.parameters()).dtype,
        )
        out = litevae.decode(z)
        pixels = out.sample if hasattr(out, "sample") else out
    raw = pixels[0].float().cpu()
    raw_lat = latent.float().cpu()
    stats = {
        "pixel_min": raw.min().item(),
        "pixel_max": raw.max().item(),
        "pixel_mean": raw.mean().item(),
        "latent_std_per_channel": raw_lat.reshape(raw_lat.shape[0], -1).std(dim=-1).tolist(),
    }
    # De-normalize from [-1, 1] to [0, 1]. Clamp handles any overshoot.
    img = ((raw + 1.0) / 2.0).clamp(0, 1)
    arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr), stats


def decode_voice_latent(latent: torch.Tensor, smg_decoder, vocoder, speaker_embedding: torch.Tensor, sample_rate: int) -> tuple[int, np.ndarray]:
    device = next(smg_decoder.parameters()).device
    dtype = next(smg_decoder.parameters()).dtype
    z = latent.to(device=device, dtype=dtype).unsqueeze(0)
    spk = speaker_embedding.to(device=device, dtype=dtype).unsqueeze(0)
    with torch.no_grad():
        mel = smg_decoder.decode(z=z, speaker_embedding=spk, features=z)
    if isinstance(mel, tuple):
        mel = mel[0]
    if isinstance(mel, dict):
        mel = mel.get("reconstructed", next(iter(mel.values())))
    if mel.dim() == 4:
        mel = mel.squeeze(1)
    mel = mel.to(next(vocoder.parameters()).device)
    with torch.no_grad():
        wav = vocoder(mel)["pred_waveform"]
    if wav.dim() > 1:
        wav = wav.squeeze(0)
    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak
    return sample_rate, wav.cpu().float().numpy()


_AUDIO_PLAYER_POOL = 4


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    from transformers import AutoTokenizer
    # Text tokenizer MUST match the trained spine: SmolLM2 (49152 vocab) when a
    # pretrained text encoder is used, else the historical Mistral tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_tokenizer_name(args=args)
    )
    shared_window_buffer = SharedWindowBuffer()

    print(f"Loading world model from {args.checkpoint_path}...")
    include_modes = [m.strip() for m in args.include_modes.split(",")]
    overrides = {"include_modes": include_modes}
    # M-RoPE carries no weights, so scale_side cannot be detected from the checkpoint and the
    # loader refuses to guess. Rate must match TRAINING (7.5 for the LibriHeavy runs, 6.0 for
    # the older ones) — a mismatch evaluates under a coordinate system the model never saw.
    # mrope_voice_rate / mrope_scale_side are flat world-config fields and ride in overrides.
    # use_mrope is NESTED (recurrent_block_config.block_config), as are the voice head width
    # and feature width, and from_config passes overrides straight to the world config
    # constructor -- so those go through a registered mutated config instead, mirroring
    # visualize.load_world_model. Only engages when the voice/M-RoPE flags are supplied, so
    # the text and image paths are byte-identical without them.
    if args.mrope_scale_side is not None:
        overrides["mrope_scale_side"] = args.mrope_scale_side
    if args.mrope_voice_rate is not None:
        overrides["mrope_voice_rate"] = float(args.mrope_voice_rate)

    config_name = args.config
    if args.voice_codebook_path or (args.mrope_scale_side not in (None, "off")):
        import copy as _copy
        from megatransformer.config.world.world_model import WORLD_MODEL_CONFIGS
        _cfg = _copy.deepcopy(WORLD_MODEL_CONFIGS[args.config])
        if args.mrope_scale_side not in (None, "off"):
            if not model_loading_utils.detect_world_mrope(args.checkpoint_path):
                print("[mrope] WARNING: --mrope_scale_side given but no rotary_global/local "
                      "weights found in the checkpoint — it was NOT trained with M-RoPE.")
            _cfg.recurrent_block_config.block_config.use_mrope = True
            print(f"[mrope] use_mrope=True, scale_side={args.mrope_scale_side}, "
                  f"rate={args.mrope_voice_rate or 6.0}")
        if args.voice_codebook_path:
            from megatransformer.utils.codebook import load_codebook as _lcb
            _cb = _lcb(args.voice_codebook_path)
            _K, _D = int(_cb.shape[0]), int(_cb.shape[1])
            # The unit head is K+1-way (EOV = class K), or K+2 on a bistream checkpoint
            # (fill_token = K+1). Prefer the width actually stored in the checkpoint: the flag
            # that decided it lives in the training CLI, which this script never sees. Getting
            # it wrong shape-mismatches the head, which strict=False then silently DROPS.
            _w = model_loading_utils.detect_voice_unit_vocab_size(args.checkpoint_path)
            _cfg.voice_coda_config.unit_vocab_size = _w if _w is not None else _K + 1
            _cfg.voice_prelude_config.feature_channels = _D
            _cfg.voice_coda_config.feature_channels = _D
            print(f"[voice] discrete head: K={_K}, unit_vocab_size="
                  f"{_cfg.voice_coda_config.unit_vocab_size}, feature_channels={_D}")
        config_name = args.config + "_chat_eval"
        WORLD_MODEL_CONFIGS[config_name] = _cfg
    if args.tie_word_embeddings:
        overrides["tie_word_embeddings"] = True
    if args.text_encoder_model:
        # Mirror training/eval_sdxl_adapter: a pretrained-LLM spine sets
        # special_token_base=native vocab, native eos, + the text_encoder dict.
        # from_config re-runs __post_init__ so the interleaver placeholder ids
        # are re-derived for the new base (BOI/EOI/IPH at base+4/+5/+8).
        from transformers import AutoConfig
        _llm = AutoConfig.from_pretrained(args.text_encoder_model)
        _eos = _llm.eos_token_id
        if _eos is None:
            _eos = tokenizer.eos_token_id
        overrides["special_token_base"] = int(_llm.vocab_size)
        overrides["eos_token_id"] = int(_eos)
        overrides["text_encoder"] = {
            "model": args.text_encoder_model,
            "freeze": True,
            "translator_hidden_mult": 2.0,
            "n_special_tokens": constants.N_SPECIAL_TOKENS,
        }
    model = model_loading_utils.load_model(
        MegaTransformerWorldModel, config_name,
        checkpoint_path=args.checkpoint_path,
        overrides=overrides, device=device,
    )
    model.eval()

    # Resolve control-token ids from the loaded model's actual base, so BO*/PH/EO*
    # match the checkpoint (32000 for Mistral, 49152 for SmolLM2). Everything that
    # emits/detects these ids downstream uses `sp` / `placeholder_triplet`.
    _mcfg = model.module.config if hasattr(model, "module") else model.config
    special_token_base = int(getattr(_mcfg, "special_token_base", constants.SPECIAL_TOKEN_BASE))
    sp = constants.special_token_ids(special_token_base)
    placeholder_triplet = _placeholder_triplet(sp)
    print(f"special_token_base={special_token_base} "
          f"(BOI={sp.BOI}, IPH={sp.IMAGE_PLACEHOLDER}, EOI={sp.EOI})")

    # Apply DiT latent scaling overrides if provided. Mirrors
    # scripts/train/world/training.py:1019-1038 so the same training CLI values
    # can be re-injected at inference for DiffusionBridgeImageDecoder. If the
    # checkpoint already persisted the correct buffer, omit the flags; if it
    # didn't (or you want to override), pass the same values you trained with.
    from megatransformer.model.image.diffusion_decoder import DiffusionBridgeImageDecoder
    if isinstance(getattr(model, "image_generator", None), DiffusionBridgeImageDecoder):
        new_scale = None
        if args.image_latent_channel_scales is not None:
            scales = [float(s) for s in args.image_latent_channel_scales.split(",")]
            n = model.image_generator.config.latent_channels
            if len(scales) != n:
                raise ValueError(f"--image_latent_channel_scales has {len(scales)} values but latent_channels={n}")
            new_scale = torch.tensor(scales, dtype=torch.float)
        elif args.image_latent_scale is not None:
            n = model.image_generator.config.latent_channels
            new_scale = torch.full((n,), float(args.image_latent_scale))
        if new_scale is not None:
            buf = model.image_generator.latent_scale
            new_scale = new_scale.view(1, -1, 1, 1).to(device=buf.device, dtype=buf.dtype)
            buf.copy_(new_scale)
        # Diagnostic: print the active latent_scale so mismatches with training
        # are visible. If this is all-ones and you trained with non-trivial
        # scales, your checkpoint didn't persist the buffer — pass the flag.
        active = model.image_generator.latent_scale.detach().flatten().cpu().tolist()
        print(f"DiT image_latent_scale (active): {[f'{v:.3f}' for v in active]}")

    # --- Optional encoders/decoders ---
    sive = None
    if args.sive_checkpoint_path:
        print(f"Loading SIVE ({args.sive_config}) from {args.sive_checkpoint_path}")
        sive = model_loading_utils.load_model(
            SpeakerInvariantVoiceEncoder, args.sive_config,
            checkpoint_path=args.sive_checkpoint_path,
            device=device,
            overrides={"num_speakers": args.sive_num_speakers},
        )
        sive.eval()

    # Gated on include_modes, not just on the decoder flag: --image_vae_decoder_config
    # defaults to "litevae", so a VOICE-ONLY run used to import olvae and die on a box that
    # has no reason to have it installed. Same for the vocoder below.
    litevae = None
    if args.image_vae_decoder_config == "litevae" and "image" in include_modes:
        print("Loading LiteVAE (encoder + decoder)...")
        from megatransformer.scripts.data.image.vae.preprocess import _load_litevae
        litevae = _load_litevae("litevae", device=device)
        litevae.eval()

    # SDXL-adapter image path: if the model predicts CLIP conditioning (not a
    # latent), load a frozen SDXL pipeline to render pixels. fp16-safe VAE —
    # SDXL's stock VAE decodes to black/NaN in fp16.
    from megatransformer.model.world.world_model import SDXLConditioningAdapter
    sdxl_pipe = None
    sdxl_neg = None
    if isinstance(getattr(model, "image_generator", None), SDXLConditioningAdapter):
        print(f"Image generator is SDXLConditioningAdapter — loading SDXL ({args.sdxl_model})...")
        from diffusers import StableDiffusionXLPipeline, AutoencoderKL, DPMSolverMultistepScheduler
        _sdxl_vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
            args.sdxl_model, vae=_sdxl_vae, torch_dtype=torch.float16, use_safetensors=True).to(device)
        # DPM++ 2M Karras is the project default (crisper than stock Euler at equal steps).
        sdxl_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            sdxl_pipe.scheduler.config, use_karras_sigmas=True)
        sdxl_pipe.set_progress_bar_config(disable=True)
        _neg_pe, _, _neg_pp, _ = sdxl_pipe.encode_prompt(
            prompt="", device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)
        sdxl_neg = (_neg_pe, _neg_pp)

    @torch.no_grad()
    def render_sdxl_cond(seq: torch.Tensor, pool: torch.Tensor, steps: int, seed: int) -> Image.Image:
        """Render the adapter's predicted conditioning (seq 77x2048, pooled 1280) via frozen SDXL."""
        g = torch.Generator(device=device).manual_seed(int(seed))
        neg_pe, neg_pp = sdxl_neg
        return sdxl_pipe(
            prompt_embeds=seq.unsqueeze(0).half(),
            pooled_prompt_embeds=pool.unsqueeze(0).half(),
            negative_prompt_embeds=neg_pe.half(),
            negative_pooled_prompt_embeds=neg_pp.half(),
            num_inference_steps=int(steps), guidance_scale=args.sdxl_guidance,
            height=1024, width=1024, generator=g).images[0]

    # Z-Image-adapter path: if the model predicts Qwen3 conditioning (not CLIP/latent),
    # load frozen Z-Image-Turbo. Only one of sdxl_pipe / zimage_pipe is ever non-None.
    from megatransformer.model.world.world_model import ZImageConditioningAdapter
    zimage_pipe = None
    # Z-Image is a rectified-flow DiT with a fixed FlowMatchEulerDiscreteScheduler and no sampler
    # argument, so the sampler control is inert on this path (it drives the LiteVAE DiT decoder).
    _IS_ZIMAGE = isinstance(getattr(model, "image_generator", None), ZImageConditioningAdapter)
    if _IS_ZIMAGE:
        print(f"Image generator is ZImageConditioningAdapter — loading Z-Image ({args.zimage_model})...")
        from diffusers import ZImagePipeline
        zimage_pipe = ZImagePipeline.from_pretrained(args.zimage_model, torch_dtype=torch.bfloat16)
        _has_accel = True
        try:
            import accelerate  # noqa: F401
        except ImportError:
            _has_accel = False
        _mode = getattr(args, "zimage_offload", "auto")
        if _mode == "on" and not _has_accel:
            raise SystemExit("--zimage_offload on requires accelerate>=0.17 "
                             "(install the `demo` or `image` extra), or pass --zimage_offload off")
        if _mode == "off" or not _has_accel:
            why = "--zimage_offload off" if _mode == "off" else "accelerate not installed"
            print(f"[zimage] no CPU offload ({why}); loading pipeline onto {device}", flush=True)
            zimage_pipe = zimage_pipe.to(device)
        else:
            zimage_pipe.enable_model_cpu_offload()
        zimage_pipe.set_progress_bar_config(disable=True)
        _gen_head = (getattr(model.image_generator, "flow_head", None)
                     or getattr(model.image_generator, "ar_flow_head", None))
        if args.zimage_output_gain != 1.0:
            if _gen_head is not None:
                raise SystemExit(
                    "--zimage_output_gain is a POINT-head correction (it undoes MSE shrinkage by "
                    "1/alpha). This checkpoint has a generative head, which already samples at full "
                    "dispersion -- applying both double-corrects. Use --flow_guidance instead.")
            model.image_generator.output_gain = args.zimage_output_gain
            print(f"[zimage] output_gain={args.zimage_output_gain} (dispersion correction)")
        if _gen_head is not None:
            w = 3.0 if args.flow_guidance is None else float(args.flow_guidance)
            _gen_head.guidance = w
            print(f"[zimage] generative head detected ({type(_gen_head).__name__}); "
                  f"flow_guidance={w}" + ("  (default; unguided would cost ~0.07 CLIPScore)"
                                          if args.flow_guidance is None else ""))
            if args.flow_seed is not None:
                model.image_generator.flow_generator = torch.Generator(device=device).manual_seed(int(args.flow_seed))
                print(f"[zimage] flow_seed={args.flow_seed} (reproducible draws)")
        elif args.flow_guidance is not None:
            print("[zimage] --flow_guidance ignored: this checkpoint has no generative head")

    @torch.no_grad()
    def render_zimage_cond(seq: torch.Tensor, steps: int, seed: int) -> Image.Image:
        """Render the adapter's predicted Qwen3 conditioning (seq, 2560) via frozen Z-Image (8-step, no CFG)."""
        g = torch.Generator(device=device).manual_seed(int(seed))
        return zimage_pipe(
            prompt_embeds=[seq.to(torch.bfloat16)],
            num_inference_steps=int(steps), guidance_scale=0.0,
            height=1024, width=1024, generator=g).images[0]

    smg_decoder = None
    if args.voice_smg_checkpoint_path:
        print(f"Loading voice SMG ({args.voice_smg_config})...")
        from megatransformer.model.smg.smg import SMG
        smg_overrides = {}
        if args.voice_smg_sive_encoder_dim is not None:
            smg_overrides["sive_encoder_dim"] = args.voice_smg_sive_encoder_dim
        smg_decoder = model_loading_utils.load_model(
            SMG, args.voice_smg_config,
            checkpoint_path=args.voice_smg_checkpoint_path,
            device=device, strict=False, overrides=smg_overrides,
        )
        smg_decoder.eval()

    # The CosyVoice 2 path renders with its own flow+HiFT, so HiFiGAN is dead weight there;
    # it is only needed by the legacy SIVE/SMG voice stack.
    vocoder = None
    _need_vocoder = ("voice" in include_modes or "audio" in include_modes) and not args.voice_cosyvoice2_model_dir
    if _need_vocoder and (args.vocoder_config or args.vocoder_checkpoint_path):
        vocoder = model_loading_utils.load_vocoder(
            args.vocoder_checkpoint_path, args.vocoder_config, shared_window_buffer,
        )
        if vocoder is not None:
            vocoder = vocoder.to(device)

    # CosyVoice 2 discrete voice path. When enabled it REPLACES SIVE/SMG/vocoder for voice;
    # set_voice_codebook is what makes generate() emit unit ids (and fixes EOV = K).
    cv2_voice = None
    if args.voice_cosyvoice2_model_dir:
        cv2_voice = CosyVoice2Voice(args, device)
        model.set_voice_codebook(cv2_voice.codebook)
        print(f"[cosyvoice2] decoder + codebook loaded: K={cv2_voice.K}, "
              f"dim={cv2_voice.codebook.shape[1]}, EOV id={cv2_voice.K}, "
              f"sr={cv2_voice.sample_rate}")
        if sive is not None or smg_decoder is not None:
            print("[cosyvoice2] NOTE: SIVE/SMG also supplied; the CV2 path takes precedence "
                  "for voice. Text and image paths are unaffected.")

    # Speaker/prompt live HERE, not in gr.State. Gradio can treat a returned State object
    # that was mutated in place as unchanged and skip the update, so on_submit was reading a
    # stale dict and reporting "no speaker" immediately after the control confirmed one. A
    # closure dict has no such semantics; this UI is single-session anyway.
    speaker_store: dict = {}

    static_speaker_emb = None
    if args.static_speaker_embedding_path:
        static_speaker_emb = torch.load(
            args.static_speaker_embedding_path, map_location="cpu", weights_only=True,
        )

    print("All models loaded.\n")

    # --- Event handlers ---
    def on_files_uploaded(files, msg_text, state):
        if files is None:
            return msg_text, render_file_list(state), state
        state = dict(state) if state else {}
        for f in files:
            path = f.name if hasattr(f, "name") else f
            mtype = classify_media(path)
            if mtype is None:
                gr.Warning(f"Skipping unsupported file: {os.path.basename(path)}")
                continue
            if mtype == "image":
                if litevae is None:
                    gr.Warning("LiteVAE not loaded — cannot accept image uploads")
                    continue
                tensor = encode_image_file(path, litevae, args.image_size, device)
            else:  # voice
                if sive is None:
                    gr.Warning("SIVE not loaded — cannot accept voice uploads")
                    continue
                if cv2_voice is not None:
                    # Encode the LONGEST prompt the slider can ask for; decode trims it.
                    # Doing it the other way round would pin the prompt length to whatever the
                    # slider read at upload time and make the control inert after the fact.
                    tensor, _spk, _pid, _pmel = encode_voice_file_cv2(
                        path, cv2_voice, _PROMPT_MAX_SECONDS)
                    speaker_store.setdefault("prompt_ids", _pid)
                    speaker_store.setdefault("prompt_mel", _pmel)
                    # An uploaded voice also supplies its OWN speaker embedding, which beats
                    # the static one for cloning: campplus of the actual clip rather than a
                    # pinned reference. Last upload wins.
                    speaker_store.setdefault("emb", _spk)
                else:
                    tensor = encode_voice_file(path, sive, shared_window_buffer, args, device)
            ref = safe_ref_name(path, state)
            state[ref] = {"type": mtype, "tensor": tensor, "path": path}
            suffix = f" @{ref}"
            msg_text = (msg_text + suffix) if msg_text else f"@{ref}"
        return msg_text, render_file_list(state), state

    def on_image_pasted(image, msg_text, state):
        """Clipboard-pasted image from the dedicated gr.Image zone."""
        if image is None:
            return msg_text, render_file_list(state), state, None
        if litevae is None:
            gr.Warning("LiteVAE not loaded — cannot accept image paste")
            return msg_text, render_file_list(state), state, None
        state = dict(state) if state else {}
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB").resize((args.image_size, args.image_size), Image.Resampling.LANCZOS)
        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        with torch.no_grad():
            latent = litevae.encode(tensor.unsqueeze(0).to(device)).mode()[0].detach().cpu()
        ref = safe_ref_name(f"pasted_{int(time.time())}.png", state)
        state[ref] = {"type": "image", "tensor": latent, "path": None}
        suffix = f" @{ref}"
        msg_text = (msg_text + suffix) if msg_text else f"@{ref}"
        return msg_text, render_file_list(state), state, None  # clear paste zone

    def on_speaker_ref(path, state):
        """Set the speaker embedding (+ zero-shot prompt) from a reference clip."""
        if not path:
            speaker_store.clear()
            return state, "*No reference voice — voice output cannot be rendered.*"
        if cv2_voice is None:
            return state, "*Reference voice needs the CosyVoice 2 path (--voice_cosyvoice2_model_dir).*"
        try:
            _, spk, pid, pmel = encode_voice_file_cv2(path, cv2_voice, _PROMPT_MAX_SECONDS)
        except Exception as e:
            return state, f"*Reference voice failed: {type(e).__name__}: {e}*"
        speaker_store["emb"] = spk
        speaker_store["prompt_ids"] = pid
        speaker_store["prompt_mel"] = pmel
        secs = (pmel.shape[0] / 50.0) if pmel is not None else 0.0
        return state, (f"**Speaker set** — campplus-192 (norm {spk.norm():.1f}), "
                       f"{secs:.1f}s of prompt available "
                       f"(trimmed by the *prompt seconds* slider at generation).")

    def on_clear(state):
        return ("", render_file_list({}), {}, "", [], [], "", None,
                *[gr.update(value=None, visible=False) for _ in range(_AUDIO_PLAYER_POOL)])

    import tempfile
    wav_tmpdir = tempfile.mkdtemp(prefix="mm_chat_wavs_")

    def on_submit(
        msg_text,
        state,
        gen_hint,
        temperature_in,
        top_p_in,
        top_k_in,
        max_new_tokens_in,
        voice_budget_in,
        audio_budget_in,
        seed_in,
        image_iter_override_in,
        image_steps_in,
        image_sampler_in,
        voice_temp_in,
        voice_ras_win_in,
        voice_ras_tau_in,
        voice_prompt_sec_in,
        exit_criteria_in,
        exit_threshold_in,
        trunk_iters_in,
    ):
        if not msg_text or not msg_text.strip():
            # Must match the outputs wiring: 5 values + one update per player.
            return ("", [], [], "Empty prompt.", None,
                    *[gr.update(value=None, visible=False) for _ in range(_AUDIO_PLAYER_POOL)])
        state = state or {}

        # Normalize UI values into the generate() API. 0 / None / negative
        # means "disabled" for top_p / top_k / seed, matching the underlying
        # generate() contract (Optional sentinels).
        temperature = float(temperature_in) if temperature_in is not None else args.temperature
        top_p = float(top_p_in) if top_p_in and top_p_in > 0.0 else None
        top_k = int(top_k_in) if top_k_in and top_k_in > 0 else None
        max_new_tokens = int(max_new_tokens_in) if max_new_tokens_in else args.max_new_tokens
        from megatransformer.scripts.train.train import media_frame_budget
        _vb = args.voice_token_budget if args.voice_token_budget is not None else media_frame_budget(args, "voice")
        _ab = args.audio_token_budget if args.audio_token_budget is not None else media_frame_budget(args, "audio")
        voice_budget = int(voice_budget_in) if voice_budget_in else _vb
        audio_budget = int(audio_budget_in) if audio_budget_in else _ab
        # 0/negative → "use model default" (mean_thinking_steps + KL early-exit)
        image_iter_override: Optional[int] = None
        if image_iter_override_in is not None and int(image_iter_override_in) > 0:
            image_iter_override = int(image_iter_override_in)
        elif args.image_iteration_override is not None and args.image_iteration_override > 0:
            image_iter_override = args.image_iteration_override

        # Diffusion sampler params. 0/negative on steps → decoder default.
        image_num_steps: Optional[int] = None
        if image_steps_in is not None and int(image_steps_in) > 0:
            image_num_steps = int(image_steps_in)
        elif args.image_num_inference_steps is not None and args.image_num_inference_steps > 0:
            image_num_steps = args.image_num_inference_steps
        image_sampler_choice: str = (image_sampler_in or args.image_sampler or "euler").lower()
        if image_sampler_choice.startswith("n/a"):      # Z-Image placeholder; never reaches the DiT
            image_sampler_choice = "euler"
        if seed_in is not None and int(seed_in) >= 0:
            torch.manual_seed(int(seed_in))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed_in))

        token_ids, media_sequence = parse_prompt(msg_text, state, tokenizer, placeholder_triplet)

        # Optional trailing BO* "nudge" for text→media — the model can still
        # emit multiple media blocks autoregressively regardless of this.
        if gen_hint == "voice":
            token_ids.append(sp.BOV)
        elif gen_hint == "image":
            token_ids.append(sp.BOI)
        elif gen_hint == "audio":
            token_ids.append(sp.BOA)

        voice_tensors = [t for m, t in media_sequence if m == "voice"]
        audio_tensors = [t for m, t in media_sequence if m == "audio"]
        image_tensors = [t for m, t in media_sequence if m == "image"]

        voice_inputs, voice_lengths = stack_media(voice_tensors, pad_time_dim=True, device=device, dtype=dtype)
        audio_inputs, audio_lengths = stack_media(audio_tensors, pad_time_dim=True, device=device, dtype=dtype)
        image_inputs, _ = stack_media(image_tensors, pad_time_dim=False, device=device, dtype=dtype)

        prompt = torch.tensor([token_ids], dtype=torch.long, device=device)
        status_lines = [
            f"Prompt: {len(token_ids)} tokens "
            f"({len(voice_tensors)}v, {len(audio_tensors)}a, {len(image_tensors)}i refs)",
        ]
        # Surface the active latent_scale every call so scale drift is visible.
        try:
            from megatransformer.model.image.diffusion_decoder import DiffusionBridgeImageDecoder
            if isinstance(getattr(model, "image_generator", None), DiffusionBridgeImageDecoder):
                active_scale = model.image_generator.latent_scale.detach().flatten().cpu().tolist()
                status_lines.append(
                    "DiT latent_scale: [" + ", ".join(f"{v:.3f}" for v in active_scale) + "]"
                )
        except Exception:
            pass

        # Report effective sampling params so repro is obvious.
        status_lines.append(
            f"Sampling: T={temperature:.2f}, top_p={top_p}, top_k={top_k}, "
            f"max_new_tokens={max_new_tokens}, voice_budget={voice_budget}, "
            f"audio_budget={audio_budget}, seed={int(seed_in) if seed_in is not None and int(seed_in) >= 0 else 'none'}, "
            f"image_iter_override={image_iter_override if image_iter_override is not None else 'off'}, "
            + (
                # Z-Image is a RECTIFIED-FLOW DiT: ZImagePipeline hard-declares
                # FlowMatchEulerDiscreteScheduler and exposes no sampler argument, so the sampler
                # dropdown controls nothing here (it is for the LiteVAE DiT path). Turbo is also
                # DISTILLED for 8 steps at guidance 0, so the step box has a narrow useful range.
                f"image_sampler=n/a (Z-Image: flow-matching Euler, fixed), "
                f"image_steps={image_num_steps if image_num_steps is not None else args.zimage_gen_steps}"
                f"{'' if image_num_steps is not None else ' (Turbo design point)'}"
                if _IS_ZIMAGE else
                f"image_sampler={image_sampler_choice}, "
                f"image_num_inference_steps={image_num_steps if image_num_steps is not None else 'default'}"
            )
        )

        with torch.no_grad():
            with autocast(device, dtype=dtype, enabled=args.bf16):
                # Voice sampling knobs only exist on the discrete path; passing them on a
                # SIVE checkpoint would be a silent no-op at best.
                _voice_gen_kwargs = {}
                if cv2_voice is not None:
                    # ⚠️ These MUST come from the UI inputs, not from `args`. Until
                    # 2026-09-16 they read args.* -- the command-line defaults fixed at
                    # launch -- so every voice sampling slider in the UI was INERT: temp
                    # 0.0-1.5, RAS win 0-40 and tau 0.0-1.0 all produced byte-identical
                    # sampling, and observed variation between runs was only the random
                    # seed. The status line even printed the slider values back, so it
                    # looked applied. Fall back to args.* only when a control is absent.
                    def _ui(v, fallback):
                        return fallback if v is None else v
                    _voice_gen_kwargs = dict(
                        voice_temperature=float(_ui(voice_temp_in, args.voice_temperature)),
                        voice_ras_win=int(_ui(voice_ras_win_in, args.voice_ras_win) or 0),
                        voice_ras_tau=float(_ui(voice_ras_tau_in, args.voice_ras_tau) or 0.0),
                        voice_min_frame_ratio=args.voice_min_frame_ratio,
                    )
                # RECURRENT EXIT CRITERION + TRUNK DEPTH, set from the UI.
                # This is the single largest quality lever measured on world-voice: the
                # shipped default `kl_divergence` compares post-norm ACTIVATIONS with
                # F.kl_div, a signed quantity, so it exits by sign accident and freezes
                # arbitrary positions mid-computation. Measured ck90000 greedy n=48 x2:
                # LCS 0.7519 at that default vs 0.9103 at full depth ("none").
                #
                # Applied to the block rather than passed to generate() because the
                # criterion is module state, and mean_thinking_steps is what every call
                # site defaults to -- so setting it here covers text, voice AND image
                # rather than only the image override that already existed.
                #
                # `mean_thinking_steps` is misleadingly named: the Poisson log-normal
                # sampling it feeds is TRAINING-only. At eval, n_k_steps() returns it
                # verbatim as (n, k=0) -- i.e. it is the hard ITERATION CAP, and the exit
                # criterion decides whether to stop before reaching it. Confirmed by
                # measurement: the `none` criterion runs at exactly 32.00 iterations.
                try:
                    from megatransformer.model import recurrent_criteria as _rc
                    _blk = model.recurrent_block
                    if not hasattr(_blk, "_ui_default_steps"):
                        _blk._ui_default_steps = _blk.mean_thinking_steps
                    _crit = str(exit_criteria_in or "none")
                    _thr = float(exit_threshold_in or 0) or None
                    if _crit == "none":
                        _blk.exit_criteria = _rc.NoOpCriteria()
                    elif _crit == "logit_kl":
                        _blk.exit_criteria = _rc.LogitKLCriteria(_thr if _thr else 5e-4)
                    elif _crit == "latent_diff":
                        _blk.exit_criteria = _rc.LatentDiffCriteria(_thr if _thr else 0.03)
                    elif _crit == "kl_divergence (legacy, broken)":
                        _blk.exit_criteria = _rc.KLDivergenceCriteria(_thr if _thr else 1e-4)
                    _it = int(trunk_iters_in or 0)
                    _blk.mean_thinking_steps = _it if _it > 0 else _blk._ui_default_steps
                except Exception as _e:
                    print(f"[chat] exit-criteria setup failed ({type(_e).__name__}: {_e}); "
                          f"leaving the model's own criterion in place", flush=True)

                outputs = model.generate(
                    text_input_ids=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    audio_token_budget=audio_budget,
                    voice_token_budget=voice_budget,
                    # SUPPRESS THE BISTREAM FILL TOKEN. A bistream checkpoint has a K+2-wide
                    # unit head, and generate() only bans the fill id while the current chunk
                    # is shorter than max(1, voice_min_chunk_frames) -- i.e. with the default
                    # 0, only at the very first frame. From frame 2 on, a spurious fill can be
                    # sampled during this UNISTREAM generation, and generate() reads a fill
                    # with no further transcript as end-of-utterance: the clip stops
                    # mid-sentence for no visible reason.
                    #
                    # The UI always generates unistream (it has no chunk plan and no
                    # continuation transcript), so the fill is never legitimate here. Setting
                    # the floor to the whole budget bans it for the entire utterance. Inert on
                    # a unistream checkpoint, whose head has no fill id at all.
                    voice_min_chunk_frames=(voice_budget or 0) + 1,
                    audio_inputs=audio_inputs,
                    audio_lengths=audio_lengths,
                    voice_inputs=voice_inputs,
                    voice_lengths=voice_lengths,
                    image_inputs=image_inputs,
                    precomputed_latents=True,
                    image_iteration_override=image_iter_override,
                    image_num_inference_steps=image_num_steps,
                    image_sampler=image_sampler_choice,
                    **_voice_gen_kwargs,
                )

        # Generation stopped because the trunk ran out of context, not because the model
        # chose to stop. Silently truncated output looks identical to a short answer, and an
        # image costs ~64 trunk positions in one jump, so this is easy to hit after one.
        if outputs.get("hit_context_limit"):
            _lim = getattr(
                model.config.recurrent_block_config.block_config,
                "max_position_embeddings", "?")
            status_lines.append(
                f"⚠️ Stopped at the trunk's context limit ({_lim} positions), not at EOS — "
                f"output is TRUNCATED. An image consumes "
                f"~{getattr(model, '_n_image_gen_positions', None) or 64} positions, so a long "
                f"chat plus an image reaches it quickly. Start a new conversation or lower "
                f"max_new_tokens."
            )

        # Authoritative counts from generate() — spurious BO*/EO* sampled by
        # the text coda don't show up here, only completed media blocks do.
        real_img = int(outputs["image_counts"][0].item()) if outputs.get("image_counts") is not None else 0
        real_voice = int(outputs["voice_counts"][0].item()) if outputs.get("voice_counts") is not None else 0
        real_audio = int(outputs["audio_counts"][0].item()) if outputs.get("audio_counts") is not None else 0

        gen_text = ""
        gen_ids = outputs.get("generated_token_ids")
        if gen_ids is not None:
            gen_tokens = gen_ids[0].tolist()
            gen_text = render_generated_text(gen_tokens, tokenizer, sp, real_img, real_voice, real_audio)
            # Compare stream-observed EO* count vs real count to surface
            # spurious sampling by the text coda (a known artifact for
            # undertrained checkpoints that haven't learned to never emit
            # these reserved tokens as regular vocab entries).
            eoi_count = sum(1 for t in gen_tokens if t == sp.EOI)
            eov_count = sum(1 for t in gen_tokens if t == sp.EOV)
            eoa_count = sum(1 for t in gen_tokens if t == sp.EOA)
            spurious = (eoi_count - real_img) + (eov_count - real_voice) + (eoa_count - real_audio)
            status_lines.append(
                f"Token stream: {eoi_count} EOI / {eov_count} EOV / {eoa_count} EOA "
                f"(spurious EO* sampled: {spurious})"
            )

        # Decode every completed image in order
        gallery_images: list[tuple[Image.Image, str]] = []
        image_preds = outputs.get("image_latent_preds")
        image_counts = outputs.get("image_counts")
        image_iters_per_item = outputs.get("image_recurrent_iterations") or []
        # SDXL-adapter path: generate() returns predicted CLIP conditioning
        # (image_clip_cond = List[List[(seq 77x2048, pooled 1280)]]) rather than
        # a latent; render pixels here with the frozen SDXL pipeline.
        image_clip_cond = outputs.get("image_clip_cond")
        if sdxl_pipe is not None and image_clip_cond is not None and image_clip_cond[0]:
            conds = image_clip_cond[0]
            sdxl_steps = image_num_steps if image_num_steps is not None else args.sdxl_gen_steps
            base_seed = int(seed_in) if (seed_in is not None and int(seed_in) >= 0) else 1000
            status_lines.append(f"image_clip_cond: {len(conds)} image(s); SDXL steps={sdxl_steps}, "
                                f"guidance={args.sdxl_guidance}")
            if image_iters_per_item and image_iters_per_item[0]:
                iters_str = ", ".join(str(i) for i in image_iters_per_item[0])
                status_lines.append(f"Image recurrent iterations per block: [{iters_str}]")
            for k, (seq_pred, pool_pred) in enumerate(conds):
                try:
                    img = render_sdxl_cond(seq_pred.float(), pool_pred.float(), sdxl_steps, base_seed + k)
                    gallery_images.append((img, f"image {k + 1}"))
                except Exception as e:
                    status_lines.append(f"Image {k + 1} SDXL render failed "
                                        f"(seq={tuple(seq_pred.shape)}): {type(e).__name__}: {e}")
            status_lines.append(f"Rendered {len(gallery_images)}/{len(conds)} image(s) via SDXL")
        elif zimage_pipe is not None and image_clip_cond is not None and image_clip_cond[0]:
            # Z-Image-adapter path: image_clip_cond = List[List[(seq 2560, None)]] (no pooled).
            conds = image_clip_cond[0]
            z_steps = image_num_steps if image_num_steps is not None else args.zimage_gen_steps
            base_seed = int(seed_in) if (seed_in is not None and int(seed_in) >= 0) else 1000
            status_lines.append(f"image_clip_cond: {len(conds)} image(s); Z-Image steps={z_steps}, guidance=0")
            if image_iters_per_item and image_iters_per_item[0]:
                iters_str = ", ".join(str(i) for i in image_iters_per_item[0])
                status_lines.append(f"Image recurrent iterations per block: [{iters_str}]")
            for k, (seq_pred, _pooled) in enumerate(conds):
                try:
                    img = render_zimage_cond(seq_pred.float(), z_steps, base_seed + k)
                    gallery_images.append((img, f"image {k + 1}"))
                except Exception as e:
                    status_lines.append(f"Image {k + 1} Z-Image render failed "
                                        f"(seq={tuple(seq_pred.shape)}): {type(e).__name__}: {e}")
            status_lines.append(f"Rendered {len(gallery_images)}/{len(conds)} image(s) via Z-Image")
        elif image_preds is not None and image_counts is not None:
            n_img = int(image_counts[0].item())
            status_lines.append(f"image_latent_preds: shape={tuple(image_preds.shape) if image_preds.numel() else 'empty'}, counts[0]={n_img}")
            if image_iters_per_item and image_iters_per_item[0]:
                iters_str = ", ".join(str(i) for i in image_iters_per_item[0])
                status_lines.append(f"Image recurrent iterations per block: [{iters_str}]")
            if litevae is None:
                status_lines.append("LiteVAE not loaded — skipping image decode")
            else:
                for k in range(n_img):
                    latent = image_preds[0, k]
                    try:
                        img, stats = decode_image_latent(litevae, latent, device)
                        gallery_images.append((img, f"image {k + 1}"))
                        # Report raw pixel range + latent per-channel std so
                        # scale mismatches are visible. If pixel_max/min are
                        # wildly outside [-1, 1], the DiT output is in scaled
                        # space (latent_scale wasn't applied); if the latent
                        # per-channel std deviates strongly from your
                        # training LiteVAE-measured std, the scales don't
                        # match the checkpoint.
                        lat_std = stats["latent_std_per_channel"]
                        lat_std_str = "[" + ", ".join(f"{v:.2f}" for v in lat_std) + "]"
                        status_lines.append(
                            f"Image {k + 1}: pixel range [{stats['pixel_min']:+.2f}, "
                            f"{stats['pixel_max']:+.2f}] (mean {stats['pixel_mean']:+.2f}); "
                            f"latent std/channel {lat_std_str}"
                        )
                    except Exception as e:
                        status_lines.append(f"Image {k + 1} decode failed (shape={tuple(latent.shape)}): {type(e).__name__}: {e}")
                status_lines.append(f"Decoded {len(gallery_images)}/{n_img} image(s)")

        # Decode every completed voice in order and save as .wav files
        voice_wav_paths: list[str] = []
        voice_preds = outputs.get("voice_latent_preds")
        voice_counts = outputs.get("voice_counts")
        voice_lens_out = outputs.get("voice_lengths")

        if cv2_voice is not None:
            # DISCRETE path. generate() emits per-utterance id segments; prefer them over the
            # flat trace, which concatenates every block and would render several utterances
            # as one clip.
            segs = outputs.get("voice_unit_id_segments")
            if segs and segs[0]:
                id_blocks = [list(x) for x in segs[0]]
            else:
                _tr = (outputs.get("voice_unit_id_trace") or [[]])[0]
                id_blocks = [list(_tr)] if len(_tr) else []
            spk = speaker_store.get("emb")
            if spk is None:
                spk = static_speaker_emb
            if id_blocks and spk is None:
                status_lines.append(
                    f"⚠️ {len(id_blocks)} voice block(s) WERE generated but could not be "
                    f"rendered: no speaker. The frozen CosyVoice 2 decoder needs a campplus-192 "
                    f"vector to know who is speaking. Set one with the **Reference voice** "
                    f"control (right-hand panel), or launch with "
                    f"--static_speaker_embedding_path.")
            elif id_blocks:
                if spk.reshape(-1).numel() != 192:
                    status_lines.append(
                        f"⚠️ speaker embedding is {spk.reshape(-1).numel()}-d, but the CosyVoice 2 "
                        f"decoder expects campplus-192. Output will be off-manifold.")
                for k, ids in enumerate(id_blocks):
                    try:
                        _pid, _pmel = trim_prompt(
                            speaker_store.get("prompt_ids"),
                            speaker_store.get("prompt_mel"),
                            float(voice_prompt_sec_in))
                        got = cv2_voice.decode(ids, spk, prompt_ids=_pid, prompt_feat=_pmel)
                        if got is None:
                            status_lines.append(f"Voice {k + 1}: no renderable units")
                            continue
                        sr, wav_np = got
                        path = os.path.join(wav_tmpdir, f"voice_{int(time.time() * 1000)}_{k}.wav")
                        _audio_save(path, torch.from_numpy(wav_np), sr)
                        voice_wav_paths.append(path)
                    except Exception as e:
                        status_lines.append(f"Voice {k + 1} decode failed: {e}")
                _ps = float(voice_prompt_sec_in)
                _has_prompt = speaker_store.get("prompt_ids") is not None and _ps > 0
                status_lines.append(
                    f"Decoded {len(voice_wav_paths)} voice clip(s) via CosyVoice 2 "
                    f"(T={float(voice_temp_in):g}, RAS win={int(voice_ras_win_in or 0)}, "
                    f"prompt={f'{_ps:g}s' if _has_prompt else 'off (embedding only)'})")
        elif voice_preds is not None and voice_counts is not None:
            n_voice = int(voice_counts[0].item())
            if n_voice and (smg_decoder is None or vocoder is None):
                status_lines.append(f"{n_voice} voice block(s) generated but SMG/vocoder missing — skipping decode")
            elif n_voice and static_speaker_emb is None:
                status_lines.append(f"{n_voice} voice block(s) generated but no --static_speaker_embedding_path — skipping decode")
            elif n_voice:
                for k in range(n_voice):
                    latent = voice_preds[0, k]
                    if voice_lens_out is not None:
                        T = int(voice_lens_out[0, k].item())
                        latent = latent[:, :T]
                    try:
                        sr, wav_np = decode_voice_latent(
                            latent, smg_decoder, vocoder, static_speaker_emb, args.sample_rate,
                        )
                        path = os.path.join(wav_tmpdir, f"voice_{int(time.time() * 1000)}_{k}.wav")
                        _audio_save(path, torch.from_numpy(wav_np), sr)
                        voice_wav_paths.append(path)
                    except Exception as e:
                        status_lines.append(f"Voice {k + 1} decode failed: {e}")
                status_lines.append(f"Decoded {n_voice} voice clip(s)")

        # Audio output is diagnostic — no dedicated decoder wired; report count
        audio_counts = outputs.get("audio_counts")
        if audio_counts is not None and int(audio_counts[0].item()):
            status_lines.append(f"{int(audio_counts[0].item())} audio block(s) generated (no audio decoder wired)")

        # Stop-head logit traces — raw signal diagnostic. A "healthy" trace
        # should trend upward over the voice/audio frames and cross 0 (where
        # sigmoid=0.5) somewhere near the end. Flat at ~-5 means the stop head
        # never fired; monotonic rise without crossing 0 means it's trying
        # but distributionally off; late crossing near the budget means it's
        # positional, not content-based.
        def _summarize_trace(trace: list[float], label: str) -> Optional[str]:
            if not trace:
                return None
            n = len(trace)
            mn, mx, last = min(trace), max(trace), trace[-1]
            crossed = next((i for i, v in enumerate(trace) if v > 0.0), None)
            cross_str = f"crossed 0 at frame {crossed}" if crossed is not None else "never crossed 0"
            # Sample 8 evenly-spaced frames for a rough shape preview
            stride = max(1, n // 8)
            sample = [f"{i}:{trace[i]:+.2f}" for i in range(0, n, stride)][:8]
            return (
                f"{label} stop_logits: n={n}, min={mn:+.2f}, max={mx:+.2f}, "
                f"last={last:+.2f}, {cross_str}; samples [" + ", ".join(sample) + "]"
            )
        voice_trace = outputs.get("voice_stop_logit_trace", [[]])[0]
        audio_trace = outputs.get("audio_stop_logit_trace", [[]])[0]
        for line in (_summarize_trace(voice_trace, "voice"), _summarize_trace(audio_trace, "audio")):
            if line:
                status_lines.append(line)

        # One update per pooled player: fill the first N, hide the rest.
        _players = [
            gr.update(value=(voice_wav_paths[i] if i < len(voice_wav_paths) else None),
                      visible=(i < len(voice_wav_paths)))
            for i in range(_AUDIO_PLAYER_POOL)
        ]
        # Iteration plot. `cap` is what the trunk was actually allowed this run -- the UI's
        # trunk-iterations box overrides mean_thinking_steps, so read it back off the block
        # rather than assuming the config default, or the cap line lands in the wrong place.
        try:
            _cap = int(getattr(model.recurrent_block, "mean_thinking_steps", 0)) or None
        except Exception:
            _cap = None
        try:
            _iter_fig = render_iteration_plot(outputs, _cap)
        except Exception as _e:
            print(f"[chat] iteration plot failed ({type(_e).__name__}: {_e})", flush=True)
            _iter_fig = None

        return (gen_text, gallery_images, voice_wav_paths,
                "\n".join(status_lines), _iter_fig, *_players)

    # --- UI ---
    with gr.Blocks(title="MegaTransformer Multimodal Chat") as demo:
        gr.Markdown(
            "# MegaTransformer multimodal chat\n"
            "Upload images (PNG/JPG/…) or audio (WAV/MP3/…) — a `@filename.ext` reference is "
            "appended to the prompt. Move it anywhere in the text by cut/paste; on submit each "
            "`@ref` is interleaved inline with the corresponding media tensor."
        )

        state = gr.State({})

        with gr.Row():
            with gr.Column(scale=3):
                msg_box = gr.Textbox(
                    lines=6, label="Prompt",
                    placeholder="Type a message. Uploaded files become @refs that you can move around.",
                )
                with gr.Row():
                    gen_hint = gr.Radio(
                        choices=["text", "voice", "image", "audio"],
                        value="text",
                        label="Output modality (appends a trailing BO* to coerce the pretrained model into emitting that modality; post-finetune this becomes optional)",
                    )
                    submit_btn = gr.Button("Generate", variant="primary")
                    clear_btn = gr.Button("Clear")

                with gr.Accordion("Sampling parameters", open=False):
                    with gr.Row():
                        # TEXT ONLY. These three reach generate() as `temperature`/`top_p`/
                        # `top_k`, which are consumed by _sample_tokens() at both of its call
                        # sites -- and both sample from the TEXT coda's logits. Voice uses the
                        # separate voice_* controls below; image uses its own diffusion steps
                        # and sampler. Naming them bare invited the reading that they were
                        # global.
                        temperature_slider = gr.Slider(
                            minimum=0.05, maximum=2.0, step=0.05,
                            value=args.temperature,
                            label="TEXT temperature (text tokens only; 0 = greedy)",
                        )
                        top_p_slider = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.01,
                            value=args.top_p,
                            label="TEXT top_p (text tokens only; 0 = off)",
                        )
                        top_k_slider = gr.Slider(
                            minimum=0, maximum=500, step=1,
                            value=0,
                            label="TEXT top_k (text tokens only; 0 = off)",
                        )
                    with gr.Row():
                        # VOICE unit sampler -- separate from the text sampler above. Measured
                        # at ck78000 (n=48): T=0.6 is the only setting with 0% collapsed
                        # clips, greedy (0.0) wins length-tracking (hit 89.6% vs 83.3%,
                        # duration r 0.741 vs 0.702) but cuts off more. RAS is LOAD-BEARING,
                        # not a refinement: the free-running argmax is "repeat the previous
                        # unit" ~99% of the time, so at win=0 the model drones.
                        voice_temp_slider = gr.Slider(
                            minimum=0.0, maximum=1.5, step=0.05,
                            value=args.voice_temperature,
                            label="voice temperature (0 = greedy, BEST measured; needs RAS win > 0)",
                        )
                        voice_ras_win_slider = gr.Slider(
                            minimum=0, maximum=40, step=1, value=args.voice_ras_win,
                            label="voice RAS window (0 = OFF -> drones; 10 = CosyVoice 2)",
                        )
                        voice_ras_tau_slider = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.05, value=args.voice_ras_tau,
                            label="voice RAS tau",
                        )
                        voice_prompt_sec_slider = gr.Slider(
                            minimum=0.0, maximum=10.0, step=0.5,
                            value=args.voice_prompt_seconds,
                            label="prompt seconds from upload (0 = embedding only)",
                        )
                    with gr.Row():
                        max_new_tokens_num = gr.Number(
                            value=args.max_new_tokens, precision=0,
                            label="max_new_tokens",
                        )
                        voice_budget_num = gr.Number(
                            value=args.voice_token_budget, precision=0,
                            label="voice_token_budget",
                        )
                        audio_budget_num = gr.Number(
                            value=args.audio_token_budget, precision=0,
                            label="audio_token_budget",
                        )
                        seed_num = gr.Number(
                            value=-1, precision=0,
                            label="seed (-1 = random)",
                        )
                        image_iter_override_num = gr.Number(
                            value=(args.image_iteration_override or 0),
                            precision=0,
                            label="IMAGE trunk iterations (0 = off; overrides the global cap for image only)",
                        )
                    with gr.Row():
                        exit_criteria_dd = gr.Dropdown(
                            choices=["none", "logit_kl", "latent_diff",
                                     "kl_divergence (legacy, broken)"],
                            value="none",
                            label="recurrent exit criterion  (none = full depth, BEST measured)",
                        )
                        exit_threshold_num = gr.Number(
                            value=0.0,
                            label="exit threshold (0 = criterion default; logit_kl 5e-4, latent_diff 0.03)",
                        )
                        trunk_iters_num = gr.Number(
                            value=8, precision=0,
                            label="max trunk iterations (8 = saturation, BEST measured; 0 = model default)",
                        )
                    with gr.Row():
                        image_steps_num = gr.Number(
                            value=(args.image_num_inference_steps or 0),
                            precision=0,
                            label="image diffusion steps (0 = decoder default)",
                        )
                        # inert on the Z-Image path -- greyed out rather than silently ignored
                        image_sampler_dd = gr.Dropdown(
                            choices=(["n/a (Z-Image: flow-matching Euler, fixed)"] if _IS_ZIMAGE
                                     else ["euler", "heun", "midpoint"]),
                            value=("n/a (Z-Image: flow-matching Euler, fixed)" if _IS_ZIMAGE
                                   else args.image_sampler),
                            interactive=(not _IS_ZIMAGE),
                            label=("image sampler — not used by Z-Image" if _IS_ZIMAGE
                                   else "image diffusion sampler (heun/midpoint = 2× NFE/step)"),
                        )

                status_box = gr.Textbox(label="Status", interactive=False, lines=3)
                # Adaptive-compute readout. Flat = the criterion is not adapting; hollow
                # markers = positions that hit the cap without converging (budget-limited,
                # not criterion-limited). See render_iteration_plot.
                out_iter_plot = gr.Plot(label="Recurrent iterations per position")

                gr.Markdown("### Outputs")
                out_text = gr.Textbox(
                    label="Generated text (with [image N] / [voice N] / [audio N] markers in position)",
                    lines=5,
                )
                # Show a 1024x1024 render at NATIVE resolution -- no larger, no smaller.
                # Three settings, each fixing a different way this goes wrong:
                #   columns=1      at columns=3 each cell is ~1/3 of the enclosing scale=3
                #                  column (~460px on a 1920px window), so the render is
                #                  downscaled by WIDTH no matter how tall the box is.
                #   scale-down     "contain" fits the box in BOTH directions, which ENLARGES
                #                  a 1024 image when the container is bigger -- on a 1440p
                #                  screen it grew to fill the viewport. "scale-down" is
                #                  contain-but-never-upscale, so 1024 renders at 1024.
                #   preview=False  the preview pane sizes to its own content and ignores
                #                  `height`. allow_preview defaults True, so clicking an
                #                  image still opens it enlarged.
                # height is 1024 plus room for the caption strip. Multiple images stack and
                # scroll -- inherent to native size, and preferable to silent downscaling.
                out_gallery = gr.Gallery(label="Generated images", columns=1, height=1040,
                                         object_fit="scale-down", preview=False)
                # PLAYBACK + DOWNLOAD. gr.Files is download-only -- it renders a file list
                # with no transport, so clips could not be auditioned without saving them
                # first. gr.Audio plays exactly one file, so a fixed pool is used and the
                # unused slots are hidden. The pool is capped; gr.Files below still carries
                # EVERY clip, so nothing is lost when a response exceeds the pool.
                out_audio_players = [
                    gr.Audio(label=f"voice clip {i + 1}", visible=False,
                             type="filepath", interactive=False)
                    for i in range(_AUDIO_PLAYER_POOL)
                ]
                out_audio_files = gr.Files(label="Generated voice clips (.wav) — download all")

            with gr.Column(scale=1):
                file_list_md = gr.Markdown(render_file_list({}))
                upload = gr.File(
                    file_count="multiple", label="Attach files",
                    file_types=[f".{e}" for e in list(IMAGE_EXTS) + list(AUDIO_EXTS)],
                )
                paste_zone = gr.Image(
                    sources=["clipboard", "upload"], type="pil",
                    label="Paste image here (Cmd/Ctrl-V)",
                    height=180,
                )
                # SEPARATE from "Attach files": attaching a voice clip inserts an @ref and
                # feeds it to the model as transcription INPUT. This one only sets who the
                # output sounds like, and never touches the prompt. Without it (and without
                # --static_speaker_embedding_path) the frozen decoder has no speaker to render
                # as, so generated units cannot be turned into audio at all.
                speaker_ref = gr.Audio(
                    sources=["upload", "microphone"], type="filepath",
                    label="Reference voice (sets the speaker — required to hear output)",
                )
                speaker_ref_md = gr.Markdown("*No reference voice — voice output cannot be rendered.*")

        speaker_ref.change(
            on_speaker_ref,
            inputs=[speaker_ref, state],
            outputs=[state, speaker_ref_md],
        )
        upload.upload(
            on_files_uploaded,
            inputs=[upload, msg_box, state],
            outputs=[msg_box, file_list_md, state],
        )
        paste_zone.change(
            on_image_pasted,
            inputs=[paste_zone, msg_box, state],
            outputs=[msg_box, file_list_md, state, paste_zone],
        )
        submit_btn.click(
            on_submit,
            inputs=[
                msg_box, state, gen_hint,
                temperature_slider, top_p_slider, top_k_slider,
                max_new_tokens_num, voice_budget_num, audio_budget_num,
                seed_num, image_iter_override_num,
                image_steps_num, image_sampler_dd,
                voice_temp_slider, voice_ras_win_slider,
                voice_ras_tau_slider, voice_prompt_sec_slider,
                exit_criteria_dd, exit_threshold_num, trunk_iters_num,
            ],
            outputs=[out_text, out_gallery, out_audio_files, status_box,
                     out_iter_plot, *out_audio_players],
        )
        clear_btn.click(
            on_clear,
            inputs=[state],
            outputs=[msg_box, file_list_md, state, out_text, out_gallery,
                     out_audio_files, status_box, out_iter_plot, *out_audio_players],
        )

    demo.launch(share=args.share, server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
