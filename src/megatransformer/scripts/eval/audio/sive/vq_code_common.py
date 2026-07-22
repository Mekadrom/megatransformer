"""Shared plumbing for the VQ-code structure probes.

The three probes (code_diversity / code_redundancy / code_tail_profile) all need the same
thing: load a SIVE checkpoint FAITHFULLY, run val utterances through it, and keep each
utterance's VQ code sequence (trimmed to valid frames) alongside its mel.

Kept in one place because the loading is the part that silently goes wrong: load_model runs
strict=False/allow_size_mismatch=True, so a mis-specified VQ variant does not raise -- it
quietly yields a random codebook and confident nonsense. See detect_sive_variant.
"""
import argparse

import torch
from tqdm import tqdm

from megatransformer.model.voice.sive.sive import SpeakerInvariantVoiceEncoder
from megatransformer.scripts.data.voice.dataset import VoiceShardedDataset
from megatransformer.utils.audio_utils import SharedWindowBuffer, extract_mels
from megatransformer.utils.model_loading_utils import detect_sive_variant, load_model


def add_common_args(ap: argparse.ArgumentParser):
    ap.add_argument("--checkpoint", action="append", default=[], metavar="name=path",
                    help="repeatable; 'name=path' to contrast several checkpoints in one run")
    ap.add_argument("--config", default="small_deep_3xdownsample_conv2d_attentive")
    ap.add_argument("--num_speakers", type=int, default=3610)
    ap.add_argument("--val_cache_dir", default="./cached_datasets/voice_sive_gender_val_merged/")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_utts", type=int, default=150, help="utterances to scan (shard order)")
    ap.add_argument("--min_frames", type=int, default=16,
                    help="skip utterances shorter than this many FEATURE frames")
    ap.add_argument("--min_waveform", type=int, default=8000, help="skip waveforms shorter than this")
    ap.add_argument("--shard_cache_size", type=int, default=2)
    ap.add_argument("--vq_cosine", action="store_true",
                    help="checkpoint was trained with cosine-distance VQ. CANNOT be auto-detected "
                         "(no weights of its own) and it CHANGES the code assignments -- pass it "
                         "for any --vq_cosine run or the codes read here are not the codes trained.")
    ap.add_argument("--final_norm_type", default=None,
                    help="match a non-default final norm (e.g. rmsnorm/none) or features are garbage")
    ap.add_argument("--voice_sample_rate", type=int, default=16000)
    ap.add_argument("--voice_n_mels", type=int, default=80)
    ap.add_argument("--voice_n_fft", type=int, default=1024)
    ap.add_argument("--voice_hop_length", type=int, default=256)
    return ap


def parse_checkpoints(args):
    """-> list of (name, path). Accepts 'name=path' or a bare path."""
    out = []
    for spec in args.checkpoint:
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            name, path = spec.rstrip("/").split("/")[-1], spec
        out.append((name, path))
    if not out:
        raise SystemExit("no --checkpoint given")
    return out


def load_sive(args, ckpt_path):
    """Load a SIVE checkpoint with the VQ variant auto-detected from its own weights."""
    overrides = {"num_speakers": args.num_speakers}
    overrides.update(detect_sive_variant(ckpt_path))
    if args.vq_cosine:
        overrides["vq_cosine"] = True
        print("  --vq_cosine set: quantizing by cosine distance (normalized features+codebook)")
    elif overrides.get("use_vq"):
        print("  [note] assuming NON-cosine VQ; pass --vq_cosine if this checkpoint used it")
    if args.final_norm_type is not None:
        overrides["final_norm_type"] = args.final_norm_type
    model = load_model(SpeakerInvariantVoiceEncoder, args.config, checkpoint_path=ckpt_path,
                       device=args.device, overrides=overrides,
                       strict=False, allow_size_mismatch=True)
    if not getattr(model.config, "use_vq", False):
        raise SystemExit(f"{ckpt_path} has no VQ codebook -- these probes read discrete codes")
    return model.eval()


@torch.no_grad()
def code_sequences(model, args, want_mel=False):
    """Run val utterances through the model; return a list of dicts.

    Each entry: {"codes": LongTensor[T_feat] (valid frames only),
                 "mel": ndarray[n_mels, T_mel] if want_mel else None}
    Utterances are visited in shard order (sequential index) so the shard LRU cache is not
    thrashed -- an unbounded out-of-order scan is what previously hung the SIVE eval callback.
    """
    dataset = VoiceShardedDataset(shard_dir=args.val_cache_dir, cache_size=args.shard_cache_size,
                                  columns=["waveforms", "mel_specs", "speaker_ids", "gender_ids"])
    buf = SharedWindowBuffer()
    out = []
    n = min(args.n_utts, len(dataset)) if args.n_utts > 0 else len(dataset)
    for i in tqdm(range(n), desc="scanning utterances"):
        s = dataset[i]
        if "mel_spec" in s:
            mel, L = s["mel_spec"], int(s["mel_length"])
            mel = mel[:, :L]
        else:
            wl = int(s["waveform_length"])
            if wl < args.min_waveform:
                continue
            mel = extract_mels(buf, s["waveform"][:wl].to(torch.float32),
                               sr=args.voice_sample_rate, n_mels=args.voice_n_mels,
                               n_fft=args.voice_n_fft, hop_length=args.voice_hop_length)
            L = mel.shape[-1]
        mel_d = mel.to(args.device)
        res = model(mel_d.unsqueeze(0), lengths=torch.tensor([L], device=args.device), grl_alpha=0.0)
        if res.get("vq_indices") is None:
            continue
        fl = int(res["feature_lengths"][0])
        if fl < args.min_frames:
            continue
        out.append({"codes": res["vq_indices"][0, :fl].cpu(),
                    "mel": mel[:, :L].float().cpu().numpy() if want_mel else None})
    if not out:
        raise SystemExit("no usable utterances -- check --val_cache_dir / --min_frames")
    return out
