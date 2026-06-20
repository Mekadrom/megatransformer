"""Evaluate image transcription (image → text) using CLIPScore.

For each image in the evaluation dataset, generates a text caption from
the world model and scores it against the original image using CLIP
image-text cosine similarity.

Usage:
    python -m megatransformer.scripts.eval.world.eval_image_transcription --checkpoint_path runs/my_run/checkpoint-3000 --config small_sum_dit --cache_dir ../cached_datasets/sive --include_modes text,image --max_samples 100 --bf16
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.amp import autocast

from megatransformer.model.world.world_model import MegaTransformerWorldModel
from megatransformer.utils import model_loading_utils
from megatransformer.utils.constants import (
    BOI_TOKEN_ID, EOI_TOKEN_ID, IMAGE_PLACEHOLDER_TOKEN_ID, EOS_TOKEN_ID,
)


def parse_args():
    p = argparse.ArgumentParser(description="Image transcription eval (CLIPScore)")
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--config", type=str, default="small_sum_dit")
    p.add_argument("--include_modes", type=str, default="text,image")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--text_cache_dir", type=str, default=None)
    p.add_argument("--image_cache_dir", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--use_memorization_dataset", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--tie_word_embeddings", action="store_true")
    p.add_argument("--clip_model", type=str, default="ViT-B-32",
                   help="OpenCLIP model name")
    p.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k",
                   help="OpenCLIP pretrained weights")
    p.add_argument("--image_vae_decoder_config", type=str, default=None,
                   help="'litevae' to load LiteVAE for decoding latents to pixels")
    p.add_argument("--image_vae_decoder_path", type=str, default=None)
    p.add_argument("--split", type=str, default="val", help="Dataset split (train/val)")
    p.add_argument("--log_dir", type=str, default=None, help="TensorBoard log dir for metrics")
    p.add_argument("--step", type=int, default=None, help="Step number (inferred from checkpoint path if omitted)")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_world_model(args, device):
    include_modes = [m.strip() for m in args.include_modes.split(",")]
    overrides = {"include_modes": include_modes}
    if args.tie_word_embeddings:
        overrides["tie_word_embeddings"] = True
    return model_loading_utils.load_model(
        MegaTransformerWorldModel, args.config,
        checkpoint_path=args.checkpoint_path,
        overrides=overrides, device=device,
    )


def load_dataset(args, split="val"):
    include_modes = [m.strip() for m in args.include_modes.split(",")]

    def resolve(specific, base, s):
        d = specific or base
        if d is None:
            return None
        for candidate in [d + "_" + s, d]:
            if os.path.isdir(candidate):
                return candidate
        return None

    text_dir = resolve(args.text_cache_dir, args.cache_dir, split) if "text" in include_modes else None
    image_dir = resolve(args.image_cache_dir, args.cache_dir, split) if "image" in include_modes else None

    if args.use_memorization_dataset:
        from megatransformer.scripts.data.world.memorization_dataset import MultimodalMemorizationDataset
        return MultimodalMemorizationDataset(
            text_shard_dir=text_dir, image_shard_dir=image_dir,
            max_samples=args.max_samples,
        )
    else:
        from megatransformer.scripts.data.world.dataset import MultimodalShardedDataset
        return MultimodalShardedDataset(
            text_shard_dir=text_dir, image_shard_dir=image_dir,
            cache_size=32, max_samples=args.max_samples,
        )


def load_image_decoder(args):
    if args.image_vae_decoder_config == "litevae":
        from megatransformer.scripts.data.image.vae.preprocess import _load_litevae
        decoder = _load_litevae("litevae", device="cpu")
        decoder.eval()
        return decoder
    elif args.image_vae_decoder_path:
        from megatransformer.model.image.vae.vae import ImageVAEDecoder
        decoder = model_loading_utils.load_model(
            ImageVAEDecoder, args.image_vae_decoder_config or "small",
            checkpoint_path=args.image_vae_decoder_path, strict=False,
        )
        decoder.eval()
        return decoder
    return None


def decode_latent_to_pixels(decoder, latent, device):
    """Decode VAE latent (C, H, W) → pixel image (3, H', W') in [0, 1]."""
    with torch.no_grad():
        z = latent.unsqueeze(0).to(device=next(decoder.parameters()).device,
                                    dtype=next(decoder.parameters()).dtype)
        if hasattr(decoder, 'decode'):
            out = decoder.decode(z)
            pixels = out.sample if hasattr(out, 'sample') else out
        else:
            pixels = decoder(z)
        return pixels[0].float().clamp(0, 1).cpu()


def decode_tokens(token_ids, tokenizer):
    text_ids = [t for t in token_ids if t < 32000 and t != 0]
    return tokenizer.decode(text_ids, skip_special_tokens=True)


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    # Load CLIP
    print("Loading CLIP model...")
    import open_clip
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained, device=device,
    )
    clip_tokenizer = open_clip.get_tokenizer(args.clip_model)
    clip_model.eval()

    # Load tokenizer for decoding generated text
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    # Load image VAE decoder (latents → pixels for CLIP)
    image_decoder = load_image_decoder(args)
    if image_decoder is None:
        print("ERROR: need --image_vae_decoder_config litevae or --image_vae_decoder_path to decode latents for CLIP")
        sys.exit(1)

    # Load world model
    print(f"Loading world model from {args.checkpoint_path}...")
    model = load_world_model(args, device)
    model.to(device)
    model.eval()

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args, split=args.split)
    print(f"Dataset: {len(dataset)} samples")

    # Eval loop
    clip_scores = []
    from torchvision import transforms
    # CLIP expects 224x224 normalized images
    clip_normalize = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    for i in range(len(dataset)):
        sample = dataset[i]
        if "image_image" not in sample:
            continue

        image_latent = sample["image_image"]  # (C, H, W) latent
        text_token_ids = sample.get("text_token_ids")
        if text_token_ids is None:
            continue

        # Build transcription prompt: [BOI] [IMAGE_PH] [EOI]
        prompt = torch.tensor(
            [[BOI_TOKEN_ID, IMAGE_PLACEHOLDER_TOKEN_ID, EOI_TOKEN_ID]],
            dtype=torch.long, device=device,
        )
        image_input = image_latent.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, C, H, W)

        # Generate caption
        with torch.no_grad():
            with autocast(device, dtype=dtype, enabled=args.bf16):
                outputs = model.generate(
                    text_input_ids=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    image_inputs=image_input,
                    precomputed_latents=True,
                )

        gen_ids = outputs.get("generated_token_ids")
        if gen_ids is None:
            continue
        caption = decode_tokens(gen_ids[0].tolist(), tokenizer)

        # Decode image latent to pixels for CLIP
        pixels = decode_latent_to_pixels(image_decoder, image_latent, device)  # (3, H, W)
        clip_image = clip_normalize(pixels).unsqueeze(0).to(device)  # (1, 3, 224, 224)

        # CLIP score
        with torch.no_grad():
            text_tokens = clip_tokenizer([caption]).to(device)
            image_features = clip_model.encode_image(clip_image)
            text_features = clip_model.encode_text(text_tokens)
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            score = (image_features @ text_features.T).item()

        clip_scores.append(score)

        # Get target text for comparison
        text_length = sample.get("text_text_length", len(text_token_ids))
        if isinstance(text_length, torch.Tensor):
            text_length = text_length.item()
        target_text = decode_tokens(text_token_ids[:text_length].tolist(), tokenizer)

        print(f"[{i}] CLIPScore={score:.4f}")
        print(f"     Generated: {caption[:120]}")
        print(f"     Target:    {target_text[:120]}")
        print()

    if clip_scores:
        scores = torch.tensor(clip_scores)
        print(f"\n{'='*60}")
        print(f"Image Transcription Results ({len(clip_scores)} samples)")
        print(f"  Mean CLIPScore: {scores.mean():.4f}")
        print(f"  Std:            {scores.std():.4f}")
        print(f"  Min:            {scores.min():.4f}")
        print(f"  Max:            {scores.max():.4f}")
        print(f"{'='*60}")

        from megatransformer.scripts.eval.world.eval_utils import infer_step_from_checkpoint, init_eval_metrics, log_eval_scalars
        step = args.step if args.step is not None else infer_step_from_checkpoint(args.checkpoint_path)
        init_eval_metrics(args.log_dir, args.checkpoint_path)
        log_eval_scalars({
            "eval/image_transcription_clipscore_mean": scores.mean().item(),
            "eval/image_transcription_clipscore_std": scores.std().item(),
        }, step)
    else:
        print("No image samples found in dataset.")


if __name__ == "__main__":
    main()
