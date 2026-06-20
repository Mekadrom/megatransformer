"""Evaluate image synthesis (text → image) using FID and CLIPScore.

For each text sample, generates an image via the world model and:
1. Computes CLIPScore between the generated image and the source text
2. Collects generated + real images for FID computation

Requires an image VAE decoder to convert latents to pixels.

Usage:
    python -m megatransformer.scripts.eval.world.eval_image_synthesis --checkpoint_path runs/my_run/checkpoint-3000 --config small_sum_dit --cache_dir ../cached_datasets/sive --include_modes text,image --image_vae_decoder_config litevae --max_samples 100 --bf16
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.amp import autocast

from megatransformer.model.world.world_model import MegaTransformerWorldModel
from megatransformer.utils import model_loading_utils
from megatransformer.utils.constants import BOI_TOKEN_ID


def parse_args():
    p = argparse.ArgumentParser(description="Image synthesis eval (FID + CLIPScore)")
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--config", type=str, default="small_sum_dit")
    p.add_argument("--include_modes", type=str, default="text,image")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--text_cache_dir", type=str, default=None)
    p.add_argument("--image_cache_dir", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--use_memorization_dataset", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--tie_word_embeddings", action="store_true")
    p.add_argument("--clip_model", type=str, default="ViT-B-32")
    p.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k")
    p.add_argument("--image_vae_decoder_config", type=str, default=None)
    p.add_argument("--image_vae_decoder_path", type=str, default=None)
    p.add_argument("--save_images", type=str, default=None,
                   help="Directory to save generated images")
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
    with torch.no_grad():
        z = latent.unsqueeze(0).to(device=next(decoder.parameters()).device,
                                    dtype=next(decoder.parameters()).dtype)
        if hasattr(decoder, 'decode'):
            out = decoder.decode(z)
            pixels = out.sample if hasattr(out, 'sample') else out
        else:
            pixels = decoder(z)
        return pixels[0].float().clamp(0, 1).cpu()


def encode_static_prompt(text, suffix_tokens, tokenizer, max_new_tokens, max_seq_len, device):
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    max_prompt = max_seq_len - max_new_tokens - len(suffix_tokens)
    token_ids = token_ids[:max(1, max_prompt)]
    all_ids = token_ids + suffix_tokens
    return torch.tensor([all_ids], dtype=torch.long, device=device)


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

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    image_decoder = load_image_decoder(args)
    if image_decoder is None:
        print("ERROR: need --image_vae_decoder_config litevae or --image_vae_decoder_path")
        sys.exit(1)

    print(f"Loading world model from {args.checkpoint_path}...")
    model = load_world_model(args, device)
    model.to(device)
    model.eval()

    print("Loading dataset...")
    dataset = load_dataset(args, split=args.split)
    print(f"Dataset: {len(dataset)} samples")

    if args.save_images:
        os.makedirs(args.save_images, exist_ok=True)

    from torchvision import transforms
    clip_normalize = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    clip_scores = []
    # Collect InceptionV3 features for FID if torchmetrics available
    real_images = []
    gen_images = []

    for i in range(len(dataset)):
        sample = dataset[i]
        if "image_image" not in sample:
            continue

        image_latent = sample["image_image"]

        # Get text prompt
        text = sample.get("text_text", "")
        if not text:
            text_token_ids = sample.get("text_token_ids")
            if text_token_ids is not None:
                text_length = sample.get("text_text_length", len(text_token_ids))
                if isinstance(text_length, torch.Tensor):
                    text_length = text_length.item()
                text_ids = [t for t in text_token_ids[:text_length].tolist() if t < 32000 and t != 0]
                text = tokenizer.decode(text_ids, skip_special_tokens=True)
        if isinstance(text, list):
            text = text[0] if text else ""
        text = str(text).strip()
        if not text:
            continue

        # Build synthesis prompt: [text] [BOI]
        prompt = encode_static_prompt(
            text[:500], [BOI_TOKEN_ID], tokenizer,
            args.max_new_tokens, 1024, device,
        )

        # Generate image
        with torch.no_grad():
            with autocast(device, dtype=dtype, enabled=args.bf16):
                outputs = model.generate(
                    text_input_ids=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )

        image_preds = outputs.get("image_latent_preds")
        if image_preds is None or image_preds.numel() == 0:
            print(f"[{i}] No image generated, skipping")
            continue

        gen_latent = image_preds[0, 0]  # (C, H, W)

        # Decode both to pixels
        gen_pixels = decode_latent_to_pixels(image_decoder, gen_latent, device)
        real_pixels = decode_latent_to_pixels(image_decoder, image_latent, device)

        # Save images
        if args.save_images:
            from torchvision.utils import save_image
            save_image(gen_pixels, os.path.join(args.save_images, f"gen_{i}.png"))
            save_image(real_pixels, os.path.join(args.save_images, f"real_{i}.png"))

        # CLIPScore: generated image vs source text
        clip_image = clip_normalize(gen_pixels).unsqueeze(0).to(device)
        with torch.no_grad():
            text_tokens = clip_tokenizer([text[:77]]).to(device)
            image_features = clip_model.encode_image(clip_image)
            text_features = clip_model.encode_text(text_tokens)
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            score = (image_features @ text_features.T).item()

        clip_scores.append(score)

        # Collect for FID
        real_images.append(real_pixels)
        gen_images.append(gen_pixels)

        print(f"[{i}] CLIPScore={score:.4f}  prompt: {text[:80]}")

    print(f"\n{'='*60}")
    print(f"Image Synthesis Results ({len(clip_scores)} samples)")

    if clip_scores:
        scores = torch.tensor(clip_scores)
        print(f"  Mean CLIPScore: {scores.mean():.4f}")
        print(f"  Std:            {scores.std():.4f}")
        print(f"  Min:            {scores.min():.4f}")
        print(f"  Max:            {scores.max():.4f}")

    # FID (needs enough samples to be meaningful, typically 2048+)
    if len(real_images) >= 2:
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            fid = FrechetInceptionDistance(feature=2048, normalize=True)
            # FID expects uint8 (batch, 3, H, W) or float [0,1] with normalize=True
            for real, gen in zip(real_images, gen_images):
                # Resize to 299x299 for InceptionV3
                real_resized = F.interpolate(real.unsqueeze(0), size=(299, 299), mode="bilinear", align_corners=False)
                gen_resized = F.interpolate(gen.unsqueeze(0), size=(299, 299), mode="bilinear", align_corners=False)
                fid.update(real_resized, real=True)
                fid.update(gen_resized, real=False)
            fid_score = fid.compute().item()
            print(f"  FID:            {fid_score:.2f}")
            if len(real_images) < 2048:
                print(f"  (FID with {len(real_images)} samples is unreliable; need ~2048+ for stable estimates)")
        except ImportError:
            print("  FID: skipped (torchmetrics not available)")
        except Exception as e:
            print(f"  FID: failed ({e})")

    # Log to TensorBoard
    from megatransformer.scripts.eval.world.eval_utils import infer_step_from_checkpoint, init_eval_metrics, log_eval_scalars
    step = args.step if args.step is not None else infer_step_from_checkpoint(args.checkpoint_path)
    init_eval_metrics(args.log_dir, args.checkpoint_path)
    metrics_dict = {}
    if clip_scores:
        scores = torch.tensor(clip_scores)
        metrics_dict["eval/image_synthesis_clipscore_mean"] = scores.mean().item()
    if 'fid_score' in dir():
        metrics_dict["eval/image_synthesis_fid"] = fid_score
    if metrics_dict:
        log_eval_scalars(metrics_dict, step)

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
