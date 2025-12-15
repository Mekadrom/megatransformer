from typing import Dict, List
import torch
import torch.nn.functional as F
import os
import json

from torch.utils.data import Dataset


class CachedImageDiffusionDataset(Dataset):
    """
    Dataset that loads preprocessed image diffusion data from cached .pt files.
    Expects files with: latent_mu, text_embeddings, text_attention_mask
    """

    def __init__(
        self,
        cache_dir: str,
    ):
        self.cache_dir = cache_dir

        # Load config
        config_path = os.path.join(cache_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)
        else:
            self.config = {}

        # Find all cached files
        self.file_paths = sorted([
            os.path.join(cache_dir, f)
            for f in os.listdir(cache_dir)
            if f.endswith(".pt")
        ])

        print(f"Found {len(self.file_paths)} cached examples in {cache_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx])

        # Require latent_mu for latent diffusion
        if "latent_mu" not in data:
            raise ValueError(
                f"latent_mu not found in {self.file_paths[idx]}. "
                "Ensure dataset was preprocessed with VAE encoding (--vae_checkpoint)."
            )

        result = {
            "text_attention_mask": data["text_attention_mask"],
            "text_embeddings": data["text_embeddings"],
            "latent_mu": data["latent_mu"],
            "latent_shape": data.get("latent_shape", list(data["latent_mu"].shape)),
        }

        # Include original image if available (for visualization)
        if "image" in data:
            result["image"] = data["image"]

        # Include label if available (for class-conditional)
        if "label" in data:
            result["label"] = data["label"]

        return result


class ImageDiffusionDataCollator:
    """Data collator for image diffusion training (latent-only)."""
    def __init__(
        self,
        max_conditions: int = 512,
    ):
        self.max_conditions = max_conditions

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        attention_masks = []
        text_embeddings = []
        latent_mus = []
        images = []
        labels = []
        has_images = False
        has_labels = False

        for ex in examples:
            if ex is None:
                continue

            # Handle text embeddings and attention mask
            text_embedding = ex["text_embeddings"]
            attention_mask = ex["text_attention_mask"]

            if text_embedding.shape[0] < self.max_conditions:
                pad_length = self.max_conditions - text_embedding.shape[0]
                text_embedding = F.pad(text_embedding, (0, 0, 0, pad_length), value=0)
                attention_mask = F.pad(attention_mask, (0, pad_length), value=0)
            elif text_embedding.shape[0] > self.max_conditions:
                text_embedding = text_embedding[:self.max_conditions, :]
                attention_mask = attention_mask[:self.max_conditions]

            attention_masks.append(attention_mask)
            text_embeddings.append(text_embedding)

            # Latent
            latent_mus.append(ex["latent_mu"])

            # Optional image (for visualization)
            if "image" in ex:
                has_images = True
                images.append(ex["image"])

            # Optional label
            if "label" in ex:
                has_labels = True
                labels.append(ex["label"])

        # Stack tensors
        batch = {
            "attention_mask": torch.stack(attention_masks),
            "text_embeddings": torch.stack(text_embeddings),
            "latent_mu": torch.stack(latent_mus),
        }

        if has_images:
            batch["image"] = torch.stack(images)

        if has_labels:
            batch["label"] = torch.tensor(labels)

        return batch
