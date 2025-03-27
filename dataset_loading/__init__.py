from dataset_loading import audio_multimodal, generic_text, image_multimodal, wikitext
from transformers import PreTrainedTokenizer
from typing import Literal


text_train_dataset_name = "gair-prox/FineWeb-pro"
text_train_dataset_config_name = None
text_train_split = "train"

text_validation_dataset_name = "wikitext"
text_validation_dataset_config_name = "wikitext-2-v1"
text_validation_split = "validation"

audio_train_dataset_name = "mozilla-foundation/common_voice_11_0"
# audio_train_dataset_name = "facebook/voxpopuli"
audio_train_dataset_config_name = "en"
audio_train_split = "train"

audio_validation_dataset_name = "mozilla-foundation/common_voice_11_0"
# audio_validation_dataset_name = "facebook/voxpopuli"
audio_validation_dataset_config_name = "en"
audio_validation_split = "validation"

image_train_dataset_name = "laion/laion400m"
image_train_dataset_config_name = None
image_train_split = "train"

image_validation_dataset_name = "laion/gpt4v-dataset"
image_validation_dataset_config_name = None
image_validation_split = "train"


lookup = {
    "text": {
        "dataset_name": text_train_dataset_name,
        "dataset_config_name": text_train_dataset_config_name,
        "dataset_split": text_train_split,
        "validation_dataset_name": text_validation_dataset_name,
        "validation_dataset_config_name": text_validation_dataset_config_name,
        "validation_dataset_split": text_validation_split,
    },
    "audio": {
        "dataset_name": audio_train_dataset_name,
        "dataset_config_name": audio_train_dataset_config_name,
        "dataset_split": audio_train_split,
        "validation_dataset_name": audio_validation_dataset_name,
        "validation_dataset_config_name": audio_validation_dataset_config_name,
        "validation_dataset_split": audio_validation_split,
    },
    "image": {
        "dataset_name": image_train_dataset_name,
        "dataset_config_name": image_train_dataset_config_name,
        "dataset_split": image_train_split,
        "validation_dataset_name": image_validation_dataset_name,
        "validation_dataset_config_name": image_validation_dataset_config_name,
        "validation_dataset_split": image_validation_split,
    },
}

def load_text_only_dataset(tokenizer, max_position_embeddings, dataset_name, dataset_config_name, dataset_split, streaming=False, cache_dir=None):
    if "wikitext" in dataset_name.lower():
        dataset = wikitext.load_text_dataset(
            dataset_name,
            dataset_config_name,
            dataset_split,
            tokenizer,
            max_position_embeddings,
            streaming=False,
            cache_dir=cache_dir,
        )
    else:
        dataset = generic_text.load_text_dataset(
            dataset_name,
            dataset_config_name,
            dataset_split,
            tokenizer,
            max_position_embeddings,
            streaming=streaming,
            cache_dir=cache_dir,
        )
    return dataset

def load_audio_dataset(sample_rate, n_mels, n_fft, hop_length, max_frames, tokenizer, dataset_name, dataset_config_name, dataset_split, streaming=False, cache_dir=None):
    if "mozilla" in dataset_name.lower() or "commonvoice" in dataset_name.lower() or "voxpopuli" in dataset_name.lower():
        dataset = audio_multimodal.load_audio_dataset(
            dataset_name,
            dataset_config_name,
            dataset_split,
            tokenizer,
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            max_frames=max_frames,
            batch_size=100,
            streaming=streaming,
            cache_dir=cache_dir,
        )
    else:
        raise ValueError(f"Unsupported audio dataset: {dataset_name}")
    return dataset

def load_image_dataset(tokenizer, dataset_name, dataset_config_name, dataset_split, image_size, streaming=False, cache_dir=None):
    if "laion" in dataset_name.lower():
        dataset = image_multimodal.load_image_dataset(
            dataset_name,
            dataset_config_name,
            dataset_split,
            tokenizer,
            image_size=image_size,
            streaming=streaming,
            cache_dir=cache_dir,
        )
    else:
        raise ValueError(f"Unsupported image dataset: {dataset_name}")
    return dataset

def load_dataset(
    tokenizer: PreTrainedTokenizer,
    max_position_embeddings: int,
    split: str = "train",
    dataset_type: Literal["text", "image", "audio"] = "text",
    sample_rate: int = None,
    n_mels: int = None,
    n_fft: int = None,
    hop_length: int = None,
    audio_max_frames: float = None,
    image_size: int = None,
    streaming: bool = False,
    cache_dir: str = None,
):
    lookup_prefix = "" if split == "train" else "validation_"

    dataset_name = lookup[dataset_type][f"{lookup_prefix}dataset_name"]
    dataset_config_name = lookup[dataset_type][f"{lookup_prefix}dataset_config_name"]
    split = lookup[dataset_type][f"{lookup_prefix}dataset_split"]

    print(f"Loading dataset {dataset_name} with config {dataset_config_name}, split {split} for {dataset_type}.")
    if dataset_type == "text":
        return load_text_only_dataset(tokenizer, max_position_embeddings, dataset_name, dataset_config_name, split, streaming=streaming, cache_dir=cache_dir)
    elif dataset_type == "audio":
        return load_audio_dataset(sample_rate, n_mels, n_fft, hop_length, audio_max_frames, tokenizer, dataset_name, dataset_config_name, split, streaming=streaming, cache_dir=cache_dir)
    elif dataset_type == "image":
        return load_image_dataset(tokenizer, dataset_name, dataset_config_name, split, image_size, streaming=streaming, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
