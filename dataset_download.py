from datasets import load_dataset, config

config.HF_DATASETS_DOWNLOAD_WORKERS = 16

dataset = load_dataset(
    "fixie-ai/common_voice_17_0",
    "en",
    split="train",
    cache_dir="./cached_datasets/cv17_cache",
    num_proc=16,
)
dataset.save_to_disk("./cached_datasets/cv17_local")
