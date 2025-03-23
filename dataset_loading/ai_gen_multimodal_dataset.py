from datasets import load_dataset, Audio
from dataset_loading import audio_multimodal, image_multimodal, generic_text
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer

import dataset_loading
import h5py
import json
import numpy as np
import os
import pickle
import torch


class MultimodalDatasetProcessor:
    """Process and cache datasets to disk for faster loading."""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_position_embeddings: int,
        cache_dir: str = "cached_datasets",
        image_size: int = 224,
        audio_n_mels: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_position_embeddings = max_position_embeddings
        self.cache_dir = cache_dir
        self.image_size = image_size
        self.audio_n_mels = audio_n_mels
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Special token IDs
        self.begin_image_token_id = tokenizer.convert_tokens_to_ids("<|IMAGE|>")
        self.end_image_token_id = tokenizer.convert_tokens_to_ids("<|/IMAGE|>")
        self.begin_audio_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO|>")
        self.end_audio_token_id = tokenizer.convert_tokens_to_ids("<|/AUDIO|>")
    
    def _get_cache_path(self, dataset_name, split="train"):
        """Get the path for cached dataset."""
        # Clean dataset name for filesystem
        clean_name = dataset_name.replace("/", "_")
        return os.path.join(self.cache_dir, f"{clean_name}_{split}.h5")
    
    def _get_metadata_path(self, dataset_name, split="train"):
        """Get the path for dataset metadata."""
        clean_name = dataset_name.replace("/", "_")
        return os.path.join(self.cache_dir, f"{clean_name}_{split}_metadata.json")
    
    def _save_metadata(self, dataset_name, split, metadata):
        """Save dataset metadata."""
        path = self._get_metadata_path(dataset_name, split)
        with open(path, 'w') as f:
            json.dump(metadata, f)
    
    def _load_metadata(self, dataset_name, split):
        """Load dataset metadata."""
        path = self._get_metadata_path(dataset_name, split)
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None
    
    def is_cached(self, dataset_name, split="train"):
        """Check if dataset is already cached."""
        cache_path = self._get_cache_path(dataset_name, split)
        metadata_path = self._get_metadata_path(dataset_name, split)
        return os.path.exists(cache_path) and os.path.exists(metadata_path)
    
    def process_text_dataset(self, max_samples=None):
        """Process and cache text dataset."""
        if self.is_cached(dataset_loading.text_train_dataset_name):
            print(f"Text dataset already cached at {self._get_cache_path(dataset_loading.text_train_dataset_name)}")
            return
        
        print(f"Processing text dataset {dataset_loading.text_train_dataset_name}...")
        
        dataset = dataset_loading.load_dataset(self.tokenizer, self.max_position_embeddings, "text", streaming=True, cache_dir=self.cache_dir)["train"]
        
        # Determine number of samples to process
        total_samples = len(dataset) if hasattr(dataset, "__len__") else max_samples or float('inf')
        
        # Create cache file
        cache_path = self._get_cache_path(dataset_loading.text_train_dataset_name)
        with h5py.File(cache_path, 'w') as f:
            # Create datasets for input_ids and attention_mask
            # Use variable-length strings to store lists of different lengths
            dt = h5py.special_dtype(vlen=np.int32)
            input_ids_dset = f.create_dataset('input_ids', (total_samples,), dtype=dt)
            attn_mask_dset = f.create_dataset('attention_mask', (total_samples,), dtype=dt)
            
            # Process samples
            for i, sample in enumerate(tqdm(dataset, total=total_samples)):
                if i >= total_samples:
                    break
                    
                input_ids = np.array(sample['input_ids'][0], dtype=np.int32)
                attn_mask = np.array(sample['attention_mask'][0], dtype=np.int32)
                
                input_ids_dset[i] = input_ids
                attn_mask_dset[i] = attn_mask
        
        # Save metadata
        metadata = {
            "size": total_samples,
            "max_seq_length": self.max_position_embeddings,
            "tokenizer": self.tokenizer.name_or_path,
        }
        self._save_metadata(dataset_loading.text_train_dataset_name, "train", metadata)
        
        print(f"Text dataset cached with {total_samples} samples")
    
    def process_audio_dataset(self, max_samples=None):
        """Process and cache audio dataset."""
        if self.is_cached(dataset_loading.audio_train_dataset_name):
            print(f"Audio dataset already cached at {self._get_cache_path(dataset_loading.audio_train_dataset_name)}")
            return
        
        print(f"Processing audio dataset {dataset_loading.audio_train_dataset_name}...")
        
        dataset = dataset_loading.load_dataset(self.tokenizer, self.max_position_embeddings, "audio", cache_dir=self.cache_dir)["train"]
        
        # Determine number of samples to process
        total_samples = len(dataset) if hasattr(dataset, "__len__") else max_samples or float('inf')
        
        # Create cache file
        cache_path = self._get_cache_path(dataset_loading.audio_train_dataset_name)
        with h5py.File(cache_path, 'w') as f:
            # Create datasets
            dt = h5py.special_dtype(vlen=np.int32)
            input_ids_dset = f.create_dataset('input_ids', (total_samples,), dtype=dt)
            
            # Get the shape of audio features from first sample
            sample = dataset[0]
            audio_shape = sample['audio_raw_inputs'].shape
            audio_dtype = sample['audio_raw_inputs'].dtype
            
            # Create fixed-size datasets for audio features
            audio_dset = f.create_dataset('audio_raw_inputs', 
                                         (total_samples, *audio_shape), 
                                         dtype=audio_dtype)
            audio_labels_dset = f.create_dataset('audio_labels', 
                                                (total_samples, *audio_shape), 
                                                dtype=audio_dtype)
            
            # Process samples
            for i, sample in enumerate(tqdm(dataset, total=total_samples)):
                if i >= total_samples:
                    break
                
                input_ids = np.array(sample['input_ids'], dtype=np.int32)
                audio_raw = sample['audio_raw_inputs']
                audio_labels = sample['audio_labels']
                
                input_ids_dset[i] = input_ids
                audio_dset[i] = audio_raw
                audio_labels_dset[i] = audio_labels
        
        # Save metadata
        metadata = {
            "size": total_samples,
            "audio_shape": audio_shape,
            "audio_dtype": str(audio_dtype),
            "max_seq_length": self.max_position_embeddings,
            "tokenizer": self.tokenizer.name_or_path,
        }
        self._save_metadata(dataset_loading.audio_train_dataset_name, "train", metadata)
        
        print(f"Audio dataset cached with {total_samples} samples")
    
    def process_image_dataset(self, max_samples=None):
        """Process and cache image dataset."""
        if self.is_cached(dataset_loading.image_train_dataset_name):
            print(f"Image dataset already cached at {self._get_cache_path(dataset_loading.image_train_dataset_name)}")
            return
        
        print(f"Processing image dataset {dataset_loading.image_train_dataset_name}...")
        
        dataset = dataset_loading.load_dataset(self.tokenizer, self.max_position_embeddings, "image", image_size=self.image_size, streaming=True, cache_dir=self.cache_dir)["train"]
        
        # Create cache file
        cache_path = self._get_cache_path(dataset_loading.image_train_dataset_name)
        
        # Determine number of samples to process
        total_samples = max_samples or 100000  # Default limit for large image datasets
        
        with h5py.File(cache_path, 'w') as f:
            # Create datasets
            dt = h5py.special_dtype(vlen=np.int32)
            input_ids_dset = f.create_dataset('input_ids', (total_samples,), dtype=dt)
            
            # Process first valid sample to get image shape
            image_shape = None
            for sample in dataset:
                if 'image_raw_inputs' in sample and sample['image_raw_inputs'] is not None:
                    image_shape = sample['image_raw_inputs'].shape
                    break
            
            if image_shape is None:
                raise ValueError("Could not find a valid image sample in the dataset")
            
            # Create fixed-size datasets for image features
            img_dset = f.create_dataset('image_raw_inputs', 
                                       (total_samples, *image_shape), 
                                       dtype=np.float32)
            img_labels_dset = f.create_dataset('image_labels', 
                                             (total_samples, *image_shape), 
                                             dtype=np.float32)
            
            # Process samples
            valid_count = 0
            for sample in tqdm(dataset):
                # Check if we have a valid image
                if ('image_raw_inputs' not in sample or 
                    sample['image_raw_inputs'] is None or 
                    'input_ids' not in sample):
                    continue
                
                if valid_count >= total_samples:
                    break
                
                try:
                    input_ids = np.array(sample['input_ids'], dtype=np.int32)
                    image_raw = sample['image_raw_inputs'].numpy()
                    image_labels = sample['image_labels'].numpy()
                    
                    input_ids_dset[valid_count] = input_ids
                    img_dset[valid_count] = image_raw
                    img_labels_dset[valid_count] = image_labels
                    
                    valid_count += 1
                except (ValueError, AttributeError, TypeError) as e:
                    # Skip invalid samples
                    print(f"Skipping sample due to error: {e}")
                    continue
        
        # Save metadata
        metadata = {
            "size": valid_count,
            "image_shape": image_shape,
            "max_seq_length": self.max_position_embeddings,
            "tokenizer": self.tokenizer.name_or_path,
        }
        self._save_metadata(dataset_loading.image_train_dataset_name, "train", metadata)
        
        print(f"Image dataset cached with {valid_count} samples")
    
    def process_all_datasets(self, max_text_samples=None, max_audio_samples=None, max_image_samples=None):
        """Process and cache all datasets."""
        self.process_text_dataset(max_text_samples)
        self.process_audio_dataset(max_audio_samples)
        self.process_image_dataset(max_image_samples)
        print("All datasets processed and cached!")


class CachedMultimodalDataset(Dataset):
    """Dataset that loads from cached data."""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        cache_dir: str = "cached_datasets",
        text_ratio: float = 0.7,
        audio_ratio: float = 0.15,
        image_ratio: float = 0.15,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        
        # Validate cache directory
        if not os.path.exists(cache_dir):
            raise ValueError(f"Cache directory {cache_dir} does not exist. Run processor first.")
        
        # Initialize random state
        self.random_state = np.random.RandomState(seed)
        
        # Set sampling ratios
        assert abs(text_ratio + audio_ratio + image_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        self.text_ratio = text_ratio
        self.audio_ratio = audio_ratio
        self.image_ratio = image_ratio
        
        # Load dataset metadata
        self.text_metadata = self._load_metadata(dataset_loading.text_train_dataset_name)
        self.audio_metadata = self._load_metadata(dataset_loading.audio_train_dataset_name)
        self.image_metadata = self._load_metadata(dataset_loading.image_train_dataset_name)
        
        if not all([self.text_metadata, self.audio_metadata, self.image_metadata]):
            raise ValueError("Missing metadata for one or more datasets. Run processor first.")
        
        # Open HDF5 files
        self.text_file = h5py.File(self._get_cache_path(dataset_loading.text_train_dataset_name), 'r')
        self.audio_file = h5py.File(self._get_cache_path(dataset_loading.audio_train_dataset_name), 'r')
        self.image_file = h5py.File(self._get_cache_path(dataset_loading.image_train_dataset_name), 'r')
        
        # Store sizes
        self.text_size = self.text_metadata["size"]
        self.audio_size = self.audio_metadata["size"]
        self.image_size = self.image_metadata["size"]
        
        # Track image and audio placeholders
        self.begin_image_token_id = tokenizer.convert_tokens_to_ids("<|IMAGE|>")
        self.end_image_token_id = tokenizer.convert_tokens_to_ids("<|/IMAGE|>")
        self.begin_audio_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO|>")
        self.end_audio_token_id = tokenizer.convert_tokens_to_ids("<|/AUDIO|>")
        
        print(f"Loaded cached datasets:")
        print(f"  Text: {self.text_size} samples")
        print(f"  Audio: {self.audio_size} samples")
        print(f"  Image: {self.image_size} samples")
    
    def _get_cache_path(self, dataset_name, split="train"):
        """Get the path for cached dataset."""
        return os.path.join(self.cache_dir, f"{dataset_name}_{split}.h5")
    
    def _load_metadata(self, dataset_name, split="train"):
        """Load dataset metadata."""
        path = os.path.join(self.cache_dir, f"{dataset_name}_{split}_metadata.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None
    
    def __len__(self):
        """Return dataset length."""
        return self.text_size + self.audio_size + self.image_size
    
    def _get_text_example(self):
        """Get a text example."""
        idx = self.random_state.randint(0, self.text_size - 1)
        
        # Get data from HDF5
        input_ids = self.text_file['input_ids'][idx]
        attention_mask = self.text_file['attention_mask'][idx] if 'attention_mask' in self.text_file else np.ones_like(input_ids)
        
        return {
            "input_ids": [input_ids],
            "attention_mask": [attention_mask],
            "begin_image_token_id": self.begin_image_token_id,
            "end_image_token_id": self.end_image_token_id,
            "begin_audio_token_id": self.begin_audio_token_id,
            "end_audio_token_id": self.end_audio_token_id,
        }
    
    def _get_audio_example(self):
        """Get an audio example."""
        idx = self.random_state.randint(0, self.audio_size - 1)
        
        # Get data from HDF5
        input_ids = self.audio_file['input_ids'][idx]
        audio_raw = self.audio_file['audio_raw_inputs'][idx]
        audio_labels = self.audio_file['audio_labels'][idx]
        
        # Convert numpy arrays to tensors
        audio_raw = torch.from_numpy(audio_raw)
        audio_labels = torch.from_numpy(audio_labels)
        
        return {
            "input_ids": [input_ids],
            "audio_raw_inputs": audio_raw,
            "audio_labels": audio_labels,
            "begin_image_token_id": self.begin_image_token_id,
            "end_image_token_id": self.end_image_token_id,
            "begin_audio_token_id": self.begin_audio_token_id,
            "end_audio_token_id": self.end_audio_token_id,
        }
    
    def _get_image_example(self):
        """Get an image example."""
        idx = self.random_state.randint(0, self.image_size - 1)
        
        # Get data from HDF5
        input_ids = self.image_file['input_ids'][idx]
        image_raw = self.image_file['image_raw_inputs'][idx]
        image_labels = self.image_file['image_labels'][idx]
        
        # Convert numpy arrays to tensors
        image_raw = torch.from_numpy(image_raw)
        image_labels = torch.from_numpy(image_labels)
        
        return {
            "input_ids": [input_ids],
            "image_raw_inputs": image_raw,
            "image_labels": image_labels,
            "begin_image_token_id": self.begin_image_token_id,
            "end_image_token_id": self.end_image_token_id,
            "begin_audio_token_id": self.begin_audio_token_id,
            "end_audio_token_id": self.end_audio_token_id,
        }
    
    def __getitem__(self, idx):
        """Get an item based on sampling ratios."""
        # Sample based on ratios
        rand_val = self.random_state.random()
        
        if rand_val < self.text_ratio:
            # Return text example
            example = self._get_text_example()
            # Add empty placeholders for audio and image
            example["audio_raw_inputs"] = None
            example["audio_labels"] = None
            example["image_raw_inputs"] = None
            example["image_labels"] = None
            return example
            
        elif rand_val < self.text_ratio + self.audio_ratio:
            # Return audio example
            example = self._get_audio_example()
            # Add empty placeholders for image
            example["image_raw_inputs"] = None
            example["image_labels"] = None
            return example
            
        else:
            # Return image example
            example = self._get_image_example()
            # Add empty placeholders for audio
            example["audio_raw_inputs"] = None
            example["audio_labels"] = None
            return example
    
    def close(self):
        """Close HDF5 files."""
        self.text_file.close()
        self.audio_file.close()
        self.image_file.close()


def multimodal_collate_fn(examples, pad_token_id: int):
    """
    Custom collate function for multimodal data.
    
    Args:
        examples: List of examples from the dataset
        pad_token_id: Padding token ID from tokenizer
        
    Returns:
        Batch with properly formatted tensors
    """
    # First separate examples by modality indicators
    text_only_examples = []
    image_examples = []
    audio_examples = []
    
    for ex in examples:
        has_image = (ex.get("image_raw_inputs") is not None and 
                    len(ex.get("input_ids", [])) > 0 and
                    (ex["begin_image_token_id"] in ex["input_ids"] or 
                     ex["end_image_token_id"] in ex["input_ids"]))
                     
        has_audio = (ex.get("audio_raw_inputs") is not None and 
                    len(ex.get("input_ids", [])) > 0 and
                    (ex["begin_audio_token_id"] in ex["input_ids"] or 
                     ex["end_audio_token_id"] in ex["input_ids"]))
        
        if has_image:
            image_examples.append(ex)
        elif has_audio:
            audio_examples.append(ex)
        else:
            text_only_examples.append(ex)
    
    # Get max sequence length for padding
    max_len = max([len(ex["input_ids"][0]) for ex in examples if len(ex.get("input_ids", [])) > 0])
    
    # Initialize batch
    batch = {
        "input_ids": [],
        "attention_mask": [],
        "image_raw_inputs": [],
        "image_labels": [],
        "audio_raw_inputs": [],
        "audio_labels": [],
    }
    
    # Process all examples to get input_ids and attention_mask
    for ex in examples:
        if len(ex.get("input_ids", [])) > 0:
            input_ids = ex["input_ids"][0]  # Get first sequence (should only be one)
            attention_mask = ex.get("attention_mask", [1] * len(input_ids))[0]
            
            # Pad sequences
            padding_len = max_len - len(input_ids)
            if padding_len > 0:
                input_ids = input_ids + [pad_token_id] * padding_len
                attention_mask = attention_mask + [0] * padding_len
                
            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
    
    # Collect image and audio data
    for ex in image_examples:
        if ex.get("image_raw_inputs") is not None:
            batch["image_raw_inputs"].append(ex["image_raw_inputs"])
            batch["image_labels"].append(ex["image_labels"])
    
    for ex in audio_examples:
        if ex.get("audio_raw_inputs") is not None:
            batch["audio_raw_inputs"].append(ex["audio_raw_inputs"])
            batch["audio_labels"].append(ex["audio_labels"])
    
    # Convert to tensors
    batch["input_ids"] = torch.tensor(batch["input_ids"], dtype=torch.long)
    batch["attention_mask"] = torch.tensor(batch["attention_mask"], dtype=torch.long)
    batch["labels"] = batch["input_ids"].clone()
    
    # Stack tensors for image and audio if available
    if batch["image_raw_inputs"]:
        batch["image_raw_inputs"] = torch.stack(batch["image_raw_inputs"])
        batch["image_labels"] = torch.stack(batch["image_labels"])
    else:
        # Empty tensors with correct shape
        batch["image_raw_inputs"] = torch.empty((0, 3, 224, 224))
        batch["image_labels"] = torch.empty((0, 3, 224, 224))
        
    if batch["audio_raw_inputs"]:
        batch["audio_raw_inputs"] = torch.stack(batch["audio_raw_inputs"]) \
            if isinstance(batch["audio_raw_inputs"][0], torch.Tensor) else \
            torch.tensor(np.stack(batch["audio_raw_inputs"]))
        batch["audio_labels"] = torch.stack(batch["audio_labels"]) \
            if isinstance(batch["audio_labels"][0], torch.Tensor) else \
            torch.tensor(np.stack(batch["audio_labels"]))
    else:
        # Empty tensors with correct shape
        batch["audio_raw_inputs"] = torch.empty((0, 1, 128, 0))
        batch["audio_labels"] = torch.empty((0, 1, 128, 0))
    
    return batch

def create_multimodal_dataloader_from_cache(
    tokenizer: PreTrainedTokenizer,
    cache_dir: str = "cached_datasets",
    batch_size: int = 8,
    text_ratio: float = 0.6,
    audio_ratio: float = 0.2,
    image_ratio: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
):
    """
    Create a dataloader from cached datasets.
    
    Args:
        tokenizer: Tokenizer to use for all datasets
        cache_dir: Directory containing cached datasets
        batch_size: Batch size for training
        text_ratio/audio_ratio/image_ratio: Sampling ratios for each modality
        num_workers: Number of workers for dataloader
        seed: Random seed for reproducibility
        
    Returns:
        DataLoader for multimodal training
    """
    # Create dataset
    dataset = CachedMultimodalDataset(
        tokenizer=tokenizer,
        cache_dir=cache_dir,
        text_ratio=text_ratio,
        audio_ratio=audio_ratio,
        image_ratio=image_ratio,
        seed=seed,
    )
    
    # Create dataloader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda examples: multimodal_collate_fn(examples, tokenizer.pad_token_id),
        pin_memory=True,
    )
    
    return dataloader, dataset  # Return dataset so we can close HDF5 files


# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer
    import argparse
    import datasets
    from datasets import config

    # Increase download timeout (default is 100s)
    config.HF_DATASETS_DOWNLOADED_TIMEOUT = 600  # 10 minutes

    # Add this at the beginning of your script
    config.HF_DATASETS_OFFLINE = False
    config.HF_DATASETS_HTTP_MAX_RETRIES = 5 
    
    parser = argparse.ArgumentParser(description="Process and cache multimodal datasets")
    parser.add_argument("--process", action="store_true", help="Process and cache datasets")
    parser.add_argument("--max_text", type=int, default=100000, help="Max text samples to process")
    parser.add_argument("--max_audio", type=int, default=None, help="Max audio samples to process")
    parser.add_argument("--max_image", type=int, default=10000, help="Max image samples to process")
    parser.add_argument("--dataset_cache_dir", type=str, default="cached_datasets", help="Cache directory")
    parser.add_argument("--max_position_embeddings", type=int, default=8192, help="Max position embeddings")
    parser.add_argument("--tokenizer_name", type=str, default="mistralai/Mistral-7B-v0.1", help="Tokenizer name")

    args = parser.parse_args()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens
    special_tokens = {
        "additional_special_tokens": ["<|IMAGE|>", "<|/IMAGE|>", "<|AUDIO|>", "<|/AUDIO|>"]
    }
    tokenizer.add_special_tokens(special_tokens)
    
    if args.process:
        # Process and cache datasets
        processor = MultimodalDatasetProcessor(
            tokenizer=tokenizer,
            max_position_embeddings=args.max_position_embeddings,
            cache_dir=args.dataset_cache_dir,
            image_size=224,
            audio_n_mels=128,
        )
        
        processor.process_all_datasets(
            max_text_samples=args.max_text,
            max_audio_samples=args.max_audio,
            max_image_samples=args.max_image,
        )
    else:
        # Create dataloader from cache
        dataloader, dataset = create_multimodal_dataloader_from_cache(
            tokenizer=tokenizer,
            cache_dir=args.dataset_cache_dir,
            batch_size=4,
        )
        
        # Check a batch
        for batch in dataloader:
            print("Batch contents:")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {v.shape}, {v.dtype}")
                else:
                    print(f"  {k}: {type(v)}")
            
            # Number of images and audios in this batch
            print(f"Number of images: {batch['image_raw_inputs'].size(0)}")
            print(f"Number of audios: {batch['audio_raw_inputs'].size(0)}")
            
            break  # Just check one batch
        
        # Close HDF5 files
        dataset.close()
