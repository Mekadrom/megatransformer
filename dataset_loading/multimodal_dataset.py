from torch.utils.data import IterableDataset
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer
from typing import Iterator, Any, List, Dict, Union, Optional

import datasets
import dataset_loading
import random
import time
import torch


datasets.config.HF_DATASETS_TIMEOUT = 900 

class LimitedStreamDataset(IterableDataset):
    """Wrapper for an IterableDataset that limits the number of samples"""
    
    def __init__(self,
                 dataset: IterableDataset,
                 limit: int,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 on_error: Optional[callable] = None):
        """
        Args:
            dataset: The base streaming dataset
            limit: Maximum number of samples to yield
        """
        super().__init__()
        self.dataset = dataset
        self.limit = limit

        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.on_error = on_error
    
    def __iter__(self):
        counter = 0
        for item in self.dataset:
            for retry in range(self.max_retries + 1):
                try:
                    yield item
                    counter += 1
                    if counter >= self.limit:
                        break
                except StopIteration:
                    return
                except Exception as e:
                    if retry >= self.max_retries:
                        print(f"Failed after {self.max_retries} retries: {e}")
                        if self.on_error:
                            self.on_error(e)
                        break
                    else:
                        print(f"Retrying due to error: {e}")
                        time.sleep(self.retry_delay * (2 ** retry))

class MultimodalDataset(IterableDataset):
    def __init__(
        self,
        approximated_length: int,
        tokenizer: PreTrainedTokenizer,
        image_size: int,
        cache_dir: str = "cached_datasets",
        text_weight: float = 1.0,
        audio_weight: float = 1.0,
        image_weight: float = 1.0,
        text_examples: int = 100000,
        audio_examples: int = 100000,
        image_examples: int = 100000,
        split: str = "train",
        seed: int = 42,
        max_position_embeddings: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.text_weight = text_weight
        self.audio_weight = audio_weight
        self.image_weight = image_weight
        self.split = split
        self.max_position_embeddings = max_position_embeddings
        self.rng = random.Random(seed)
        self.approximated_length = approximated_length

        self.delegate_datasets = []
        weights = []

        if self.text_weight > 0:
            dataset = dataset_loading.load_dataset(
                self.tokenizer,
                self.max_position_embeddings,
                split,
                "text",
                streaming=True,
                cache_dir=self.cache_dir
            )
            if text_examples > 0:
                dataset = LimitedStreamDataset(dataset, text_examples)
            self.delegate_datasets.append(dataset)
            weights.append(self.text_weight)

        if self.audio_weight > 0:
            dataset = dataset_loading.load_dataset(
                self.tokenizer,
                self.max_position_embeddings,
                split,
                "audio",
                streaming=True,
                cache_dir=self.cache_dir
            )

            if audio_examples > 0:
                dataset = LimitedStreamDataset(dataset, audio_examples)
            self.delegate_datasets.append(dataset)
            weights.append(self.audio_weight)

        if self.image_weight > 0:
            dataset = dataset_loading.load_dataset(
                self.tokenizer,
                self.max_position_embeddings,
                split,
                "image",
                image_size=image_size,
                streaming=True,
                cache_dir=self.cache_dir
            )

            if image_examples > 0:
                dataset = LimitedStreamDataset(dataset, image_examples)
            self.delegate_datasets.append(dataset)
            weights.append(self.image_weight)

        self.num_datasets = len(self.delegate_datasets)

        total = sum(weights)
        self.weights = [w / total for w in weights]

        self.iterators = None

    def __len__(self) -> int:
        return self.approximated_length # can sometimes be the real length

    def __iter__(self) -> Iterator[Any]:
        # Create iterators for all delegate datasets
        self.iterators = [iter(dataset) for dataset in self.delegate_datasets]
        
        # Create a worker-specific infinite generator
        return self._sample_generator()
    
    def _sample_generator(self) -> Iterator[Any]:
        """Infinite generator that yields random samples from delegate datasets"""
        while True:
            try:
                # Randomly select a dataset based on weights
                dataset_idx = random.choices(range(self.num_datasets), weights=self.weights, k=1)[0]
                
                # Get next item from the selected dataset
                yield next(self.iterators[dataset_idx])
                
            except StopIteration:
                # If one iterator is exhausted, reinitialize it
                self.iterators[dataset_idx] = iter(self.delegate_datasets[dataset_idx])
                
                # Try again with the fresh iterator
                try:
                    yield next(self.iterators[dataset_idx])
                except StopIteration:
                    # If still empty, the dataset might truly be empty
                    # Skip this dataset by reducing its weight to 0 and renormalizing
                    if sum(self.weights) - self.weights[dataset_idx] > 0:
                        self.weights[dataset_idx] = 0
                        total = sum(self.weights)
                        if total > 0:
                            self.weights = [w / total for w in self.weights]
                    
                    # If all datasets are exhausted, break the infinite loop
                    if sum(self.weights) == 0:
                        break

class DataCollatorForMultimodalLanguageModeling(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_position_embeddings, image_size, audio_max_frames, *args, **kwargs):
        super().__init__(tokenizer=tokenizer, *args, **kwargs)
        self.max_position_embeddings = max_position_embeddings
        self.image_size = image_size
        self.audio_max_frames = audio_max_frames

    """DataCollatorForLanguageModeling but additional dataset dict keys are allowed through."""
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        all_input_ids = []
        all_audio_raw_inputs = []
        all_image_raw_inputs = []

        for example in examples:
            if "audio_raw_inputs" not in example:
                example["audio_raw_inputs"] = torch.zeros((128, 1))
            if "image_raw_inputs" not in example:
                example["image_raw_inputs"] = torch.zeros((3, self.image_size, self.image_size))

            all_input_ids.append(example["input_ids"])
            all_audio_raw_inputs.append(example["audio_raw_inputs"])
            all_image_raw_inputs.append(example["image_raw_inputs"])

        # Pad sequences to the same length
        all_input_ids = [torch.nn.functional.pad(ids, (0, self.max_position_embeddings - len(ids)), value=self.tokenizer.pad_token_id) for ids in all_input_ids]
        all_input_ids = torch.stack(all_input_ids)

        all_labels = all_input_ids.clone()
        all_labels[all_input_ids == self.tokenizer.pad_token_id] = -100

        all_audio_raw_inputs = [torch.nn.functional.pad(audio, (0, self.audio_max_frames - audio.shape[-1]), value=0).unsqueeze(0) for audio in all_audio_raw_inputs]
        all_audio_raw_inputs = torch.stack(all_audio_raw_inputs)
        all_audio_labels = all_audio_raw_inputs.clone()

        all_image_raw_inputs = torch.stack(all_image_raw_inputs, dim=0)
        all_image_labels = all_image_raw_inputs.clone()

        # bullshit function expects every example to consist of only tensors of the same shapes per key
        # batch = super().torch_call(examples)

        batch = {
            "input_ids": all_input_ids,
            "labels": all_labels,
            "audio_raw_inputs": all_audio_raw_inputs,
            "audio_labels": all_audio_labels,
            "image_raw_inputs": all_image_raw_inputs,
            "image_labels": all_image_labels,
        }

        # print({k: v.shape for k, v in batch.items()})

        return batch
