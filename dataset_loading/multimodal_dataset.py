from torch.utils.data import IterableDataset
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer
from typing import Iterator, Any, List, Dict, Union, Optional

import dataset_loading
import random
import time
import torch


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
        config,  # megatransformer_utils.MegaTransformerConfig,
        approximated_length: int,
        tokenizer: PreTrainedTokenizer,
        sample_rate: int,
        n_mels: int,
        n_fft: int,
        hop_length: int,
        audio_max_frames: float,
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
        self.config = config
        self.approximated_length = approximated_length
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.text_weight = text_weight
        self.audio_weight = audio_weight
        self.image_weight = image_weight
        self.split = split
        self.max_position_embeddings = max_position_embeddings

        self.seed_epoch = 0

        self.rng = random.Random(seed)

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
                sample_rate=sample_rate,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
                audio_max_frames=audio_max_frames,
                streaming=True,
                cache_dir=self.cache_dir,
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
                cache_dir=self.cache_dir,
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
        """Generator with deterministic sampling using epoch and iteration for seed"""
        # Track iterations for deterministic sampling
        iteration = 0
        
        while True:
            try:
                # Use a deterministic seed based on iteration number
                # This ensures all processes make the same random choice
                seed = hash(f"epoch_{self.config.current_epoch}_iter_{self.config.current_global_step}") % (2**32)
                iteration += 1
                
                # All processes will choose the same dataset
                dataset_idx = random.Random(seed).choices(range(self.num_datasets), weights=self.weights, k=1)[0]
                # if torch.distributed.is_initialized():
                #     print(f"Rank {torch.distributed.get_rank()} - epoch {self.epoch} - iter {iteration} - dataset {dataset_idx} - seed {seed}")
                yield next(self.iterators[dataset_idx])
            except StopIteration:
                # Handle iterator exhaustion
                self.iterators[dataset_idx] = iter(self.delegate_datasets[dataset_idx])
                
                try:
                    yield next(self.iterators[dataset_idx])
                except StopIteration:
                    if sum(self.weights) - self.weights[dataset_idx] > 0:
                        self.weights[dataset_idx] = 0
                        self.weights = [w / sum(self.weights) for w in self.weights] if sum(self.weights) > 0 else []
                    
                    if sum(self.weights) == 0:
                        break

class DataCollatorForMultimodalLanguageModeling(DataCollatorForLanguageModeling):
    """DataCollatorForLanguageModeling but additional dataset dict keys are allowed through."""

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 max_position_embeddings=512,
                 image_size=None,
                 audio_max_frames=None,
                 audio_max_waveform_length=None,
                 modes=["text", "audio", "image"],
                 *args,
                 **kwargs):
        super().__init__(tokenizer=tokenizer, *args, **kwargs)

        self.max_position_embeddings = max_position_embeddings
        self.image_size = image_size
        self.audio_max_frames = audio_max_frames
        self.audio_max_waveform_length = audio_max_waveform_length

        self.modes = modes

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        all_input_ids = []

        all_audio_raw_inputs = []
        audio_waveform_labels = []

        all_image_raw_inputs = []
        all_image_labels = []

        for example in examples:
            if not "audio_raw_inputs" in example and not "image_raw_inputs" in example:
                # text only example, returns a list of batch examples; convert to tensors
                input_ids = example["input_ids"]

                if not isinstance(input_ids, torch.Tensor):
                    input_ids = torch.tensor(input_ids)
            else:
                input_ids = example["input_ids"]

            if "audio" in self.modes:
                if "audio_raw_inputs" not in example:
                    # mels, 1 length
                    example["audio_raw_inputs"] = torch.zeros((128, 1))
                    # waveform, 1 channel and 1 sample
                    example["audio_waveform_labels"] = torch.zeros((1))
            if "image" in self.modes:
                if "image_raw_inputs" not in example:
                    example["image_raw_inputs"] = torch.zeros((3, self.image_size, self.image_size))

            all_input_ids.append(input_ids)

            if "audio_raw_inputs" in example:
                all_audio_raw_inputs.append(example["audio_raw_inputs"])
                audio_waveform_labels.append(example["audio_waveform_labels"])

            if "image_raw_inputs" in example:
                all_image_raw_inputs.append(example["image_raw_inputs"])
                all_image_labels.append(example["image_raw_inputs"])

        # Pad sequences to the same length
        all_input_ids = [torch.nn.functional.pad(ids, (0, self.max_position_embeddings - len(ids)), value=self.tokenizer.pad_token_id) for ids in all_input_ids]
        all_input_ids = torch.stack(all_input_ids)

        # bullshit function expects every example to consist of only tensors of the same shapes per key
        # batch = super().torch_call(examples)

        batch = {
            "input_ids": all_input_ids,
        }

        if "text" in self.modes:
            all_labels = all_input_ids.clone()
            all_labels[all_labels == self.tokenizer.pad_token_id] = -100
            batch.update({
                "labels": all_labels,
            })

        if "audio" in self.modes:
            all_audio_raw_inputs = [torch.nn.functional.pad(audio, (0, self.audio_max_frames - audio.shape[-1]), value=0).unsqueeze(0) for audio in all_audio_raw_inputs if audio is not None]
            all_audio_raw_inputs = torch.stack(all_audio_raw_inputs).unsqueeze(1)

            audio_mel_spec_labels = all_audio_raw_inputs.clone()
            audio_waveform_labels = [torch.nn.functional.pad(audio, (0, self.audio_max_waveform_length - audio.shape[-1]), value=0).unsqueeze(0) for audio in audio_waveform_labels if audio is not None]

            audio_waveform_labels = torch.stack(audio_waveform_labels)
            batch.update({
                "audio_raw_inputs": all_audio_raw_inputs,
                "audio_mel_spec_labels": audio_mel_spec_labels,
                "audio_waveform_labels": audio_waveform_labels,
            })

        if "image" in self.modes:
            all_image_raw_inputs = torch.stack(all_image_raw_inputs, dim=0).unsqueeze(1)
            all_image_labels = torch.stack(all_image_labels, dim=0).unsqueeze(1)
            batch.update({
                "image_raw_inputs": all_image_raw_inputs,
                "image_labels": all_image_labels,
            })

        return batch
