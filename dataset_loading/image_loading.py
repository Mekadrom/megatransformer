from io import BytesIO
from datasets import load_dataset
from PIL import Image
from transformers import PreTrainedTokenizer
from torchvision import transforms
from typing import Optional

import os
import requests
import torch


misses = 0

class DropAlpha(object):
    def __call__(self, tensor):
        if tensor.shape[0] == 4:  # If RGBA
            return tensor[:3]
        return tensor

def get_transform(image_size):
    if image_size is None:
        raise ValueError("Image size must be specified for image transformations")
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        # transforms.RandomResizedCrop((image_size, image_size), scale=(0.8, 1.0)),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        DropAlpha(),
        # normalize to [0, 1]
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

def fetch_and_transform_image(image_url, transform):
    try:
        response = requests.get(image_url, timeout=(1, 5), headers=HEADERS)
        response.raise_for_status()  # Catch HTTP errors like 404, 403
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return transform(image)
    except Exception as e:
        # print(f"Error fetching or transforming image: {e} for URL {image_url}")
        global misses
        misses += 1
        return None

def hash_url(url: str) -> str:
    replaced = url.replace("/", "_").replace(":", "_")
    return str(abs(hash(replaced)))

def cache_or_fetch_and_transform_image(dataset_name, dataset_config_name, image_url, transform):
    cache_dir = os.path.join("image_cache", dataset_name.replace("/", "_"))
    if dataset_config_name:
        cache_dir = os.path.join(cache_dir, dataset_config_name)
    os.makedirs(cache_dir, exist_ok=True)
    image_filename = os.path.join(cache_dir, hash_url(image_url) + ".png")
    if os.path.exists(image_filename):
        try:
            image = Image.open(image_filename).convert('RGB')
            return transform(image)
        except Exception as e:
            print(f"Error loading cached image: {e} for file {image_filename}")
            global misses
            misses += 1
            return None
    else:
        image_tensor = fetch_and_transform_image(image_url, transform)
        if image_tensor is not None:
            try:
                # Save the image to cache
                image = transforms.ToPILImage()(image_tensor)
                image.save(image_filename)
            except Exception as e:
                print(f"Error saving image to cache: {e} for file {image_filename}")
        return image_tensor

def load_image_dataset(dataset_name: str,
                       dataset_config_name: Optional[str],
                       split: str,
                       tokenizer: PreTrainedTokenizer,
                       image_size: int,
                       streaming=False,
                       cache_dir=None):
    print(f"Loading dataset {dataset_name} with config {dataset_config_name}, split {split} for image.")

    dataset = load_dataset(dataset_name, dataset_config_name, streaming=streaming, cache_dir=cache_dir, split=split, trust_remote_code=True)

    begin_image_token_id = tokenizer.convert_tokens_to_ids("<|IMAGE|>")
    end_image_token_id = tokenizer.convert_tokens_to_ids("<|/IMAGE|>")

    assert isinstance(begin_image_token_id, int), f"Image placeholder token should be an integer, got {type(begin_image_token_id)}"
    assert isinstance(end_image_token_id, int), f"Image placeholder token should be an integer, got {type(end_image_token_id)}"

    transform = get_transform(image_size)

    def process_function(examples):
        captions = examples["caption"] if "caption" in examples else examples["text"] if "text" in examples else examples["text_input"] if "text_input" in examples else None
        
        image_urls = examples["url"] if "url" in examples else examples["image_url"] if "image_url" in examples else examples["image"] if "image" in examples else examples["link"] if "link" in examples else None
        
        all_input_ids = []
        all_image_raw_inputs = []

        for caption, image_url in zip(captions, image_urls):
            image_raw_input = cache_or_fetch_and_transform_image(dataset_name, dataset_config_name, image_url, transform)
            if caption is None or image_raw_input is None:
                # add dummy example
                image_raw_input = torch.zeros((3, image_size, image_size), dtype=torch.float32)
                input_ids = [begin_image_token_id, end_image_token_id]
                all_input_ids.append(torch.tensor(input_ids))
                all_input_ids.append(torch.tensor(input_ids))
            else:
                tokenized = tokenizer(text=caption, add_special_tokens=True)
                input_ids = tokenized.input_ids

                # append the image placeholder token
                # before input_ids: describe image (image followed by text)
                # after input_ids: generate image (text followed by image)
                # model will interleave embeds between the begin and end tokens; they need to be appended in the text input beforehand so that the embedding process applies appropriately to each token
                image_transcription_input_ids = [begin_image_token_id, end_image_token_id] + input_ids
                image_generation_input_ids = input_ids + [begin_image_token_id, end_image_token_id]

                all_input_ids.append(torch.tensor(image_transcription_input_ids))
                all_input_ids.append(torch.tensor(image_generation_input_ids))

            all_image_raw_inputs.append(image_raw_input)
            all_image_raw_inputs.append(image_raw_input)
        
        # Pad sequences to the same length
        max_length = max([len(ids) for ids in all_input_ids])
        all_input_ids = [torch.nn.functional.pad(ids, (0, max_length - len(ids)), value=0) for ids in all_input_ids]

        # images require no padding

        return {
            "input_ids": torch.stack(all_input_ids),
            "image_raw_inputs": all_image_raw_inputs,
            "image_labels": all_image_raw_inputs,
        }

    return dataset.map(process_function, batched=True, batch_size=256, remove_columns=dataset.column_names)