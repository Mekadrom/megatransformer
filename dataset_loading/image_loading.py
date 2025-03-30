from datasets import load_dataset
from PIL import Image
from transformers import PreTrainedTokenizer
from torchvision import transforms
from typing import Optional

import requests
import torch

def get_transform(image_size):
    if image_size is None:
        raise ValueError("Image size must be specified for image transformations")
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ])

def fetch_and_transform_image(image_url, transform):
    try:
        image = Image.open(requests.get(
            image_url,
            timeout=10,
            stream=True
        ).raw)
        image = image.convert("RGB")
        image = transform(image)
        return image
    except Exception as e:
        # print(f"Error fetching or transforming image: {e} for URL {image_url}")
        return None
    
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
            image_raw_input = fetch_and_transform_image(image_url, transform)
            if image_raw_input is not None:
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
    return dataset.map(process_function, batched=True, batch_size=200, remove_columns=dataset.column_names)
