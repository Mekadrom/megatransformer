import torch


class ImageVAEDataCollator:
    def __init__(
        self,
        training: bool = True,
    ):
        self.training = training

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        images = []
        for ex in examples:
            if ex is None:
                continue

            image = ex["image"]
            images.append(image)

        # Stack tensors
        image_batch = torch.stack(images)

        batch = {
            "image": image_batch,
        }

        return batch
