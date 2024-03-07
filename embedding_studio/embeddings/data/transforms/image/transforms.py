import logging
from typing import Callable, Optional

from datasets import Dataset

from embedding_studio.embeddings.data.transforms.image.clip_original import (
    center_crop_transform,
)

logger = logging.getLogger(__name__)


def image_transforms(
    examples: Dataset,
    transform: Optional[Callable] = center_crop_transform,
    n_pixels: Optional[int] = 224,
    image_field_name: Optional[str] = "item",
    pixel_values_name: Optional[str] = "pixel_values",
    del_images: Optional[bool] = False,
) -> Dataset:
    """

    :param examples:
    :param transform:
    :param n_pixels:
    :param image_field_name:
    :param pixel_values_name:
    :param del_images:
    :return:
    """
    if not isinstance(n_pixels, int) or n_pixels <= 0:
        raise ValueError(
            f"Num of pixels {n_pixels} should be a positive integer"
        )

    if image_field_name in examples:
        transform_func = transform(n_pixels)
        examples[pixel_values_name] = [
            transform_func(img.convert("RGB"))
            for img in examples[image_field_name]
        ]
        if del_images:
            logger.debug(f"Delete {image_field_name} field from dataset")
            del examples[image_field_name]

    else:
        raise ValueError(
            f"Image field name {image_field_name} is not found in the provided dataset"
        )

    return examples
