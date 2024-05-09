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
    Applies a transformation to the images in a dataset.

    :param examples: The input dataset containing images.
    :param transform: A callable that applies a transformation to an image.
                      Default is center_crop_transform.
    :param n_pixels: The size of the square image to produce. Should be a positive integer.
                     Default is 224 pixels.
    :param image_field_name: The name of the field in the dataset that contains the images.
                             Default is "item".
    :param pixel_values_name: The name of the field to store the transformed images in.
                              Default is "pixel_values".
    :param del_images: Whether to delete the original images from the dataset after transforming.
                       Default is False.

    :return: The updated dataset with transformed images.
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
