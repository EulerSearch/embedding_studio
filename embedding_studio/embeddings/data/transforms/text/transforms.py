import logging
from typing import Callable, Optional

from datasets import Dataset

from embedding_studio.embeddings.data.transforms.text.dummy import do_nothing

logger = logging.getLogger(__name__)


def text_transforms(
    examples: Dataset,
    transform: Optional[Callable] = do_nothing,
    raw_text_field_name: Optional[str] = "item",
    text_values_name: Optional[str] = "text",
) -> Dataset:
    """Apply transform function on text_values_name column to get a new raw_text_field_name column.

    :param examples: dataset to add a new column into.
    :param transform: transform function that applied to raw_text_field_name column to get raw_text_field_name column (default: do_nothing)
    :param raw_text_field_name: original text column to be transformed (default: "item")
    :param text_values_name: text field to be used for embedding model fine-tuning (default: "text")
    :return:
    """
    examples[text_values_name] = [
        transform(row) for row in examples[raw_text_field_name]
    ]

    return examples
