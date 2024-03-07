import logging
from typing import Callable, Optional

from datasets import Dataset

from embedding_studio.embeddings.data.transforms.dict.line_from_dict import (
    get_text_line_from_dict,
)

logger = logging.getLogger(__name__)


def dict_transforms(
    examples: Dataset,
    transform: Optional[Callable] = get_text_line_from_dict,
    text_values_name: Optional[str] = "text",
) -> Dataset:
    """Add text value from a dataset row.

    :param examples: dataset to add text column.
    :param transform: function of creating text out of dataset's row (default: get_text_line_from_dict).
    :param text_values_name: name of the text column.
    :return: dataset with a text column.
    """
    examples[text_values_name] = [transform(row) for row in examples]

    return examples
