from typing import List

from embedding_studio.data_storage.loaders.item_meta import ItemMeta
from embedding_studio.embeddings.features.fine_tuning_input import (
    FineTuningInput,
)


class FineTuningInputWithItems:
    """
    A container class that combines fine-tuning input with associated items.

    This class enables associating metadata items with fine-tuning inputs
    for more comprehensive model training.

    :param input: The fine-tuning input data
    :param items: List of metadata items associated with the input
    """

    def __init__(self, input: FineTuningInput, items: List[ItemMeta]):
        self.input = input
        self.items = items
