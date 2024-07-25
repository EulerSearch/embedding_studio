from typing import List

from embedding_studio.data_storage.loaders.item_meta import ItemMeta
from embedding_studio.embeddings.features.fine_tuning_input import (
    FineTuningInput,
)


class FineTuningInputWithItems:
    def __init__(self, input: FineTuningInput, items: List[ItemMeta]):
        self.input = input
        self.items = items
