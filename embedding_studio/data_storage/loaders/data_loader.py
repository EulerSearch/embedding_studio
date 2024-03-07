from abc import ABC, abstractmethod
from typing import List

from datasets import Dataset

from embedding_studio.data_storage.loaders.item_meta import ItemMeta


class DataLoader(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def load(self, items_data: List[ItemMeta]) -> Dataset:
        raise NotImplemented
