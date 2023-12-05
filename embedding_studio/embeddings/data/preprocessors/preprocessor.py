from abc import ABC, abstractmethod

from datasets import DatasetDict


class ItemsDatasetDictPreprocessor(ABC):
    """Interface to items dataset dict preprocessing"""

    @abstractmethod
    def convert(self, dataset: DatasetDict) -> DatasetDict:
        raise NotImplemented()

    @abstractmethod
    def get_id_field_name(self) -> str:
        raise NotImplemented()
