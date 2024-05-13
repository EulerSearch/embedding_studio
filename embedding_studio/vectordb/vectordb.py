from abc import ABC, abstractmethod
from typing import List, Optional

from embedding_studio.models.embeddings.collections import CollectionStateInfo
from embedding_studio.models.embeddings.models import EmbeddingModel
from embedding_studio.vectordb.collection import Collection


class VectorDb(ABC):
    @abstractmethod
    def list_collections(self) -> List[CollectionStateInfo]:
        raise NotImplementedError()

    @abstractmethod
    def get_blue_collection(self) -> Optional[Collection]:
        raise NotImplementedError()

    @abstractmethod
    def get_collection(self, collection_id: str) -> Collection:
        raise NotImplementedError()

    @abstractmethod
    def create_collection(
        self, model: EmbeddingModel, collection_id: Optional[str] = None
    ) -> Collection:
        raise NotImplementedError()

    @abstractmethod
    def delete_collection(self, collection_id: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def set_blue_collection(self, collection_id: str) -> None:
        raise NotImplementedError()
