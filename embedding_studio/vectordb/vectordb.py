from abc import ABC, abstractmethod
from typing import List, Optional

from embedding_studio.models.embeddings.collections import CollectionStateInfo
from embedding_studio.models.embeddings.models import (
    EmbeddingModelInfo,
    SearchIndexInfo,
)
from embedding_studio.vectordb.collection import Collection


class VectorDb(ABC):
    @abstractmethod
    def list_collections(self) -> List[CollectionStateInfo]:
        raise NotImplementedError()

    @abstractmethod
    def get_blue_collection(self) -> Optional[Collection]:
        raise NotImplementedError()

    @abstractmethod
    def get_collection(
        self, embedding_model: EmbeddingModelInfo
    ) -> Collection:
        raise NotImplementedError()

    @abstractmethod
    def create_collection(
        self,
        embedding_model: EmbeddingModelInfo,
        search_index_info: SearchIndexInfo,
    ) -> Collection:
        raise NotImplementedError()

    @abstractmethod
    def collection_exists(self, embedding_model: EmbeddingModelInfo) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def delete_collection(self, embedding_model: EmbeddingModelInfo) -> None:
        raise NotImplementedError()

    @abstractmethod
    def set_blue_collection(self, embedding_model: EmbeddingModelInfo) -> None:
        raise NotImplementedError()
