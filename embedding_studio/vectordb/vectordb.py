from abc import ABC, abstractmethod
from typing import List, Optional

from embedding_studio.models.embeddings.collections import CollectionStateInfo
from embedding_studio.models.embeddings.models import (
    EmbeddingModelInfo,
    SearchIndexInfo,
)
from embedding_studio.vectordb.collection import Collection, QueryCollection


class VectorDb(ABC):
    def get_query_collection_id(self, collection_name: str) -> str:
        return f"{collection_name}_q"

    @abstractmethod
    def update_info(self):
        raise NotImplementedError()

    @abstractmethod
    def list_collections(self) -> List[CollectionStateInfo]:
        raise NotImplementedError()

    @abstractmethod
    def list_query_collections(self) -> List[CollectionStateInfo]:
        raise NotImplementedError()

    @abstractmethod
    def get_blue_collection(self) -> Optional[Collection]:
        raise NotImplementedError()

    @abstractmethod
    def get_blue_query_collection(self) -> Optional[QueryCollection]:
        raise NotImplementedError()

    @abstractmethod
    def get_collection(
        self, embedding_model: EmbeddingModelInfo
    ) -> Collection:
        raise NotImplementedError()

    @abstractmethod
    def get_query_collection(
        self, embedding_model: EmbeddingModelInfo
    ) -> QueryCollection:
        raise NotImplementedError()

    @abstractmethod
    def create_collection(
        self,
        embedding_model: EmbeddingModelInfo,
        search_index_info: SearchIndexInfo,
    ) -> Collection:
        raise NotImplementedError()

    @abstractmethod
    def create_query_collection(
        self,
        embedding_model: EmbeddingModelInfo,
        search_index_info: SearchIndexInfo,
    ) -> QueryCollection:
        raise NotImplementedError()

    def get_or_create_collection(
        self,
        embedding_model: EmbeddingModelInfo,
        search_index_info: SearchIndexInfo,
    ) -> Collection:
        if not self.collection_exists(embedding_model):
            return self.create_collection(embedding_model, search_index_info)
        else:
            return self.get_collection(embedding_model)

    def get_or_create_query_collection(
        self,
        embedding_model: EmbeddingModelInfo,
        search_index_info: SearchIndexInfo,
    ) -> QueryCollection:
        if not self.query_collection_exists(embedding_model):
            return self.create_query_collection(
                embedding_model, search_index_info
            )
        else:
            return self.get_query_collection(embedding_model)

    @abstractmethod
    def collection_exists(self, embedding_model: EmbeddingModelInfo) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def query_collection_exists(
        self, embedding_model: EmbeddingModelInfo
    ) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def delete_collection(self, embedding_model: EmbeddingModelInfo) -> None:
        raise NotImplementedError()

    @abstractmethod
    def delete_query_collection(
        self, embedding_model: EmbeddingModelInfo
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def set_blue_collection(self, embedding_model: EmbeddingModelInfo) -> None:
        raise NotImplementedError()
