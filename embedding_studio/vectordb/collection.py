from abc import ABC, abstractmethod
from typing import List, Optional

from embedding_studio.models.embeddings.collections import CollectionStateInfo
from embedding_studio.models.embeddings.objects import Object, SearchResults
from embedding_studio.models.payload.models import PayloadFilter


class Collection(ABC):
    @abstractmethod
    def get_state_info(self) -> CollectionStateInfo:
        raise NotImplementedError()

    @abstractmethod
    def insert(self, objects: List[Object]) -> None:
        raise NotImplementedError()

    @abstractmethod
    def create_index(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def upsert(
        self,
        objects: List[Object],
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def delete(self, object_ids: List[str]) -> None:
        raise NotImplementedError()

    @abstractmethod
    def find_by_ids(self, object_ids: List[str]) -> List[Object]:
        raise NotImplementedError()

    @abstractmethod
    def find_similarities(
        self,
        query_vector: List[float],
        limit: int,
        offset: Optional[int] = None,
        max_distance: Optional[float] = None,
        payload_filter: Optional[PayloadFilter] = None,
    ) -> SearchResults:
        raise NotImplementedError()

    @abstractmethod
    def find_by_payload_filter(
        self,
        payload_filter: PayloadFilter,
        limit: int,
        offset: Optional[int] = None,
    ) -> SearchResults:
        raise NotImplementedError()
