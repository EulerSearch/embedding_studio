from abc import ABC, abstractmethod
from typing import List, Optional

from embedding_studio.models.embeddings.collections import (
    CollectionInfo,
    CollectionStateInfo,
)
from embedding_studio.models.embeddings.models import EmbeddingModelInfo
from embedding_studio.vectordb.collection import Collection, QueryCollection
from embedding_studio.vectordb.optimization import Optimization


class VectorDb(ABC):
    """
    Abstract base class representing a vector database system.

    This class serves as an interface for vector database operations including
    creating, retrieving, and managing collections of vector embeddings.

    :param optimizations: List of optimizations to apply to regular collections
    :param query_optimizations: List of optimizations to apply to query collections
    """

    def __init__(
        self,
        optimizations: Optional[List[Optimization]] = None,
        query_optimizations: Optional[List[Optimization]] = None,
    ):
        self._optimizations = optimizations if optimizations else []
        self._query_optimizations = (
            query_optimizations if query_optimizations else []
        )

    def get_query_collection_id(self, collection_name: str) -> str:
        """
        Generate a query collection ID based on a collection name.

        :param collection_name: Name of the base collection
        :return: Generated query collection ID string with '_q' suffix
        """
        return f"{collection_name}_q"

    @abstractmethod
    def update_info(self):
        """
        Update the information about collections in the vector database.

        This method should refresh any cached information about collections.

        :return: None

        Example implementation:
        ```python
        def update_info(self):
            # Refresh collection cache from the underlying storage
            self._collection_cache.invalidate_cache()
        ```
        """
        raise NotImplementedError()

    def add_optimization(self, optimization: Optimization):
        """
        Add an optimization strategy to be applied to regular collections.

        :param optimization: An optimization strategy object
        :return: None
        """
        self._optimizations.append(optimization)

    def add_query_optimization(self, optimization: Optimization):
        """
        Add an optimization strategy to be applied to query collections.

        :param optimization: An optimization strategy object
        :return: None
        """
        self._query_optimizations.append(optimization)

    def apply_optimizations(self):
        """
        Apply all registered optimizations to each collection in the database.

        This method applies each optimization in self._optimizations to all collections
        and updates the collection information to track which optimizations have been applied.

        :return: None
        """
        for collection_info in self.list_collections():
            for optimization in self._optimizations:
                collection = self.get_collection(
                    collection_info.embedding_model.id
                )
                if (
                    optimization.name
                    not in collection_info.applied_optimizations
                ):
                    optimization(collection)
                    collection_info.applied_optimizations.append(
                        optimization.name
                    )
                    self.save_collection_info(collection_info)

    def apply_query_optimizations(self):
        """
        Apply all registered query optimizations to each query collection in the database.

        This method applies each optimization in self._query_optimizations to all query collections
        and updates the collection information to track which optimizations have been applied.

        :return: None
        """
        for collection_info in self.list_query_collections():
            for optimization in self._query_optimizations:
                collection = self.get_query_collection(
                    collection_info.embedding_model.id
                )
                if (
                    optimization.name
                    not in collection_info.applied_optimizations
                ):
                    optimization(collection)
                    collection_info.applied_optimizations.append(
                        optimization.name
                    )
                    self.save_query_collection_info(collection_info)

    @abstractmethod
    def save_collection_info(self, collection_info: CollectionInfo):
        """
        Save or update collection information in the database.

        :param collection_info: Collection information to save
        :return: None

        Example implementation:
        ```python
        def save_collection_info(self, collection_info: CollectionInfo):
            # Update collection info in storage
            self._collection_cache.update_collection(collection_info)
        ```
        """
        raise NotImplementedError()

    @abstractmethod
    def save_query_collection_info(self, collection_info: CollectionInfo):
        """
        Save or update query collection information in the database.

        :param collection_info: Query collection information to save
        :return: None

        Example implementation:
        ```python
        def save_query_collection_info(self, collection_info: CollectionInfo):
            # Update query collection info in storage
            self._collection_cache.update_query_collection(collection_info)
        ```
        """
        raise NotImplementedError()

    @abstractmethod
    def list_collections(self) -> List[CollectionStateInfo]:
        """
        List all regular collections in the database.

        :return: List of collection state information objects

        Example implementation:
        ```python
        def list_collections(self) -> List[CollectionStateInfo]:
            # Return all collection state info from cache
            return self._collection_cache.list_collections()
        ```
        """
        raise NotImplementedError()

    @abstractmethod
    def list_query_collections(self) -> List[CollectionStateInfo]:
        """
        List all query collections in the database.

        :return: List of query collection state information objects

        Example implementation:
        ```python
        def list_query_collections(self) -> List[CollectionStateInfo]:
            # Return all query collection state info from cache
            return self._collection_cache.list_query_collections()
        ```
        """
        raise NotImplementedError()

    @abstractmethod
    def get_blue_collection(self) -> Optional[Collection]:
        """
        Get the designated 'blue' collection, which is the primary active collection.

        :return: The blue collection or None if not set

        Example implementation:
        ```python
        def get_blue_collection(self) -> Optional[Collection]:
            blue_info = self._collection_cache.get_blue_collection()
            if not blue_info:
                return None
            return self.get_collection(blue_info.embedding_model.id)
        ```
        """
        raise NotImplementedError()

    @abstractmethod
    def get_blue_query_collection(self) -> Optional[QueryCollection]:
        """
        Get the designated 'blue' query collection, which is the primary active query collection.

        :return: The blue query collection or None if not set

        Example implementation:
        ```python
        def get_blue_query_collection(self) -> Optional[QueryCollection]:
            blue_info = self._collection_cache.get_blue_query_collection()
            if not blue_info:
                return None
            return self.get_query_collection(blue_info.embedding_model.id)
        ```
        """
        raise NotImplementedError()

    @abstractmethod
    def get_collection(self, embedding_model_id: str) -> Collection:
        """
        Get a collection by its embedding model ID.

        :param embedding_model_id: The ID of the embedding model used by the collection
        :return: The requested collection

        Example implementation:
        ```python
        def get_collection(self, embedding_model_id: str) -> Collection:
            info = self._collection_cache.get_collection(embedding_model_id)
            if not info:
                raise CollectionNotFoundError(embedding_model_id)
            return self._collection_factory.create_collection(info)
        ```
        """
        raise NotImplementedError()

    @abstractmethod
    def get_query_collection(self, embedding_model_id: str) -> QueryCollection:
        """
        Get a query collection by its embedding model ID.

        :param embedding_model_id: The ID of the embedding model used by the query collection
        :return: The requested query collection

        Example implementation:
        ```python
        def get_query_collection(self, embedding_model_id: str) -> QueryCollection:
            query_collection_id = self.get_query_collection_id(embedding_model_id)
            info = self._collection_cache.get_collection(query_collection_id)
            if not info:
                raise CollectionNotFoundError(query_collection_id)
            return self._collection_factory.create_query_collection(info)
        ```
        """
        raise NotImplementedError()

    @abstractmethod
    def _create_collection(
        self, embedding_model: EmbeddingModelInfo
    ) -> Collection:
        """
        Internal method to create a new collection for a specific embedding model.

        :param embedding_model: Information about the embedding model
        :return: The newly created collection

        Example implementation:
        ```python
        def _create_collection(self, embedding_model: EmbeddingModelInfo) -> Collection:
            collection_info = CollectionInfo(
                collection_id=embedding_model.id,
                embedding_model=embedding_model,
                applied_optimizations=[]
            )
            self._collection_cache.add_collection(collection_info)
            return self._collection_factory.create_collection(collection_info)
        ```
        """
        raise NotImplementedError()

    def create_collection(
        self, embedding_model: EmbeddingModelInfo
    ) -> Collection:
        """
        Create a new collection for a specific embedding model and apply optimizations.

        :param embedding_model: Information about the embedding model
        :return: The newly created and optimized collection
        """
        collection = self._create_collection(embedding_model)
        for optimization in self._optimizations:
            optimization(collection)

        return collection

    @abstractmethod
    def _create_query_collection(
        self, embedding_model: EmbeddingModelInfo
    ) -> QueryCollection:
        """
        Internal method to create a new query collection for a specific embedding model.

        :param embedding_model: Information about the embedding model
        :return: The newly created query collection

        Example implementation:
        ```python
        def _create_query_collection(self, embedding_model: EmbeddingModelInfo) -> QueryCollection:
            query_collection_id = self.get_query_collection_id(embedding_model.id)
            collection_info = CollectionInfo(
                collection_id=query_collection_id,
                embedding_model=embedding_model,
                applied_optimizations=[]
            )
            self._collection_cache.add_query_collection(collection_info)
            return self._collection_factory.create_query_collection(collection_info)
        ```
        """
        raise NotImplementedError()

    def create_query_collection(
        self, embedding_model: EmbeddingModelInfo
    ) -> QueryCollection:
        """
        Create a new query collection for a specific embedding model and apply optimizations.

        :param embedding_model: Information about the embedding model
        :return: The newly created and optimized query collection
        """
        query_collection = self._create_query_collection(embedding_model)
        for optimization in self._query_optimizations:
            optimization(query_collection)

        return query_collection

    def get_or_create_collection(
        self, embedding_model: EmbeddingModelInfo
    ) -> Collection:
        """
        Get an existing collection or create a new one if it doesn't exist.

        :param embedding_model: Information about the embedding model
        :return: The existing or newly created collection
        """
        if not self.collection_exists(embedding_model.id):
            return self.create_collection(embedding_model)
        else:
            return self.get_collection(embedding_model.id)

    def get_or_create_query_collection(
        self, embedding_model: EmbeddingModelInfo
    ) -> QueryCollection:
        """
        Get an existing query collection or create a new one if it doesn't exist.

        :param embedding_model: Information about the embedding model
        :return: The existing or newly created query collection
        """
        if not self.query_collection_exists(embedding_model.id):
            return self.create_query_collection(embedding_model)
        else:
            return self.get_query_collection(embedding_model.id)

    @abstractmethod
    def collection_exists(self, embedding_model_id: str) -> bool:
        """
        Check if a collection exists for the given embedding model ID.

        :param embedding_model_id: The ID of the embedding model
        :return: True if the collection exists, False otherwise

        Example implementation:
        ```python
        def collection_exists(self, embedding_model_id: str) -> bool:
            collection = self._collection_cache.get_collection(embedding_model_id)
            return collection is not None and not collection.contains_queries
        ```
        """
        raise NotImplementedError()

    @abstractmethod
    def query_collection_exists(self, embedding_model_id: str) -> bool:
        """
        Check if a query collection exists for the given embedding model ID.

        :param embedding_model_id: The ID of the embedding model
        :return: True if the query collection exists, False otherwise

        Example implementation:
        ```python
        def query_collection_exists(self, embedding_model_id: str) -> bool:
            query_collection_id = self.get_query_collection_id(embedding_model_id)
            collection = self._collection_cache.get_collection(query_collection_id)
            return collection is not None and collection.contains_queries
        ```
        """
        raise NotImplementedError()

    @abstractmethod
    def delete_collection(self, embedding_model_id: str) -> None:
        """
        Delete a collection with the given embedding model ID.

        :param embedding_model_id: The ID of the embedding model
        :return: None

        Example implementation:
        ```python
        def delete_collection(self, embedding_model_id: str) -> None:
            blue_collection = self._collection_cache.get_blue_collection()
            if blue_collection and blue_collection.embedding_model.id == embedding_model_id:
                raise DeleteBlueCollectionError()

            # Delete collection from storage
            self._collection_storage.delete_collection(embedding_model_id)
            # Update cache
            self._collection_cache.delete_collection(embedding_model_id)
        ```
        """
        raise NotImplementedError()

    @abstractmethod
    def delete_query_collection(self, embedding_model_id: str) -> None:
        """
        Delete a query collection with the given embedding model ID.

        :param embedding_model_id: The ID of the embedding model
        :return: None

        Example implementation:
        ```python
        def delete_query_collection(self, embedding_model_id: str) -> None:
            query_collection_id = self.get_query_collection_id(embedding_model_id)
            blue_query_collection = self._collection_cache.get_blue_query_collection()
            if blue_query_collection and blue_query_collection.embedding_model.id == query_collection_id:
                raise DeleteBlueCollectionError()

            # Delete query collection from storage
            self._collection_storage.delete_collection(query_collection_id)
            # Update cache
            self._collection_cache.delete_collection(query_collection_id)
        ```
        """
        raise NotImplementedError()

    @abstractmethod
    def set_blue_collection(self, embedding_model_id: str) -> None:
        """
        Set the 'blue' collection to the one with the given embedding model ID.

        This marks a collection as the primary active collection in the system.

        :param embedding_model_id: The ID of the embedding model
        :return: None

        Example implementation:
        ```python
        def set_blue_collection(self, embedding_model_id: str) -> None:
            query_collection_id = self.get_query_collection_id(embedding_model_id)
            self._collection_cache.set_blue_collection(embedding_model_id, query_collection_id)
        ```
        """
        raise NotImplementedError()
