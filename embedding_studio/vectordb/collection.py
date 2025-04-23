from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, List, Optional, Tuple

from embedding_studio.models.embeddings.collections import (
    CollectionInfo,
    CollectionStateInfo,
)
from embedding_studio.models.embeddings.objects import (
    Object,
    ObjectsCommonDataBatch,
    ObjectWithDistance,
    SearchResults,
)
from embedding_studio.models.payload.models import PayloadFilter
from embedding_studio.models.sort_by.models import SortByOptions


class Collection(ABC):
    """
    Abstract base class representing a vector collection.

    A collection stores and provides access to vector embeddings and their associated
    metadata, allowing for operations like insertion, retrieval, and similarity search.
    """

    @abstractmethod
    def get_info(self) -> CollectionInfo:
        """
        Get information about the collection.

        :return: Collection information object

        Example implementation:
        ```python
        def get_info(self) -> CollectionInfo:
            return self.collection_info
        ```
        """
        raise NotImplementedError()

    @abstractmethod
    def get_state_info(self) -> CollectionStateInfo:
        """
        Get the current state information of the collection.

        :return: Collection state information object

        Example implementation:
        ```python
        def get_state_info(self) -> CollectionStateInfo:
            info = self.get_info()
            return CollectionStateInfo(
                **info.model_dump(),
                work_state=CollectionWorkState.GREEN
            )
        ```
        """
        raise NotImplementedError()

    @abstractmethod
    @contextmanager
    def lock_objects(self, object_ids: List[str]):
        """
        Context manager to lock the specified objects.

        This method acquires locks for the specified objects to prevent
        concurrent modifications during critical operations.

        :param object_ids: List of object IDs to lock

        Example implementation:
        ```python
        @contextmanager
        def lock_objects(self, object_ids: List[str]):
            # Sort IDs to prevent deadlocks
            sorted_ids = sorted(object_ids)
            locks = []
            try:
                # Acquire locks
                for obj_id in sorted_ids:
                    lock = self._lock_manager.acquire_lock(f"obj:{obj_id}")
                    locks.append(lock)
                yield
            finally:
                # Release locks in reverse order
                for lock in reversed(locks):
                    self._lock_manager.release_lock(lock)
        ```
        """
        raise NotImplementedError()

    @abstractmethod
    def insert(self, objects: List[Object]) -> None:
        """
        Insert objects into the collection.

        :param objects: List of objects to insert
        :return: None

        Example implementation:
        ```python
        def insert(self, objects: List[Object]) -> None:
            object_ids = [obj.id for obj in objects]
            with self.lock_objects(object_ids):
                for obj in objects:
                    self._storage.insert_one(obj)
        ```
        """
        raise NotImplementedError()

    @abstractmethod
    def create_index(self) -> None:
        """
        Create an index for this collection to optimize similarity searches.

        :return: None

        Example implementation:
        ```python
        def create_index(self) -> None:
            if not self._index_exists():
                self._storage.create_index(self.get_info().collection_id)
                self._collection_cache.set_index_state(self.get_info().collection_id, True)
        ```
        """
        raise NotImplementedError()

    @abstractmethod
    def upsert(self, objects: List[Object], shrink_parts: bool = True) -> None:
        """
        Update existing objects or insert new ones.

        :param objects: List of objects to upsert
        :param shrink_parts: Whether to optimize storage after upsert
        :return: None

        Example implementation:
        ```python
        def upsert(self, objects: List[Object], shrink_parts: bool = True) -> None:
            object_ids = [obj.id for obj in objects]
            with self.lock_objects(object_ids):
                existing_objects = self.find_by_ids(object_ids)
                existing_ids = {obj.id for obj in existing_objects}

                # Update existing objects
                for obj in objects:
                    if obj.id in existing_ids:
                        self._storage.update_one(obj)
                    else:
                        self._storage.insert_one(obj)

                if shrink_parts:
                    self._storage.optimize()
        ```
        """
        raise NotImplementedError()

    @abstractmethod
    def delete(self, object_ids: List[str]) -> None:
        """
        Delete objects from the collection.

        :param object_ids: List of object IDs to delete
        :return: None

        Example implementation:
        ```python
        def delete(self, object_ids: List[str]) -> None:
            with self.lock_objects(object_ids):
                for obj_id in object_ids:
                    self._storage.delete_one(obj_id)
        ```
        """
        raise NotImplementedError()

    @abstractmethod
    def find_by_ids(self, object_ids: List[str]) -> List[Object]:
        """
        Find objects by their IDs.

        :param object_ids: List of object IDs to find
        :return: List of found objects

        Example implementation:
        ```python
        def find_by_ids(self, object_ids: List[str]) -> List[Object]:
            results = []
            for obj_id in object_ids:
                obj = self._storage.find_one(obj_id)
                if obj:
                    results.append(obj)
            return results
        ```
        """
        raise NotImplementedError()

    @abstractmethod
    def find_by_original_ids(self, object_ids: List[str]) -> List[Object]:
        """
        Find objects by their original IDs.

        :param object_ids: List of original object IDs to find
        :return: List of found objects

        Example implementation:
        ```python
        def find_by_original_ids(self, object_ids: List[str]) -> List[Object]:
            return self._storage.find(
                filter={"original_id": {"$in": object_ids}}
            )
        ```
        """
        raise NotImplementedError()

    @abstractmethod
    def get_total(self) -> int:
        """
        Get the total number of objects in the collection.

        :return: Total object count

        Example implementation:
        ```python
        def get_total(self) -> int:
            return self._storage.count()
        ```
        """
        raise NotImplementedError()

    @abstractmethod
    def get_objects_common_data_batch(
        self,
        limit: int,
        offset: Optional[int] = None,
    ) -> ObjectsCommonDataBatch:
        """
        Get a batch of common data for objects in the collection.

        :param limit: Maximum number of objects to return
        :param offset: Number of objects to skip
        :return: Batch of common object data

        Example implementation:
        ```python
        def get_objects_common_data_batch(
            self,
            limit: int,
            offset: Optional[int] = None,
        ) -> ObjectsCommonDataBatch:
            offset = offset or 0
            objects = self._storage.find_many(
                skip=offset,
                limit=limit,
                projection={"id": 1, "original_id": 1, "payload": 1}
            )

            return ObjectsCommonDataBatch(
                objects=objects,
                total=self.get_total()
            )
        ```
        """
        raise NotImplementedError()

    @abstractmethod
    def find_similarities(
        self,
        query_vector: List[float],
        limit: int,
        offset: Optional[int] = None,
        max_distance: Optional[float] = None,
        payload_filter: Optional[PayloadFilter] = None,
        sort_by: Optional[SortByOptions] = None,
        user_id: Optional[str] = None,
        similarity_first: bool = False,
        meta_info: Any = None,
    ) -> SearchResults:
        """
        Find similar vectors based on a query vector.

        :param query_vector: Vector to find similarities for
        :param limit: Maximum number of results to return
        :param offset: Number of results to skip
        :param max_distance: Maximum distance threshold for similarity
        :param payload_filter: Filter to apply to object payloads
        :param sort_by: Options for sorting results
        :param user_id: ID of the user performing the search
        :param similarity_first: Whether to prioritize similarity in results
        :param meta_info: Additional metadata for the search
        :return: Search results

        Example implementation:
        ```python
        def find_similarities(
            self,
            query_vector: List[float],
            limit: int,
            offset: Optional[int] = None,
            max_distance: Optional[float] = None,
            payload_filter: Optional[PayloadFilter] = None,
            sort_by: Optional[SortByOptions] = None,
            user_id: Optional[str] = None,
            similarity_first: bool = False,
            meta_info: Any = None,
        ) -> SearchResults:
            objects, search_meta = self.find_similar_objects(
                query_vector=query_vector,
                limit=limit,
                offset=offset,
                max_distance=max_distance,
                payload_filter=payload_filter,
                sort_by=sort_by,
                user_id=user_id,
                similarity_first=similarity_first,
                meta_info=meta_info
            )

            return SearchResults(
                objects=[obj.object for obj in objects],
                distances=[obj.distance for obj in objects],
                meta=search_meta
            )
        ```
        """
        raise NotImplementedError()

    @abstractmethod
    def find_similar_objects(
        self,
        query_vector: List[float],
        limit: int,
        offset: Optional[int] = None,
        max_distance: Optional[float] = None,
        payload_filter: Optional[PayloadFilter] = None,
        sort_by: Optional[SortByOptions] = None,
        user_id: Optional[str] = None,
        with_vectors: bool = False,
        similarity_first: bool = False,
        meta_info: Any = None,
    ) -> Tuple[List[ObjectWithDistance], Any]:
        """
        Find similar objects based on a query vector.

        :param query_vector: Vector to find similarities for
        :param limit: Maximum number of results to return
        :param offset: Number of results to skip
        :param max_distance: Maximum distance threshold for similarity
        :param payload_filter: Filter to apply to object payloads
        :param sort_by: Options for sorting results
        :param user_id: ID of the user performing the search
        :param with_vectors: Whether to include vectors in results
        :param similarity_first: Whether to prioritize similarity in results
        :param meta_info: Additional metadata for the search
        :return: Tuple of (list of objects with distances, search metadata)

        Example implementation:
        ```python
        def find_similar_objects(
            self,
            query_vector: List[float],
            limit: int,
            offset: Optional[int] = None,
            max_distance: Optional[float] = None,
            payload_filter: Optional[PayloadFilter] = None,
            sort_by: Optional[SortByOptions] = None,
            user_id: Optional[str] = None,
            with_vectors: bool = False,
            similarity_first: bool = False,
            meta_info: Any = None,
        ) -> Tuple[List[ObjectWithDistance], Any]:
            offset = offset or 0

            # Prepare search parameters
            search_params = {
                "vector": query_vector,
                "limit": limit,
                "offset": offset,
                "with_payload": True,
                "with_vector": with_vectors,
            }

            if max_distance is not None:
                search_params["score_threshold"] = 1.0 - max_distance

            if payload_filter is not None:
                search_params["filter"] = payload_filter.to_filter_dict()

            # Execute search
            results = self._index.search(**search_params)

            # Process results
            objects_with_distance = []
            for hit in results.get("results", []):
                obj = Object(
                    id=hit["id"],
                    vector=hit.get("vector"),
                    payload=hit.get("payload", {})
                )
                distance = 1.0 - hit.get("score", 0.0)
                objects_with_distance.append(ObjectWithDistance(
                    object=obj,
                    distance=distance
                ))

            return objects_with_distance, results.get("meta", {})
        ```
        """
        raise NotImplementedError()

    @abstractmethod
    def find_by_payload_filter(
        self,
        payload_filter: PayloadFilter,
        limit: int,
        offset: Optional[int] = None,
        sort_by: Optional[SortByOptions] = None,
    ) -> SearchResults:
        """
        Find objects by applying a filter to their payloads.

        :param payload_filter: Filter to apply to object payloads
        :param limit: Maximum number of results to return
        :param offset: Number of results to skip
        :param sort_by: Options for sorting results
        :return: Search results

        Example implementation:
        ```python
        def find_by_payload_filter(
            self,
            payload_filter: PayloadFilter,
            limit: int,
            offset: Optional[int] = None,
            sort_by: Optional[SortByOptions] = None,
        ) -> SearchResults:
            offset = offset or 0

            # Convert filter to storage-specific format
            filter_dict = payload_filter.to_filter_dict()

            # Apply sorting if specified
            sort_options = None
            if sort_by:
                sort_options = [(sort_by.field, 1 if sort_by.ascending else -1)]

            # Execute query
            objects = self._storage.find_many(
                filter=filter_dict,
                skip=offset,
                limit=limit,
                sort=sort_options
            )

            return SearchResults(
                objects=objects,
                distances=[0.0] * len(objects),  # No distances for filter-based search
                meta={"filtered_count": len(objects)}
            )
        ```
        """
        raise NotImplementedError()

    @abstractmethod
    def count_by_payload_filter(self, payload_filter: PayloadFilter) -> int:
        """
        Count objects that match a payload filter.

        :param payload_filter: Filter to apply to object payloads
        :return: Count of matching objects

        Example implementation:
        ```python
        def count_by_payload_filter(
                self,
                payload_filter: PayloadFilter
        ) -> int:
            # Convert filter to storage-specific format
            filter_dict = payload_filter.to_filter_dict()

            # Execute count query
            return self._storage.count(filter=filter_dict)
        ```
        """
        raise NotImplementedError()


class QueryCollection(Collection):
    """
    Abstract base class extending Collection with query-specific functionality.

    A QueryCollection is specialized for storing and retrieving query vectors
    and associated data, often used for query analysis and optimization.
    """

    @abstractmethod
    def get_objects_by_session_id(self, session_id: str) -> Object:
        """
        Get objects associated with a specific session ID.

        :param session_id: The session ID to search for
        :return: Object associated with the session

        Example implementation:
        ```python
        def get_objects_by_session_id(self, session_id: str) -> Object:
            # Find objects with the specified session ID in payload
            filter_dict = {"payload.session_id": session_id}
            objects = self._storage.find_many(filter=filter_dict)

            if not objects:
                return None

            # Return the first matching object
            return objects[0]
        ```
        """
        raise NotImplementedError()
