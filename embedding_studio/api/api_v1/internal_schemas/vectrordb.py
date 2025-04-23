from typing import List, Optional

from pydantic import BaseModel

from embedding_studio.models.embeddings.collections import CollectionStateInfo
from embedding_studio.models.embeddings.objects import Object


class CreateCollectionRequest(BaseModel):
    """
    Request schema for creating a new vector collection for a specific model.
    Specifies which embedding model's vectors will be stored in this collection.
    Initiates the creation of both standard and query-optimized collections.
    """

    embedding_model_id: str


class CreateIndexRequest(BaseModel):
    """
    Request schema for building search indexes on existing vector collections.
    Identifies which embedding model's collection requires indexing.
    """

    embedding_model_id: str


class DeleteCollectionRequest(BaseModel):
    """
    Request schema for removing vector collections and their associated indexes.
    Specifies which embedding model's collection should be deleted.
    Handles cleanup of both main collections and their query-optimized variants.
    """

    embedding_model_id: str


class GetCollectionInfoRequest(BaseModel):
    """
    Request schema for retrieving metadata about a specific vector collection.
    Identifies which embedding model's collection information is needed.
    Provides access to collection status, index state, and operational metrics.
    """

    embedding_model_id: str


class SetBlueCollectionRequest(BaseModel):
    """
    Request schema for promoting a collection to 'blue' (active/primary) status.
    Specifies which embedding model's collection should become the primary one.
    Implements blue-green deployment pattern for vector collection switching.
    """

    embedding_model_id: str


class ListCollectionsResponse(BaseModel):
    """
    Response structure containing information about available vector collections.
    Provides a comprehensive view of all collections in the vector database.
    Contains collection state, index status, and embedding model details.
    Supports discovery and inventory management of vector storage resources.
    """

    collections: List[CollectionStateInfo]


class InsertObjectsRequest(BaseModel):
    """
    Request schema for adding new vector objects to a collection.
    Specifies the target embedding model and objects with their vectors.
    """

    objects: List[Object]
    embedding_model_id: str


class UpsertObjectsRequest(BaseModel):
    """
    Request schema for adding or updating vector objects in a collection.
    Combines insertion and update operations with idempotent semantics.
    Includes options for vector part management and optimization.
    """

    objects: List[Object]
    shrink_parts: bool = True
    embedding_model_id: str


class DeleteObjectRequest(BaseModel):
    """
    Request schema for removing specific vector objects from a collection.
    Identifies objects by ID within a specific embedding model's collection.
    """

    object_ids: List[str]
    embedding_model_id: str


class FindObjectsByIdsRequest(BaseModel):
    """
    Request schema for directly retrieving vector objects by their identifiers.
    Specifies which objects to fetch from a particular embedding model's collection.
    Bypasses similarity search for exact-match retrieval operations.
    """

    object_ids: List[str]
    embedding_model_id: str


class FindSimilarObjectsRequest(BaseModel):
    """
    Request schema for performing vector similarity searches.
    Contains the query vector to match against stored vectors in a collection.
    Includes search parameters like limit, offset, and maximum distance.
    """

    query_vector: List[float]
    limit: int
    offset: Optional[int] = None
    max_distance: Optional[float] = None
    embedding_model_id: str
