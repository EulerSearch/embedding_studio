from typing import List, Optional

from pydantic import BaseModel

from embedding_studio.models.embeddings.collections import CollectionStateInfo
from embedding_studio.models.embeddings.models import (
    EmbeddingModelInfo,
    SearchIndexInfo,
)
from embedding_studio.models.embeddings.objects import Object


class CreateCollectionRequest(BaseModel):
    embedding_model: EmbeddingModelInfo


class CreateIndexRequest(BaseModel):
    embedding_model: EmbeddingModelInfo


class DeleteCollectionRequest(BaseModel):
    embedding_model: EmbeddingModelInfo


class GetCollectionInfoRequest(BaseModel):
    embedding_model: EmbeddingModelInfo


class SetBlueCollectionRequest(BaseModel):
    embedding_model: EmbeddingModelInfo
    search_index_info: SearchIndexInfo


class ListCollectionsResponse(BaseModel):
    collections: List[CollectionStateInfo]


class InsertObjectsRequest(BaseModel):
    objects: List[Object]
    embedding_model: Optional[EmbeddingModelInfo] = None


class UpsertObjectsRequest(BaseModel):
    objects: List[Object]
    shrink_parts: bool = True
    embedding_model: Optional[EmbeddingModelInfo] = None


class DeleteObjectRequest(BaseModel):
    object_ids: List[str]
    embedding_model: Optional[EmbeddingModelInfo] = None


class FindObjectsByIdsRequest(BaseModel):
    object_ids: List[str]
    embedding_model: Optional[EmbeddingModelInfo] = None


class FindSimilarObjectsRequest(BaseModel):
    query_vector: List[float]
    limit: int
    offset: Optional[int] = None
    max_distance: Optional[float] = None
    embedding_model: Optional[EmbeddingModelInfo] = None
