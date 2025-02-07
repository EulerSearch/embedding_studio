from typing import List, Optional

from pydantic import BaseModel

from embedding_studio.models.embeddings.collections import CollectionStateInfo
from embedding_studio.models.embeddings.objects import Object


class CreateCollectionRequest(BaseModel):
    embedding_model_id: str


class CreateIndexRequest(BaseModel):
    embedding_model_id: str


class DeleteCollectionRequest(BaseModel):
    embedding_model_id: str


class GetCollectionInfoRequest(BaseModel):
    embedding_model_id: str


class SetBlueCollectionRequest(BaseModel):
    embedding_model_id: str


class ListCollectionsResponse(BaseModel):
    collections: List[CollectionStateInfo]


class InsertObjectsRequest(BaseModel):
    objects: List[Object]
    embedding_model_id: str


class UpsertObjectsRequest(BaseModel):
    objects: List[Object]
    shrink_parts: bool = True
    embedding_model_id: str


class DeleteObjectRequest(BaseModel):
    object_ids: List[str]
    embedding_model_id: str


class FindObjectsByIdsRequest(BaseModel):
    object_ids: List[str]
    embedding_model_id: str


class FindSimilarObjectsRequest(BaseModel):
    query_vector: List[float]
    limit: int
    offset: Optional[int] = None
    max_distance: Optional[float] = None
    embedding_model_id: str
