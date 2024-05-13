from typing import List, Optional

from pydantic import BaseModel

from embedding_studio.models.embeddings.collections import CollectionStateInfo
from embedding_studio.models.embeddings.models import EmbeddingModel
from embedding_studio.models.embeddings.objects import Object


class CreateCollectionRequest(BaseModel):
    model: EmbeddingModel
    collection_id: Optional[str] = None


class ListCollectionsResponse(BaseModel):
    collections: List[CollectionStateInfo]


class InsertObjectsRequest(BaseModel):
    objects: List[Object]


class UpsertObjectsRequest(BaseModel):
    objects: List[Object]
    shrink_parts: bool = True


class DeleteObjectRequest(BaseModel):
    object_ids: List[str]


class FindObjectsByIdsRequest(BaseModel):
    object_ids: List[str]


class FindSimilarObjectsRequest(BaseModel):
    query_vector: List[float]
    limit: int
    offset: Optional[int] = None
    max_distance: Optional[float] = None
