from enum import Enum

from pydantic import AwareDatetime, BaseModel

from embedding_studio.models.embeddings.models import EmbeddingModel


class CollectionWorkState(str, Enum):
    GREEN = "green"
    BLUE = "blue"


class CollectionInfo(BaseModel):
    collection_id: str
    embedding_model: EmbeddingModel


class CollectionStateInfo(CollectionInfo):
    created_at: AwareDatetime
    index_created: bool
    work_state: CollectionWorkState
