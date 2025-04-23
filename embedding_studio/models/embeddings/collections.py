from enum import Enum
from typing import List

from pydantic import AwareDatetime, BaseModel, Field

from embedding_studio.models.embeddings.models import EmbeddingModelInfo


class CollectionWorkState(str, Enum):
    """
    An enum that defines the operational state of a collection in the vector database -
    either "green" (standard) or "blue" (active/primary).
    Used to track which collection is currently active and serving production queries.
    """

    GREEN = "green"
    BLUE = "blue"


class CollectionInfo(BaseModel):
    """
    A base model containing essential information about a vector collection,
    including its identifier and the embedding model it uses.
    Serves as a reference point for creating, retrieving, and managing collections.
    """

    collection_id: str
    embedding_model: EmbeddingModelInfo


class CollectionStateInfo(CollectionInfo):
    """
    Extends CollectionInfo with additional state information such as creation time, index status, and work state.
    Used for monitoring collection status and providing operational insights.

    It's specifically designed for caching collection metadata to avoid repeated database queries.
    The CollectionInfoCache class maintains these objects in memory after initial loading,
    providing fast access to collection states without hitting the database for every request.
    """

    created_at: AwareDatetime
    index_created: bool
    work_state: CollectionWorkState
    applied_optimizations: List[str] = Field(default_factory=list)
