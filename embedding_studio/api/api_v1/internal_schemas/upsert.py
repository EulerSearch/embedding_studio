from typing import List

from pydantic import Field

from embedding_studio.api.api_v1.internal_schemas.task import (
    BaseInternalTaskRequest,
    BaseInternalTaskResponse,
)
from embedding_studio.api.api_v1.schemas.upsert import DataItem, FailedDataItem


class UpsertionTaskRunRequest(BaseInternalTaskRequest):
    """
    Request schema for adding or updating vector embeddings in the database.
    Contains the collection of data items to be processed by the embedding model.
    Maps to a specific embedding model that will transform data into vector space.
    Supports atomic operations through transaction-like task identity tracking.
    """

    items: List[DataItem] = Field(...)


class UpsertionTaskResponse(BaseInternalTaskResponse):
    """
    Response structure detailing vector embedding addition/update results.
    Records items that couldn't be processed along with specific failure reasons.
    Provides operational metadata for tracking the task's lifecycle and outcomes.
    Enables targeted retry strategies by isolating problematic data entries.
    """

    failed_items: List[FailedDataItem] = Field(...)
