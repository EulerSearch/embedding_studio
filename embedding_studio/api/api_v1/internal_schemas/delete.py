from typing import List

from pydantic import Field

from embedding_studio.api.api_v1.internal_schemas.task import (
    BaseInternalTaskRequest,
    BaseInternalTaskResponse,
)
from embedding_studio.api.api_v1.schemas.delete import FailedItemIdWithDetail


class DeletionTaskRunRequest(BaseInternalTaskRequest):
    """
    Request schema for removing specific objects from vector storage.
    Identifies target objects by ID and links them to a specific embedding model.
    Enables systematic data maintenance and cleanup operations.
    Supports bulk operations for efficient vector database management.
    """

    object_ids: List[str] = Field(...)


class DeletionTaskResponse(BaseInternalTaskResponse):
    """
    Response structure detailing vector embedding deletion results.
    Documents objects that couldn't be deleted with their corresponding error details.
    Maintains context about which embedding model was targeted by the operation.
    Facilitates complete auditing of deletion operations for data governance.
    """

    failed_item_ids: List[FailedItemIdWithDetail] = Field(...)
