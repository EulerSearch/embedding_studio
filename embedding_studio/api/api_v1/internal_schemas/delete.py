from typing import List

from pydantic import Field

from embedding_studio.api.api_v1.internal_schemas.task import (
    BaseInternalTaskRequest,
    BaseInternalTaskResponse,
)
from embedding_studio.api.api_v1.schemas.delete import FailedItemIdWithDetail


class DeletionTaskRunRequest(BaseInternalTaskRequest):
    object_ids: List[str] = Field(...)


class DeletionTaskResponse(BaseInternalTaskResponse):
    failed_item_ids: List[FailedItemIdWithDetail] = Field(...)
