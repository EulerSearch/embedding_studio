from typing import List

from pydantic import BaseModel, Field

from embedding_studio.api.api_v1.schemas.task import (
    BaseTaskRequest,
    BaseTaskResponse,
)


class FailedItemIdWithDetail(BaseModel):
    object_id: str
    detail: str = Field(
        description="Detailed error message explaining the failure"
    )


class DeletionTaskRunRequest(BaseTaskRequest):
    object_ids: List[str] = Field(...)


class DeletionTaskResponse(BaseTaskResponse):
    failed_item_ids: List[FailedItemIdWithDetail] = Field(...)
