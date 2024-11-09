from typing import List, Optional

from pydantic import BaseModel, Field

from embedding_studio.api.api_v1.schemas.task import (
    BaseTaskRequest,
    BaseTaskResponse,
)
from embedding_studio.api.api_v1.schemas.upsert import FailedDataItem


class ModelParams(BaseModel):
    embedding_model_id: str = Field(...)
    fine_tuning_method: str = Field(...)


class ReindexTaskRunRequest(BaseTaskRequest):
    source: ModelParams = Field(...)
    dest: ModelParams = Field(...)


class ReindexTaskResponse(BaseTaskResponse):
    progress: float = Field(
        default=0.0,
        ge=0.0,
        description="Task progress represented as a percentage (0-100)",
    )
    detail: Optional[str] = Field(
        None,
        max_length=500,
        description="Details or information about the current status of the task",
    )
    count: Optional[int] = Field(
        default=0, ge=0, description="Number of processed items so far"
    )
    total: Optional[int] = Field(
        None, ge=0, description="Total number of items to process"
    )

    children: List[str] = Field(default_factory=list)
    failed_items: List[FailedDataItem] = Field(...)
