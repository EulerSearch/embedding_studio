from typing import List, Optional

from pydantic import BaseModel, Field

from embedding_studio.api.api_v1.schemas.task import (
    BaseTaskRequest,
    BaseTaskResponse,
)
from embedding_studio.api.api_v1.schemas.upsert import FailedDataItem


class ModelParams(BaseModel):
    """
    Concise identifier for a specific embedding model in the system.
    Acts as a reference point that routes operations to the correct model instance.
    Used consistently across different endpoints to specify model targeting.
    Essential for systems that support multiple embedding models simultaneously.
    """

    embedding_model_id: str = Field(...)


class ReindexTaskRunRequest(BaseTaskRequest):
    """
    Request schema for migrating data between different embedding models.
    Specifies source and destination models for comprehensive data transfer.
    Includes deployment configuration for controlled production transitions.
    Defines conflict resolution behavior for deterministic migration outcomes.
    """

    source: ModelParams = Field(...)
    dest: ModelParams = Field(...)

    deploy_as_blue: Optional[bool] = Field(...)
    wait_on_conflict: Optional[bool] = Field(...)


class ReindexTaskResponse(BaseTaskResponse):
    """
    Response structure tracking embedding model migration progress and results.
    Provides detailed metrics on completion status and item processing counts.
    Records failed migrations with specific error contexts for targeted resolution.
    Maintains configuration context for deployment and conflict handling preferences.
    """

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

    deploy_as_blue: Optional[bool] = Field(...)
    wait_on_conflict: Optional[bool] = Field(...)
