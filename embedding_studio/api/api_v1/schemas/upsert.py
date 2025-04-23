import enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from embedding_studio.api.api_v1.schemas.task import (
    BaseTaskRequest,
    BaseTaskResponse,
)


class UpsertionFailureStage(str, enum.Enum):
    """
    Enum representing points in the upsert pipeline where failures may occur.
    Enables precise diagnosis of different failure types during data processing.
    Supports targeted improvements to specific processing stages.
    Critical for effective monitoring and optimization of the upsert pipeline.
    """

    on_downloading = "on_downloading"
    on_inference = "on_inference"
    on_splitting = "on_splitting"
    on_upsert = "on_upsert"
    other = "other"


class DataItem(BaseModel):
    """
    Core data structure for objects being inserted or updated in the system.
    Combines unique identifier with optional payload and metadata.
    Provides flexible structure for various content types and information models.
    Central to the system's data ingestion and management capabilities.
    """

    object_id: str
    payload: Optional[Dict[str, Any]] = None
    item_info: Optional[Dict[str, Any]] = None


class FailedDataItem(DataItem):
    """
    Extends DataItem with detailed failure information for diagnostics.
    Captures both the error message and the stage where failure occurred.
    Enables precise troubleshooting of upsert pipeline issues.
    Supports reliable error handling and reporting for data operations.
    """

    detail: str = Field(
        ..., description="Detailed error message explaining the failure"
    )
    failure_stage: UpsertionFailureStage = UpsertionFailureStage.other


class UpsertionTaskRunRequest(BaseTaskRequest):
    """
    Defines a batch upsert operation for multiple data items.
    Supports efficient bulk addition or update of content in the vector database.
    Enables atomic management of vector database contents.
    Critical for maintaining data freshness and consistency.
    """

    items: List[DataItem] = Field(...)


class UpsertionTaskResponse(BaseTaskResponse):
    """
    Reports the outcome of a batch upsert operation.
    Provides detailed error information for any failed items.
    Enables clients to determine which items need retry or special handling.
    Supports reliable upsert workflows with comprehensive reporting.
    """

    failed_items: List[FailedDataItem] = Field(...)
