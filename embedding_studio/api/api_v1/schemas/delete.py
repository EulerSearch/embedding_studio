from typing import List

from pydantic import BaseModel, Field

from embedding_studio.api.api_v1.schemas.task import (
    BaseTaskRequest,
    BaseTaskResponse,
)


class FailedItemIdWithDetail(BaseModel):
    """
    Detailed error reporting structure for failed deletion operations.
    Associates specific error messages with affected object identifiers.
    Enables precise diagnosis and resolution of deletion failures.
    Supports transparent error reporting to clients and operators.
    """

    object_id: str
    detail: str = Field(
        description="Detailed error message explaining the failure"
    )


class DeletionTaskRunRequest(BaseTaskRequest):
    """
    Defines a batch deletion operation across multiple objects.
    Supports efficient bulk removal of content from the vector database.
    Enables atomic management of vector database contents.
    Critical for maintaining data freshness and compliance requirements.
    """

    object_ids: List[str] = Field(...)


class DeletionTaskResponse(BaseTaskResponse):
    """
    Reports the outcome of a batch deletion operation.
    Provides detailed error information for any failed deletions.
    Enables clients to determine which items need retry or special handling.
    Supports reliable deletion workflows with comprehensive reporting.
    """

    failed_item_ids: List[FailedItemIdWithDetail] = Field(...)
