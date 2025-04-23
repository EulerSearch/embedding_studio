from pydantic import Field

from embedding_studio.api.api_v1.schemas.task import (
    BaseTaskRequest,
    BaseTaskResponse,
)


class BaseInternalTaskRequest(BaseTaskRequest):
    """
    Defines the structure for system-internal task operations on embedding models.
    Extends public task requests with model targeting for restricted operations.
    Creates a boundary between public API operations and system-level functions.
    Ensures consistent format across specialized internal tasks like upsert and delete.
    """

    embedding_model_id: str = Field(
        description="Deployed embedding embedding_model ID"
    )


class BaseInternalTaskResponse(BaseTaskResponse):
    """
    Standardized response structure for internal model operations.
    Maintains consistent response patterns across different system components.
    Links operation results with their target embedding model for traceability.
    Provides necessary context for asynchronous task monitoring and reporting.
    """

    embedding_model_id: str = Field(...)
