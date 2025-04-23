import enum
from typing import Optional

from pydantic import AwareDatetime, BaseModel, Field


class TaskStatus(str, enum.Enum):
    """
    Enum representing the lifecycle states of asynchronous tasks in the system.
    Provides standardized status tracking across different task types.
    Enables consistent monitoring and reporting of task progress.
    Critical for managing user expectations and system reliability.
    """

    pending = "pending"
    processing = "processing"
    done = "done"
    canceled = "canceled"
    failed = "failed"
    refused = "refused"


class BaseTaskRequest(BaseModel):
    """
    Foundation for all task creation requests with idempotency support.
    Enables clients to safely retry operations without duplicating work.
    Provides consistent interface across different task types.
    Supports robust distributed task processing patterns.
    """

    task_id: Optional[str] = Field(
        None,
        description="Optional custom task ID for idempotent task creation",
        alias="id",
    )


class BaseTaskResponse(BaseModel):
    """
    Standard response structure for all task-related operations.
    Provides essential tracking metadata including timestamps and status.
    Enables consistent monitoring and auditing of task execution.
    Forms the foundation for all task-specific response types.
    """

    task_id: str = Field(..., alias="id")
    status: TaskStatus = Field(...)
    created_at: AwareDatetime = Field(...)
    updated_at: AwareDatetime = Field(...)
