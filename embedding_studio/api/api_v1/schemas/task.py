import enum
from typing import Optional

from pydantic import AwareDatetime, BaseModel, Field


class TaskStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    done = "done"
    canceled = "canceled"
    failed = "failed"


class BaseTaskRequest(BaseModel):
    task_id: Optional[str] = Field(
        None,
        description="Optional custom task ID for idempotent task creation",
        alias="id",
    )


class BaseTaskResponse(BaseModel):
    task_id: str = Field(..., alias="id")
    status: TaskStatus = Field(...)
    created_at: AwareDatetime = Field(...)
    updated_at: AwareDatetime = Field(...)
