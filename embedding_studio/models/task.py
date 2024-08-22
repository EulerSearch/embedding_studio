import enum
from typing import Optional

from bson import ObjectId
from pydantic import AwareDatetime, BaseModel, Field

from embedding_studio.db.common import PyObjectId
from embedding_studio.utils.datetime_utils import current_time


class BaseTaskCreateSchema(BaseModel):
    id: Optional[PyObjectId] = Field(default=ObjectId, alias="_id")
    embedding_model_id: str = Field(...)
    fine_tuning_method: str = Field(...)


class TaskStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    done = "done"
    canceled = "canceled"
    failed = "failed"


class BaseTaskInDb(BaseModel):
    id: Optional[PyObjectId] = Field(default=ObjectId, alias="_id")
    broker_id: Optional[str] = Field(default=None)


class BaseModelOperationTask(BaseModel):
    fine_tuning_method: str = Field(...)
    embedding_model_id: str = Field(...)
    status: TaskStatus = Field(default=TaskStatus.pending)
    created_at: AwareDatetime = Field(default_factory=current_time)
    updated_at: AwareDatetime = Field(default_factory=current_time)
