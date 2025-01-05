import enum
from typing import Optional

from bson import ObjectId
from pydantic import AwareDatetime, BaseModel, Field, constr

from embedding_studio.db.common import PyObjectId
from embedding_studio.utils.datetime_utils import current_time


class ModelParams(BaseModel):
    embedding_model_id: str = Field(...)
    fine_tuning_method: str = Field(...)


class BaseTaskMetadata(BaseModel):
    id: Optional[PyObjectId] = Field(default=ObjectId, alias="_id")

    parent_id: Optional[PyObjectId] = Field(default=None)


class BaseTaskCreateSchema(ModelParams, BaseTaskMetadata):
    ...


class TaskStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    done = "done"
    canceled = "canceled"
    failed = "failed"
    refused = "refused"


class BaseTaskInDb(BaseModel):
    id: Optional[PyObjectId] = Field(default=ObjectId, alias="_id")
    broker_id: Optional[str] = Field(default=None)

    parent_id: Optional[PyObjectId] = Field(default=None)


class BaseTaskInfo(BaseModel):
    status: TaskStatus = Field(default=TaskStatus.pending)
    detail: Optional[constr(max_length=1500)] = Field(
        None,
        description="Details or information about the current status of the task",
    )

    created_at: AwareDatetime = Field(default_factory=current_time)
    updated_at: AwareDatetime = Field(default_factory=current_time)

    parent_id: Optional[PyObjectId] = Field(default=None)


class BaseModelOperationTask(BaseTaskInfo, ModelParams):
    ...
