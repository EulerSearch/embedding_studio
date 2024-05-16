import enum
from typing import Any, Dict, Optional

from bson import ObjectId
from pydantic import AwareDatetime, BaseModel, Field

from embedding_studio.db.common import PyObjectId
from embedding_studio.utils.datetime_utils import current_time


class ModelDeploymentStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    done = "done"
    canceled = "canceled"
    error = "error"


class ModelDeploymentTask(BaseModel):
    fine_tuning_method: str = Field(...)
    embedding_model_id: str = Field(...)
    status: ModelDeploymentStatus = Field(
        default=ModelDeploymentStatus.pending
    )
    created_at: AwareDatetime = Field(default_factory=current_time)
    updated_at: AwareDatetime = Field(default_factory=current_time)
    metadata: Optional[Dict[str, Any]] = None


class ModelDeploymentTaskInDb(ModelDeploymentTask):
    id: Optional[PyObjectId] = Field(default=ObjectId, alias="_id")
    broker_id: Optional[str] = Field(default=None)


class ModelDeletionTask(BaseModel):
    fine_tuning_method: str = Field(...)
    embedding_model_id: str = Field(...)
    status: ModelDeploymentStatus = Field(
        default=ModelDeploymentStatus.pending
    )
    created_at: AwareDatetime = Field(default_factory=current_time)
    updated_at: AwareDatetime = Field(default_factory=current_time)
    metadata: Optional[Dict[str, Any]] = None


class ModelDeletionTaskInDb(ModelDeletionTask):
    id: Optional[PyObjectId] = Field(default=ObjectId, alias="_id")
    broker_id: Optional[str] = Field(default=None)
