import enum
from typing import Any, Dict, Optional

from bson import ObjectId
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from embedding_studio.db.common import PyObjectId
from embedding_studio.utils.datetime_utils import current_time


class FineTuningStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    done = "done"
    canceled = "canceled"
    error = "error"


class FineTuningTask(BaseModel):
    fine_tuning_method: str = Field(...)
    status: FineTuningStatus = Field(default=FineTuningStatus.pending)
    created_at: AwareDatetime = Field(default_factory=current_time)
    updated_at: AwareDatetime = Field(default_factory=current_time)
    batch_id: Optional[str] = None
    embedding_model_id: Optional[str] = None
    best_run_id: Optional[str] = None
    best_model_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    idempotency_key: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


class FineTuningTaskInDb(FineTuningTask):
    id: Optional[PyObjectId] = Field(default=ObjectId, alias="_id")
    broker_id: Optional[str] = Field(default=None)
