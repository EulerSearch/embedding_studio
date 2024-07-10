import enum
from typing import Any, Dict, List, Optional

from bson import ObjectId
from pydantic import AwareDatetime, BaseModel, Field

from embedding_studio.db.common import PyObjectId
from embedding_studio.utils.datetime_utils import current_time


class UpsertionStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    done = "done"
    canceled = "canceled"
    error = "error"


class UpsertionFailureStage(str, enum.Enum):
    on_downloading = "on_downloading"
    on_inference = "on_inference"
    on_splitting = "on_splitting"
    on_upsert = "on_upsert"
    other = "other"


class DataItem(BaseModel):
    object_id: str
    payload: Optional[Dict[str, Any]] = None
    item_info: Optional[Dict[str, Any]] = None


class FailedDataItem(DataItem):
    detail: str = Field(
        description="Detailed error message explaining the failure"
    )
    failure_stage: UpsertionFailureStage = UpsertionFailureStage.other


class UpsertionTask(BaseModel):
    fine_tuning_method: str = Field(...)
    embedding_model_id: str = Field(...)
    status: UpsertionStatus = Field(default=UpsertionStatus.pending)
    items: List[DataItem] = Field(...)
    failed_items: List[FailedDataItem] = Field(default_factory=list)
    created_at: AwareDatetime = Field(default_factory=current_time)
    updated_at: AwareDatetime = Field(default_factory=current_time)


class UpsertionTaskInDb(UpsertionTask):
    id: Optional[PyObjectId] = Field(default=ObjectId, alias="_id")
    broker_id: Optional[str] = Field(default=None)
