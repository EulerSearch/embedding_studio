import enum
from datetime import datetime
from typing import Dict, Optional

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field

from embedding_studio.db.common import PyObjectId


class FineTuningStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    done = "done"
    canceled = "canceled"
    error = "error"


class FineTuningTask(BaseModel):
    status: FineTuningStatus = Field(default=FineTuningStatus.pending)
    start_at: datetime = Field(...)
    end_at: datetime = Field(...)
    metadata: dict = Field(...)


class FineTuningTaskInDb(FineTuningTask):
    id: Optional[PyObjectId] = Field(default=ObjectId, alias="_id")

    model_config = ConfigDict(populate_by_name=True)


class FineTuningTaskCreate(FineTuningTask):
    start_at: datetime
    end_at: datetime
    metadata: Optional[Dict] = None


class FineTuningTaskUpdate(FineTuningTask):
    pass
