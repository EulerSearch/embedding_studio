import uuid
from typing import Dict, Optional

from bson import ObjectId
from pydantic import BaseModel, Field

from embedding_studio.db.common import PyObjectId
from embedding_studio.models.fine_tuning import FineTuningTask


class FineTuningTaskCreate(BaseModel):
    fine_tuning_method: str
    batch_id: Optional[str] = None
    metadata: Optional[Dict] = None
    idempotency_key: Optional[uuid.UUID] = None


# TODO:
#   1. remove dependency on internal model (FineTuningTask)
#   2. id shouldn't be optional
class FineTuningTaskResponse(FineTuningTask):
    id: Optional[PyObjectId] = Field(default=ObjectId, alias="_id")
