from enum import Enum
from typing import Optional

from pydantic import BaseModel


class SessionBatchStatus(str, Enum):
    collecting = "collecting"
    released = "released"
    fine_tuning = "fine_tuning"
    archiving = "archiving"
    archived = "archived"


class SessionBatch(BaseModel):
    batch_id: str
    session_counter: int
    created_at: int
    status: SessionBatchStatus
    release_id: Optional[str] = None
    released_at: Optional[int] = None
