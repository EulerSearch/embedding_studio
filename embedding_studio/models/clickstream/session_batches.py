from enum import Enum
from typing import Optional

from pydantic import BaseModel


class SessionBatchStatus(str, Enum):
    """
    An enum that represents the possible states of a session batch, including collecting (active),
    released (finalized), fine_tuning (being used for model training), archiving (in process of being archived),
    and archived (stored for historical purposes).
    """

    collecting = "collecting"
    released = "released"
    fine_tuning = "fine_tuning"
    archiving = "archiving"
    archived = "archived"


class SessionBatch(BaseModel):
    """
    A model for grouping multiple user sessions together, containing a batch identifier,
    session counter, timestamps for creation and release events, batch status,
    and an optional release identifier.
    """

    batch_id: str
    session_counter: int
    created_at: int
    status: SessionBatchStatus
    release_id: Optional[str] = None
    released_at: Optional[int] = None
