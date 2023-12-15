from typing import Any, Dict, Optional

from pydantic import BaseModel


class SessionEvent(BaseModel):
    event_id: str
    session_id: str
    object_id: str
    event_type: str
    created_at: int
    meta: Optional[Dict[str, Any]] = None


class DbSessionEvent(SessionEvent):
    db_id: str
