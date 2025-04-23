from typing import Any, Dict, Optional

from pydantic import BaseModel


class SessionEvent(BaseModel):
    """
    A model representing a user interaction event within a session,
    capturing the event identifier, session identifier, object identifier,
    event type, creation timestamp, and optional metadata.
    """

    event_id: str
    session_id: str
    object_id: str
    event_type: str
    created_at: int
    meta: Optional[Dict[str, Any]] = None


class DbSessionEvent(SessionEvent):
    """
    An extension of SessionEvent that includes an additional
    database-specific identifier field for storage purposes.
    """

    db_id: str
