from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from embedding_studio.models.clickstream.session_events import SessionEvent


class SearchResultItem(BaseModel):
    object_id: str
    rank: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None


class Session(BaseModel):
    session_id: str
    search_query: str
    created_at: int
    search_results: List[SearchResultItem]
    search_meta: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    is_irrelevant: bool = False


class RegisteredSession(Session):
    batch_id: str
    session_number: int


class SessionWithEvents(RegisteredSession):
    events: List[SessionEvent]
