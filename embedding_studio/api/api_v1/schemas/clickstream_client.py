from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class SearchResultItem(BaseModel):
    object_id: str
    rank: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None


class SessionCreateRequest(BaseModel):
    session_id: str
    search_query: str
    search_results: List[SearchResultItem]
    search_meta: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    created_at: Optional[int] = None


class NewSessionEvent(BaseModel):
    event_id: str
    object_id: str
    event_type: str = "click"
    created_at: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None


class SessionEvent(NewSessionEvent):
    created_at: int


class SessionAddEventsRequest(BaseModel):
    session_id: str
    events: List[NewSessionEvent]


class SessionMarkIrrelevantRequest(BaseModel):
    session_id: str


class SessionGetResponse(SessionCreateRequest):
    created_at: int
    is_irrelevant: bool
    events: List[SessionEvent]
