from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from embedding_studio.models.clickstream.session_events import SessionEvent
from embedding_studio.models.payload.models import PayloadFilter
from embedding_studio.models.sort_by.models import SortByOptions


class SearchResultItem(BaseModel):
    """
    A lightweight model representing an individual search result within a session,
    containing the object identifier, optional ranking score, and optional metadata.
    """

    object_id: str
    rank: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None


class Session(BaseModel):
    """
    A model capturing a user's search session, including the session identifier,
    search query, creation timestamp, search results, optional metadata,
    filter criteria, sorting preferences, user identifier, and flags
    for relevance and search type.
    """

    session_id: str
    search_query: Any
    created_at: int
    search_results: List[SearchResultItem]
    search_meta: Optional[Dict[str, Any]] = None
    payload_filter: Optional[PayloadFilter] = None
    sort_by: Optional[SortByOptions] = None
    user_id: Optional[str] = None
    is_irrelevant: bool = False
    is_payload_search: bool = False


class RegisteredSession(Session):
    """
    An extension of Session that has been formally registered in the system,
    adding batch information and session numbering within that batch.
    """

    session_id: str = Field(alias="_id", validation_alias="session_id")
    batch_id: str
    session_number: int

    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",  # Ignore extra fields
        arbitrary_types_allowed=True,
    )


class SessionWithEvents(RegisteredSession):
    """
    Further extends RegisteredSession to include a list of associated
    SessionEvents, providing a complete view of user interactions within the session.
    """

    events: List[SessionEvent]
