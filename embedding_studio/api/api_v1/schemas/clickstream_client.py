from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from embedding_studio.api.api_v1.schemas.payload_filter import PayloadFilter
from embedding_studio.api.api_v1.schemas.sorting_options import SortByOptions


class SearchResultItem(BaseModel):
    """
    Represents an individual search result with its rank and metadata.
    Used to track each result shown to a user during a search session.
    Enables analysis of which results were presented versus which were
    interacted with, forming the basis for retrieval evaluation.
    """

    object_id: str
    rank: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None


class SessionCreateRequest(BaseModel):
    """
    Defines the creation of a new search session with all relevant details.
    Captures search intent, displayed results, and optional contextual data.
    Central to tracking user search journeys from query to result selection.
    Forms the foundation for search quality analytics and improvements.
    """

    session_id: str
    search_query: str
    search_results: List[SearchResultItem]
    search_meta: Optional[Dict[str, Any]] = None
    payload_filter: Optional[PayloadFilter] = None
    sort_by: Optional[SortByOptions] = None
    user_id: Optional[str] = None
    created_at: Optional[int] = None


class NewSessionEvent(BaseModel):
    """
    Represents a new user interaction event within a search session.
    Primarily tracks user clicks but can be extended to other interaction types.
    Provides crucial signals for understanding user behavior and preferences.
    Core component for implicit feedback collection in search systems.
    """

    event_id: str
    object_id: str
    event_type: str = "click"
    created_at: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None


class SessionEvent(NewSessionEvent):
    """
    Extends NewSessionEvent with a mandatory timestamp for chronological analysis.
    Represents a persisted and validated interaction event in the system.
    Enables temporal analysis of user behavior patterns within sessions.
    Critical for understanding the sequence and timing of user interactions.
    """

    created_at: int


class SessionAddEventsRequest(BaseModel):
    """
    Container for submitting multiple interaction events for a specific session.
    Supports batch recording of user interactions for efficiency.
    Enables capturing complex interaction patterns across multiple results.
    Maintains association between events and their parent session.
    """

    session_id: str
    events: List[NewSessionEvent]


class SessionMarkIrrelevantRequest(BaseModel):
    """
    Request to flag an entire session as irrelevant for analytics purposes.
    Helps filter out sessions that shouldn't influence search improvement efforts.
    Prevents noisy or anomalous sessions from skewing quality metrics.
    Important for maintaining clean training data for search optimization.
    """

    session_id: str


class SessionGetResponse(SessionCreateRequest):
    """
    Complete representation of a session with all its associated events.
    Provides a holistic view of the user's search journey and interactions.
    Includes relevance flag to indicate validity for analytics purposes.
    Essential for detailed analysis of search quality and user satisfaction.
    """

    created_at: int
    is_irrelevant: bool
    events: List[SessionEvent]
