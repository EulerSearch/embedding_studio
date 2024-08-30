from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from embedding_studio.api.api_v1.schemas.payload_filter import PayloadFilter


class SimilaritySearchRequest(BaseModel):
    search_query: Any
    limit: int
    offset: Optional[int] = None
    max_distance: Optional[float] = None
    filter: Optional[PayloadFilter] = None
    create_session: bool = False
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class SearchResult(BaseModel):
    object_id: str
    distance: float
    payload: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None


class SimilaritySearchResponse(BaseModel):
    next_page_offset: Optional[int] = None
    session_id: Optional[str] = None
    search_results: List[SearchResult]
