from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from embedding_studio.api.api_v1.schemas.payload_filter import PayloadFilter
from embedding_studio.api.api_v1.schemas.sorting_options import SortByOptions


class PayloadCountRequest(BaseModel):
    """
    Request model for counting objects based on payload filter criteria.
    Supports optional payload filtering without similarity matching.
    Enables efficient document counting operations with filter conditions.
    Foundation for faceted search and result set quantification.
    """

    search_query: Any
    filter: Optional[PayloadFilter] = None


class CountRequest(PayloadCountRequest):
    """
    Extends PayloadCountRequest to include distance-based filtering.
    Allows counting objects that also satisfy similarity thresholds.
    Enables hybrid filtering by both content criteria and vector similarity.
    Supports advanced analytics on vector database contents.
    """

    max_distance: Optional[float] = None


class SimilaritySearchRequest(BaseModel):
    """
    Comprehensive request model for vector similarity search operations.
    Combines query vectors, payload filtering, and pagination controls.
    Supports session tracking and user identification for analytics.
    Configures search behavior including similarity thresholds and sorting.
    Essential for powering semantic search experiences.

    :param search_query: The query to search for, can be text or structured data
    :param limit: Maximum number of results to return in the response
    :param offset: Number of results to skip for pagination
    :param max_distance: Maximum allowable distance between query and result vectors
    :param filter: Optional structured filter conditions for payload fields
    :param create_session: Whether to create a persistent session for this search
    :param user_id: Optional identifier for the user performing the search
    :param session_id: Optional identifier for an existing session to use
    :param sort_by: Optional sorting configuration for the results
    :param similarity_first: Whether to prioritize similarity over other sorting criteria
    :param meta_info: Optional additional metadata to associate with the search
    """

    search_query: Any
    limit: int
    offset: Optional[int] = None
    max_distance: Optional[float] = None
    filter: Optional[PayloadFilter] = None
    create_session: bool = False
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    sort_by: Optional[SortByOptions] = None
    similarity_first: bool = Field(default=False)
    meta_info: Optional[Any] = None


class PayloadSearchRequest(SimilaritySearchRequest):
    """
    Specialization of SimilaritySearchRequest for payload-only searches.
    Enables content-based searches without requiring a query vector.
    Supports traditional database-style filtering in vector databases.
    Complements similarity search for comprehensive retrieval capabilities.
    """


class SearchResult(BaseModel):
    """
    Represents a single item in search results with relevance information.
    Contains object identifier, similarity score, and associated content.
    Provides both structured payload data and metadata for the matched item.
    Essential response component for all search operations.
    """

    object_id: str
    distance: float
    payload: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None


class SimilaritySearchResponse(BaseModel):
    """
    Container for search results with pagination information.
    Tracks session context for user journey and analytics.
    Provides complete result set with item details and relevance scores.
    Supports efficient client-side rendering and result processing.
    """

    next_page_offset: Optional[int] = None
    session_id: Optional[str] = None
    search_results: List[SearchResult]
    total_count: Optional[int] = None
    meta_info: Optional[Any] = None


class CountResponse(BaseModel):
    """
    Simple response model providing count information for database queries.
    Returns the total number of items matching filter criteria.
    Supports faceted search, analytics, and UI pagination components.
    Enables efficient querying of database statistics.
    """

    total_count: int
