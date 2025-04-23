from typing import Any, List

from pydantic import BaseModel

from embedding_studio.api.api_v1.schemas.similarity_search import SearchResult


class QueryParsingRequest(BaseModel):
    """
    Container for a search query to be parsed and analyzed.
    Accepts any data type to support flexible query formats.
    Serves as the input to query parsing and categorization services.
    Enables query understanding and semantic interpretation workflows.
    """

    search_query: Any


class QueryParsingCategoriesResponse(BaseModel):
    """
    Response containing categories relevant to the parsed search query.
    Returns a list of matching categories with similarity scores.
    Provides metadata and payload information for each matched category.
    Enables intelligent query categorization and semantic understanding.
    """

    categories: List[SearchResult]
