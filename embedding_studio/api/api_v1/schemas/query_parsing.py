from typing import Any, List

from pydantic import BaseModel

from embedding_studio.api.api_v1.schemas.similarity_search import SearchResult


class QueryParsingRequest(BaseModel):
    search_query: Any


class QueryParsingCategoriesResponse(BaseModel):
    categories: List[SearchResult]
