from abc import ABC, abstractmethod
from typing import Optional

from embedding_studio.models.suggesting import SuggestingRequest


class QueryGenerator(ABC):
    """
    Abstract base class for all query generators used in the suggestion system.

    Query generators are responsible for creating Redis search query strings from
    SuggestingRequest objects. These query strings are used to retrieve relevant
    suggestion phrases from the Redis database.

    All concrete implementations must override the __call__ method to define their
    specific query generation logic.
    """

    @abstractmethod
    def __call__(
        self,
        request: SuggestingRequest,
        top_k: int = 10,
        domain: Optional[str] = None,
        soft_match: bool = False,
    ) -> str:
        """
        Generate a suggestion query based on the given request.

        :param request: The suggesting request containing input data
        :param top_k: The maximum number of suggestions to return. Defaults to 10
        :param domain: The domain of the suggestion query. Defaults to None
        :param soft_match: Whether to perform a soft match with more flexible matching criteria
        :return: A string representing a Redis suggestion query

        Example implementation:
        ```python
        def __call__(
            self,
            request: SuggestingRequest,
            top_k: int = 10,
            domain: Optional[str] = None,
            soft_match: bool = False,
        ) -> str:
            # Extract chunks from the request
            found_chunks = request.chunks.found_chunks or []
            next_chunk = request.chunks.next_chunk or ""

            # Build the query string based on the chunks
            query_parts = []
            for i, chunk in enumerate(found_chunks):
                query_parts.append(f"@chunk_{i}:{chunk}")

            if next_chunk:
                query_parts.append(f"@chunk_{len(found_chunks)}:{next_chunk}*")

            # Combine all parts with AND
            query = " ".join(query_parts)

            # Add domain filter if specified
            if domain:
                domain_query = "{__" + domain.replace("-", "_") + "__}"
                query = f"(@domains:{domain_query}) ({query})"

            return query
        ```
        """
