from typing import Optional

from embedding_studio.models.suggesting import SuggestingRequest
from embedding_studio.suggesting.redis.query_generator import QueryGenerator


class MostProbableSuggestsQueryGenerator(QueryGenerator):
    """
    A query generator that creates queries to retrieve the most probable suggestions
    based only on domain information.

    This generator is used as a fallback when there are no found chunks or next chunks,
    but a domain is specified. It returns suggestions sorted by probability without
    text matching criteria.
    """

    def __init__(
        self,
        max_chunks: int = 20,
    ):
        """
        Initialize the Query Generator.

        :param max_chunks: Maximum number of chunks that each document can have.
        """
        super().__init__()
        self.max_chunks = max_chunks

    def __call__(
        self,
        request: SuggestingRequest,
        top_k: int = 10,
        domain: Optional[str] = None,
        soft_match: bool = False,
    ) -> str:
        """
        Generate a query that filters only by domain, returning the most probable
        suggestions when no text context is available.

        If no domain is specified, returns a query matching all domains.

        :param request: The suggesting request (not used in this generator)
        :param top_k: The maximum number of suggestions to return
        :param domain: The domain to filter suggestions by
        :param soft_match: Whether to use soft matching (not used in this generator)
        :return: A Redis search query string filtering by domain only
        """
        if domain:
            domain_query = "{__" + domain.replace("-", "_") + "__}"
            return f"@domains:{domain_query}"
        else:
            return "@domains:*"
