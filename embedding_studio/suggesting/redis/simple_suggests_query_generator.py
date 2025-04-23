from typing import Optional

from embedding_studio.models.suggesting import SuggestingRequest
from embedding_studio.suggesting.redis.query_generator import QueryGenerator


class SimpleSuggestsQueryGenerator(QueryGenerator):
    """
    A query generator that creates simple prefix-based queries for Redis search.

    This generator is used when there are no found chunks but only a next chunk,
    creating a query that matches documents where the first chunk starts with
    the provided text.
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
        Generate a simple Redis search query that matches documents where the first chunk
        starts with the provided next_chunk text.

        The query is formatted differently based on whether soft matching is enabled:
        - For soft matching: performs a contains search (@chunk_0:%text%)
        - For regular matching: performs a prefix search (@chunk_0:text*)

        :param request: The suggesting request containing the next chunk to match
        :param top_k: The maximum number of suggestions to return
        :param domain: Optional domain to filter the suggestions
        :param soft_match: Whether to perform a soft (contains) match instead of a prefix match
        :return: A Redis search query string
        ```
        """
        chunk = request.chunks.next_chunk.lower()
        final_query = ""
        if soft_match and len(chunk) > 1:
            final_query = f"@chunk_0:%{chunk}%"

        else:
            final_query = f"@chunk_0:{chunk}*"

        if domain:
            domain_query = "{__" + domain.replace("-", "_") + "__}"
            return f"(@domains:{domain_query}) ({final_query})"
        else:
            return final_query
