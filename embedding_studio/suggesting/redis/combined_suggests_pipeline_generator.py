from typing import Optional

from embedding_studio.models.suggesting import SuggestingRequest
from embedding_studio.suggesting.redis.query_generator import QueryGenerator


class CombinedSuggestsQueryGenerator(QueryGenerator):
    """
    A query generator that creates complex queries combining exact matches for found chunks
    and prefix matches for the next chunk.

    This generator is used when both found chunks and a next chunk are present,
    implementing a shifting-based pattern that finds matches at different positions
    within documents while also matching the next chunk as a prefix.
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
        Generate a complex query that combines exact matching for found chunks and
        prefix matching for the next chunk.

        The query implements a shifting-based pattern that:
        1. For each possible shift position, creates an AND condition matching:
           - Exact matches for all found chunks at their respective positions
           - A prefix match for the next chunk at the next position
        2. ORs these shift-based clauses together
        3. Adds a pure prefix match case for when no shift applies

        :param request: The suggesting request containing found chunks and next chunk
        :param top_k: The maximum number of suggestions to return
        :param domain: Optional domain to filter the suggestions
        :param soft_match: Whether to use fuzzy matching (contains) instead of exact matching
        :return: A Redis search query string implementing the combined matching logic
        """
        found_chunks = request.chunks.found_chunks or []

        # Truncate found_chunks to 'max_chunks' from the right, like your Mongo code:
        if len(found_chunks) > self.max_chunks:
            found_chunks = found_chunks[-self.max_chunks :]

        n_chunks = len(found_chunks)
        if n_chunks == 0:
            return ""

        # This is the "prefix text" we want to match in @chunk_{shift_size}
        prefix_text = request.chunks.next_chunk or ""
        prefix_span = request.spans.next_chunk_span

        or_clauses = []
        for index in range(n_chunks):
            shift_size = n_chunks - index
            subclauses = []

            # 1) Exact matches for chunk_0..chunk_{shift_size-1}
            for pos in range(shift_size):
                chunk_text = found_chunks[index + pos]
                if soft_match:
                    subclauses.append(
                        f"((@chunk_{pos}:%{chunk_text}%) | (@chunk_{pos}:{chunk_text}*))"
                    )
                else:
                    subclauses.append(f"@chunk_{pos}:{chunk_text}")

            # 2) Add a prefix match for chunk_{shift_size}, if we have a prefix
            #    (Be sure we don't exceed max_chunks)
            if prefix_text and shift_size < self.max_chunks:
                # E.g. @chunk_3:prefix_text*
                # The trailing asterisk is RediSearch's prefix operator.
                if soft_match and len(prefix_span) > 1:
                    subclauses.append(f"@chunk_{shift_size}:%{prefix_text}%")

                else:
                    subclauses.append(f"@chunk_{shift_size}:{prefix_text}*")

            # Combine subclauses with spaces => AND them in RediSearch
            and_clause = " ".join(subclauses)
            or_clauses.append(f"({and_clause})")
        # Wrap in parentheses so we can OR across different shifts
        or_clauses = " | ".join(or_clauses)

        # Suppose you want each prefix of next_chunk to match exactly in `search_0`.
        # Or if you want "prefix search," you can do @search_0:prefix*.
        # The original snippet does exact partial matching for each prefix length.
        prefix = request.chunks.next_chunk
        prefix_clauses = ""
        # Do exact match: @search_0:"prefix"
        # (If you want real prefix search, do @search_0:prefix*)
        if soft_match and len(prefix_span) > 1:
            prefix_clauses = f"(@chunk_0:%{prefix}%)"
        else:
            prefix_clauses = f"(@chunk_0:{prefix}*)"

        # If you'd also like a "pure prefix" case for when no shift applies,
        # you could handle that outside the loop if needed.

        # Finally, OR all shift-based clauses
        final_query = f"{or_clauses} | {prefix_clauses}"

        if domain:
            domain_query = "{__" + domain.replace("-", "_") + "__}"
            return f"(@domains:{domain_query}) ({final_query})"
        else:
            return final_query
