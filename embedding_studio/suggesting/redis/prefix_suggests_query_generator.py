from typing import Optional

from embedding_studio.models.suggesting import SuggestingRequest
from embedding_studio.suggesting.redis.query_generator import QueryGenerator


class PrefixSuggestsQueryGenerator(QueryGenerator):
    """
    A query generator that creates queries to match exact prefixes in Redis.

    This generator is used when there are found chunks but no next chunk.
    It implements a shifting-OR logic that allows matching phrases at different
    positions, handling cases where the user has already typed part of a suggestion.
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
        Build a RediSearch query string that mimics the shifting-OR logic
        from your MongoDB pipeline, ignoring case variations.

        Specifically:
        - Limit found_chunks to 'max_chunks' from the right (like your pipeline).
        - For each shift 'index' in [0..N), build an AND of:
            ( @chunk_0:found_chunks[index],
              @chunk_1:found_chunks[index+1],
              ...
              @n_chunks:[k +inf]
            )
        - OR all these shift-based clauses together.

        If found_chunks = ["foo", "bar", "baz"], the final query might look like:
          ( @n_chunks:[3 +inf] @chunk_0:"foo" @chunk_1:"bar" @chunk_2:"baz" )
          | ( @n_chunks:[2 +inf] @chunk_0:"bar" @chunk_1:"baz" )
          | ( @n_chunks:[1 +inf] @chunk_0:"baz" )
        """
        found_chunks = request.chunks.found_chunks
        if len(found_chunks) > self.max_chunks:
            found_chunks = request.chunks.found_chunks[-self.max_chunks :]

        n_chunks = len(found_chunks)
        if n_chunks == 0:
            return ""

        if not found_chunks:
            return ""

        # Truncate to max_chunks from the right
        if len(found_chunks) > self.max_chunks:
            found_chunks = found_chunks[-self.max_chunks :]

        n_chunks = len(found_chunks)
        if n_chunks == 0:
            return ""

        or_clauses = []
        # For each possible shift (index), build an AND subclause
        for index in range(n_chunks):
            subclauses = []
            # If we shift by 'index', how many chunk matches do we produce?
            # E.g. if index=0 and found_chunks=["foo","bar"], then chunk_0=foo, chunk_1=bar
            # If index=1, then chunk_0=bar
            shift_size = n_chunks - index  # how many matches from that shift
            for pos in range(shift_size):
                # pos goes from 0..(shift_size-1)
                # found_chunks[index + pos] should match chunk_{pos}
                chunk_text = found_chunks[index + pos]
                # Match the exact chunk (ignoring case expansions)
                if soft_match and len(chunk_text) > 1:
                    subclauses.append(f"@chunk_{pos}:%{chunk_text}%")
                else:
                    subclauses.append(f"@chunk_{pos}:{chunk_text}")

            # Combine them with a space => RediSearch interprets as AND
            and_clause = " ".join(subclauses)
            # Wrap in parentheses so we can OR it with other shifts
            or_clauses.append(f"({and_clause})")

        # Combine all shift-based clauses with OR => '|'
        final_query = " | ".join(or_clauses)
        if domain:
            domain_query = "{__" + domain.replace("-", "_") + "__}"
            return f"(@domains:{domain_query}) ({final_query})"
        else:
            return final_query
