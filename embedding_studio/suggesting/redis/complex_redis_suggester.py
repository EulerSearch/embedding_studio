from embedding_studio.models.suggesting import SuggestingRequest
from embedding_studio.suggesting.redis.combined_suggests_pipeline_generator import (
    CombinedSuggestsQueryGenerator,
)
from embedding_studio.suggesting.redis.most_probable_suggests_pipeline_generator import (
    MostProbableSuggestsQueryGenerator,
)
from embedding_studio.suggesting.redis.prefix_suggests_query_generator import (
    PrefixSuggestsQueryGenerator,
)
from embedding_studio.suggesting.redis.simple_suggests_query_generator import (
    SimpleSuggestsQueryGenerator,
)
from embedding_studio.suggesting.redis.suggester import RedisSuggester
from embedding_studio.suggesting.tokenizer import SuggestingTokenizer


class ComplexRedisSuggester(RedisSuggester):
    """
    A complex suggester that combines functionalities from the
    MostProbableSuggestsQueryGenerator, CombinedSuggestsQueryGenerator,
    PrefixSuggestsQueryGenerator, and SimpleSuggestsQueryGenerator classes.
    """

    def __init__(
        self,
        redis_url: str,
        tokenizer: SuggestingTokenizer,
        index_name: str = "suggestion_phrases",
        max_chunks: int = 20,
    ):
        """
        Initialize the MongoSuggester.

        :param redis_url:
            The Redis client URL to use for storing and retrieving documents.
        :param tokenizer:
            The SuggestingTokenizer responsible for splitting phrases into chunks.
        :param index_name:
            Name of the index storing suggestion data.
        :param max_chunks: Maximum number of chunks that each document can have.
        """
        super(ComplexRedisSuggester, self).__init__(
            redis_url, tokenizer, index_name, max_chunks
        )
        self._most_probable_query_generator = (
            MostProbableSuggestsQueryGenerator(self._max_chunks)
        )
        self._simple_query_generator = SimpleSuggestsQueryGenerator(
            self._max_chunks
        )
        self._prefix_query_generator = PrefixSuggestsQueryGenerator(
            self._max_chunks
        )
        self._combined_query_generator = CombinedSuggestsQueryGenerator(
            self._max_chunks
        )

    def _generate_query(
        self,
        request: SuggestingRequest,
        top_k: int = 10,
        soft_match: bool = False,
    ) -> str:
        if (
            len(request.chunks.found_chunks) == 0
            and len(request.chunks.next_chunk) > 0
        ):
            return self._simple_query_generator(
                request=request,
                top_k=top_k,
                domain=request.domain,
                soft_match=soft_match,
            )

        elif (
            len(request.chunks.found_chunks) > 0
            and len(request.chunks.next_chunk) == 0
        ):
            return self._prefix_query_generator(
                request=request,
                top_k=top_k,
                domain=request.domain,
                soft_match=soft_match,
            )

        elif (
            len(request.chunks.found_chunks) > 0
            and len(request.chunks.next_chunk) > 0
        ):
            return self._combined_query_generator(
                request=request,
                top_k=top_k,
                domain=request.domain,
                soft_match=soft_match,
            )

        elif request.domain:
            return self._most_probable_query_generator(
                request=request,
                top_k=top_k,
                domain=request.domain,
                soft_match=soft_match,
            )

        else:
            return ""
