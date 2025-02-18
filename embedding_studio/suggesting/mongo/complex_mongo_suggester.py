from typing import List

from pymongo.database import Database

from embedding_studio.models.suggesting import SuggestingRequest
from embedding_studio.suggesting.mongo.full_suggests_pipeline_generator import (
    FullSuggestsPipelineGenerator,
)
from embedding_studio.suggesting.mongo.prefix_suggests_pipeline_generator import (
    PrefixSuggestsPipelineGenerator,
)
from embedding_studio.suggesting.mongo.simple_suggests_pipeline_generator import (
    SimpleSuggestsPipelineGenerator,
)
from embedding_studio.suggesting.mongo.suggester import MongoSuggester
from embedding_studio.suggesting.tokenizer import SuggestingTokenizer


class ComplexMongoSuggester(MongoSuggester):
    """
    A complex suggester that combines functionalities from the
    SimpleSuggestsPipelineGenerator, PrefixSuggestsPipelineGenerator, and FullSuggestsPipelineGenerator classes.
    This class demonstrates multiple inheritance and allows for more flexible
    or combined query logic when generating suggestions.
    """

    def __init__(
        self,
        mongo_database: Database,
        tokenizer: SuggestingTokenizer,
        collection_name: str = "suggestion_phrases",
        max_chunks: int = 20,
    ):
        """
        Initialize the MongoSuggester.

        :param mongo_database: The MongoDB Database object to use for retrieving documents.
        :param tokenizer:
            The SuggestingTokenizer responsible for splitting phrases into chunks.
        :param collection_name: Name of the collection storing suggestion data.
        :param max_chunks: Maximum number of chunks that each document can have.
        """

        # Explicitly call each parent constructor.
        super().__init__(
            mongo_database, tokenizer, collection_name, max_chunks
        )
        self._simple_pipeline_generator = SimpleSuggestsPipelineGenerator(
            max_chunks
        )
        self._prefix_pipeline_generator = PrefixSuggestsPipelineGenerator(
            max_chunks
        )
        self._full_pipeline_generator = FullSuggestsPipelineGenerator(
            max_chunks
        )

    def _generate_pipeline(
        self, request: SuggestingRequest, top_k: int = 10
    ) -> List[dict]:
        """
        Construct the MongoDB aggregation pipeline by combining or overriding
        the logic from the parent classes in any custom manner needed.

        :param request:
            A SuggestingRequest containing context about previously found chunks
            and the next chunk to be matched.
        :param top_k:
            The number of results to return. Defaults to 10.
        :return:
            A list of MongoDB aggregation pipeline stages (dictionaries).
        """
        if len(request.found_chunks) == 0 and len(request.next_chunk) > 0:
            return self._simple_pipeline_generator.generate_pipeline(
                request=request, top_k=top_k
            )

        elif len(request.found_chunks) > 0 and len(request.next_chunk) == 0:
            return self._prefix_pipeline_generator.generate_pipeline(
                request=request, top_k=top_k
            )

        elif len(request.found_chunks) > 0 and len(request.next_chunk) > 0:
            return self._full_pipeline_generator.generate_pipeline(
                request=request, top_k=top_k
            )

        else:
            return []
