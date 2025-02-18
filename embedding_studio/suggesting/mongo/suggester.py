from typing import List

from pymongo.collection import Collection
from pymongo.database import Database

from embedding_studio.models.suggesting import Suggest, SuggestingRequest
from embedding_studio.suggesting.abstract_suggester import AbstractSuggester
from embedding_studio.suggesting.abtract_phrase_manager import (
    AbstractSuggestionPhraseManager,
)
from embedding_studio.suggesting.mongo.phrases_manager import (
    MongoSuggestionPhraseManager,
)
from embedding_studio.suggesting.tokenizer import SuggestingTokenizer


def _get_or_create_collection(
    mongo_database: Database, collection_name: str
) -> Collection:
    """
    Get an existing MongoDB collection if it exists, otherwise create it.

    :param mongo_database:
        The MongoDB database in which to look for or create the collection.
    :param collection_name:
        The name of the collection to get or create.
    :return:
        The MongoDB Collection object.
    """
    list_of_collections = mongo_database.list_collection_names()
    if collection_name in list_of_collections:
        collection = mongo_database.get_collection(collection_name)
    else:
        collection = mongo_database.create_collection(collection_name)

    return collection


# TODO: move suggester into in-memory trie, which is going to be stored in mongo.


class MongoSuggester(AbstractSuggester):
    """
    Provides a MongoDB-based implementation for retrieving suggestions.
    The class defines how to transform retrieved MongoDB documents into
    Suggest objects and relies on subclasses to implement the pipeline logic.
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
        self._mongo_database = mongo_database
        self._collection = _get_or_create_collection(
            mongo_database, collection_name
        )
        self._max_chunks = max_chunks

        self._phrases_manager = MongoSuggestionPhraseManager(
            self._collection, tokenizer, self._max_chunks
        )

    @property
    def phrases_manager(self) -> AbstractSuggestionPhraseManager:
        return self._phrases_manager

    def _generate_pipeline(
        self, request: SuggestingRequest, top_k: int = 10
    ) -> List[dict]:
        """
        Construct the MongoDB aggregation pipeline needed to fetch the top suggestions.
        Must be overridden by subclasses to define the query logic.

        :param request: The SuggestingRequest containing chunks and other context.
        :param top_k: The number of suggestions to retrieve.
        :return: A list of MongoDB pipeline stages (dicts).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _generate_pipeline"
        )

    def _doc_to_suggest(self, doc: dict) -> Suggest:
        """
        Transform one aggregated MongoDB document into a Suggest object.

        :param doc: A single document from the aggregation pipeline, which
                    should contain 'chunks', 'match_info', 'prob', and 'labels'.
        :return: A Suggest object populated based on the document data.
        """
        # Convert a dict of chunk fields (chunk_0, chunk_1, ...) to a Python list
        raw_chunks = []
        for i in range(self._max_chunks):
            chunk_val = doc["chunks"].get(f"chunk_{i}")
            if chunk_val:
                raw_chunks.append(chunk_val)
            else:
                break

        # Extract match position, length, and type from match_info
        match_position = doc["match_info"][
            "position"
        ]  # Where the matched sequence starts
        match_type = doc["match_info"]["type"]  # "exact", "prefix", or "fuzzy"

        # Identify the matched portion and what preceded it (prefix)
        matched_chunks = raw_chunks[match_position + 1 :]
        prefix_chunks = raw_chunks[: match_position + 1]

        # Create and return the Suggest object
        return Suggest(
            chunks=matched_chunks,
            prefix_chunks=prefix_chunks,
            match_type=match_type,
            prob=doc["prob"],
            labels=doc["labels"],
        )

    def get_topk_suggestions(
        self, request: SuggestingRequest, top_k: int = 10
    ) -> List[Suggest]:
        """
        Retrieve the top-k suggestions for the given request by running the generated pipeline.

        :param request: The SuggestingRequest containing the context for which suggestions are needed.
        :param top_k: How many suggestions to retrieve.
        :return: A list of Suggest objects that satisfy the request criteria.
        """
        pipeline = self._generate_pipeline(request, top_k)
        if not pipeline:
            return []

        # Execute the pipeline and transform each resulting document

        docs = self._collection.aggregate(pipeline)
        docs = [doc for doc in docs]
        result = [self._doc_to_suggest(doc) for doc in docs]
        return result
