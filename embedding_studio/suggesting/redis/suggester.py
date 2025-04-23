import concurrent.futures
from difflib import get_close_matches
from typing import List

from redisearch import Query

from embedding_studio.models.suggesting import Suggest, SuggestingRequest
from embedding_studio.suggesting.abstract_suggester import AbstractSuggester
from embedding_studio.suggesting.abtract_phrase_manager import (
    AbstractSuggestionPhraseManager,
)
from embedding_studio.suggesting.redis.phrases_manager import (
    RedisSuggestionPhraseManager,
)
from embedding_studio.suggesting.tokenizer import SuggestingTokenizer


class RedisSuggester(AbstractSuggester):
    """
    Provides a MongoDB-based implementation for retrieving suggestions.
    The class defines how to transform retrieved MongoDB documents into
    Suggest objects and relies on subclasses to implement the pipeline logic.
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
        self._max_chunks = max_chunks
        self._phrases_manager = RedisSuggestionPhraseManager(
            redis_url=redis_url,
            tokenizer=tokenizer,
            index_name=index_name,
            max_chunks=self._max_chunks,
        )

    @property
    def phrases_manager(self) -> AbstractSuggestionPhraseManager:
        return self._phrases_manager

    def _generate_query(
        self,
        request: SuggestingRequest,
        top_k: int = 10,
        soft_match: bool = False,
    ) -> str:
        """
        Construct the MongoDB aggregation pipeline needed to fetch the top suggestions.
        Must be overridden by subclasses to define the query logic.

        :param request: The SuggestingRequest containing chunks and other context.
        :param top_k: The number of suggestions to retrieve.
        :param soft_match: Do the soft match.
        :return: A redis query string.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _generate_query"
        )

    def _find_match_position_prefix(
        self, raw_chunks: List[str], found_chunk: str
    ) -> int:
        """
        Return the index in 'raw_chunks' of the first chunk that
        starts with 'found_chunk' (case-sensitive), or -1 if none.

        :param raw_chunks: List of chunk strings to search through
        :param found_chunk: The chunk prefix to find
        :return: Index of the first matching chunk, or -1 if not found
        """
        for i, chunk_val in enumerate(raw_chunks):
            if chunk_val.startswith(found_chunk):
                return i
        return -1

    def _find_match_position_soft(
        self, raw_chunks: List[str], found_chunk: str, max_distance: int = 1
    ) -> int:
        """
        Find the chunk that is closest to 'found_chunk' using fuzzy matching.

        Uses difflib.get_close_matches to find approximate matches with
        at least 80% similarity.

        :param raw_chunks: List of chunk strings to search through
        :param found_chunk: The chunk to find a close match for
        :param max_distance: Maximum edit distance for matches (not directly used)
        :return: Index of the best fuzzy match, or -1 if no good match found
        """
        # get_close_matches won't directly limit "single-edit distance," but it does
        # approximate similarity. We'll do a quick pass to find best matches:
        close = get_close_matches(found_chunk, raw_chunks, n=1, cutoff=0.8)
        # 'cutoff=0.8' means "80% similarity" from difflib's perspective, *not*
        # strictly "<=1 edit distance," but it’s a workable example.
        if close:
            # close[0] is the best match
            # find its index
            return raw_chunks.index(close[0])
        return -1

    def _find_match_position_exact(
        self, raw_chunks: List[str], found_chunk: str
    ) -> int:
        """
        Return the index of 'found_chunk' in 'raw_chunks' if present, else -1.

        :param raw_chunks: List of chunk strings to search through
        :param found_chunk: The chunk to find an exact match for
        :return: Index of the exact match, or -1 if not found
        """
        try:
            return raw_chunks.index(found_chunk)
        except ValueError:
            return -1

    def _doc_to_suggest(
        self, doc: dict, request: SuggestingRequest
    ) -> Suggest:
        """
        Convert a Redis document to a Suggest object.

        The conversion process:
        1. Extract raw chunks from the document
        2. Try to find a match for the last found chunk using exact, prefix, or fuzzy matching
        3. Split the raw chunks into prefix and matched chunks based on the match position
        4. Create a Suggest object with the appropriate match type and content

        :param doc: The Redis document to convert
        :param request: The original suggesting request
        :return: A Suggest object representing the suggestion
        """

        # 1) Build the list of raw_chunks from 'doc'
        raw_chunks = []
        for i in range(self._max_chunks):
            val = doc.get(f"chunk_{i}")
            if val:
                raw_chunks.append(val)
            else:
                break

        found_chunks = request.chunks.found_chunks or []
        if not found_chunks:
            # No found chunks => treat entire raw_chunks as a prefix
            return Suggest(
                chunks=[],
                prefix_chunks=raw_chunks,
                match_type="exact",
                prob=doc.get("prob", 0.0),
                labels=doc.get("labels", "").split("\n"),
            )

        # We'll try to match the *last* found chunk
        target = found_chunks[-1]

        # 2) Attempt different match strategies in order
        pos = self._find_match_position_exact(raw_chunks, target)
        match_type = "exact"
        if pos < 0:
            # Next, try prefix
            pos = self._find_match_position_prefix(raw_chunks, target)
            match_type = "prefix"
        if pos < 0:
            # Finally, try soft
            pos = self._find_match_position_soft(raw_chunks, target)
            match_type = "fuzzy"

        # If still not found, fallback to the last chunk
        if pos < 0:
            pos = len(raw_chunks) - 1 if raw_chunks else 0
            match_type = "unknown"

        # 3) Slice raw_chunks into prefix vs matched
        prefix_chunks = raw_chunks[: pos + 1]
        matched_chunks = raw_chunks[pos + 1 :]

        # 4) Build and return Suggest
        return Suggest(
            chunks=matched_chunks,
            prefix_chunks=prefix_chunks,
            match_type=match_type,
            prob=doc.get("prob", 0.0),
            labels=doc.get("labels", "").split("\n"),
        )

    def _search_docs(self, text_query: str, top_k: int) -> List:
        """
        Run the RediSearch query and return matching documents.

        Executes the provided text query against the Redis search index,
        sorting results by probability in descending order.

        :param text_query: The RediSearch query string to execute
        :param top_k: The maximum number of results to return (multiplied by 100 for initial fetch)
        :return: A list of document objects from RediSearch matching the query
        """
        if not text_query:
            return []
        query = (
            Query(text_query).sort_by("prob", asc=False).paging(0, top_k * 100)
        )
        return self.phrases_manager.search_client.search(query).docs

    def get_topk_suggestions(
        self, request: SuggestingRequest, top_k: int = 10
    ) -> List[Suggest]:
        """
        Retrieve the top-k suggestions for the given request.

        This method implements a parallel search strategy:
        1. Submits both strict and soft queries concurrently
        2. Waits for the strict query to complete first
        3. If strict returns enough results, cancels the soft query
        4. Otherwise, waits for the soft query and merges the results
        5. Deduplicates and prioritizes results based on labels and scores

        :param request: The suggesting request containing context for suggestions
        :param top_k: The maximum number of suggestions to return
        :return: A list of Suggest objects representing the top suggestions
        """
        strict_text_query = self._generate_query(
            request, top_k, soft_match=False
        )
        soft_text_query = self._generate_query(request, top_k, soft_match=True)

        # If both queries are empty, no point in proceeding
        if not strict_text_query and not soft_text_query:
            return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Schedule both queries
            f_strict = executor.submit(
                self._search_docs, strict_text_query, top_k
            )
            f_soft = executor.submit(self._search_docs, soft_text_query, top_k)

            # 1) Wait on the strict future
            strict_docs = f_strict.result()

            # 2) Check if we already have enough
            if len(strict_docs) >= top_k:
                # Cancel the soft query; skip waiting for it
                f_soft.cancel()
                soft_docs = []
            else:
                # Not enough strict docs => get soft docs as well
                soft_docs = f_soft.result()

        # Merge them in a "strict-first" list
        combined_docs = strict_docs + soft_docs

        seen_labels = set()
        seen_suggestions = set()
        final_docs = []

        for doc in combined_docs:
            label_ids = doc.label_ids.split(" ") if doc.label_ids else []
            if doc.phrase in seen_suggestions:
                continue

            seen_suggestions.add(doc.phrase)

            # If at least one label_id hasn't been seen, we include the doc
            if any(lbl_id not in seen_labels for lbl_id in label_ids):
                # Mark all these label_ids as seen
                for lbl_id in label_ids:
                    seen_labels.add(lbl_id)
                final_docs.append(doc)

            if len(final_docs) >= top_k:
                break

        # Convert final docs into Suggest objects
        return [
            self._doc_to_suggest(doc.__dict__, request)
            for doc in final_docs[:top_k]
        ]
