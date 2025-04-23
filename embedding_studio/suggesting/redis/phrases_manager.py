from typing import List, Optional

from redis import Redis
from redis.connection import ConnectionPool
from redisearch import Client, NumericField, TagField, TextField

from embedding_studio.models.suggesting import (
    Chunk,
    SearchDocument,
    SuggestingPhrase,
    SuggestingRequest,
    SuggestingRequestChunks,
    SuggestingRequestSpans,
)
from embedding_studio.suggesting.abtract_phrase_manager import (
    AbstractSuggestionPhraseManager,
)
from embedding_studio.suggesting.tokenizer import SuggestingTokenizer
from embedding_studio.utils.redis_utils import ft_escape_punctuation


class RedisSuggestionPhraseManager(AbstractSuggestionPhraseManager):
    """
    Manages suggestion phrases in a Redis database by tokenizing them into chunks,
    creating indexes for efficient retrieval, and providing CRUD operations.
    """

    def __init__(
        self,
        redis_url: str,
        tokenizer: SuggestingTokenizer,
        index_name: str = "suggestion_phrases",
        max_chunks: int = 20,
    ):
        """
        Initialize the SuggestionPhraseManager with a tokenizer, Redis client,
        and indexes. Automatically creates indexes for chunk fields.

        :param redis_url:
            The Redis client URL to use for storing and retrieving documents.
        :param tokenizer:
            The SuggestingTokenizer responsible for splitting phrases into chunks.
        :param index_name:
            Name of the index storing suggestion data.
        :param max_chunks:
            The maximum number of chunks each document can have. Defaults to 20.
        :raises ValueError:
            If max_chunks is greater than 20.
        """
        if max_chunks > 20:
            raise ValueError("max_chunks cannot be greater than 20")

        self._tokenizer = tokenizer
        self._max_chunks = max_chunks

        self._redis_connection_pool = ConnectionPool.from_url(redis_url)
        self._redis_client = Redis(connection_pool=self._redis_connection_pool)

        self._search_client = None
        self._create_main_index(index_name)

        self._index_name = index_name

    @property
    def redis_client(self) -> Redis:
        return self._redis_client

    @property
    def search_client(self) -> Client:
        return self._search_client

    @property
    def domains_search_client(self) -> Client:
        return self._domains_search_client

    def _create_main_index(self, index_name: str):
        """Create the main search index for phrases"""
        self._search_client = Client(index_name, conn=self._redis_client)

        try:
            # Prepare fields for the index
            fields = []

            # Add chunk fields
            for i in range(self._max_chunks):
                fields.append(
                    TextField(f"chunk_{i}", weight=(i + 1), no_stem=True)
                )

            # Add other fields
            fields.extend(
                [
                    TextField(
                        "phrase", weight=1.0, no_stem=True
                    ),  # Set weight for relevance scoring
                    NumericField("n_chunks", sortable=True, no_index=False),
                    NumericField(
                        "prob", sortable=True
                    ),  # Important for your sorting
                    TagField("label_ids", separator=" "),
                    TagField("domains", separator=" "),
                    NumericField("is_original_phrase", sortable=True),
                ]
            )

            # Create the index
            self._search_client.create_index(fields)

            self._redis_client.execute_command(
                "FT.CONFIG", "SET", "MINPREFIX", "1"
            )
            print("Index is created")

        except Exception as e:
            print(e)

    def convert_phrase_to_request(
        self, phrase: str, domain: Optional[str] = None
    ) -> SuggestingRequest:
        if len(phrase.strip()) == 0:
            return SuggestingRequest(
                chunks=SuggestingRequestChunks(
                    found_chunks=[],
                    next_chunk="",
                ),
                spans=SuggestingRequestSpans(
                    found_chunk_spans=[], next_chunk_span=None
                ),
                domain=domain,
            )

        """Convert a phrase to a suggesting request"""
        # TODO: add LRU-cache
        tokens, spans = self._tokenizer.tokenize_with_spans(phrase.lower())
        chunks = [ft_escape_punctuation(t) for t in tokens]
        next_chunk = ""
        next_chunk_span = None

        if not (
            phrase.endswith(" ")
            or phrase.endswith("\t")
            or phrase.endswith("\n")
        ):
            next_chunk = chunks[-1]
            next_chunk_span = spans[-1]

            chunks = chunks[:-1]
            spans = spans[:-1]

        if len(chunks) > self._max_chunks:
            chunks = chunks[-self._max_chunks :]
            spans = spans[-self._max_chunks :]

        return SuggestingRequest(
            chunks=SuggestingRequestChunks(
                found_chunks=chunks,
                next_chunk=next_chunk,
            ),
            spans=SuggestingRequestSpans(
                found_chunk_spans=spans, next_chunk_span=next_chunk_span
            ),
            domain=domain,
        )

    def _convert_phrase(self, suggesting_phrase: SuggestingPhrase) -> dict:
        """
        Convert a SuggestingPhrase into a Redis-compatible format.

        :param suggesting_phrase:
            The SuggestingPhrase object to be converted.
        :return:
            A dictionary with fields for Redis.
        """
        tokenized = self._tokenizer.tokenize(suggesting_phrase.phrase)
        chunks = []

        for token in tokenized:
            # Generate prefixes for search
            prefixes = [token[:j] for j in range(1, len(token) + 1)]
            chunks.append(Chunk(value=token, search_field=prefixes))

        # Create document
        doc = SearchDocument(
            phrase=suggesting_phrase.phrase,
            chunks=chunks,
            labels=suggesting_phrase.labels,
            prob=suggesting_phrase.prob,
            domains=suggesting_phrase.domains,
        )

        return doc.get_flattened_dict()

    def add(self, phrases: List[SuggestingPhrase]) -> List[str]:
        """
        Insert multiple SuggestingPhrase documents into Redis.

        :param phrases:
            A list of SuggestingPhrase objects to be inserted.
        :return:
            A list of newly inserted document IDs (as strings).
        """
        inserted_ids = []

        for suggesting_phrase in phrases:
            # Convert to Redis format
            redis_doc = self._convert_phrase(suggesting_phrase)

            # Skip if too many chunks
            if redis_doc["n_chunks"] > self._max_chunks:
                continue

            # Create document ID
            doc_id = redis_doc.pop("id")

            # Store in Redis
            key = f"{self._index_name}:{doc_id}"
            self._redis_client.hset(key, mapping=redis_doc)
            self._redis_client.persist(key)

            inserted_ids.append(doc_id)

        return inserted_ids

    def delete(self, phrase_ids: List[str]) -> None:
        """
        Delete documents by their IDs.

        :param phrase_ids:
            A list of phrase IDs specifying which documents to delete.
        :return:
            None
        """
        pipe = self._redis_client.pipeline()

        for phrase_id in phrase_ids:
            doc_id = SearchDocument.hash_string(phrase_id)
            key = f"{self._index_name}:{doc_id}"
            pipe.delete(key)

        pipe.execute()

    def update_probability(
        self, phrase_id: str, new_probability: float
    ) -> None:
        """
        Update the probability score for a specific phrase document.

        :param phrase_id:
            The string ID of the document to update.
        :param new_probability:
            The new probability value to set.
        :return:
            None
        """
        if new_probability < 0 or new_probability > 1:
            raise ValueError(f"Invalid probability value {new_probability}")

        doc_id = SearchDocument.hash_string(phrase_id)
        key = f"{self._index_name}:{doc_id}"

        # Check if document exists
        if not self._redis_client.exists(key):
            raise ValueError(f"No document found with id={phrase_id}")

        # Update probability
        self._redis_client.hset(key, "prob", str(new_probability))

    def add_labels(self, phrase_id: str, labels: List[str]) -> None:
        """
        Add labels to a document without duplicating existing labels.

        :param phrase_id:
            The ID of the document to update.
        :param labels:
            The list of labels to add to the document's 'labels' array.
        :return:
            None
        """
        doc_id = SearchDocument.hash_string(phrase_id)
        key = f"{self._index_name}:{doc_id}"

        # Check if document exists
        if not self._redis_client.exists(key):
            raise ValueError(f"No document found with id={phrase_id}")

        doc_data = self._redis_client.hgetall(key)
        if not doc_data:
            raise ValueError(f"No document found with id={phrase_id}")

        current_labels = doc_data.get(b"labels", "").split("\n")
        current_labels += labels
        current_labels = list(set(current_labels))

        new_labels = "\n".join(current_labels)
        new_label_ids = " ".join(
            [SearchDocument.hash_string(label) for label in current_labels]
        )

        pipe = self._redis_client.pipeline()
        # Set individual label fields
        pipe.hset(key, f"labels", new_labels)
        pipe.hset(key, f"label_ids", new_label_ids)
        pipe.execute()

    def remove_labels(self, phrase_id: str, labels: List[str]) -> None:
        """
        Remove specified labels from a phrase document.

        :param phrase_id:
            The ID of the document whose labels will be removed.
        :param labels:
            The list of labels to remove.
        :return:
            None
        """
        doc_id = SearchDocument.hash_string(phrase_id)
        key = f"{self._index_name}:{doc_id}"

        # Check if document exists
        if not self._redis_client.exists(key):
            raise ValueError(f"No document found with id={phrase_id}")

        doc_data = self._redis_client.hgetall(key)
        if not doc_data:
            raise ValueError(f"No document found with id={phrase_id}")

        # Redis returns bytes for both keys/values by default, so decode them
        # consistently if you haven't already.
        current_labels_raw = doc_data.get(b"labels", b"").decode("utf-8")
        current_labels = (
            current_labels_raw.split("\n") if current_labels_raw else []
        )

        # Filter out labels that need removal
        updated_labels = [lbl for lbl in current_labels if lbl not in labels]

        # Remove duplicates, just in case
        updated_labels = list(set(updated_labels))

        # Build the new stored values for 'labels' (newline-separated)
        # and 'label_ids' (space-separated MD5 hashes).
        new_labels = "\n".join(updated_labels)
        new_label_ids = " ".join(
            [SearchDocument.hash_string(label) for label in updated_labels]
        )

        pipe = self._redis_client.pipeline()
        pipe.hset(key, "labels", new_labels)
        pipe.hset(key, "label_ids", new_label_ids)
        pipe.execute()

    def remove_all_label_values(self, labels: List[str]) -> None:
        """
        Remove the specified labels from all documents containing them.
        Uses pagination to handle large result sets efficiently.

        :param labels:
            The labels to remove from any matching document.
        :return:
            None
        """
        # 1. Build a textual query to find any doc whose label_ids field contains the MD5 of any label
        label_hashes = [SearchDocument.hash_string(lbl) for lbl in labels]

        # Example: @label_ids:{hash1} | @label_ids:{hash2} ...
        joined_terms = " | ".join(f"@label_ids:{{{h}}}" for h in label_hashes)

        batch_size = 100
        offset = 0

        while True:
            # 2. Fetch a batch of documents matching the query
            results = self._search_client.search(
                joined_terms, offset=offset, num=batch_size
            )
            if not results.docs:
                break  # No more results

            pipe = self._redis_client.pipeline()
            for doc in results.docs:
                doc_id = doc.id.split(":", 1)[
                    -1
                ]  # Extract ID from "indexName:<doc_id>"
                key = f"{self._index_name}:{doc_id}"

                doc_data = self._redis_client.hgetall(key)
                if not doc_data:
                    continue

                # Decode the newline-separated labels field
                current_labels_raw = doc_data.get(b"labels", b"").decode(
                    "utf-8"
                )
                current_labels = (
                    current_labels_raw.split("\n")
                    if current_labels_raw
                    else []
                )

                # Remove all labels in our removal set
                updated_labels = [
                    lbl for lbl in current_labels if lbl not in labels
                ]
                # remove duplicates just in case
                updated_labels = list(set(updated_labels))

                # Build updated label_ids from updated_labels
                updated_label_ids = " ".join(
                    [SearchDocument.hash_string(lbl) for lbl in updated_labels]
                )

                # Re-encode the newline-separated label field
                updated_labels_str = "\n".join(updated_labels)

                # Update Redis in pipeline
                pipe.hset(key, "labels", updated_labels_str)
                pipe.hset(key, "label_ids", updated_label_ids)

            pipe.execute()

            offset += batch_size
            # If fewer docs than batch_size returned, we're done
            if len(results.docs) < batch_size:
                break

    def add_domains(self, phrase_id: str, domains: List[str]) -> None:
        """
        Add domains to a document without duplicating existing labels.

        :param phrase_id:
            The ID of the document to update.
        :param domains:
            The list of domains to add to the document's 'domains' array.
        :return:
            None
        """
        doc_id = SearchDocument.hash_string(phrase_id)
        key = f"{self._index_name}:{doc_id}"

        # Check if document exists
        if not self._redis_client.exists(key):
            raise ValueError(f"No document found with id={phrase_id}")

        doc_data = self._redis_client.hgetall(key)
        if not doc_data:
            raise ValueError(f"No document found with id={phrase_id}")

        current_domains = doc_data.get(b"domains", "").split("\n")
        current_domains += [
            f"__{domain.replace('-', '_')}__" for domain in domains
        ]
        current_domains = list(set(current_domains))

        pipe = self._redis_client.pipeline()
        pipe.hset(key, "domains", " ".join(current_domains))

        pipe.execute()

    def remove_domains(self, phrase_id: str, domains: List[str]) -> None:
        """
        Remove specified domains from a phrase document.

        :param phrase_id:
            The ID of the document whose domains will be removed.
        :param domains:
            The list of domains to remove.
        :return:
            None
        """
        doc_id = SearchDocument.hash_string(phrase_id)
        key = f"{self._index_name}:{doc_id}"

        # Check if document exists
        if not self._redis_client.exists(key):
            raise ValueError(f"No document found with id={phrase_id}")

        doc_data = self._redis_client.hgetall(key)
        if not doc_data:
            raise ValueError(f"No document found with id={phrase_id}")

        # Decode the existing 'domains' field and split by newline
        # (matching your add_domains method's code snippet).
        current_domains_raw = doc_data.get(b"domains", b"").decode("utf-8")
        current_domains = (
            current_domains_raw.split("\n") if current_domains_raw else []
        )

        # Convert each domain in 'domains' into the __domain__ format
        # e.g., "foo-bar" => "__foo_bar__"
        domain_tags_to_remove = [f"__{d.replace('-', '_')}__" for d in domains]

        # Filter out the ones we want to remove
        updated_domains = [
            d for d in current_domains if d not in domain_tags_to_remove
        ]

        # Deduplicate if desired
        updated_domains = list(set(updated_domains))

        # Write them back as a space-joined string (just like in add_domains).
        updated_domains_str = " ".join(updated_domains)

        pipe = self._redis_client.pipeline()
        pipe.hset(key, "domains", updated_domains_str)
        pipe.execute()

    def remove_all_domain_values(self, domains: List[str]) -> None:
        """
        Remove the specified domains from all documents containing them.
        Uses pagination to handle large result sets efficiently.

        :param domains:
            The domains to remove from any matching document.
        :return:
            None
        """
        # 1. Build a query to find documents whose 'domains' text field contains any of the target domains.
        #    For instance, each domain is stored as __my_domain__, so we search for that substring.
        #    Example: @domains:(__my_domain__|__my_other_domain__)
        domain_tags = [f"__{d.replace('-', '_')}__" for d in domains]
        or_conditions = "|".join(domain_tags)
        query = f"@domains:({or_conditions})"

        batch_size = 100
        offset = 0

        while True:
            # 2. Fetch a batch of documents matching the query
            results = self._search_client.search(
                query, offset=offset, num=batch_size
            )
            if not results.docs:
                break

            pipe = self._redis_client.pipeline()

            for doc in results.docs:
                # The doc.id is something like "indexName:docId", so extract the docId part
                doc_id = doc.id.split(":", 1)[-1]
                key = f"{self._index_name}:{doc_id}"

                doc_data = self._redis_client.hgetall(key)
                if not doc_data:
                    continue

                # 3. Decode the existing 'domains' field
                current_domains_raw = doc_data.get(b"domains", b"").decode(
                    "utf-8"
                )
                current_domains = (
                    current_domains_raw.split("\n")
                    if current_domains_raw
                    else []
                )

                # 4. Build the set of domain tags to remove
                domain_tags_to_remove = set(
                    f"__{d.replace('-', '_')}__" for d in domains
                )

                # 5. Filter out unwanted domains
                updated_domains = [
                    dom
                    for dom in current_domains
                    if dom not in domain_tags_to_remove
                ]

                # Remove duplicates, if desired
                updated_domains = list(set(updated_domains))

                # 6. Write the updated domains back
                #    (Note that add_domains uses " ".join(...), while here we read with split("\n").)
                updated_domains_str = " ".join(updated_domains)
                pipe.hset(key, "domains", updated_domains_str)

            pipe.execute()

            offset += batch_size
            if len(results.docs) < batch_size:
                break

    def get_info_by_id(self, phrase_id: str) -> SearchDocument:
        """
        Fetch the phrase document by its ID and return a SearchDocument object.

        :param phrase_id:
            The string ID of the document.
        :return:
            A SearchDocument containing phrase, chunks, labels, and probability.
        :raises ValueError:
            If no document is found for the provided phrase_id.
        """
        doc_id = (
            SearchDocument.hash_string(phrase_id)
            if not phrase_id.startswith(self._index_name)
            else phrase_id
        )
        key = (
            f"{self._index_name}:{doc_id}"
            if not doc_id.startswith(self._index_name)
            else doc_id
        )

        # Get document from Redis
        doc_data = self._redis_client.hgetall(key)
        if not doc_data:
            raise ValueError(f"No document found with id={phrase_id}")

        # Build the SearchDocument
        return SearchDocument.from_flattened_dict(doc_data)

    def list_phrases(
        self, offset: int = 0, limit: int = 100
    ) -> List[SearchDocument]:
        """
        Return a paginated list of full phrase documents, rehydrated into SearchDocument objects.

        :param offset:
            Number of documents to offset (offset).
        :param limit:
            Maximum number of documents to return.
        :return:
            A list of SearchDocument objects, each containing phrase, chunks, labels, and prob.
        """
        # Search all documents
        query = "*"  # Match everything
        results = self._search_client.search(query, offset=offset, num=limit)

        # Convert results to SearchDocument objects
        documents = []
        for doc in results.docs:
            documents.append(self.get_info_by_id(doc.id))

        return documents
