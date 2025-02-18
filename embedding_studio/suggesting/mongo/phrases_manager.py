from typing import List, Optional

from bson import ObjectId
from pymongo.collection import Collection

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


class MongoSuggestionPhraseManager(AbstractSuggestionPhraseManager):
    """
    Manages suggestion phrases in a MongoDB collection by tokenizing them into chunks,
    creating indexes for efficient retrieval, and providing CRUD operations.
    """

    def __init__(
        self,
        collection: Collection,
        tokenizer: SuggestingTokenizer,
        max_chunks: int = 20,
    ):
        """
        Initialize the SuggestionPhraseManager with a tokenizer, MongoDB database,
        and a target collection. Automatically creates indexes for chunk fields.

        :param collection:
            The MongoDB collection used to store and query suggestion documents.
        :param tokenizer:
            The SuggestingTokenizer responsible for splitting phrases into chunks.
        :param max_chunks:
            The maximum number of chunks each document can have. Defaults to 30.
        :raises ValueError:
            If max_chunks is greater than 20.
        """
        if max_chunks > 20:
            raise ValueError("max_chunks cannot be greater than 20")

        self._tokenizer = tokenizer
        self._max_chunks = max_chunks
        self.collection = collection
        self._create_indexes()

    def convert_phrase_to_request(self, phrase: str) -> SuggestingRequest:
        # TODO: add LRU-cache
        tokens, spans = self._tokenizer.tokenize_with_spans(phrase)
        chunks = tokens
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
        )

    def _create_indexes(self) -> None:
        """
        Create necessary indexes on the collection for chunk fields,
        search prefixes, document length, probability, and compound queries,
        if they do not already exist.
        """
        # 1. Retrieve existing index names so we can avoid duplicates or naming conflicts.
        existing_index_names = {
            idx["name"] for idx in self.collection.list_indexes()
        }

        # 2. For each chunk field and search field
        for i in range(self._max_chunks):
            chunk_index_name = f"chunk_{i}_index"
            search_index_name = f"search_{i}_index"

            if chunk_index_name not in existing_index_names:
                self.collection.create_index(
                    [(f"chunk_{i}", 1)],
                    name=chunk_index_name,
                )

            if search_index_name not in existing_index_names:
                self.collection.create_index(
                    [(f"search_{i}", 1)],
                    name=search_index_name,
                )

        # 3. Other indexes
        if "length_index" not in existing_index_names:
            self.collection.create_index(
                [("n_chunks", 1)],
                name="length_index",
            )

        if "phrase_index" not in existing_index_names:
            self.collection.create_index(
                [("phrase", 1)],
                name="phrase_index",
            )

        if "probability_index" not in existing_index_names:
            self.collection.create_index(
                [("prob", -1)],
                name="probability_index",
            )

        if "length_first_chunk_index" not in existing_index_names:
            self.collection.create_index(
                [("n_chunks", 1), ("chunk_0", 1)],
                name="length_first_chunk_index",
            )

    def _convert_phrase(self, suggesting_phrase: SuggestingPhrase) -> dict:
        """
        Convert a SuggestingPhrase into a flattened dictionary suitable for MongoDB.
        Each token is stored as a separate 'chunk_{i}', and each prefix as 'search_{i}'.

        :param suggesting_phrase:
            The SuggestingPhrase object to be converted.
        :return:
            A dictionary with fields for the original phrase, labels, probability,
            and chunk/prefix pairs for each token.
        """
        tokenized = self._tokenizer.tokenize(suggesting_phrase.phrase)
        chunks = []
        for token in tokenized:
            # Generate prefixes for search
            prefixes = [token[:j] for j in range(1, len(token) + 1)]
            chunks.append(Chunk(value=token, search_field=prefixes))

        doc = SearchDocument(
            phrase=suggesting_phrase.phrase,
            chunks=chunks,
            labels=suggesting_phrase.labels,
            prob=suggesting_phrase.prob,
        )
        return doc.get_flattened_dict()

    def add(self, phrases: List[SuggestingPhrase]) -> List[str]:
        """
        Insert multiple SuggestingPhrase documents into the collection.

        :param phrases:
            A list of SuggestingPhrase objects to be inserted.
        :return:
            A list of newly inserted document IDs (as strings).
        :raise BulkWriteError:
            In case if something went wrong during insert_many.
        """
        documents = [
            self._convert_phrase(suggesting_phrase)
            for suggesting_phrase in phrases
        ]

        result = self.collection.insert_many(documents, ordered=False)
        return [str(inserted_id) for inserted_id in result.inserted_ids]

    def find_phrases_by_values(
        self, phrase_texts: List[str]
    ) -> List[Optional[SearchDocument]]:
        """
        Look up documents by their 'phrase' text field and return them as SearchDocument objects.

        :param phrase_texts:
            A list of strings for which to find matching 'phrase' fields in the database.
        :return:
            A list of SearchDocument objects or None if a matching phrase was not found.
        """
        results = []
        for phrase in phrase_texts:
            doc = self.collection.find_one({"phrase": phrase})
            if doc:
                # Convert flat document back to SearchDocument format
                chunks = []
                for i in range(doc["n_chunks"]):
                    chunks.append(
                        Chunk(
                            value=doc[f"chunk_{i}"],
                            search_field=doc[f"search_{i}"],
                        )
                    )
                results.append(
                    SearchDocument(
                        phrase=doc["phrase"],
                        chunks=chunks,
                        labels=doc.get("labels", []),
                        prob=doc.get("prob"),
                    )
                )
            else:
                results.append(None)
        return results

    def delete_by_value(self, phrase_texts: List[str]) -> None:
        """
        Delete documents from the collection matching the given phrase values.

        :param phrase_texts:
            A list of phrase strings identifying which documents to remove.
        :return:
            None
        """
        self.collection.delete_many({"phrase": {"$in": phrase_texts}})

    def delete(self, phrase_ids: List[str]) -> None:
        """
        Delete documents by their MongoDB _id values.

        :param phrase_ids:
            A list of _id strings specifying which documents to delete.
        :return:
            None
        """
        self.collection.delete_many({"_id": {"$in": phrase_ids}})

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

        result = self.collection.update_one(
            {"_id": phrase_id}, {"$set": {"prob": new_probability}}
        )
        if result.matched_count == 0:
            raise ValueError(f"No document found with _id={phrase_id}")

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
        result = self.collection.update_one(
            {"_id": phrase_id}, {"$addToSet": {"labels": {"$each": labels}}}
        )

        if result.matched_count == 0:
            raise ValueError(f"No document found with _id={phrase_id}")

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
        result = self.collection.update_one(
            {"_id": phrase_id}, {"$pull": {"labels": {"$in": labels}}}
        )
        if result.matched_count == 0:
            raise ValueError(f"No document found with _id={phrase_id}")

    def remove_all_label_values(self, labels: List[str]) -> None:
        """
        Remove the specified labels from all documents containing them.

        :param labels:
            The labels to remove from any matching document.
        :return:
            None
        """
        self.collection.update_many(
            {"labels": {"$in": labels}}, {"$pull": {"labels": {"$in": labels}}}
        )

    def get_info_by_id(self, phrase_id: str) -> SearchDocument:
        """
        Fetch the phrase document by its MongoDB _id, reconstruct its chunks,
        and return a SearchDocument object.

        :param phrase_id:
            The string representation of the document's ObjectId.
        :return:
            A SearchDocument containing phrase, chunks, labels, and probability.
        :raises ValueError:
            If no document is found for the provided phrase_id.
        """
        # Convert the string ID into an ObjectId if necessary:
        # (If your _id field is stored as a string and not an ObjectId, remove ObjectId(...) below.)
        try:
            obj_id = ObjectId(phrase_id)
        except:
            # If you truly store _id as a string, you can skip this conversion.
            # For safety, you can handle an invalid ObjectId format here.
            obj_id = phrase_id

        doc = self.collection.find_one({"_id": obj_id})
        if not doc:
            raise ValueError(f"No document found with _id={phrase_id}")

        # Reconstruct the chunk list from the flattened fields
        chunks = []
        for i in range(doc["n_chunks"]):
            chunks.append(
                Chunk(
                    value=doc[f"chunk_{i}"],
                    search_field=doc[f"search_{i}"],
                )
            )

        # Build and return the SearchDocument
        return SearchDocument(
            phrase=doc["phrase"],
            chunks=chunks,
            labels=doc.get("labels", []),
            prob=doc.get("prob", 1.0),
        )

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
        cursor = self.collection.find({}).skip(offset).limit(limit)

        results = []
        for doc in cursor:
            # Reconstruct the chunk list from the flattened fields
            chunks = []
            for i in range(doc["n_chunks"]):
                chunks.append(
                    Chunk(
                        value=doc[f"chunk_{i}"],
                        search_field=doc[f"search_{i}"],
                    )
                )

            # Build the SearchDocument object
            search_doc = SearchDocument(
                phrase=doc["phrase"],
                chunks=chunks,
                labels=doc.get("labels", []),
                prob=doc.get("prob", 1.0),
            )
            results.append(search_doc)

        return results
