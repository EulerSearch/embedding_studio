from abc import ABC, abstractmethod
from typing import List, Optional

from embedding_studio.models.suggesting import (
    SearchDocument,
    SuggestingPhrase,
    SuggestingRequest,
)


class AbstractSuggestionPhraseManager(ABC):
    """
    Abstract interface for managing suggestion phrases in a MongoDB collection.

    This class defines the contract for phrase managers that handle storage,
    retrieval, and manipulation of phrases used for suggestions. Implementations
    should manage phrases in a MongoDB collection.
    """

    @abstractmethod
    def convert_phrase_to_request(
        self, phrase: str, domain: Optional[str] = None
    ) -> SuggestingRequest:
        """
        Convert a phrase into a structured suggestion request.

        :param phrase: The input phrase to convert
        :param domain: Optional domain to associate with the request
        :return: A SuggestingRequest object created from the phrase

        Example implementation:
        ```python
        def convert_phrase_to_request(
            self, phrase: str, domain: Optional[str] = None
        ) -> SuggestingRequest:
            # Create a basic request with the phrase text
            request = SuggestingRequest(text=phrase)

            # Add domain if provided
            if domain:
                request.domain = domain

            return request
        ```
        """

    @abstractmethod
    def add(self, phrases: List[SuggestingPhrase]) -> List[str]:
        """
        Insert multiple suggesting phrase documents into the collection.

        :param phrases: List of SuggestingPhrase objects to be added
        :return: List of string IDs for the newly added phrases

        Example implementation:
        ```python
        def add(self, phrases: List[SuggestingPhrase]) -> List[str]:
            # Convert phrases to documents
            documents = [phrase.dict() for phrase in phrases]

            # Insert documents into MongoDB collection
            result = self._collection.insert_many(documents)

            # Return inserted IDs as strings
            return [str(id) for id in result.inserted_ids]
        ```
        """

    @abstractmethod
    def delete(self, phrase_ids: List[str]) -> None:
        """
        Delete documents by their MongoDB _id values.

        :param phrase_ids: List of string IDs identifying the phrases to delete

        Example implementation:
        ```python
        def delete(self, phrase_ids: List[str]) -> None:
            # Convert string IDs to ObjectId
            object_ids = [ObjectId(id) for id in phrase_ids]

            # Delete the documents with these IDs
            self._collection.delete_many({"_id": {"$in": object_ids}})
        ```
        """

    @abstractmethod
    def update_probability(
        self, phrase_id: str, new_probability: float
    ) -> None:
        """
        Update the probability score for a specific phrase document.

        :param phrase_id: String ID of the phrase to update
        :param new_probability: New probability value to set

        Example implementation:
        ```python
        def update_probability(self, phrase_id: str, new_probability: float) -> None:
            # Convert string ID to ObjectId
            object_id = ObjectId(phrase_id)

            # Update the probability field
            self._collection.update_one(
                {"_id": object_id},
                {"$set": {"probability": new_probability}}
            )
        ```
        """

    @abstractmethod
    def add_labels(self, phrase_id: str, labels: List[str]) -> None:
        """
        Add labels to a document without duplicating existing labels.

        :param phrase_id: String ID of the phrase to update
        :param labels: List of label strings to add

        Example implementation:
        ```python
        def add_labels(self, phrase_id: str, labels: List[str]) -> None:
            # Convert string ID to ObjectId
            object_id = ObjectId(phrase_id)

            # Add labels using $addToSet to avoid duplicates
            self._collection.update_one(
                {"_id": object_id},
                {"$addToSet": {"labels": {"$each": labels}}}
            )
        ```
        """

    @abstractmethod
    def remove_labels(self, phrase_id: str, labels: List[str]) -> None:
        """
        Remove specified labels from a phrase document.

        :param phrase_id: String ID of the phrase to update
        :param labels: List of label strings to remove

        Example implementation:
        ```python
        def remove_labels(self, phrase_id: str, labels: List[str]) -> None:
            # Convert string ID to ObjectId
            object_id = ObjectId(phrase_id)

            # Remove the specified labels
            self._collection.update_one(
                {"_id": object_id},
                {"$pull": {"labels": {"$in": labels}}}
            )
        ```
        """

    @abstractmethod
    def remove_all_label_values(self, labels: List[str]) -> None:
        """
        Remove specified labels from all documents containing them.

        :param labels: List of label strings to remove from all documents

        Example implementation:
        ```python
        def remove_all_label_values(self, labels: List[str]) -> None:
            # Remove the specified labels from all documents
            self._collection.update_many(
                {"labels": {"$in": labels}},
                {"$pull": {"labels": {"$in": labels}}}
            )
        ```
        """

    @abstractmethod
    def add_domains(self, phrase_id: str, domains: List[str]) -> None:
        """
        Add domains to a document without duplicating existing domains.

        :param phrase_id: String ID of the phrase to update
        :param domains: List of domain strings to add

        Example implementation:
        ```python
        def add_domains(self, phrase_id: str, domains: List[str]) -> None:
            # Convert string ID to ObjectId
            object_id = ObjectId(phrase_id)

            # Add domains using $addToSet to avoid duplicates
            self._collection.update_one(
                {"_id": object_id},
                {"$addToSet": {"domains": {"$each": domains}}}
            )
        ```
        """

    @abstractmethod
    def remove_domains(self, phrase_id: str, domains: List[str]) -> None:
        """
        Remove specified domains from a phrase document.

        :param phrase_id: String ID of the phrase to update
        :param domains: List of domain strings to remove

        Example implementation:
        ```python
        def remove_domains(self, phrase_id: str, domains: List[str]) -> None:
            # Convert string ID to ObjectId
            object_id = ObjectId(phrase_id)

            # Remove the specified domains
            self._collection.update_one(
                {"_id": object_id},
                {"$pull": {"domains": {"$in": domains}}}
            )
        ```
        """

    @abstractmethod
    def remove_all_domain_values(self, domains: List[str]) -> None:
        """
        Remove the specified domains from all documents containing them.

        :param domains: List of domain strings to remove from all documents

        Example implementation:
        ```python
        def remove_all_domain_values(self, domains: List[str]) -> None:
            # Remove the specified domains from all documents
            self._collection.update_many(
                {"domains": {"$in": domains}},
                {"$pull": {"domains": {"$in": domains}}}
            )
        ```
        """

    @abstractmethod
    def get_info_by_id(self, phrase_id: str) -> SearchDocument:
        """
        Fetch a phrase document by its MongoDB _id and return a SearchDocument object.

        :param phrase_id: String ID of the phrase to retrieve
        :return: A SearchDocument containing the phrase information

        Example implementation:
        ```python
        def get_info_by_id(self, phrase_id: str) -> SearchDocument:
            # Convert string ID to ObjectId
            object_id = ObjectId(phrase_id)

            # Find the document
            document = self._collection.find_one({"_id": object_id})

            if not document:
                raise ValueError(f"No document found with ID: {phrase_id}")

            # Convert MongoDB document to SearchDocument
            return SearchDocument(
                id=str(document["_id"]),
                text=document["text"],
                labels=document.get("labels", []),
                domains=document.get("domains", []),
                probability=document.get("probability", 0.0)
            )
        ```
        """

    @abstractmethod
    def list_phrases(
        self, offset: int = 0, limit: int = 100
    ) -> List[SearchDocument]:
        """
        Return a paginated list of full phrase documents.

        :param offset: Number of documents to skip (for pagination)
        :param limit: Maximum number of documents to return
        :return: List of SearchDocument objects

        Example implementation:
        ```python
        def list_phrases(self, offset: int = 0, limit: int = 100) -> List[SearchDocument]:
            # Query with pagination
            cursor = self._collection.find().skip(offset).limit(limit)

            # Convert MongoDB documents to SearchDocument objects
            result = []
            for doc in cursor:
                result.append(SearchDocument(
                    id=str(doc["_id"]),
                    text=doc["text"],
                    labels=doc.get("labels", []),
                    domains=doc.get("domains", []),
                    probability=doc.get("probability", 0.0)
                ))

            return result
        ```
        """
