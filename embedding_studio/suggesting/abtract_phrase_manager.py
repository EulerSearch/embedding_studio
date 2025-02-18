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
    """

    @abstractmethod
    def convert_phrase_to_request(self, phrase: str) -> SuggestingRequest:
        """Convert a phrase into a structured suggestion request."""

    @abstractmethod
    def add(self, phrases: List[SuggestingPhrase]) -> List[str]:
        """Insert multiple suggesting phrase documents into the collection."""

    @abstractmethod
    def find_phrases_by_values(
        self, phrase_texts: List[str]
    ) -> List[Optional[SearchDocument]]:
        """Find documents for given phrase texts.
        'None' means document for with relevant value is not found"""

    @abstractmethod
    def delete_by_value(self, phrase_texts: List[str]) -> None:
        """Delete documents by phrase value."""

    @abstractmethod
    def delete(self, phrase_ids: List[str]) -> None:
        """Delete documents by their MongoDB _id values."""

    @abstractmethod
    def update_probability(
        self, phrase_id: str, new_probability: float
    ) -> None:
        """Update the probability score for a specific phrase document."""

    @abstractmethod
    def add_labels(self, phrase_id: str, labels: List[str]) -> None:
        """Add labels to a document without duplicating existing labels."""

    @abstractmethod
    def remove_labels(self, phrase_id: str, labels: List[str]) -> None:
        """Remove specified labels from a phrase document."""

    @abstractmethod
    def remove_all_label_values(self, labels: List[str]) -> None:
        """Remove specified labels from all documents containing them."""

    @abstractmethod
    def get_info_by_id(self, phrase_id: str) -> SearchDocument:
        """Fetch a phrase document by its MongoDB _id and return a SearchDocument object."""

    @abstractmethod
    def list_phrases(
        self, offset: int = 0, limit: int = 100
    ) -> List[SearchDocument]:
        """Return a paginated list of full phrase documents."""
