from abc import ABC, abstractmethod
from typing import List

from embedding_studio.models.suggesting import Suggest, SuggestingRequest
from embedding_studio.suggesting.abtract_phrase_manager import (
    AbstractSuggestionPhraseManager,
)


class AbstractSuggester(ABC):
    """
    An abstract base class that defines the interface for all suggesters.

    This class provides the foundation for implementing different suggestion
    strategies. Any concrete suggester must implement the `get_topk_suggestions`
    method and the `phrases_manager` property.
    """

    @abstractmethod
    def get_topk_suggestions(
        self, request: SuggestingRequest, top_k: int = 10
    ) -> List[Suggest]:
        """
        Returns the top-k suggestions based on the given request.

        :param request: The request containing details such as the chunks found so far
                        and the upcoming chunk to be suggested
        :param top_k: The number of suggestions to retrieve. Defaults to 10
        :return: A list of Suggest objects, each representing a suggested continuation
                 or phrase that matches the criteria

        Example implementation:
        ```python
        def get_topk_suggestions(self, request: SuggestingRequest, top_k: int = 10) -> List[Suggest]:
            # Retrieve candidate phrases from the phrase manager
            candidate_phrases = self.phrases_manager.get_candidate_phrases(request.text)

            # Score and rank the candidates
            scored_phrases = [(self._score_phrase(phrase, request), phrase) for phrase in candidate_phrases]
            scored_phrases.sort(reverse=True)  # Sort by score in descending order

            # Convert top phrases to Suggest objects
            results = []
            for score, phrase in scored_phrases[:top_k]:
                results.append(Suggest(text=phrase.text, score=score))

            return results
        ```
        """

    @property
    @abstractmethod
    def phrases_manager(self) -> AbstractSuggestionPhraseManager:
        """
        Returns the phrase manager used by this suggester.

        :return: An instance of AbstractSuggestionPhraseManager that manages
                 the phrases used for suggestions

        Example implementation:
        ```python
        @property
        def phrases_manager(self) -> AbstractSuggestionPhraseManager:
            return self._phrases_manager
        ```
        """
