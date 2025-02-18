from abc import ABC, abstractmethod
from typing import List

from embedding_studio.models.suggesting import Suggest, SuggestingRequest
from embedding_studio.suggesting.abtract_phrase_manager import (
    AbstractSuggestionPhraseManager,
)


class AbstractSuggester(ABC):
    """
    An abstract base class that defines the interface for all suggesters.
    Any concrete suggester must implement the `get_topk_suggestions` method.
    """

    @abstractmethod
    def get_topk_suggestions(
        self, request: SuggestingRequest, top_k: int = 10
    ) -> List[Suggest]:
        """
        Returns the top-k suggestions based on the given request.

        :param request:
            The request containing details such as the chunks found so far
            and the upcoming chunk to be suggested.

        :param top_k:
            The number of suggestions to retrieve. Defaults to 10.

        :return:
            A list of Suggest objects, each representing a suggested continuation
            or phrase that matches the criteria.
        """

    @property
    @abstractmethod
    def phrases_manager(self) -> AbstractSuggestionPhraseManager:
        pass
