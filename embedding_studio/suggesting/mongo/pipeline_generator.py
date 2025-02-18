from abc import ABC, abstractmethod
from typing import List

from embedding_studio.models.suggesting import SuggestingRequest


class AbstractPipelineGenerator(ABC):
    """
    Abstract base class for generating suggestion pipelines based on a request.
    """

    @abstractmethod
    def generate_pipeline(
        self, request: SuggestingRequest, top_k: int = 10
    ) -> List[dict]:
        """
        Generate a suggestion pipeline based on the given request.

        :param request: The suggesting request containing input data.
        :param top_k: The maximum number of suggestions to return. Defaults to 10.
        :return: A list of dictionaries representing the generated pipeline steps.
        """
