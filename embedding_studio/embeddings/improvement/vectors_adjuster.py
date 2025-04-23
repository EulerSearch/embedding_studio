from abc import ABC, abstractmethod
from typing import List

from embedding_studio.embeddings.data.clickstream.improvement_input import (
    ImprovementInput,
)


class VectorsAdjuster(ABC):
    @abstractmethod
    def adjust_vectors(
        self, data_for_improvement: List[ImprovementInput]
    ) -> List[ImprovementInput]:
        """
        Adjust embedding vectors based on provided improvement data.

        This method takes a list of improvement inputs and adjusts the vectors
        contained within them according to the specific adjustment strategy
        implemented by the concrete class.

        :param data_for_improvement: A list of ImprovementInput objects containing
                                     queries and their corresponding clicked and
                                     non-clicked elements with embeddings to adjust
        :return: The updated list of ImprovementInput objects with adjusted vectors

        Example implementation:
        ```
        def adjust_vectors(self, data_for_improvement: List[ImprovementInput]) -> List[ImprovementInput]:
            # Simple implementation that increases similarity between query and clicked items
            # and decreases similarity between query and non-clicked items

            for input_data in data_for_improvement:
                query_vector = input_data.query.vector

                # Move clicked vectors closer to query vector
                for clicked_element in input_data.clicked_elements:
                    # Simple adjustment: move vector slightly toward query vector
                    clicked_element.vector = clicked_element.vector + 0.1 * (query_vector - clicked_element.vector)

                # Move non-clicked vectors further from query vector
                for non_clicked_element in input_data.non_clicked_elements:
                    # Simple adjustment: move vector slightly away from query vector
                    non_clicked_element.vector = non_clicked_element.vector - 0.1 * (query_vector - non_clicked_element.vector)

            return data_for_improvement
        ```
        """
