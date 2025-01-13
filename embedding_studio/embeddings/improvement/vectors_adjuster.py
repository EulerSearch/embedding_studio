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
        pass
