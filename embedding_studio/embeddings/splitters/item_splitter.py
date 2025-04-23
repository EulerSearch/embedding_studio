from abc import ABC, abstractmethod
from typing import Any, List


class ItemSplitter(ABC):
    """Interface for a method of splitting an item into subitems.

    This abstract class defines the contract for item splitters, which are used to break down
    items into smaller components when they are too large to process in their entirety.

    Examples of implementations include text splitters that divide documents into paragraphs,
    image splitters that segment images into smaller patches, etc.
    """

    @abstractmethod
    def __call__(self, item: Any) -> List[Any]:
        """Split an item into a list of subitems.

        :param item: The original item to be split
        :return: A list containing the subitems

        Example implementation:
        ```
        def __call__(self, item: str) -> List[str]:
            # Simple paragraph splitter implementation
            paragraphs = item.split("\n\n")
            return [p for p in paragraphs if p.strip()]
        ```
        """
        raise NotImplementedError
