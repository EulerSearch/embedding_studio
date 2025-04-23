from typing import Any, List

from embedding_studio.embeddings.splitters.item_splitter import ItemSplitter


class NoSplitter(ItemSplitter):
    """The simplest item splitter implementation that returns the original item as a single-element list.

    This splitter doesn't actually split anything but maintains compatibility with the ItemSplitter
    interface. It's useful as a default or when splitting is optional but the interface expects
    an ItemSplitter.
    """

    def __call__(self, item: Any) -> List[Any]:
        """Return the original item wrapped in a list without splitting.

        :param item: The original item
        :return: A list containing only the original item
        """
        return [
            item,
        ]
