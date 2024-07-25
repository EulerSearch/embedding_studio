from typing import Any, List

from embedding_studio.embeddings.splitters.item_splitter import ItemSplitter


class NoSplitter(ItemSplitter):
    """The dummiest items_set_splitter ever -> return the original item as a list."""

    def __call__(self, item: Any) -> List[Any]:
        return [
            item,
        ]
