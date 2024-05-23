from typing import List

from embedding_studio.embeddings.splitters.item_splitter import ItemSplitter


class SeparatorSplitter(ItemSplitter):
    def __init__(self, separators: List[str] = None):
        """Simple iterative split of text into chunks via list of separators.

        :param separators: list of used separators (default: None)
                           None means usage of just one symbol ':'
        """
        self.separators = (
            separators
            if separators
            else [
                ";",
            ]
        )

    def __call__(self, item: str) -> List[str]:
        result = [
            item,
        ]
        for separator in self.separators:
            result_split = []
            for part in result:
                result_split += part.split(separator)

            result = result_split

        return result
