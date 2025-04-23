from typing import List

from embedding_studio.embeddings.splitters.item_splitter import ItemSplitter


class SeparatorSplitter(ItemSplitter):
    def __init__(self, separators: List[str] = None):
        """Split text into chunks using a list of separator characters.

        This splitter performs sequential splitting of text using each separator in the provided list.
        Each chunk from a split is further split by the next separator in the list, creating a nested
        splitting effect.

        :param separators: List of separator strings to use for splitting (default: [";"])
        """
        self.separators = (
            separators
            if separators
            else [
                ";",
            ]
        )

    def __call__(self, item: str) -> List[str]:
        """Split text using each provided separator in sequence.

        The splitting process is iterative - first split by the first separator,
        then each resulting chunk is split by the next separator, and so on.

        :param item: Text string to split
        :return: List of text chunks after applying all separators

        Example:
        ```
        # With separators = [';', ',']
        splitter = SeparatorSplitter([';', ','])

        # For input: "apple;banana,orange;grape,melon"
        # First splits by ';' to get: ["apple", "banana,orange", "grape,melon"]
        # Then splits each chunk by ',' to get: ["apple", "banana", "orange", "grape", "melon"]
        ```
        """
        result = [
            item,
        ]
        for separator in self.separators:
            result_split = []
            for part in result:
                result_split += part.split(separator)

            result = result_split

        return result
