import re
from typing import List, Union

from embedding_studio.embeddings.splitters.item_splitter import ItemSplitter


class PatternSplitter(ItemSplitter):
    def __init__(self, regexp: Union[str, re.Pattern]):
        """Split text into chunks, which match with a provided regexp pattern.

        :param regexp: string or pre-compiled regexp pattern.
        """
        self.regexp = re.compile(regexp) if isinstance(regexp, str) else regexp

    def __call__(self, item: str) -> List[str]:
        return [m.string for m in self.regexp.finditer(item)]
