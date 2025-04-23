import re
from typing import List, Union

from embedding_studio.embeddings.splitters.item_splitter import ItemSplitter


class PatternSplitter(ItemSplitter):
    def __init__(self, regexp: Union[str, re.Pattern]):
        """Split text into chunks that match a provided regular expression pattern.

        This splitter uses regular expressions to identify and extract matching parts from the text.
        It's useful for extracting structured data or specific patterns from unstructured text.

        :param regexp: String or pre-compiled regexp pattern used to identify text chunks
        """
        self.regexp = re.compile(regexp) if isinstance(regexp, str) else regexp

    def __call__(self, item: str) -> List[str]:
        """Extract all text parts that match the regular expression pattern.

        :param item: Text string to process
        :return: List of text chunks that match the provided regular expression pattern

        Example:
        ```
        # Using a pattern to extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        splitter = PatternSplitter(email_pattern)

        # For input: "Contact us at support@example.com or sales@example.com"
        # Returns: ['support@example.com', 'sales@example.com']
        ```
        """
        return [m.group(0) for m in self.regexp.finditer(item)]
