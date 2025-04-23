import json
from typing import List

from embedding_studio.embeddings.splitters.item_splitter import ItemSplitter


class JSONSplitter(ItemSplitter):
    """Splits text that was created by combining dictionary fields into a single line.

    This splitter is designed to work with text that was generated using
    embedding_studio.embeddings.data.transforms.dict.line_from_dict.get_text_line_from_dict.
    It creates a regex pattern that matches each field and its value, allowing for splitting
    the combined text back into its individual field components.

    :param field_names: List of dictionary field names to match in the text
    :param field_name_separator: Character used to separate field names from values (default: ':')
    :param text_quotes: Character used to enclose field values (default: '"')
    :param separator: Character used to separate different field-value pairs (default: ',')
    """

    def __init__(self, field_names: List[str]):
        """Split json into chunks by inner items."""
        self.field_names = field_names

    def __call__(self, item: str) -> List[str]:
        """Split a JSON string into multiple JSON strings, each containing a single field.

        :param item: A JSON string to split
        :return: List of JSON strings, each containing a single requested field from the original

        Example:
        ```
        # For input: '{"title": "Example", "text": "Some text", "author": "John"}'
        # With field_names = ["title", "text"]
        # Returns: ['{"title": "Example"}', '{"text": "Some text"}']
        ```
        """

        data = json.loads(item)
        parts = []
        for field in self.field_names:
            if field in data:
                parts.append(json.dumps({field: data[field]}))

        return parts
