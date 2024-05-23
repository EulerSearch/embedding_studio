import re
from typing import List

from embedding_studio.embeddings.splitters.text.pattern_splitter import (
    PatternSplitter,
)


class FieldCombinedSplitter(PatternSplitter):
    def __init__(
        self,
        field_names: List[str],
        field_name_separator: str = ":",
        text_quotes: str = '"',
        separator: str = ",",
    ):
        """If you used embedding_studio.embeddings.data.transforms.dict.line_from_dict.get_text_line_from_dict
        to combine a dict into a solid text line, this is the most natural way to split a final text.

        :param field_names: list of used dict fields
        :param field_name_separator: field name separtor (default: ':', example: 'field_name: text')
        :param text_quotes: symbol used a quotes for text (defult: '"', example: 'field_name: "text"')
        :param separator: symbol used to separate different values (default: ',')
        """
        self.field_names = field_names
        self.field_name_separator = field_name_separator
        self.text_quotes = text_quotes
        self.separator = separator

        regexp = re.compile(
            "|".join(
                [
                    f"({name}{self.field_name_separator} {self.text_quotes}"
                    f"[^{self.text_quotes}]*{self.text_quotes}({self.separator})?)"
                    for name in self.field_names
                ]
            ),
            flags=re.IGNORECASE,
        )
        super(FieldCombinedSplitter, self).__init__(regexp)
