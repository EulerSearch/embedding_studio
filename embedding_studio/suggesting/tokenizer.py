import re
from typing import List, Tuple

from embedding_studio.models.suggesting import Span


class SuggestingTokenizer:
    def __init__(self):
        # Matches word characters, digits, or any single non-whitespace symbol
        self._pattern = re.compile(r"\w+|\d+|[^\w\s]")

    def tokenize(self, text: str) -> List[str]:
        """
        Return a list of tokens as strings (no spans).
        """
        return self._pattern.findall(text)

    def tokenize_with_spans(self, text: str) -> Tuple[List[str], List[Span]]:
        """
        Return a tuple:
          - A list of tokens.
          - A list of (start, end) spans for each token in 'text'.
        """
        tokens: List[str] = []
        spans: List[Span] = []

        for match in self._pattern.finditer(text):
            tokens.append(match.group())
            spans.append(Span(start=match.start(), end=match.end()))

        return tokens, spans
