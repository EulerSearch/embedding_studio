import re
from typing import List, Tuple

from embedding_studio.models.suggesting import Span


class SuggestingTokenizer:
    """
    A tokenizer class that splits text into tokens based on a regex pattern.

    This tokenizer identifies words, digits, and single non-whitespace symbols
    as individual tokens.
    """

    def __init__(self):
        """
        Initialize the tokenizer with a regex pattern.

        The pattern matches word characters, digits, or any single non-whitespace symbol.
        """
        # Matches word characters, digits, or any single non-whitespace symbol
        self._pattern = re.compile(r"\w+|\d+|[^\w\s]")

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text into a list of tokens.

        :param text: The input string to be tokenized
        :return: A list of token strings
        """
        return self._pattern.findall(text)

    def tokenize_with_spans(self, text: str) -> Tuple[List[str], List[Span]]:
        """
        Tokenize the input text and return both tokens and their position spans.

        :param text: The input string to be tokenized
        :return: A tuple containing:
                - A list of token strings
                - A list of Span objects with start and end positions for each token
        """
        tokens: List[str] = []
        spans: List[Span] = []

        for match in self._pattern.finditer(text):
            tokens.append(match.group())
            spans.append(Span(start=match.start(), end=match.end()))

        return tokens, spans
