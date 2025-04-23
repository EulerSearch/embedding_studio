import re
from typing import List

from nltk.tokenize import sent_tokenize

from embedding_studio.embeddings.splitters.item_splitter import ItemSplitter

CODE_BLOCK_REGEX = re.compile(
    r"(```[a-z]*\n[\s\S]*?\n```|```[a-z]*\n```)", re.MULTILINE
)
LIST_ITEM_REGEX = re.compile(r"^\d+\.\s", re.MULTILINE)


class DummySentenceSplitter(ItemSplitter):
    CODE_BLOCK_PLACEHOLDER = "CODE_BLOCK_PLACEHOLDER"

    def __init__(self):
        """Initialize a splitter that preserves code blocks while splitting sentences.

        This splitter is specifically designed for text that contains both regular text
        and code blocks (marked with triple backticks). It preserves the integrity of code blocks
        while splitting the surrounding text into sentences.
        """

    def __call__(self, item: str) -> List[str]:
        """Split text into sentences while preserving code blocks as individual chunks.

        This method performs the following steps:
        1. Identifies and extracts code blocks (text between triple backticks)
        2. Replaces code blocks with placeholders
        3. Splits the remaining text into sentences using NLTK's sent_tokenize
        4. Handles numbered list items specially to preserve their context
        5. Reconstructs the original text with code blocks intact but as separate chunks

        :param item: Text string containing both prose and potentially code blocks
        :return: List of text chunks where code blocks are preserved and prose is split into sentences

        Example:
        ```
        # For input: "This is a sentence. Here's some code:\n```python\nprint('hello')\n```\nAnother sentence."
        # Returns: ["This is a sentence.", "```python\nprint('hello')\n```", "Another sentence."]
        ```
        """
        # Extract code blocks and replace them with placeholders
        code_blocks = CODE_BLOCK_REGEX.findall(item)
        text_without_code = CODE_BLOCK_REGEX.sub(
            DummySentenceSplitter.CODE_BLOCK_PLACEHOLDER, item
        )

        # Split the remaining text into sentences or blocks
        blocks = []
        current_block = []
        for line in text_without_code.split("\n"):
            if DummySentenceSplitter.CODE_BLOCK_PLACEHOLDER in line:
                # Append the current block if it's not empty
                if current_block:
                    blocks.append(" ".join(current_block))
                    current_block = []
                # Append the code block
                blocks.append(code_blocks.pop(0))
            elif LIST_ITEM_REGEX.match(line):
                if current_block:
                    blocks.append(" ".join(current_block))
                    current_block = [line]
                else:
                    current_block.append(line)
            else:
                sentences = sent_tokenize(line)
                if current_block:
                    current_block.append(sentences.pop(0))
                blocks.extend(sentences)

        # Append any remaining sentences in the current block
        if current_block:
            blocks.append(" ".join(current_block))

        return blocks
