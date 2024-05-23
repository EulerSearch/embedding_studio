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
        pass

    def __call__(self, item: str) -> List[str]:
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
