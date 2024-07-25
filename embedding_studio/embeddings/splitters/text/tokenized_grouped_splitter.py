from typing import List, Optional

from transformers import PreTrainedTokenizer

from embedding_studio.embeddings.splitters.item_splitter import ItemSplitter


class TokenGroupTextSplitter(ItemSplitter):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        blocks_splitter: ItemSplitter,
        max_tokens: Optional[int] = None,
        split_sentences: bool = True,
    ):
        """The items_set_splitter wrapper, which group chunks the way count of tokens
         in each chunk is less than tokenizer max_tokens.

        :param tokenizer: tokenizer used in embedding model
        :param blocks_splitter: original text items_set_splitter
        :param max_tokens: maximum count of tokens in each chunk (default: None)
                           default value means using tokenizer.model_max_length
        :param split_sentences: should algorithm break a solid sentence if it's longer than max_tokens
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.blocks_splitter = blocks_splitter
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else self.tokenizer.model_max_length
        )
        self.split_sentences = split_sentences

    def __call__(self, item: str) -> List[str]:
        initial_blocks = self.blocks_splitter(item)
        grouped_blocks = []
        current_group = []
        current_token_count = 0

        # Iteratively combine chunks
        for block in initial_blocks:
            tokens = self.tokenizer.tokenize(block)
            token_count = len(tokens)

            if current_token_count + token_count > self.max_tokens:
                if current_group:
                    grouped_blocks.append(" ".join(current_group))
                    current_group = []
                    current_token_count = 0

                if token_count > self.max_tokens and self.split_sentences:
                    # Split the block into smaller parts
                    split_parts = self._block_split(block, self.max_tokens)
                    grouped_blocks.extend(split_parts)
                else:
                    grouped_blocks.append(block)
            else:
                current_group.append(block)
                current_token_count += token_count

        if current_group:
            grouped_blocks.append(" ".join(current_group))

        return grouped_blocks

    def _block_split(self, text: str, max_tokens: int) -> List[str]:
        # This method splits a text into multiple parts where each part is within the token limit
        words = text.split()
        parts = []
        current_part = []
        current_part_token_count = 0

        for word in words:
            word_tokens = self.tokenizer.tokenize(word)
            word_token_count = len(word_tokens)

            if current_part_token_count + word_token_count > max_tokens:
                parts.append(" ".join(current_part))
                current_part = [word]
                current_part_token_count = word_token_count
            else:
                current_part.append(word)
                current_part_token_count += word_token_count

        if current_part:
            parts.append(" ".join(current_part))

        return parts
