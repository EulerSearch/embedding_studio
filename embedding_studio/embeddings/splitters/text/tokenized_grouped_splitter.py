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
        """A wrapper splitter that groups text chunks to fit within token limits.

        This splitter works with another splitter to first break text into initial chunks,
        then intelligently combines or further splits those chunks to ensure each output chunk
        stays within the specified token limit. It's particularly useful for preparing text
        for models with maximum token constraints.

        :param tokenizer: The tokenizer from the target embedding model, used to count tokens
        :param blocks_splitter: The initial splitter used to break text into semantic chunks
        :param max_tokens: Maximum number of tokens allowed per chunk (default: tokenizer's model_max_length)
        :param split_sentences: Whether to split individual sentences if they exceed the token limit
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
        """Split text into token-aware chunks that respect the maximum token limit.

        This method:
        1. Uses the blocks_splitter to create initial text chunks
        2. Counts tokens in each chunk using the provided tokenizer
        3. Groups smaller chunks together if they fit within the token limit
        4. Splits larger chunks if they exceed the token limit (when split_sentences=True)

        :param item: Text string to split into token-aware chunks
        :return: List of text chunks, each within the specified token limit

        Example:
        ```
        # With a 128-token limit and a sentence splitter:
        # Input: A long document with many sentences totaling over 500 tokens
        # Output: Multiple chunks, each under 128 tokens, with sentences kept together when possible
        ```
        """
        # First, split the input text using the provided blocks_splitter
        # This creates our initial semantic chunks (like sentences, paragraphs, etc.)
        initial_blocks = self.blocks_splitter(item)

        # Initialize the list that will hold our final token-sized chunks
        grouped_blocks = []

        # Initialize a buffer to accumulate smaller chunks until they approach the token limit
        current_group = []

        # Keep track of how many tokens are in our current group
        current_token_count = 0

        # Process each initial block one by one
        for block in initial_blocks:
            # Count how many tokens are in this block using the model's tokenizer
            # This is crucial for ensuring we don't exceed the model's context window
            tokens = self.tokenizer.tokenize(block)
            token_count = len(tokens)

            # Check if adding this block would exceed our token limit
            if current_token_count + token_count > self.max_tokens:
                # If we have accumulated blocks in our current group, save them
                # before handling the current (potentially oversized) block
                if current_group:
                    # Join the accumulated blocks with spaces and add to our results
                    grouped_blocks.append(" ".join(current_group))
                    # Reset our accumulator for the next group
                    current_group = []
                    current_token_count = 0

                # Handle blocks that are themselves too large for our token limit
                if token_count > self.max_tokens and self.split_sentences:
                    # If the block itself exceeds token limit and splitting is enabled,
                    # recursively split it into smaller parts at the word level
                    split_parts = self._block_split(block, self.max_tokens)
                    # Add all these smaller parts to our results
                    grouped_blocks.extend(split_parts)
                else:
                    # If splitting is disabled or unnecessary, add the block as is
                    # This might exceed the token limit if splitting is disabled
                    grouped_blocks.append(block)
            else:
                # This block fits within our current token limit
                # Add it to our accumulator group and update token count
                current_group.append(block)
                current_token_count += token_count

        # Don't forget any leftover blocks in our accumulator
        if current_group:
            grouped_blocks.append(" ".join(current_group))

        # Return all our carefully sized chunks
        return grouped_blocks

    def _block_split(self, text: str, max_tokens: int) -> List[str]:
        """Split a single block of text into multiple parts that fit within token limits.

        This method is used when a single chunk from the initial splitter exceeds
        the token limit and needs to be broken down further at the word level.

        :param text: Text block to split
        :param max_tokens: Maximum tokens allowed per split
        :return: List of text parts, each within the token limit
        """
        # Break the text into individual words
        # This is the finest granularity we'll use for splitting
        words = text.split()

        # Initialize the list that will contain our final text parts
        parts = []

        # Initialize a buffer to collect words until we approach the token limit
        current_part = []

        # Track how many tokens are in our current part
        current_part_token_count = 0

        # Process each word one by one
        for word in words:
            # Tokenize the word to get an accurate token count
            # Some words might result in multiple tokens (e.g., "tokenization" -> ["token", "##ization"])
            word_tokens = self.tokenizer.tokenize(word)

            # Count how many tokens this word represents
            word_token_count = len(word_tokens)

            # Check if adding this word would exceed our token limit
            if current_part_token_count + word_token_count > max_tokens:
                # If we can't add this word, finalize the current part
                # Join all accumulated words with spaces and add to our results
                parts.append(" ".join(current_part))

                # Start a new part with the current word
                # This assumes each word by itself doesn't exceed the token limit
                # (which is generally a safe assumption for most languages and models)
                current_part = [word]
                current_part_token_count = word_token_count
            else:
                # This word fits within our current token count
                # Add it to our current part and update the token count
                current_part.append(word)
                current_part_token_count += word_token_count

        # Don't forget to add the last part if there's anything left
        if current_part:
            parts.append(" ".join(current_part))

        # Return all our token-sized text parts
        return parts
