## TokenGroupTextSplitter Documentation

### Class Overview
The `TokenGroupTextSplitter` class is designed to efficiently manage the splitting of text for transformer-based models that have strict token constraints. It first uses a provided splitter to break text into semantic blocks and then groups or further splits these blocks to ensure that each chunk meets a specified token limit.

### Motivation
Models typically impose a maximum token limit, and naive splitting of text can disrupt context and semantic meaning. The purpose of this class is to preserve the flow of text by intelligently combining smaller chunks or breaking larger ones down, ensuring that all parts stay within model limitations while maintaining semantic integrity.

### Inheritance
`TokenGroupTextSplitter` inherits from the `ItemSplitter` class, extending its functionality with token-level smart grouping and splitting capabilities.

### Parameters
- `tokenizer`: `PreTrainedTokenizer`; used for token counting and to determine chunk sizes.
- `blocks_splitter`: `ItemSplitter`; an initial splitter that segments the text into semantic blocks.
- `max_tokens`: `Optional[int]`; the maximum number of tokens allowed per chunk. Defaults to the tokenizer's `model_max_length` if not specified.
- `split_sentences`: `bool`; indicates whether chunks that exceed the token limit should be further split into sentences or words.

### Method: _block_split
#### Functionality
The `_block_split` method within the `TokenGroupTextSplitter` class is responsible for splitting a substantial text block into smaller parts that adhere to a specified token limit. It operates at the word level to ensure that each segment does not exceed the prescribed `max_tokens`.

#### Parameters
- `text`: A string representing the text block to be split.
- `max_tokens`: An integer that defines the maximum number of tokens permitted per part.

#### Usage
- Purpose: This method is specifically designed to break down oversized text into smaller, token-safe parts.

#### Example
When provided with a long text and a tokenizer, calling:
```python
_block_split(text, 50)
```
will return a list of text segments, each not exceeding 50 tokens.

### Usage of TokenGroupTextSplitter
The process begins by splitting the input text using the specified `blocks_splitter`. Smaller blocks are subsequently aggregated until they approach the token limit. If any block surpasses this limit, it can be recursively split based on the `split_sentences` flag to ensure that the resulting token count remains compliant with the set constraints.

#### Example
For a lengthy document, the splitter will create multiple text chunks, each maintaining a token count below the designated maximum token limit.