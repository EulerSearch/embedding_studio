## Documentation for `SuggestingTokenizer`

### Functionality
The `SuggestingTokenizer` class tokenizes an input string by applying a regex pattern that matches words, digits, and single non-whitespace symbols. It provides both a list of tokens and token spans with start and end positions for each token.

### Purpose
This class is designed to support text preprocessing for embedding suggestion tasks. Breaking text into tokens simplifies analysis and subsequent processing.

### Motivation
Tokenization is crucial for text analysis. By offering both token lists and span information, this class aids in aligning tokens with their original text positions, facilitating error tracking and debugging.

### Inheritance
`SuggestingTokenizer` directly inherits from Python's built-in object class. This minimal inheritance allows for extension and customization of the tokenization process without unnecessary complexity.

### Methods

#### `tokenize`
- **Functionality**: This method tokenizes a given text string using a pre-compiled regular expression. The regex splits the text into words, digits, and non-whitespace symbols.
- **Parameters**:
  - `text`: A string representing the input text to be tokenized.
- **Returns**: A list of strings, where each string is a token extracted from the input text.
- **Usage**: The method is used to break down text into tokens for further processing. It is useful in scenarios where text needs to be divided into meaningful elements (words, digits, punctuation).
- **Example**:
  ```python
  tokens = tokenizer.tokenize("Hello, world!")
  print(tokens)  # ["Hello", ",", "world", "!"]
  ```

#### `tokenize_with_spans`
- **Functionality**: This method tokenizes an input string and returns both the tokens and their corresponding spans as a tuple. The spans indicate the start and end positions of each token in the original string.
- **Parameters**:
  - `text`: A string that is to be tokenized.
- **Returns**: A tuple containing:
  - A list of token strings.
  - A list of Span objects with start and end positions for each token.
- **Usage**: Extract tokens along with their character spans from a given text.
- **Example**:
  ```python
  tokens, spans = tokenizer.tokenize_with_spans("Hello, world!")
  print(tokens)  # ['Hello', ',', 'world', '!']
  print(spans)   # [Span(start=0, end=5), Span(start=5, end=6), ...]
  ```