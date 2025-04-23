## Documentation for `generate_fuzzy_regex` and `combine_chunks`

### `generate_fuzzy_regex`

#### Functionality
Generates a fuzzy regex pattern that matches variations of an input text. The pattern is case-insensitive and allows for one to two character substitutions at any position in the text.

#### Parameters
- **text**: The string for which a fuzzy regex pattern is generated.

#### Usage
**Purpose**: To build a regex pattern capable of matching input strings with minor character variations, useful for fuzzy text matching.

##### Example
```python
pattern = generate_fuzzy_regex("Hello")
# pattern may look like:
# ^([Hh][a-zA-Z]{1,2}[Ee][Ll][Ll][Oo]|[Hh][Ee][a-zA-Z]{1,2}[Ll][Ll][Oo]|...)
import re
re.match(pattern, "HEllo")
```

### `combine_chunks`

#### Functionality
This function takes a list of tokens and merges them into a single string, ensuring that punctuation is attached correctly to tokens. Tokens in `punctuation_attach_left` (like '.', ',', '!', '?', ':', ';', ')', ']', or '}') are appended to the previous token with no space, while tokens in `punctuation_attach_both` (like '-' or '/') are merged with both the previous and next token. All other tokens are separated by spaces.

#### Parameters
- **chunks**: List[str]  
  A list of tokens (words and punctuation) that will be combined.

#### Returns
- **A string**:  
  The combined string with correct punctuation formatting.

#### Usage
**Purpose**: Used to combine tokenized text into a coherent sentence string while preserving punctuation rules.

##### Example
```python
tokens = ['a', 'b', '-', 'c', 'd', '.']
result = combine_chunks(tokens)
# result is "a b-c d."
```