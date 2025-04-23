## Documentation for `ft_escape_punctuation` and `ft_unescape_punctuation`

### ft_escape_punctuation

#### Functionality

Escapes punctuation characters in a given string for safe RediSearch full text search operations. The function replaces special punctuation with their corresponding escaped forms following RedisSearch tokenization rules.

#### Parameters

- `text` (str): Input string with punctuation to escape. Returns unchanged if empty.

#### Usage

- **Purpose**: Prepares search queries by escaping punctuation to prevent parsing errors in Redis full text search.

#### Example

```python
sample_text = "Hello, world! (Test)"
escaped_text = ft_escape_punctuation(sample_text)
print(escaped_text)  # Output: "Hello\, world\! \(Test\)"
```

---

### ft_unescape_punctuation

#### Functionality

Restores the original punctuation from an escaped string that was prepared for Redis full-text search.

#### Parameters

- `escaped_text`: A string with escaped punctuation.

#### Usage

- **Purpose**: Converts the escaped string back to its original form by replacing escape sequences with the corresponding punctuation.

#### Example

```python
original = "Hello, world!"
escaped = "Hello\, world!"
result = ft_unescape_punctuation(escaped)
print(result)  # Hello, world!
```