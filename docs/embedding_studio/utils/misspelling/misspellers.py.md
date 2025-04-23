# Merged Documentation

## Documentation for Text Mutation Functions

This documentation covers a set of functions designed to simulate typographical errors in words. These functions can be utilized for generating misspellings, useful in text augmentation tasks and testing spell checkers.

### Function: `adjacent_key_error`

#### Functionality
Replaces some characters in a word with one of its adjacent keys based on a keyboard layout. Randomly introduces an error according to a given probability.

#### Parameters
- `word`: The original word to be mutated.

#### Returns
- A new string with some characters replaced, simulating a typing mistake by using adjacent keyboard keys.

#### Usage
- **Purpose**: Mimics typographical errors typically caused by hitting a key adjacent to the intended one.

#### Example
```python
original = "example"
error_word = adjacent_key_error(original)
print(error_word)  # Might print: exzmple
```

---

### Function: `delete_random_character`

#### Functionality
Deletes a random character from a provided word. This function simulates human typographical errors by randomly removing one character from the input word.

#### Parameters
- `word`: The input word as a string. Must have more than one character for the deletion to occur.

#### Usage
Used to generate misspellings for text augmentation tasks.

#### Example
```python
original = "example"
result = delete_random_character(original)
print(result)  # might output "exmple" or "exaple"
```

---

### Function: `swap_characters`

#### Functionality
Swaps two characters in a string at specified positions.

#### Parameters
- `string`: The original string.
- `i`: The index of the first character.
- `j`: The index of the second character.

#### Usage
**Purpose**: Swap two characters in a string.

#### Example
```python
swap_characters("hello", 0, 1)  # Result: "ehllo"
```

---

### Function: `swap_random_adjacent_characters`

#### Functionality
Swaps two adjacent characters in the given string at random. If the string has fewer than two characters, it returns the original string.

#### Parameters
- `string`: The original string.

#### Returns
- A new string with one pair of adjacent characters swapped.

#### Usage
Useful for generating common misspellings to test spell checkers. It can be integrated into text processing pipelines.

#### Example
```python
swap_random_adjacent_characters('hello')  # might return 'hlelo' depending on the selected index.
```

---

### Function: `insert_random_character`

#### Functionality
Inserts a random lowercase character into a given word. The character is inserted at a random position, while the rest of the word remains unchanged.

#### Parameters
- `word`: The original word into which the character will be inserted.

#### Usage
Used for generating misspelled words by randomly adding extra letters.

#### Example
```python
insert_random_character("example")  # 'exxample'  (output may vary due to randomness)
```

---

### Function: `random_split`

#### Functionality
Randomly splits a word into two parts by inserting a space at a random index.

#### Parameters
- `word`: The original word as a string.

#### Usage
- **Purpose**: To divide the input word into two parts with a space in between.

#### Example
```python
Input: "example"
Output: "exam ple"
```

---

### Function: `introduce_misspellings_with_keyboard_map`

#### Functionality
This function introduces misspellings to the input text by making random editing operations based on a keyboard layout. It tokenizes the text and applies one of several error functions, simulating common typing mistakes.

#### Parameters
- `text`: The original text to modify.
- `error_rate`: The probability (0-1) that an error will be introduced for each token. Defaults to 0.1.
- `tokenizer`: An optional tokenizer (following nltk TokenizerI) used to tokenize the text. If None, a default TreebankWordTokenizer is used.

#### Usage
- **Purpose**: To simulate typical typing errors by introducing random misspellings. Useful for testing and generating synthetic typo data.

#### Example
```python
Input: "Hello world"
Possible output: "Heklo wormd"
```