## Documentation for SuggestingPhrase

### Functionality
The SuggestingPhrase class represents a candidate phrase that can be suggested to a user as they type. It stores the phrase text, associated labels, domains, and a probability score used for ranking suggestions.

### Motivation
This class was created to support search and autocomplete features, ensuring that suggestions are validated and structured. The validation on the probability score helps maintain reliable ranking of suggestions.

### Inheritance
Inherits from pydantic.BaseModel to leverage automatic data validation and model configuration, making it easier to integrate with other components of the system.

## Documentation for SuggestingPhrase.check_prob method

### Functionality
Validates that the input probability is within the range [0.0, 1.0]. If not, it raises a ValueError.

### Parameters
- prob: A float value representing the probability that should be between 0.0 and 1.0.

### Usage
- **Purpose** - Ensures the probability is a valid float for suggesting phrases.

#### Example
```python
from embedding_studio.models.suggesting import SuggestingPhrase

# Valid probability
SuggestingPhrase.check_prob(0.85)

# Invalid probability, raises ValueError
SuggestingPhrase.check_prob(1.5)
```

---

## Documentation for SearchDocument

### Functionality
SearchDocument is a model that encapsulates a suggestion phrase along with its metadata. It stores the complete phrase, labels, probability, a list of chunks, and domain information. It also provides methods to flatten the nested model for database storage and to generate string hashes for indexing.

### Motivation
This class was created to simplify the processing and storage of suggestion data. By converting the model into a flattened structure, it becomes easier to persist and query suggestion records efficiently.

### Inheritance
SearchDocument inherits from Pydantic's BaseModel, leveraging its validation, serialization, and parsing capabilities.

### Usage
- **Purpose**: Represent and manipulate a suggestion document with additional metadata for search and storage.

#### Example
```python
from embedding_studio.models.suggesting import SearchDocument
document = SearchDocument(
    phrase="hello world",
    labels=["greeting"],
    prob=0.9,
    chunks=[{"value": "hello"}, {"value": "world"}],
    domains=["general"]
)
flattened = document.get_flattened_dict()
print(flattened)
```

---

## Documentation for SearchDocument.get_flattened_dict method

### Functionality
Converts a SearchDocument instance into a flattened dictionary. It uses model_dump to extract a nested dict, then computes hash values for labels and formats domain names. It also processes each chunk using ft_escape_punctuation.

### Parameters
This method does not take any parameters.

### Returns
A dictionary with the flattened representation of the SearchDocument. It includes keys like id, phrase, n_chunks, prob, labels, label_ids, domains, is_original_phrase, and individual chunk keys.

### Usage
The method is typically used to convert the SearchDocument into a storage-friendly format.

#### Example
```python
doc = SearchDocument(...)
flat_dict = doc.get_flattened_dict()
```

---

## Documentation for SearchDocument.from_flattened_dict method

### Functionality
Creates a new instance of SearchDocument from a flattened dictionary. This method extracts required fields and reconstructs the original SearchDocument structure.

### Parameters
- `data` (dict): A flattened representation containing:
  - `phrase`: The original search phrase.
  - `n_chunks`: Total number of chunks for the document.
  - `prob` (optional): A float representing the document probability.
  - `labels` (optional): A space-separated string of labels.
  - `domains` (optional): A space-separated string of domain tags, each wrapped in double underscores.
  - `chunk_i`: A string for each chunk (with i as the index).

### Usage
This method converts a flattened dictionary back into a SearchDocument instance by rebuilding the list of chunks and converting string fields into their respective types.

#### Example
```python
flattened = {
  "phrase": "example",
  "n_chunks": 2,
  "prob": 0.9,
  "labels": "label1 label2",
  "domains": "__dom1__ __dom2__",
  "chunk_0": "first chunk",
  "chunk_1": "second chunk"
}
doc = SearchDocument.from_flattened_dict(flattened)
```

---

## Documentation for Span

### Functionality
The Span class tracks the position of text within a larger string using start and end indices. It helps identify where certain words or phrases appear in user input, facilitating matching and highlighting of suggestions.

### Motivation
The class provides a systematic way to validate and manage text spans. It ensures that the start index is less than the end, preventing errors in text processing and improving suggestion accuracy.

### Inheritance
Span inherits from pydantic's BaseModel, allowing for data validation and easy integration with other models.

### Usage
- **Purpose:** Mark regions in text for suggestion matching.

#### Example
```python
span = Span(start=0, end=5)
print(len(span))  # Outputs: 5
```

---

## Documentation for Span.check_start_end method

### Functionality
Validates the "end" attribute in a Span object, ensuring it is greater than the "start" value. The method retrieves the start value from info.data and raises a ValueError if the condition is not met.

### Parameters
- end (int): The ending index of the span. Must be greater than the start value.
- info (dict): Contains supplemental data, including the start value accessed via info.data.

### Usage
**Purpose**: To guarantee that a Span instance represents a valid range, where the start index is less than the end index.

#### Example
```python
span = Span(start=5, end=10)  # Valid span
span = Span(start=10, end=5)  # Raises ValueError
# ValueError: start must be less than end
```

---

## Documentation for Suggest

### Inheritance
Inherits from BaseModel (from pydantic) which provides data validation and model parsing.

### Functionality
Encapsulates details of a suggestion shown to users. It holds:
- chunks: List of text segments that form the suggestion.
- prefix_chunks: List of prefix parts matching user input.
- match_type: One of "exact", "exact_case_insensitive", "prefix", "fuzzy", or "unknown".
- prob: A float (0.0 to 1.0) representing the match probability.
- labels: A list of categories for filtering suggestions.

### Motivation
This class is designed to structure and transport suggestion data for effective user interface display and filtering.

### Usage
#### Example
```python
from embedding_studio.models.suggesting import Suggest

suggestion = Suggest(
    chunks=["example", "suggestion"],
    prefix_chunks=["ex", "su"],
    match_type="prefix",
    prob=0.85,
    labels=["demo", "example"]
)
print(suggestion)
```