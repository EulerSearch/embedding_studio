# Documentation for ChangeCases

## Functionality
The ChangeCases class generates multiple case variants of a text string, including original, lowercase, and uppercase formats. It is designed to augment text data for improved robustness in natural language processing (NLP) tasks.

### Motivation
Changing the text case simulates varied input conditions and helps train models that are resilient to case variations.

### Inheritance
ChangeCases inherits from AugmentationWithRandomSelection, reusing its mechanism for selecting a subset of available augmentations based on a selection_size parameter.

### Method: _raw_transform

#### Functionality
Applies case transformations to a string, producing a list of variations: original, lowercase, and uppercase.

#### Parameters
- `object`: The input string to be transformed.

#### Usage
- **Purpose**: Generate various case modifications of a string.

#### Example
```python
change_cases = ChangeCases()
print(change_cases._raw_transform("Hello"))
# Output: ["Hello", "hello", "HELLO"]
```