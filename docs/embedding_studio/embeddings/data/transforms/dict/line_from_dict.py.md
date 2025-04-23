# Merged Documentation

This documentation covers the functionalities and usage of three methods: `get_field_name`, `get_text_line_from_dict`, and `get_json_line_from_dict`.

## `get_field_name`

### Functionality
Returns a mapped field name if a `field_names` dictionary is provided. When `ignore_missed` is True, it returns the original name if no mapping is found. If False, it expects the mapping to exist and will raise a KeyError if the key is missing.

### Parameters
- `name`: The original field name as a string.
- `field_names`: Optional dictionary with field name mappings.
- `ignore_missed`: Boolean flag determining if missing keys should be ignored (default is True).

### Usage
Purpose - Retrieve a correct field name for data processing, ensuring the field key is properly mapped.

#### Example
```python
mapping = {"color": "colour"}
print(get_field_name("color", mapping))  # Output: colour
print(get_field_name("size", mapping))   # Output: size
```

---

## `get_text_line_from_dict`

### Functionality
Converts a dictionary into a formatted string for fine-tuning embedding models. The output contains each field represented as a combination of a mapped field name, a separator, and a quoted value. Field ordering can be customized or sorted based on parameters.

### Parameters
- `object`: Dictionary to convert into a string.
- `separator`: String to separate fields (default: " ").
- `field_name_separator`: Character separating field name from value (default: ":").
- `text_quote`: Character used for quoting the field value (default: '"').
- `order_fields`: If True, fields are sorted; if False, no sorting; if a list, the provided order is used.
- `ascending`: Boolean indicating if sorting is ascending (default: True).
- `field_names`: Optional dictionary mapping original field names to new ones.
- `ignore_missed`: Boolean flag to ignore fields not present during mapping (default: True).

### Usage
Convert dictionaries to text lines usable for embedding model fine-tuning. Customize field order and names as needed.

#### Example
Given a dictionary:
```python
{"name": "Alice", "age": "30"}
```
Calling:
```python
get_text_line_from_dict({"name": "Alice", "age": "30"})
```
produces a formatted string with each key-value pair.

---

## `get_json_line_from_dict`

### Functionality
This function converts a dictionary into a JSON string. If a field mapping is provided, only the specified keys are included and renamed accordingly in the output.

### Parameters
- `object`: The dictionary to be converted into a JSON string.
- `field_names`: Optional dictionary for mapping original field names to new names. Only fields present in this mapping will be included in the result if provided.

### Usage
- **Purpose**: Transform a dictionary into a JSON string, optionally filtering and renaming its fields using a mapping.

#### Example
Given:
```python
data = {"a": 1, "b": 2, "c": 3}
```
And a mapping:
```python
mapping = {"a": "alpha", "c": "gamma"}
```
The call:
```python
result = get_json_line_from_dict(data, mapping)
```
will produce:
```json
{"alpha": 1, "gamma": 3}
```