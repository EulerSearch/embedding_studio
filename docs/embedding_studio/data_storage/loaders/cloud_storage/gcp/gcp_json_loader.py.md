## Documentation for `GCPJSONLoader`

### Functionality
GCPJSONLoader extends GCPTextLoader to load JSON files from GCP Cloud Storage. It decodes byte streams using a specified encoding and parses the JSON data. Optionally, it filters the JSON to retain only specified keys.

### Parameters
- `fields_to_keep`: Optional list or set of JSON keys to retain.
- `retry_config`: Configuration for retrying failed operations.
- `features`: Optional schema for the expected JSON structure.
- `encoding`: Character encoding of the JSON file (default: "utf-8").
- `kwargs`: Additional parameters for credential configuration.

### Usage
**Purpose**: Simplify loading and processing JSON data stored in GCP Cloud Storage by enabling field filtering and robust retry mechanisms.

#### Example
```python
>>> loader = GCPJSONLoader(fields_to_keep=['name', 'age'])
>>> result = loader._get_item(io.BytesIO(b'{"name": "Alice", "age": 30}'))
>>> print(result)
{'name': 'Alice', 'age': 30}
```

### Inheritance
Inherits from `GCPTextLoader`, leveraging its cloud storage retrieval capabilities and retry strategies.

---

## Documentation for `GCPJSONLoader._get_item`

### Functionality
Extract and optionally filter a JSON object from a file object. This method ensures that the file read pointer is reset and decodes the file using the specified encoding. If filtering is enabled via 'fields_to_keep', it returns only the specified fields from the JSON data.

### Parameters
- `file` (io.BytesIO): A byte stream containing JSON data, typically from a cloud storage file.

### Returns
- `Dict` or `List[Dict]`: The parsed JSON object. If filtering is applied, the result contains only the fields specified in 'fields_to_keep'.

### Usage
**Purpose**: Load and filter JSON data from a GCP Cloud Storage file.

#### Example
```python
with open('data.json', 'rb') as f:
    loader = GCPJSONLoader(fields_to_keep={'name', 'id'})
    data = loader._get_item(io.BytesIO(f.read()))
print(data)
```

---

## Documentation for `GCPJSONLoader._filter_fields`

### Functionality
Filters a JSON object by retaining only keys present in the pre-defined list or set assigned to fields_to_keep. If fields_to_keep is None, returns the original item.

### Parameters
- `item`: JSON object (dictionary) with key-value pairs. Expected to contain the parsed JSON data.

### Usage
**Purpose**: Internally used to limit data by keeping only the relevant field names.

#### Example
For `item = {"name": "Alice", "age": 30, "city": "Paris"}` and `fields_to_keep = {"name", "city"}`, the function returns `{"name": "Alice", "city": "Paris"}`.