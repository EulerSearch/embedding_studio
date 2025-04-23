# Documentation for AwsS3JSONLoader

## Functionality
This module extends AwsS3TextLoader to provide specialized handling for JSON files stored in AWS S3. It parses JSON content and supports filtering fields based on user-defined parameters.

## Parameters
- `fields_to_keep`: Optional list or set of field names to retain in the JSON.
- `retry_config`: Optional configuration for retry strategies during loading.
- `features`: Optional specification of expected dataset features.
- `encoding`: Character encoding used for processing the JSON file.
- `kwargs`: Additional parameters for AWS S3 credentials.

## Usage
**Purpose** - To load and process JSON files from AWS S3 with optional field filtering, simplifying the extraction of relevant fields.

### Example
```python
loader = AwsS3JSONLoader(
    fields_to_keep=["id", "text"],
    aws_access_key_id="YOUR_KEY",
    aws_secret_access_key="YOUR_SECRET"
)
data = loader.load_items([
    S3FileMeta(bucket="my-bucket", file="data.json")
])
```

---

## Documentation for AwsS3JSONLoader._filter_fields

### Functionality
Filters a dictionary based on the fields_to_keep attribute. Only keys that exist in fields_to_keep are retained in the returned dictionary.

### Parameters
- item (dict): The dictionary to filter.

### Return Value
- dict: A new dictionary containing only the allowed fields from the original item.

### Usage
Use this method to remove unwanted data from JSON records loaded from S3. Ensure that the instance's fields_to_keep is set before invoking this method.

#### Example
```python
loader = AwsS3JSONLoader(fields_to_keep=["id", "text"])
filtered_item = loader._filter_fields(item)
```

---

## Documentation for AwsS3JSONLoader._get_item

### Functionality
Processes a BytesIO object containing JSON data. Overrides the parent method to decode the file, parse it into a JSON object, and filter its fields when 'fields_to_keep' is set.

### Parameters
- file: A BytesIO object that holds the downloaded JSON data.

### Usage
- **Purpose**: Convert file contents to a JSON object with optional field filtering based on the provided configuration.

#### Example
```python
# Instantiate loader to keep only 'id' and 'text'
loader = AwsS3JSONLoader(
    fields_to_keep=["id", "text"],
    aws_access_key_id="YOUR_KEY",
    aws_secret_access_key="YOUR_SECRET"
)
item = loader.load_items([
    S3FileMeta(bucket="my-bucket", file="data.json")
])
```