# Documentation for PgsqlMultiTextColumnLoader

## Functionality

The `PgsqlMultiTextColumnLoader` class extends the `PgsqlDataLoader` to load and decode text data from multiple columns in a PostgreSQL database. It is motivated by the need to handle multiple text fields, combining them into a single output for further processing. The internal method `_get_item` extracts and decodes text data from multiple columns defined in the loader, iterating over each column and decoding bytes with the specified encoding when necessary.

## Parameters

- `connection_string`: PostgreSQL connection string.
- `query_generator`: PostgreSQL query generator class.
- `text_columns`: List of column names storing text data.
- `retry_config`: Optional retry configuration for database queries.
- `features`: Optional dataset features specification.
- `encoding`: Encoding used for decoding text (default is 'utf-8').

### _get_item Parameters

- `data`: Dict
    A mapping representing a single database row with text columns. Missing columns default to empty strings.

## Returns

- Dict[str, str]: Maps column names to decoded text.

## Usage

**Purpose** - Provide a flexible interface for reading text data across multiple columns. This loader facilitates data ingestion by handling byte decoding and merging fields seamlessly. The `_get_item` method is intended for internal use to prepare row data for further processing.

#### Example

```python
data = {
    'col1': b'Hello',
    'col2': 'World'
}
loader = PgsqlMultiTextColumnLoader(...)
result = loader._get_item(data)
print(result)
```