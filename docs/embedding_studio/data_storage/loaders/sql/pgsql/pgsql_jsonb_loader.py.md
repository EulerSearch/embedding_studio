# Documentation for `PgsqlJSONBLoader`

## Class Overview

The `PgsqlJSONBLoader` class is designed to load and process JSONB data from a PostgreSQL database. It extracts the JSONB content, with the option to filter specific fields if requested.

### Parameters

- `connection_string`: PostgreSQL connection string.
- `query_generator`: Class to generate PostgreSQL queries.
- `jsonb_column`: The column name where JSONB data is stored (default is 'jsonb_data').
- `fields_to_keep`: Optional list or set of fields to extract from the JSONB data.
- `retry_config`: Optional configuration for retrying database operations.
- `features`: Expected features for the dataset.
- `**kwargs`: Additional keyword arguments.

### Usage

- **Purpose**: To provide a flexible and reliable way to load and filter JSONB data retrieved from PostgreSQL.
- **Motivation**: Facilitate the conversion of JSONB data into a standard Python dict for further processing and analysis.
- **Inheritance**: Inherits from `PgsqlDataLoader`, reusing its connection handling and query generating functionality.

#### Example

Below is an example of how to instantiate and use the loader:

```python
loader = PgsqlJSONBLoader(
    connection_string="postgresql://user:pass@host/db",
    query_generator=YourQueryGenerator,
    jsonb_column="jsonb_data",
    fields_to_keep=["field1", "field2"]
)

data = loader.load_data(query_parameters)
```

## Method: `PgsqlJSONBLoader._get_item`

### Functionality

The `_get_item` method extracts and filters JSONB data from a given dictionary. If `fields_to_keep` is defined, only the specified keys are returned.

### Parameters

- `data`: A dictionary containing a single row's data. It must include the JSONB data under the column defined by the loader (typically 'jsonb_data').

### Returns

- A dictionary with the parsed JSONB data, filtered if `fields_to_keep` is provided.

### Raises

- `ValueError`: Raised if JSONB data is missing or not a dictionary.

### Usage

- **Purpose**: To extract JSONB data from a database row and optionally filter it based on specified fields.

#### Example

```python
data = {
    "jsonb_data": {"id": 1, "name": "example", "age": 30}
}
loader = PgsqlJSONBLoader(connection_string,
                          query_generator,
                          fields_to_keep=["id", "name"])
item = loader._get_item(data)
print(item)  # Output: {'id': 1, 'name': 'example'}
```