# Documentation for `PgsqlTextLoader`

## Functionality
PgsqlTextLoader is a specialized data loader that inherits from the PgsqlDataLoader class. It is designed to retrieve text data from a PostgreSQL database. The loader extracts text from a specified column, performs necessary decoding, and ensures proper error handling if the text data is missing or in bytes format.

## Parameters
- `connection_string`: PostgreSQL connection string to establish the connection with the database.
- `query_generator`: PostgreSQL query generator class used to build queries dynamically.
- `text_column`: The database column from which text data is extracted (default is "text_data").
- `retry_config`: Optional retry strategy configuration in case of connection or query execution failures.
- `features`: Optional dataset features which describe the expected structure of the loaded data.
- `encoding`: Encoding used for decoding text when necessary (default is "utf-8").

## Usage
**Purpose**: The primary purpose of PgsqlTextLoader is to facilitate loading of text data from a PostgreSQL database. It abstracts connection handling, query formulation, and data conversion, making it easier to incorporate text data into further processing pipelines.

### Method: `_get_item`
This method extracts text data from a single row dictionary. It attempts to decode the text using the specified encoding if the data is in bytes. If the text data is missing, it logs an error and raises a ValueError.

#### Parameters
- `data`: A dictionary containing the row data, including the text data under the key defined by `text_column`.

#### Example
Assuming a row with byte data:
```python
row = {"text_data": b"Sample text"}
loader = PgsqlTextLoader(connection_string, QueryGenerator, text_column="text_data")
text = loader._get_item(row)
# text == "Sample text"
```

### Example of Using PgsqlTextLoader
```python
from embedding_studio.data_storage.loaders.sql.pgsql.pgsql_text_loader import PgsqlTextLoader

# Instantiate with connection details and a query generator
loader = PgsqlTextLoader(
    connection_string="postgresql://user:pass@host:port/db",
    query_generator=YourQueryGeneratorClass,
    text_column="description",
    encoding="utf-8"
)

# Load and process data
text_data = loader.load_data()
```