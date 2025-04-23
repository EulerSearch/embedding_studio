## Documentation for `PgsqlDataLoader`

### Functionality
PgsqlDataLoader is a loader class that retrieves data from PostgreSQL databases using SQLAlchemy. It can fetch individual items, batches, or complete datasets with robust retry logic.

### Motivation
The class is designed to standardize and simplify data retrieval from PostgreSQL. It implements configurable retry mechanisms to handle transient errors, ensuring reliability in data loading operations.

### Inheritance
PgsqlDataLoader inherits from the DataLoader class. This ensures it adheres to a common interface for loading data across different storage backends, promoting consistency and reusability in the codebase.

### Usage
- **Purpose**: To provide a unified interface for loading data from PostgreSQL, handling retries and connection management internally.

#### Example
```python
from embedding_studio.data_storage.loaders.sql.pgsql.pgsql_loader import PgsqlDataLoader

loader = PgsqlDataLoader(
    connection_string='postgresql://user:pass@localhost/db',
    query_generator=MyCustomQueryGenerator
)
data = loader.fetch_data([1, 2, 3])
```

---

## Documentation for `PgsqlDataLoader.item_meta_cls`

### Functionality
This property method returns the metadata class used by the PgsqlDataLoader. It provides the type for representing metadata associated with items loaded from a PostgreSQL database.

### Parameters
None.

### Usage
- **Purpose**: To obtain the class that defines metadata for loaded items from a PostgreSQL database.

#### Example
```python
loader = PgsqlDataLoader(connection_string, query_generator)
meta_cls = loader.item_meta_cls
print(meta_cls)
```

---

## Documentation for `PgsqlDataLoader._get_default_retry_config`

### Functionality
This method creates a default retry configuration object for PostgreSQL operations. It uses settings from the global configuration to set the maximum attempts and wait time for both general and 'fetch_data' operations.

### Parameters
This method does not accept any parameters.

### Usage
- **Purpose**: Provides a default RetryConfig with preset retry parameters for PostgreSQL operations.

#### Example
```python
loader = PgsqlDataLoader(connection_string, query_generator)
config = loader._get_default_retry_config()
print(config)
```

---

## Documentation for `PgsqlDataLoader._fetch_data`

### Functionality
Fetches data from a PostgreSQL database using a list of row IDs. This method builds a SQL query via a query generator and executes it using a SQLAlchemy engine. The results are returned as a list of dictionaries, each representing a row from the database. If an error occurred during execution, it logs the exception and returns an empty list.

### Parameters
- `row_ids` (List[int]): A list of integer row IDs for which data is to be fetched.

### Return
- List[Dict[str, Any]]: A list of dictionaries containing the data corresponding to the given row IDs. Returns an empty list if no data is found or an error occurs.

### Usage
- **Purpose**: Retrieve multiple records from a PostgreSQL database based on provided row IDs.

#### Example
```python
data = loader._fetch_data([1, 2, 3])
if data:
    for record in data:
        print(record)
else:
    print("No data found or an error occurred")
```

---

## Documentation for `PgsqlDataLoader._get_item`

### Functionality
Process raw data fetched from PostgreSQL into the required format. This method should be overridden by subclasses to enable custom transformation of data. By default, the raw data is returned as-is.

### Parameters
- `data`: The raw data from PostgreSQL to be processed.

### Usage
- **Purpose**: Convert raw PostgreSQL data into a defined format.

#### Example
```python
def _get_item(self, data: Any) -> Dict:
    return {
        "processed_item": self._process_data(data),
        "timestamp": data.get("created_at")
    }
```

---

## Documentation for `PgsqlDataLoader._get_data_from_db`

### Functionality
This method retrieves data from a PostgreSQL database using a list of item metadata objects. It fetches data based on row IDs, creates a mapping from IDs to data, and yields a tuple of the processed data and its metadata. If an item is not found, an error is logged. Exceptions are raised unless `ignore_failures` is True.

### Parameters
- `items_data`: List of PgsqlFileMeta objects with metadata for each row.
- `ignore_failures`: Boolean flag. If True, continue on failure; if False, raise an exception upon error.

### Usage
- **Purpose**: To load and process data from PostgreSQL, returning a generator of (data, metadata) tuples.

#### Example
```python
for data, meta in loader._get_data_from_db(items_data):
    print(data, meta)
```

---

## Documentation for `PgsqlDataLoader._create_item_object`

### Functionality
Creates an item object by merging raw data with its metadata. Uses an internal helper to extract data fields and sets the `item_id` key from the metadata.

### Parameters
- data (Dict): Raw data from the database query.
- item_meta (PgsqlFileMeta): Metadata including unique identifiers.

### Returns
A tuple consisting of:
- A dictionary with `item_id` and item data.
- The original `item_meta` instance.

### Usage
- **Purpose**: Standardize item creation from database results.

#### Example
```python
data = {"id": 1, "value": "sample"}
item_meta = PgsqlFileMeta(object_id=1, id=1, ...)
item_obj, meta = loader._create_item_object(data, item_meta)
```

---

## Documentation for `PgsqlDataLoader.load`

### Functionality
Loads a dataset of data from a PostgreSQL database using SQLAlchemy. It uses a generator function to yield dictionary items and then converts them into a Dataset object with specified features.

### Parameters
- `items_data`: List[PgsqlFileMeta]
  A list of item metadata objects used to fetch data from the database.

### Return Value
- `Dataset`: A Dataset object containing the loaded data.

### Usage
- **Purpose**: Load data from PostgreSQL into a Dataset for further processing.

#### Example
```python
loader = PgsqlDataLoader(
    connection_string, query_generator, features=features
)
dataset = loader.load(items_data)
```

---

## Documentation for `PgsqlDataLoader.data_generator`

### Functionality
Provides a generator that lazily yields data rows as dictionaries. It iterates over the results fetched from the database using the internal _get_data_from_db method.

### Parameters
- None

### Usage
- **Purpose**: Lazily yield dictionary items for dataset creation.

#### Example
```python
for item in data_generator():
    print(item)
```

---

## Documentation for `PgsqlDataLoader.load_items`

### Functionality
This method loads items from a PostgreSQL database. It iterates over results fetched via a database query, converting each item into a `DownloadedItem` instance that encapsulates an item id, its data, and associated metadata.

### Parameters
- `items_data`: A list of `PgsqlFileMeta` objects containing metadata for each item to be loaded.

### Returns
- A list of `DownloadedItem` objects representing the loaded items.

### Usage
- **Purpose**: Retrieve items from a PostgreSQL database for further processing.

#### Example
Assuming `pgsql_loader` is an instance of `PgsqlDataLoader`:
```python
items = pgsql_loader.load_items(meta_list)
for item in items:
    print(item.id, item.data)
```

---

## Documentation for `PgsqlDataLoader._load_batch_with_offset`

### Functionality
Load a batch of rows from PostgreSQL starting from a given offset. It queries the database for a fixed number of rows and converts each row into a DownloadedItem containing its id, data, and metadata. If an error occurs, an empty list is returned.

### Parameters
- `offset`: Integer. The starting index for loading rows.
- `batch_size`: Integer. The number of rows to load.
- `kwargs`: Additional keyword arguments.

### Return Value
- List of DownloadedItem objects.

### Usage
This method is usually invoked by the load_all generator to iteratively fetch data from PostgreSQL.

#### Example
```python
batch = loader._load_batch_with_offset(0, 100)
# Process the batch as needed
```

---

## Documentation for `PgsqlDataLoader.load_all`

### Functionality
This method is a generator that repeatedly loads batches from PostgreSQL. It retrieves data using the _load_batch_with_offset method and yields each batch until no data is returned.

### Parameters
- `batch_size`: An integer indicating the size of each batch.
- **kwargs**: Extra keyword arguments for extended configs.

### Usage
- **Purpose**: Iteratively retrieve large datasets in paged batches.

#### Example
```python
loader = PgsqlDataLoader(conn_str, QueryGenerator, ...)
for batch in loader.load_all(batch_size=100):
    process(batch)
```

---

## Documentation for `PgsqlDataLoader.total_count`

### Functionality
Returns the total number of rows in the table used by the data loader. Executes a count query on the connected PostgreSQL table and returns the result, or None if the query fails due to an error.

### Parameters
None.

### Usage
- **Purpose**: Determine the total number of rows in the table.

#### Example
Assuming 'loader' is an instance of PgsqlDataLoader:
```python
total = loader.total_count()
if total is not None:
    print(f"Total rows: {total}")
```