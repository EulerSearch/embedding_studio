# Merged Documentation

## Documentation for `profiled`

### Functionality

Encapsulates the code profiling process. It launches the profiler, captures runtime statistics, and prints them upon completion.

### Parameters

- None

### Usage

- **Purpose** - Measures the performance of a code block using Python's cProfile.

#### Example

```python
with profiled():
    # Your code to benchmark
```

---

## Documentation for `PgvectorCollection`

### General Description
PgvectorCollection is a class that manages vector embeddings in a PostgreSQL database using the pgvector extension. It extends the base Collection class and provides methods for CRUD operations, vector similarity search, and payload filtering.

### Main Purposes
- Manage and store vector embeddings efficiently in PostgreSQL.
- Enable fast similarity search with the pgvector extension.
- Provide mechanisms for CRUD operations and filtering data.

### Motivation
This class was created to leverage PostgreSQL's capabilities for handling high-dimensional vector data. By using the pgvector extension, it supports scalable and robust vector operations, making it suitable for applications like semantic search.

### Inheritance
PgvectorCollection inherits from the Collection class, ensuring that it follows a standardized API for vector database operations while introducing specialized functionality enabling vector searches.

### Usage Example

```python
from sqlalchemy import create_engine
from embedding_studio.vectordb.pgvector.collection import PgvectorCollection
from embedding_studio.vectordb.collection_info_cache import CollectionInfoCache

engine = create_engine("postgresql://user:password@localhost/db")
collection_cache = CollectionInfoCache()

collection = PgvectorCollection(
    pg_database=engine,
    collection_id="my_collection",
    collection_info_cache=collection_cache
)

# Use 'collection' to perform similarity search and other operations.
```

---

## Documentation for `PgvectorCollection.get_info`

### Functionality
This method retrieves metadata for the collection. It uses the collection info cache to return a CollectionInfo object based on the collection ID stored in the instance.

### Parameters
This method does not require any parameters.

### Usage
- **Purpose**: Retrieve collection metadata in a PostgreSQL vector database using the pgvector extension.

#### Example

```python
collection = PgvectorCollection(pg_db, coll_id, cache)
info = collection.get_info()
print(info)
```

---

## Documentation for `PgvectorCollection.get_state_info`

### Functionality
Retrieves state information for a collection stored in a PostgreSQL vector database using the pgvector extension. It returns a `CollectionStateInfo` object containing metadata and state details about the collection.

### Parameters
This method accepts no parameters.

### Usage
- **Purpose:** Retrieves the current state information for a given collection.

#### Example

```python
state_info = collection.get_state_info()
print(state_info)
```

---

## Documentation for `PgvectorCollection.lock_objects`

### Functionality
This context manager locks specified objects by their IDs within a transaction. It attempts to acquire the lock up to a maximum number of tries, waiting a fixed time between attempts. Once the lock is acquired, the method yields a session allowing safe operations on the locked objects and commits or rolls back the transaction accordingly.

### Parameters
- `object_ids`: List of object IDs to lock.
- `max_attempts`: Maximum attempts to acquire the lock. Default is 5.
- `wait_time`: Time in seconds to wait between attempts. Default is 1.0 sec.

### Usage
- **Purpose**: To safely perform operations on a subset of objects by obtaining an exclusive lock in a transactional context.

#### Example

```python
with collection.lock_objects(['id1', 'id2']) as session:
    # Perform operations with locked objects
    result = session.execute(query)
    # Commit is automatic if no exceptions occur
```

---

## Documentation for `PgvectorCollection.insert`

### Functionality
Inserts objects and their associated vector parts into the collection's database. The method converts each provided Object into its corresponding database record and inserts both the object and its parts in a single transaction to ensure consistency.

### Parameters
- `objects`: A list of Object instances to be added. Each Object contains payload data, storage metadata, and vector parts.

### Usage
- **Purpose**: To add multiple vector objects into the collection in a batch operation.

#### Example

```python
objects = [Object(...), Object(...)]
collection.insert(objects)
```

---

## Documentation for `PgvectorCollection.create_index`

### Functionality
This method creates an HNSW vector index on the object parts table's vector column. It then updates the collection's index state in the cache, ensuring that the collection recognizes the index as created. The index is built only if it does not already exist.

### Parameters
- None.

### Usage
**Purpose** - Establish a vector index for similarity searches within the collection.

#### Example

```python
# Assuming pg_collection is an instance of PgvectorCollection
pg_collection.create_index()
```

---

## Documentation for `PgvectorCollection.upsert`

### Functionality
Update or insert objects with their vector parts. This method either updates an existing record or inserts new records for objects and their corresponding vector parts. Depending on the 'shrink_parts' parameter, it either deletes the old parts before inserting new ones or performs a database upsert of the parts.

### Parameters
- `objects`: List of Object instances to upsert. Each instance includes attributes like object_id, payload, storage_meta, user_id, original_id, and a list of parts (with part_id, vector, is_average).
- `shrink_parts`: Boolean flag. If True, deletes existing parts before inserting new ones; if False, it upserts the parts.

### Usage
- **Purpose**: Manage the upsertion of objects and vector parts in the Pgvector collection to ensure database consistency during data updates or insertions.

#### Example

```python
objects = [Object(...), Object(...)]
collection.upsert(objects, shrink_parts=True)
```

---

## Documentation for `PgvectorCollection.delete`

### Functionality
Deletes objects and their parts from the collection. This method first removes parts from the parts table to avoid potential deadlocks, then deletes the corresponding objects from the main table. If any error occurs, the entire transaction is rolled back to ensure data integrity.

### Parameters
- `object_ids`: List[str] - A list of object IDs that should be deleted.

### Usage
- **Purpose**: Remove one or more objects and their associated parts from the collection in a safe, transactional manner.

#### Example

```python
object_ids = ["id1", "id2", "id3"]
collection.delete(object_ids)
```

---

## Documentation for `PgvectorCollection._reset_read_session`

### Functionality
Reconnects to the PostgreSQL database to create and return a new SQLAlchemy session for persistent read-only operations. This method ensures that read operations use a valid and active connection.

### Parameters
This method does not take any parameters.

### Returns
- A new SQLAlchemy session used for persistent read operations.

### Usage
- **Purpose**: Reset and obtain a fresh read session when the current connection is closed or outdated.

#### Example

```python
collection = PgvectorCollection(pg_database, collection_id, cache)
session = collection._reset_read_session()
result = session.execute(query)
```

---

## Documentation for `PgvectorCollection._with_read_session`

### Functionality
This method executes the provided query function using a persistent read session. It first checks if the session's connection is open, and if not, resets the session. If an error occurs during the query, it falls back to a traditional session.

### Parameters
- `query_func`: A function that accepts a session parameter and performs database queries.

### Usage
- **Purpose**: Centralizes read query execution with a fallback mechanism.

#### Example

```python
def example_query(session):
    return session.execute("SELECT * FROM my_table").fetchall()

result = collection._with_read_session(example_query)
```

---

## Documentation for `PgvectorCollection.find_by_ids`

### Functionality
This method retrieves objects from the collection using a list of IDs. It runs a database query via a persistent read session. If the connection fails, it falls back to a normal session.

### Parameters
- `object_ids`: List[str] - A list of object IDs to be found.

### Return Value
Returns a list of objects that match the provided IDs.

### Usage
- **Purpose**: Retrieve specific objects by their IDs from a PgvectorCollection instance.

#### Example

```python
objects = collection.find_by_ids(["id1", "id2", "id3"])
for obj in objects:
    print(obj)
```

---

## Documentation for `PgvectorCollection.query`

### Functionality
Executes a database query using a persistent read session. If the persistent session is closed or encounters an error, it falls back to a traditional session. This design ensures robust query execution without interruption.

### Parameters
- `query_func`: A function that accepts a session object and performs database operations. The session provided is either the persistent read session or a fallback traditional session.

### Usage
- **Purpose**: To execute database queries reliably by first using a persistent read connection, and if that fails, switching to a regular session.

#### Example

```python
# Define a query function that takes a session parameter
def my_query(session):
    result = session.execute("SELECT * FROM my_table")
    return result.fetchall()

# Execute the query through PgvectorCollection
results = collection.query(my_query)
```

---

## Documentation for `PgvectorCollection.find_by_original_ids`

### Functionality
This method retrieves objects from the database using their original IDs. It uses a read-only session to execute a query that returns rows matching the given original object IDs and converts the results into Object instances.

### Parameters
- `object_ids`: A list of strings representing the original object IDs used to identify objects in the collection.

### Usage
- **Purpose**: Fetch object instances by providing their original identifiers.

#### Example

```python
collection = PgvectorCollection(pg_database, "my_collection", collection_info_cache)
objects = collection.find_by_original_ids(["id1", "id2"])
for obj in objects:
    print(obj)
```

---

## Documentation for `PgvectorCollection.find_similarities`

### Functionality
Find objects similar to a provided query vector using similarity search on stored vectors in a PostgreSQL database. It performs a similarity comparison with optional filtering and sorting.

### Parameters
- `query_vector`: Vector (list of floats) used for computing similarity.
- `limit`: Maximum number of similar objects to return.
- `offset`: Number of objects to skip (for pagination).
- `max_distance`: Distance threshold to determine similarity.
- `payload_filter`: Filter to constrain objects based on payload.
- `sort_by`: Options to sort the returned objects.
- `user_id`: Filter objects by a specific user identifier.
- `similarity_first`: Flag to prioritize similarity in sorting.
- `meta_info`: Additional metadata for query customization.

### Usage
- **Purpose**: Retrieves objects similar to a given query vector while applying optional filtering, sorting, and pagination.

#### Example

```python
results = collection.find_similarities(
    query_vector=[0.2, 0.5, 0.3],
    limit=10,
    offset=0,
    max_distance=0.5,
    payload_filter=None,
    sort_by=None,
    user_id='user123',
    similarity_first=True,
    meta_info={'additional': 'data'}
)
print(results)
```

---

## Documentation for `PgvectorCollection.get_total`

### Functionality
Returns the total count of objects stored in the collection. When the flag is set, it counts only original objects, ignoring any derivatives.

### Parameters
- `originals_only`: Boolean flag indicating whether to count only original objects (default is True).

### Usage
- **Counting Objects** - Use this method to get a tally of objects in the collection for status or pagination.

#### Example

```python
total = collection.get_total()
print("Total objects:", total)
```

---

## Documentation for `PgvectorCollection.get_objects_common_data_batch`

### Functionality
This method retrieves a batch of common object data from the collection. It queries the total number of objects and fetches object details based on the provided limit and offset. It also calculates the next offset for pagination.

### Parameters
- `limit`: Maximum number of objects to retrieve.
- `offset`: Number of objects to skip; optional parameter.
- `originals_only`: If True, only original objects are retrieved.

### Usage
**Purpose** - Use this method to obtain a set of object data along with pagination info from a PostgreSQL vector database using the pgvector extension.

#### Example

```python
batch = collection.get_objects_common_data_batch(
    limit=50,
    offset=0,
    originals_only=True
)
print(batch.objects_info)
```

---

## Documentation for `PgvectorCollection.count_by_payload_filter`

### Functionality
Counts the number of objects that match a specific payload filter in the PostgreSQL vector database using the pgvector extension. It applies a filter condition based on the object's payload.

### Parameters
- `payload_filter`: A PayloadFilter object that defines the criteria for matching objects.

### Return
An integer representing the number of objects that satisfy the payload filter criteria.

### Usage
- **Purpose**: Quickly obtain the count of objects matching a given payload filter in your vector database.

#### Example

```python
count = collection.count_by_payload_filter(payload_filter)
```

---

## Documentation for `PgvectorQueryCollection`

### Functionality
PgvectorQueryCollection is a query-specific extension of the PgvectorCollection. It adds functionality to handle user queries and operations associated with vector searches in a PostgreSQL setup with pgvector. The class focuses on retrieving objects using a session identifier and other query parameters.

### Motivation
The class was designed to separate query logic from basic CRUD operations. This separation simplifies the codebase and allows for specialized handling of queries, including vector validations and payload filtering.

### Inheritance
PgvectorQueryCollection inherits from the following classes:
- **PgvectorCollection**: Provides core functionality for managing vector embeddings in a PostgreSQL database.
- **QueryCollection**: Defines common query operations for vector databases.

### Example
Below is a simple example demonstrating how to use the class:

```python
# Create an instance of PgvectorQueryCollection
pg_query_collection = PgvectorQueryCollection(
    pg_database=engine,
    collection_id='example_collection',
    collection_info_cache=cache
)

# Retrieve objects by session ID
objects = pg_query_collection.get_objects_by_session_id('session123')
```

---

## Documentation for `PgvectorQueryCollection.get_objects_by_session_id`

### Functionality
This method retrieves objects and their parts based on a given session ID. It performs vector validation to ensure that only valid objects are returned from the database.

### Parameters
- `session_id`: A unique identifier for the session to query.

### Usage
- **Purpose**: To fetch objects from the database using a session ID, ensuring only the applicable and validated objects are retrieved.

#### Example

```python
# Retrieve objects by session ID
objects = collection.get_objects_by_session_id("session_123")
```