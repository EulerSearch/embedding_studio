## Documentation for `PgvectorDb`

### Functionality
PgvectorDb is a vector database implementation that leverages PostgreSQL with the pgvector extension for efficient vector similarity search and storage. The class integrates a PostgreSQL engine for data management and a MongoDB database for caching collection metadata.

### Motivation and Purpose
This class is designed to manage vector data for modern embedding applications by utilizing the pgvector extension. It simplifies vector storage and search operations while optimizing performance.

### Inheritance
PgvectorDb inherits from VectorDb, which establishes a common interface for all vector database implementations, ensuring consistency and interoperability across modules.

### Usage Example
A simple example to initialize a PgvectorDb instance:

```python
from sqlalchemy import create_engine
import pymongo
from embedding_studio.vectordb.pgvector.vectordb import PgvectorDb

pg_engine = create_engine("postgresql://user:pass@localhost/db")
mongo_db = pymongo.MongoClient()["embeddings"]
vectordb = PgvectorDb(pg_database=pg_engine, embeddings_mongo_database=mongo_db)
```

---

## Documentation for `PgvectorDb._init_pgvector`

### Functionality
Initializes the pgvector extension in PostgreSQL. This method ensures the 'vector' extension is created, which enables vector similarity search in PostgreSQL.

### Parameters
This method does not accept external parameters. It uses the SQLAlchemy engine connection to run the SQL command.

### Usage
- **Purpose**: Activate the pgvector extension if not active. Essential to support vector operations.

#### Example
For an instance 'pgvector_db' of PgvectorDb:

```python
pgvector_db._init_pgvector()
```

After running, the PostgreSQL server will have the vector extension enabled.

---

## Documentation for `PgvectorDb.update_info`

### Functionality
Updates internal collection information by invalidating the cache. This forces a refresh of collection metadata from the database.

### Parameters
This method does not take any parameters.

### Usage
Call this method when collection metadata needs updating or after performing schema changes.

#### Example
```python
db = PgvectorDb(...parameters...)
db.update_info()
```

---

## Documentation for `PgvectorDb.list_collections`

### Functionality
The list_collections method retrieves metadata for all available collections stored in the PostgreSQL vector database. It accesses the collection info cache which holds the current state of each collection.

### Parameters
This method does not require any parameters.

### Usage
Call this method to obtain a list of CollectionStateInfo objects that represent the collections stored in the database.

#### Example
```python
# Create an instance of PgvectorDb with proper database connections
pg_db = create_pg_engine(...)
mongo_db = create_mongo_database(...)
db = PgvectorDb(pg_db, mongo_db)

# List all collections
collections = db.list_collections()
for coll in collections:
    print(coll)
```

---

## Documentation for `PgvectorDb.list_query_collections`

### Functionality
Lists all available query collections stored in the database. This method retrieves collection metadata from an internal cache and returns a list of CollectionStateInfo objects.

### Parameters
None.

### Usage
- **Purpose**: To fetch query collection information for operations requiring query collections.

#### Example
```python
# Example usage of list_query_collections
pg_db = PgvectorDb(pg_database, embeddings_mongo_database)
query_collections = pg_db.list_query_collections()
for qc in query_collections:
    print(qc)
```

---

## Documentation for `PgvectorDb.get_collection`

### Functionality
Retrieves a pgvector collection associated with an embedding model ID. Returns a PgvectorCollection object that holds the database connection and metadata cache.

### Parameters
- `embedding_model_id`: The identifier for the embedding model whose collection is requested.

### Usage
- **Purpose** - Fetch the collection for an embedding model for further vector operations.

#### Example
```python
collection = db.get_collection("model_xyz")
# Perform operations with collection
```

---

## Documentation for `PgvectorDb.get_query_collection`

### Functionality
This method retrieves a pgvector query collection using the embedding model ID. It returns a PgvectorQueryCollection object that provides operations for vector similarity queries.

### Parameters
- `embedding_model_id`: A string representing the ID of the embedding model associated with the query collection.

### Usage
- **Purpose**: Retrieve a query collection for performing advanced vector searches using the pgvector extension.
- **Returns**: A PgvectorQueryCollection object with query capabilities.

#### Example
```python
query_coll = pgvector_db.get_query_collection("model_id")
```

---

## Documentation for `PgvectorDb.get_blue_collection`

### Functionality
Returns the current active (blue) collection from the cache. If no blue collection exists, returns None.

### Parameters
This method does not accept any parameters.

### Usage
- **Purpose**: Retrieve the active pgvector collection. It is useful when the system needs to perform operations on the primary collection.

#### Example
```python
collection = db.get_blue_collection()
if collection is None:
    print("No active collection set")
else:
    # work with the blue collection
    pass
```

---

## Documentation for `PgvectorDb.get_blue_query_collection`

### Functionality
Returns the active (blue) query collection used for vector similarity searches. If no blue query collection exists, the method returns None.

### Parameters
This method does not take any parameters.

### Usage
Call this method on your PgvectorDb instance to retrieve the active query collection.

#### Example
```python
query_collection = pgvector_db.get_blue_query_collection()
if query_collection is not None:
    # proceed with vector query operations
    pass
```

---

## Documentation for `PgvectorDb.set_blue_collection`

### Functionality
Sets the specified collection as the active (blue) collection in the metadata store. If a query collection exists for the given model ID, it is also activated as the blue query collection.

### Parameters
- `embedding_model_id`: ID of the embedding model associated with the collection. This is used both as the collection ID and to determine the corresponding query collection.

### Usage
- **Purpose**: Designate a collection as the primary (blue) collection. This enables the system to know which collection to use by default.

#### Example
```python
pgvector_db.set_blue_collection("embedding_model_123")
```

---

## Documentation for `PgvectorDb.save_collection_info`

### Functionality
Saves or updates collection metadata in the store via the internal cache.

### Parameters
- `collection_info`: A CollectionInfo object containing the collection's metadata.

### Usage
- **Purpose** - Persist and update metadata for a collection.

#### Example
Assuming collection_info is a valid CollectionInfo object:

```python
db.save_collection_info(collection_info)
```

---

## Documentation for `PgvectorDb.save_query_collection_info`

### Functionality
Saves or updates query collection information in the metadata store. This method updates the internal cache with the latest query collection details.

### Parameters
- `collection_info`: A CollectionInfo object containing the necessary details for the query collection.

### Usage
- **Purpose** - To ensure that the metadata for query collections remains current and consistent.

#### Example
Assuming `info` is an instance of CollectionInfo, update the query collection metadata as follows:

```python
pgvector_db.save_query_collection_info(info)
```

---

## Documentation for `PgvectorDb._create_collection`

### Functionality
This method creates a new pgvector collection by building the required database tables, indexes, and SQL functions for vector search. It also registers the collection in the metadata store.

### Parameters
- `embedding_model`: An EmbeddingModelInfo object representing the model to use for building the collection.

### Return
- Returns a new PgvectorCollection object that encapsulates the collection.

### Usage
- **Purpose**: Internally used to initialize a new collection in the PostgreSQL vector database with the pgvector extension.

#### Example
Assume you have an EmbeddingModelInfo instance named model_info:

```python
collection = pgvector_db._create_collection(model_info)
```

---

## Documentation for `PgvectorDb._create_query_collection`

### Functionality
This method creates a new query collection for a given embedding model. It sets up the necessary PostgreSQL tables and indexes, and registers the collection in the metadata store.

### Parameters
- `embedding_model`: An EmbeddingModelInfo object defining the model for which the query collection is created.

### Usage
- **Purpose**: Initializes database structures for a query collection tied to the provided embedding model.

#### Example
Assuming you have an instance of PgvectorDb named `db` and an embedding model info `model_info`, you can create a query collection as:

```python
query_coll = db._create_query_collection(model_info)
```

---

## Documentation for `PgvectorDb.collection_exists`

### Functionality
Checks whether a collection exists for a given embedding model ID in the PostgreSQL vector database. It consults the internal cache and returns True if the collection metadata is found, otherwise False.

### Parameters
- `embedding_model_id`: A string representing the embedding model's identifier. The method determines if a corresponding collection exists.

### Usage
- **Purpose**: To verify the existence of a collection before performing operations that depend on its availability.

#### Example
Assuming you have an instance of PgvectorDb:

```python
exists = db.collection_exists("model_identifier")
if exists:
    print("Collection exists!")
else:
    print("No collection found.")
```

---

## Documentation for `PgvectorDb.query_collection_exists`

### Functionality
Checks if a query collection exists for the specified embedding model ID. It leverages a collection info cache to determine if the query collection has been registered in the system.

### Parameters
- `embedding_model_id`: The ID of the embedding model for which the query collection existence is checked.

### Usage
- **Purpose** - Verify whether a query collection exists.

#### Example
Assuming you have initialized a PgvectorDb instance as db. To check for a query collection, use:

```python
exists = db.query_collection_exists("model_123")
```

This returns True if the query collection exists, otherwise False.

---

## Documentation for `PgvectorDb.delete_collection`

### Functionality
Deletes a collection and its associated database objects from PostgreSQL and the metadata cache. It first checks if the collection exists and is not marked as active ("blue"). If the collection is missing, it raises a CollectionNotFoundError. If the collection is active (blue), it raises a DeleteBlueCollectionError.

### Parameters
- `embedding_model_id` (str): ID of the embedding model associated with the collection to delete.

### Raises
- `CollectionNotFoundError`: If the collection does not exist.
- `DeleteBlueCollectionError`: If attempting to delete an active "blue" collection.

### Usage
**Purpose**: Remove a collection and its related database objects.

#### Example
```python
pg_db = PgvectorDb(pg_database, mongo_db)
pg_db.delete_collection("model123")
```

---

## Documentation for `PgvectorDb.delete_query_collection`

### Functionality
Deletes a query collection and its associated database objects in a PostgreSQL database. It drops the tables and removes the query collection from the metadata cache.

### Parameters
- `embedding_model_id`: A string representing the ID of the embedding model linked to the query collection.

### Usage
- **Purpose** - Remove a query collection when it is no longer needed. This method drops the underlying tables and clears the collection from the cache. It raises a CollectionNotFoundError if the query collection does not exist, and a DeleteBlueCollectionError if it is actively used.

#### Example
```python
db = PgvectorDb(...)
db.delete_query_collection("model123")
```