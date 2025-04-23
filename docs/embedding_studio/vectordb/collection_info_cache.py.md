## Documentation for CollectionInfoCache Class

### Functionality
The CollectionInfoCache class provides an in-memory cache for collection metadata and state information stored in a MongoDB database. It manages retrieval, updates, and storage of collection data to minimize frequent database queries.

### Parameters
- `mongo_database`: A MongoDB database instance used for storing collection information.
- `db_id`: A string identifier for selecting the active database.

### Usage
- **Purpose**: To improve performance by caching collection states and metadata, thereby reducing direct database queries.
- **Motivation**: To avoid repeatedly costly database queries and ensure faster access to current collection states.
- **Inheritance**: This class directly inherits from Python's object class (i.e., no explicit inheritance).

#### Example
```python
# Example usage of CollectionInfoCache
cache = CollectionInfoCache(mongo_database, 'my_db_id')
cache.invalidate_cache()
```

---

## Documentation for `CollectionInfoCache.invalidate_cache` method

### Functionality
Refresh the in-memory cache by clearing the current collection lists and fetching the latest collection information from MongoDB. It also identifies and marks the current blue collections, sorting collections into regular and query types.

### Parameters
None.

### Usage
- **Purpose** - Ensures the in-memory state is synchronized with the database after any changes to collections.

#### Example
Assuming you have an instance named `cache` of `CollectionInfoCache`, you can refresh the cache by calling:
```python
cache.invalidate_cache()
```

---

## Documentation for `CollectionInfoCache.list_collections`

### Functionality
Returns a list of all regular collections stored internally in the cache. This method retrieves the internal list of collection state information that excludes the query collections.

### Parameters
None.

### Usage
- **Purpose**: Retrieve collection state information for regular collections.

#### Example
```python
collections = cache.list_collections()
for coll in collections:
    print(coll)
```

---

## Documentation for `CollectionInfoCache.list_query_collections`

### Functionality
Returns a list of collections used for queries. This method retrieves the query collections stored in memory.

### Parameters
None.

### Usage
Call this method to obtain all query collections from the cache.

#### Example
```python
query_collections = cache.list_query_collections()
for qc in query_collections:
    print(qc)
```

---

## Documentation for `CollectionInfoCache.get_collection`

### Functionality
This method searches for a collection by its ID among the regular collections and query collections. It returns the collection state information if found, or None otherwise.

### Parameters
- `collection_id`: The unique identifier of the collection to search.

### Usage
- **Purpose**: Retrieve the state information for a collection using its ID.

#### Example
```python
collection = cache.get_collection("my_collection_id")
if collection:
    print("Collection found")
else:
    print("Collection not found")
```

---

## Documentation for `CollectionInfoCache.get_blue_collection`

### Functionality
Retrieves the current blue (active) collection. If no blue collection is set, returns None.

### Parameters
None.

### Usage
Used to fetch the primary active collection after it is set. For instance, after calling set_blue_collection, you can obtain the blue collection as follows:

#### Example
```python
cache = CollectionInfoCache(mongo_db, "db1")
blue_coll = cache.get_blue_collection()
```

---

## Documentation for `CollectionInfoCache.get_blue_query_collection`

### Functionality
Retrieves the currently active blue query collection from the in-memory cache. This method returns the blue query collection state information if it has been set; otherwise, it returns None.

### Parameters
There are no parameters for this method.

### Usage
- **Purpose**: To obtain the active blue query collection for operations requiring query processing.

#### Example
Assuming you have an instance `cache` of CollectionInfoCache:
```python
blue_query = cache.get_blue_query_collection()
if blue_query:
    # process the blue query collection
    pass
```

---

## Documentation for `CollectionInfoCache.set_blue_collection`

### Functionality
Sets the blue (primary active) collection and its associated query collection. It first refreshes the in-memory cache to ensure it has the most recent collection data, then checks whether both the given collection and query collection exist. If either is missing, it raises a CollectionNotFoundError. If both exist, it updates the blue collection identifier record in the database and refreshes the cache again.

### Parameters
- `collection_id`: ID of the collection to set as blue.
- `query_collection_id`: ID of the query collection to set as blue.

### Usage
This method is used to mark a collection and its corresponding query collection as the primary active ones for the system. It ensures the collections exist before updating the database.

#### Example
Assuming you have collections with IDs "col1" and "query1", you can set them as blue by calling:
```python
cache.set_blue_collection("col1", "query1")
```

---

## Documentation for `set_index_state`

### Functionality
Update the index creation state for a specific collection. This method updates the flag indicating if the index on the collection has been created. It modifies the collection record via the data access object and refreshes the internal cache.

### Parameters
- `collection_id`: A string representing the unique identifier of the collection. It must match an existing collection in the database.
- `created`: A boolean flag indicating whether the index has been created. True if the index is created, false otherwise.

### Usage
- **Purpose** - To track the index creation status within the collection info.

#### Example
```python
collection_info_cache.set_index_state("collection_id", True)
```

---

## Documentation for `CollectionInfoCache.add_collection`

### Functionality
Adds a new regular collection to the database by converting the provided collection info into a database model. It sets required fields such as creation time, database id, index state, and query flag. Then, it inserts the record into MongoDB and refreshes the cache. If a duplicate is detected, a warning is logged instead of erroring.

### Parameters
- `collection_info`: A CollectionInfo instance with details for the new collection. This includes properties required by the underlying model.

### Return Value
- CollectionStateInfo: The state information of the added collection, reflecting the current state in the cache after insertion.

### Usage
Use this method to add collection metadata seamlessly while ensuring cache consistency and duplicate handling.

#### Example
```python
collection_state = cache.add_collection(collection_info)
```

---

## Documentation for `CollectionInfoCache.update_collection`

### Functionality
Update an existing regular collection in the database. This method prepares an update payload from the provided collection info and adds an updated_at timestamp. It uses MongoDB update operations to modify the collection document. If the collection is not found, a warning is logged. The cache is refreshed after the update, and the updated collection state is returned.

### Parameters
- `collection_info`: Updated information for the collection. This should contain the new metadata and fields to be updated.

### Usage
- **Purpose**: Update and persist changes for an existing regular collection in the database.

#### Example
Assuming `collection_info` is an instance with updated fields, the method can be used as follows:
```python
updated_collection = cache.update_collection(collection_info)
```

---

## Documentation for `CollectionInfoCache.add_query_collection`

### Functionality
This method adds a new query collection to the database by creating a database model from the given collection info. It sets required fields including created_at, db_id, index_created=False, and contains_queries=True. After insertion into MongoDB, the cache is refreshed and the new collection state is returned.

### Parameters
- `collection_info`: A CollectionInfo object with details for the query collection.

### Usage
- **Purpose**: To add a query collection with proper settings in the database.

#### Example
Assuming you have a valid collection_info object:
```python
state = cache.add_query_collection(collection_info)
print(state)
```

---

## Documentation for `CollectionInfoCache.update_query_collection`

### Functionality
Updates an existing query collection in the database. This method prepares an update payload from the provided collection info, adds an 'updated_at' timestamp, and enforces the query flag by setting 'contains_queries' to True. It then updates the document in MongoDB and refreshes the in-memory cache, returning the updated collection state.

### Parameters
- `collection_info`: A CollectionInfo object containing the updated information for the query collection.

### Usage
- **Purpose**: To modify the stored details of a query collection while ensuring that it is maintained as a query collection.

#### Example
```python
updated_info = CollectionInfo(...)
new_state = collection_cache.update_query_collection(updated_info)
```

---

## Documentation for `delete_collection`

### Functionality
Deletes a collection from the database using its unique collection identifier. It calls the Mongo DAO's delete method to remove the collection record from MongoDB and refresh the cache.

### Parameters
- `collection_id`: ID of the collection to delete.

### Usage
- **Purpose**: Remove an obsolete or unwanted collection from the database.

#### Example
Suppose you want to delete a collection with the ID "col_123":
```python
cache.delete_collection("col_123")
``` 

This call will delete the corresponding collection record.