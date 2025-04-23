## Documentation for `Collection`

### Functionality

Collection is an abstract base class that defines an interface for handling vector embeddings and metadata. It supports operations such as insertion, retrieval, and similarity search.

### Motivation

This interface standardizes the handling of vector embeddings and provides a template for extending storage backends.

### Inheritance

Collection inherits from 'ABC', which forces the implementation of abstract methods. Concrete collections must override these methods.

### Usage Example

```python
class MyCollection(Collection):
    def get_info(self) -> CollectionInfo:
        # return collection info
        pass

    def get_state_info(self) -> CollectionStateInfo:
        # return state info
        pass

    @contextmanager
    def lock_objects(self, object_ids: List[str]):
        # lock objects
        yield

    def insert(self, objects: List[Object]) -> None:
        # insert objects
        pass
```

---

## Documentation for `Collection.get_info`

### Functionality

The `get_info` method retrieves key metadata for a collection. It returns a `CollectionInfo` object containing details about the collection, such as its configuration, description, and related metadata.

### Parameters

- None (besides the implicit self parameter).

### Usage

- Use `get_info` to access metadata about the collection easily.

#### Example

```python
# Example usage
collection = ConcreteCollection()
info = collection.get_info()
print(info)
```

---

## Documentation for `Collection.get_state_info`

### Functionality

Returns the current state information of the collection. This method typically builds upon the output of `get_info` and then augments it with additional state details like `work_state`.

### Parameters

None.

### Returns

- A `CollectionStateInfo` object containing metadata about the current state of the collection.

### Usage

- Purpose: Retrieve up-to-date state information from a collection.

#### Example

```python
state_info = collection.get_state_info()
print(state_info)
```

---

## Documentation for `Collection.lock_objects`

### Functionality

This method acts as a context manager to lock a list of object IDs ensuring safe and exclusive access during critical operations. It acquires locks to prevent race conditions and concurrent modifications.

### Parameters

- `object_ids`: A list of IDs of objects to lock for the duration of the operation.

### Usage

- Purpose: To secure objects during sensitive operations by preventing concurrent modifications.

#### Example

```python
# Assume 'collection' is an instance of a collection class
object_ids = ["id1", "id2", "id3"]

with collection.lock_objects(object_ids):
    # Execute critical operations with locked objects
    process_objects()
```

---

## Documentation for `Collection.insert`

### Functionality

Inserts objects into the vector collection. This method takes a list of `Object` instances and adds them to the underlying storage. It uses a locking mechanism to ensure thread safety and prevent concurrent modifications during the insertion process.

### Parameters

- `objects`: List of `Object` instances to be inserted into the collection.

### Usage

- **Purpose**: Add new vector objects into the collection while maintaining data consistency with locks.

#### Example

```python
def insert(self, objects: List[Object]) -> None:
    object_ids = [obj.id for obj in objects]
    with self.lock_objects(object_ids):
        for obj in objects:
            self._storage.insert_one(obj)
```

---

## Documentation for `Collection.create_index`

### Functionality

The `create_index` method is used to create an index for the collection. This index optimizes similarity search queries by ensuring efficient retrieval of embedding vectors.

### Parameters

None.

### Usage

- **Purpose** - Initialize and create an index if it does not already exist.

#### Example

```python
def create_index(self) -> None:
    if not self._index_exists():
        self._storage.create_index(self.get_info().collection_id)
        self._collection_cache.set_index_state(
            self.get_info().collection_id, True
        )
```

---

## Documentation for `Collection.upsert`

### Functionality

Update existing objects or insert new ones. The method takes a list of objects to be upserted. It updates objects if they exist and inserts new objects otherwise. If `shrink_parts` is True, it will optimize storage after the upsert operation.

### Parameters

- `objects`: List of objects to upsert.
- `shrink_parts`: Boolean flag to optimize storage post upsert.

### Usage

Purpose: To insert or update objects while ensuring data integrity and efficient storage management.

#### Example

```python
def upsert(self, objects: List[Object], shrink_parts: bool = True) -> None:
    object_ids = [obj.id for obj in objects]
    with self.lock_objects(object_ids):
        existing = self.find_by_ids(object_ids)
        existing_ids = {obj.id for obj in existing}
        for obj in objects:
            if obj.id in existing_ids:
                self._storage.update_one(obj)
            else:
                self._storage.insert_one(obj)
        if shrink_parts:
            self._storage.optimize()
```

---

## Documentation for `Collection.delete`

### Functionality

Deletes objects identified by their IDs from the collection. This method uses a locking mechanism to prevent concurrent modifications, ensuring safe removal of the objects.

### Parameters

- `object_ids`: A list of object IDs that need to be removed.

### Usage

- **Purpose**: Remove objects from the collection based on IDs.

#### Example

```python
collection = YourCollectionImplementation()
collection.delete(["id1", "id2"])
```

---

## Documentation for `Collection.find_by_ids`

### Functionality

This method searches for objects in a collection by their IDs. It iterates over the provided list, retrieves each object from the storage, and returns a list of found objects. Objects not found are skipped.

### Parameters

- `object_ids`: List[str]. List of object IDs to search.

### Usage

- **Purpose**: Retrieve multiple objects by their unique IDs.

#### Example

```python
# Retrieve objects with specific IDs
results = collection.find_by_ids(["id1", "id2", "id3"])
```

---

## Documentation for `Collection.find_by_original_ids`

### Functionality

This method retrieves objects from the collection using their original identifiers. It queries the underlying storage by filtering with the key "original_id" and returns all matching objects.

### Parameters

- `object_ids`: List[str]. A list of original object identifiers to search for.

### Usage

- Purpose: Fetch objects based on the original IDs.

#### Example

```python
def find_by_original_ids(self, object_ids: List[str]) -> List[Object]:
    return self._storage.find(
        filter={"original_id": {"$in": object_ids}}
    )
```

---

## Documentation for `Collection.get_total`

### Functionality

This method retrieves the total number of objects stored in the collection by interacting with the underlying storage system.

### Parameters

This method does not require any parameters.

### Usage

- **Purpose**: Use this method to obtain the count of objects present in the collection for pagination or bookkeeping.

#### Example

```python
total_objects = collection.get_total()
print(f"Total objects: {total_objects}")
```

---

## Documentation for `Collection.get_objects_common_data_batch`

### Functionality

Retrieves a batch of common data for objects in the collection. This method returns a set of objects along with the total count of objects. It supports pagination using the parameters provided.

### Parameters

- `limit`: Maximum number of objects to return.
- `offset`: Number of objects to skip (default is 0 if not provided).

### Usage

- **Purpose**: Retrieve paginated common data of objects for display or processing purposes.

#### Example

```python
batch = collection.get_objects_common_data_batch(limit=10, offset=0)
print(batch.objects)
print(batch.total)
```

---

## Documentation for `find_similarities`

### Functionality

Find similar vectors based on a query vector. This method takes a vector as input and returns search results with objects, their similarity distances, and additional metadata.

### Parameters

- `query_vector`: List[float] representing the vector to compare.
- `limit`: Maximum number of results to return.
- `offset`: Number of results to skip for pagination.
- `max_distance`: Optional maximum threshold for similarity filtering.
- `payload_filter`: Optional filter for object payloads.
- `sort_by`: Optional options for sorting the results.
- `user_id`: Optional identifier for the user performing the search.
- `similarity_first`: Boolean to prioritize similarity in ranking.
- `meta_info`: Optional additional metadata for the search.

### Usage

- **Purpose**: Retrieve vectors similar to a given query and return detailed search results.

### Example

```python
results = collection.find_similarities(
    query_vector=[0.12, 0.34, 0.56],
    limit=10,
    offset=0,
    max_distance=0.3,
    payload_filter=filter_obj,
    sort_by=sort_options,
    user_id="user123",
    similarity_first=False,
    meta_info={"example": True}
)
```

---

## Documentation for `find_similar_objects`

### Functionality

This method searches for objects similar to a given vector. It supports filtering, sorting, and can include vectors in the results.

### Parameters

- `query_vector`: A list of floats representing the input vector.
- `limit`: An integer for the maximum number of results.
- `offset`: An integer specifying how many results to skip.
- `max_distance`: A float indicating the maximum allowed distance.
- `payload_filter`: An optional filter for object payloads.
- `sort_by`: Optional sorting options for the results.
- `user_id`: An optional string for the user's ID.
- `with_vectors`: A boolean to include vectors in results.
- `similarity_first`: A boolean to prioritize similarity in scoring.
- `meta_info`: Additional metadata for the search.

### Usage

- **Purpose** - Find and return objects that are similar to a given query vector. The method returns a tuple with a list of objects (with their distances) and search metadata.

#### Example

```python
results, meta = collection.find_similar_objects(
    query_vector=[0.12, 0.34, 0.56],
    limit=10,
    offset=0,
    max_distance=0.3,
    payload_filter=filter_obj,
    sort_by=sort_options,
    user_id="user123",
    with_vectors=True,
    similarity_first=False,
    meta_info={"example": True}
)
```

---

## Documentation for `Collection.find_by_payload_filter`

### Functionality

This method locates objects by applying a filter to their payloads. It converts the provided payload filter into a storage-specific format, applies sorting if specified, and returns a `SearchResults` object containing the matched objects.

### Parameters

- `payload_filter`: Filter to apply to object payloads.
- `limit`: Maximum number of matching objects to return.
- `offset`: Number of matching objects to skip (optional).
- `sort_by`: Sorting options specifying field and order (optional).

### Usage

- Purpose: To search and retrieve objects that satisfy the payload filter condition.

#### Example

```python
results = collection.find_by_payload_filter(
    payload_filter=my_filter,
    limit=10,
    offset=0,
    sort_by=SortByOptions(field='name', ascending=True)
)
```

---

## Documentation for `count_by_payload_filter`

### Functionality

Count objects that match a given payload filter. Returns the number of objects meeting the filter criteria.

### Parameters

- `payload_filter`: Instance of `PayloadFilter` carrying filter criteria for object payloads.

### Usage

- **Purpose**: Get the count of objects that satisfy the specified payload filter.

#### Example

```python
def count_by_payload_filter(self, payload_filter: PayloadFilter) -> int:
    # Convert filter to a storage format
    filter_dict = payload_filter.to_filter_dict()
    
    # Execute count query
    return self._storage.count(filter=filter_dict)
```

---

## Documentation for `QueryCollection`

### Functionality

Provides query-specific functionality on top of base collection operations for vector databases. It includes methods for retrieving and analyzing query vectors and associated data.

### Inheritance

This class inherits from the `Collection` abstract base class, serving as a foundation for query-oriented vector storage and retrieval.

### Motivation

Designed to support efficient handling of query vectors, it adds capabilities such as retrieving objects by session ID for query analysis and optimization.

### Usage

- **Purpose** - To facilitate efficient retrieval and analysis of query vectors using session-specific operations.

#### Example

An example implementation of the abstract method:

```python
class MyQueryCollection(QueryCollection):
    def get_objects_by_session_id(self, session_id: str) -> Object:
        filter_dict = {"payload.session_id": session_id}
        objects = self._storage.find_many(filter=filter_dict)
        return objects[0] if objects else None
```

---

## Documentation for `QueryCollection.get_objects_by_session_id`

### Functionality

This method retrieves an object associated with a given session ID. It performs a search on the underlying storage, filtering objects by the session ID stored in the payload. The method returns the first object found that matches the given session ID, or None if no object is found.

### Parameters

- `session_id`: The session identifier used to locate the object.

### Usage

- **Purpose**: Retrieve the first matching object for the provided session ID.

#### Example

```python
def get_objects_by_session_id(self, session_id: str) -> Object:
    filter_dict = {"payload.session_id": session_id}
    objects = self._storage.find_many(filter=filter_dict)
    if not objects:
        return None
    return objects[0]
```