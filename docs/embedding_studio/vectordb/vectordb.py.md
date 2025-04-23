## Documentation for VectorDb Class

### Functionality
VectorDb is an abstract base class that defines an interface for vector database systems. It manages collections of vector embeddings, providing methods for creation, retrieval, and optimization. This ensures consistent collection management across implementations.

### Motivation
The class exists to standardize interactions with various vector databases. By enforcing a common interface, it allows different backends to be integrated seamlessly while maintaining a unified approach to vector collection management.

### Inheritance
VectorDb is an abstract class (derived from ABC) and must be extended. Concrete implementations are required to override methods like update_info, ensuring that specific details are handled appropriately for the underlying storage or retrieval mechanism.

### Usage
To use VectorDb, create a subclass that inherits from it and overrides methods such as update_info. Additionally, you can leverage built-in optimization support by using add_optimization and add_query_optimization.

#### Example
```python
class MyVectorDb(VectorDb):
    def update_info(self):
        # custom logic to refresh collection information
        pass
```

---

## Documentation for VectorDb.get_query_collection_id

### Functionality
Generates a query collection ID by appending "_q" to the base name. This helps distinguish query collections from regular ones.

### Parameters
- collection_name: A string specifying the base collection name.

### Usage
- **Purpose:** Standardize query collection IDs in the database.

#### Example
```python
id = get_query_collection_id("my_collection")
# Returns "my_collection_q"
```

---

## Documentation for VectorDb.update_info

### Functionality
Updates the collection information in the vector database. This method refreshes cached data to ensure that current collection details are available.

### Parameters
None.

### Usage
- Use this method to refresh the collection cache so that the vector database holds up-to-date information about collections.

#### Example
```python
class MyVectorDb(VectorDb):
    def update_info(self):
        # Refresh collection cache from storage
        self._collection_cache.invalidate_cache()

db = MyVectorDb()
db.update_info()
```

---

## Documentation for VectorDb.add_optimization

### Functionality
Adds an optimization strategy for regular collections. This method accepts an optimization object and appends it to an internal list. The optimization is later applied when processing collections.

### Parameters
- `optimization`: An optimization strategy object. It is expected to be callable and have a `name` attribute, which is used to track applied optimizations.

### Usage
- **Purpose** - To register new optimization strategies that enhance regular collections in the vector database.

#### Example
```python
vector_db.add_optimization(custom_optimization)
```

---

## Documentation for VectorDb.add_query_optimization

### Functionality
This method registers an optimization strategy to be applied on query collections. It stores the provided optimization strategy in an internal list used during query operations to enhance search performance.

### Parameters
- `optimization`: An instance of the Optimization class that defines the strategy to be applied.

### Returns
None

### Usage
This method extends the behavior of query collections by allowing custom optimization strategies, thereby improving dynamic query capabilities.

#### Example
Assuming a vector database instance `vdb` and an optimization instance `opt`, you can add the optimization as follows:
```python
vdb.add_query_optimization(opt)
```

---

## Documentation for VectorDb.apply_optimizations

### Functionality
This method applies all registered optimizations to each collection in the database. It iterates over all available collections and applies any optimization strategy from the regular optimizations list that has not yet been applied. Once an optimization is applied, the collection's applied optimizations are updated to record the change.

### Parameters
This method does not accept any parameters.

### Usage
- **Purpose**: Apply registered optimizations and update collections.

#### Example
```python
# vector_db is an instance of a concrete VectorDb
vector_db.apply_optimizations()
```

---

## Documentation for VectorDb.apply_query_optimizations

### Functionality
This method applies all registered query optimization strategies to each query collection in the database. It iterates over every query collection, checks if an optimization has already been applied, executes the optimization if needed, and saves the updated state information.

### Parameters
There are no external parameters. The method operates on the class instance attributes: `self._query_optimizations` and the query collection list returned by `list_query_collections()`.

### Usage
- **Purpose:** Apply query-specific optimizations to update the state of all query collections.

#### Example
After adding query optimizations with `add_query_optimization`, simply call:
```python
vector_db.apply_query_optimizations()
```
to update all query collections with the new optimization steps.

---

## Documentation for VectorDb.save_collection_info

### Functionality
This method saves or updates collection information in the database by persisting changes to the underlying storage or cache.

### Parameters
- `collection_info`: A CollectionInfo object holding all details required to describe and manage a collection in the database.

### Usage
- **Purpose**: Update stored metadata after changing a collection.

#### Example
```python
def save_collection_info(self, collection_info: CollectionInfo):
    # Update collection info in storage
    self._collection_cache.update_collection(collection_info)
```

---

## Documentation for VectorDb.save_query_collection_info

### Functionality
Saves or updates query collection information in the vector database. This method is designed to update the query collection's metadata, typically stored in a cache or underlying storage.

### Parameters
- `collection_info`: Query collection information to save or update. It should be an instance of CollectionInfo containing all necessary metadata for the query collection.

### Usage
- **Purpose** - To record and maintain current query collection data within the vector database system.

#### Example
```python
def save_query_collection_info(self, collection_info: CollectionInfo):
    # Update query collection info in storage
    self._collection_cache.update_query_collection(collection_info)
```

---

## Documentation for VectorDb.list_collections

### Functionality
Lists all regular collections in the database and returns a list of CollectionStateInfo objects.

### Parameters
None.

### Usage
- **Purpose**: Retrieve collection state information for regular collections.

#### Example
```python
collections = vector_db_instance.list_collections()
for collection in collections:
    print(collection)
```

---

## Documentation for VectorDb.list_query_collections

### Functionality
This method returns a list of query collection state information objects. It retrieves all query collections managed by the database, typically from a cache or persistent storage.

### Parameters
This method does not take any parameters.

### Usage
- **Purpose** - To obtain query collection state information for all query collections in the database.

#### Example
An example implementation might look like:
```python
def list_query_collections(self) -> List[CollectionStateInfo]:
    return self._collection_cache.list_query_collections()
```

---

## Documentation for VectorDb.get_blue_collection

### Functionality
Retrieves the designated 'blue' collection from the database. This method returns the primary active collection if set, or None otherwise.

### Parameters
This method does not require any parameters aside from the implicit `self`.

### Usage
- **Purpose** - To access the primary active collection designated as 'blue'.

#### Example
```python
blue_collection = vectordb.get_blue_collection()
if blue_collection:
    # Process the blue collection
    print("Blue collection retrieved.")
else:
    print("No blue collection is set.")
```

---

## Documentation for VectorDb.get_blue_query_collection

### Functionality
Retrieves the primary active query collection associated with the 'blue' designation. If set in the collection cache, it returns the corresponding QueryCollection, or None otherwise.

### Parameters
None besides the instance reference.

### Usage
- **Purpose** - To obtain the active blue query collection from the collection cache.

#### Example
```python
query_collection = vector_db.get_blue_query_collection()
if query_collection:
    # Use the query_collection for operations
    pass
else:
    # Handle the absence of a blue query collection
    pass
```

---

## Documentation for VectorDb.get_collection

### Functionality
Retrieves a collection using the provided embedding model ID. The method queries the collection cache and returns the corresponding collection. If no collection is found, it raises an error.

### Parameters
- `embedding_model_id`: Identifier of the embedding model associated with the collection.

### Usage
- **Purpose**: Obtain a collection instance linked to a specific embedding model.

#### Example
```python
collection = vectordb.get_collection("model_123")
```

---

## Documentation for VectorDb.get_query_collection

### Functionality
This method retrieves a query collection based on an embedding model ID. It constructs a query collection ID by appending a suffix (typically '_q') to the provided embedding model ID and fetches the associated collection information from the cache. If the collection does not exist, a CollectionNotFoundError is raised.

### Parameters
- embedding_model_id (str): The ID of the embedding model used to obtain the query collection.

### Usage
- Purpose: Retrieve a query collection for query-oriented tasks.

#### Example
Assuming an instance 'vectordb' of a concrete VectorDb class:
```python
query_collection = vectordb.get_query_collection("model123")
```

---

## Documentation for VectorDb._create_collection

### Functionality
Internal method to create a new collection for a specific embedding model. This method prepares a CollectionInfo and adds it to the collection cache, then instantiates a Collection object using the collection factory.

### Parameters
- `embedding_model` (EmbeddingModelInfo): Information about the embedding model for which the collection is created.

### Return
- `Collection`: The newly created collection object.

### Usage
This internal method is typically used within create_collection.

#### Example
```python
def _create_collection(self, embedding_model: EmbeddingModelInfo) -> Collection:
    collection_info = CollectionInfo(
        collection_id=embedding_model.id,
        embedding_model=embedding_model,
        applied_optimizations=[]
    )
    self._collection_cache.add_collection(collection_info)
    return self._collection_factory.create_collection(collection_info)
```

---

## Documentation for VectorDb.create_collection

### Functionality
This method creates a new collection for a specified embedding model. It calls an internal method to generate the base collection and then applies all registered optimizations before returning the collection.

### Parameters
- `embedding_model`: An instance of EmbeddingModelInfo containing details about the embedding model to create the collection for.

### Usage
- **Purpose**: Creates a new collection with applied optimizations.

#### Example
```python
from embedding_studio.models.embeddings.models import EmbeddingModelInfo
# vector_db is an instance of a concrete class inheriting from VectorDb
embedding_model = EmbeddingModelInfo(id='model_1', ...)
collection = vector_db.create_collection(embedding_model)
```

---

## Documentation for VectorDb._create_query_collection

### Functionality
Creates a new query collection for a given embedding model. This internal method builds the collection info and returns a new QueryCollection instance.

### Parameters
- `embedding_model`: Object containing embedding model details.

### Usage
- **Purpose**: To create a query collection with a given model info; used internally to manage query collections.

#### Example
```python
# Example implementation of _create_query_collection
def _create_query_collection(self, embedding_model: EmbeddingModelInfo) -> QueryCollection:
    query_collection_id = self.get_query_collection_id(embedding_model.id)
    collection_info = CollectionInfo(
        collection_id=query_collection_id,
        embedding_model=embedding_model,
        applied_optimizations=[]
    )
    self._collection_cache.add_query_collection(collection_info)
    return self._collection_factory.create_query_collection(collection_info)
```

---

## Documentation for VectorDb.create_query_collection

### Functionality
Creates a new query collection using a specific embedding model and applies registered query optimizations. This ensures the query collection is enhanced and ready for use.

### Parameters
- `embedding_model`: An EmbeddingModelInfo object that provides details for creating the query collection.

### Usage
- **Purpose**: Generate a query collection with applied optimizations.

#### Example
```python
# Instantiate your concrete VectorDb implementation
vector_db = MyVectorDb()
query_collection = vector_db.create_query_collection(embedding_model)
```

---

## Documentation for VectorDb.get_or_create_collection

### Functionality
Retrieves an existing collection based on the provided embedding model information. If the collection does not exist, a new one is created.

### Parameters
- `embedding_model`: An instance of EmbeddingModelInfo with details about the embedding model.

### Usage
- **Purpose**: Obtain a collection for storing or retrieving embeddings.

#### Example
Assuming model_info is an instance of EmbeddingModelInfo:
```python
collection = vectordb.get_or_create_collection(model_info)
```

---

## Documentation for VectorDb.get_or_create_query_collection

### Functionality
This method retrieves an existing query collection or creates a new one for a given embedding model. It checks whether the query collection exists and, if not, creates one using the provided embedding model details.

### Parameters
- `embedding_model`: An object containing details about the embedding model. It is used to identify or create the appropriate query collection.

### Usage
- **Purpose**: To ensure that a query collection is available for the specified embedding model, either by retrieving an existing collection or by creating a new one.

#### Example
```python
from embedding_studio.vectordb import VectorDb
from embedding_studio.embedding_model import EmbeddingModelInfo

# Create an instance of VectorDb and an embedding model
vector_db = VectorDb()
embedding_model = EmbeddingModelInfo(id="model123")

# Retrieve or create a query collection
query_collection = vector_db.get_or_create_query_collection(embedding_model)
```

---

## Documentation for VectorDb.collection_exists

### Functionality
Checks if a collection exists for the given embedding model ID. Returns True if found and valid, False otherwise.

### Parameters
- `embedding_model_id`: The ID of the embedding model to check.

### Usage
- **Purpose** - Verify the presence of a vector collection before creation.

#### Example
```python
if vector_db.collection_exists(embedding_model_info.id):
    print("Collection exists")
else:
    print("Creating collection")
```

---

## Documentation for VectorDb.query_collection_exists

### Functionality
Checks if a query collection exists for a given embedding model ID. It returns True when the query collection is present; otherwise, it returns False. This helps determine if query operations should be performed on the collection.

### Parameters
- `embedding_model_id`: The ID for which the query collection is verified.

### Usage
- **Purpose** - To validate the availability of a query collection in the system.

#### Example
```python
exists = vector_db.query_collection_exists("model_123")
if exists:
    print("Query collection exists.")
else:
    print("No query collection found.")
```

---

## Documentation for VectorDb.delete_collection

### Functionality
Deletes a collection from the vector database using the specified embedding model ID. It removes the collection from storage and purges its cache. Attempting to delete a blue collection will raise an error.

### Parameters
- embedding_model_id: A string representing the embedding model's ID for which the collection is to be deleted.

### Usage
- Purpose: Remove a collection from the system while ensuring database consistency.

#### Example
```python
try:
    vector_db.delete_collection("model_123")
except DeleteBlueCollectionError:
    print("Cannot delete blue collection.")
```

---

## Documentation for VectorDb.delete_query_collection

### Functionality
Removes a query collection for a given embedding model ID. The method first computes the query collection identifier by appending a '_q' suffix to the provided embedding model ID using the get_query_collection_id method. It then proceeds to delete the corresponding query collection from both the storage system and internal cache.

### Parameters
- `embedding_model_id`: A string identifier for the embedding model. This is utilized to generate the query collection ID and target the correct collection for deletion.

### Usage
- **Purpose**: To remove a query-specific collection when it is no longer needed, thereby freeing storage resources and keeping the system up-to-date.

#### Example
```python
# Assuming vectordb is an instance of a concrete
# implementation of VectorDb:
vectordb.delete_query_collection("model123")
```

---

## Documentation for VectorDb.set_blue_collection

### Functionality
Sets the 'blue' collection for a given embedding model ID. This marks the collection as the primary active collection in the system.

### Parameters
- `embedding_model_id`: The ID of the embedding model whose collection is to be set as blue.

### Usage
- **Purpose**: To designate a collection as the primary active one.

#### Example
```python
def set_blue_collection(self, embedding_model_id: str) -> None:
    query_collection_id = self.get_query_collection_id(embedding_model_id)
    self._collection_cache.set_blue_collection(embedding_model_id, query_collection_id)
```