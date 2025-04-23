# Merged Documentation for Vector Collection Operations

## Documentation for `create_collection`

### Functionality
Creates a new vector collection for a specified embedding model. It validates the existence of the model's iteration, then creates both a primary collection and a query-optimized collection. If the model does not exist, a 404 error is returned.

### Parameters
- **embedding_model_id**: A unique identifier for the embedding model. Used to retrieve iteration details.
- **embedding_model**: The specification used for creating the vector collections.

### Usage
- **Purpose**: Set up collections for storing and querying vector embeddings.

#### Example
Request:
```json
{
  "embedding_model_id": "model123",
  "embedding_model": "example_model"
}
```
Response:
```json
{
  "state_info": { /* collection state details */ }
}
```

## Documentation for `create_index`

### Functionality
Builds search indexes on an existing vector collection. Creates HNSW graph indexes for both the main collection and its query-optimized variant. This step is essential for enabling efficient similarity searches and should be called after collection creation but before performing searches.

### Parameters
- **body**: CreateIndexRequest object containing:
  - **embedding_model_id**: The identifier for the embedding model.

### Usage
- **Purpose**: Enable fast vector search capabilities.
- **When to call**: After creating the collection and prior to searches.

#### Example
```python
create_index(body=CreateIndexRequest(
  embedding_model_id="model_123"
))
```

## Documentation for `create_categories_index`

### Functionality
Builds search indexes on a categories-specific vector collection. It uses the specialized categories vector database to create indexes for both the main and query collections. This process enables efficient similarity searches and is optimized for category-based operations.

### Parameters
- **body**: CreateIndexRequest - Request object containing the embedding model's identifier to select the appropriate categories vector collection.

### Usage
- **Purpose**: To create and optimize indexes for a categories-specific vector collection after its creation, ensuring fast similarity search performance.

#### Example
Assuming an embedding model identifier of "model_x", a POST request to the endpoint `/collections/categories/create-index` with the JSON payload below will trigger the indexing process:
```json
{
  "embedding_model_id": "model_x"
}
```

## Documentation for `_delete_collection`

### Functionality
Deletes a main vector collection and its associated query collection. It logs state information and handles errors by raising HTTP exceptions if the collection does not exist or if deletion fails.

### Parameters
- **vectordb (VectorDb)**: The vector database instance to perform collection operations.
- **collection_id (str)**: The identifier of the collection to delete.

### Usage
- **Purpose**: Internally manages the deletion of both the primary and query-optimized collections in the vector database. It ensures proper cleanup and error handling.

#### Example
A typical use case within an endpoint:
```python
@router.post("/collections/delete")
def delete_collection(body: DeleteCollectionRequest):
    return _delete_collection(context.vectordb, body.embedding_model_id)
```

## Documentation for `delete_collection`

### Functionality
Removes a vector collection and its associated resources by deleting both its main collection and the query-optimized variant. If the collection is not found, a 404 error is returned. Other failures result in a 500 error with detailed information.

### Parameters
- **body (DeleteCollectionRequest)**: Request object that contains the `embedding_model_id` used to identify the collection to be deleted.

### Usage
- **Purpose**: Remove a vector collection along with its query counterpart for complete resource cleanup.

#### Example
Request:
```json
{
  "embedding_model_id": "example_model_id"
}
```
Response:
```
HTTP 200 OK for success, or appropriate error code if deletion fails.
```

## Documentation for `delete_categories_collection`

### Functionality
Removes a categories-specific vector collection. Works with the specialized categories vector database to remove category vector collections and their query variants. Ensures proper cleanup of category vector data and returns detailed error information on failure.

### Parameters
- **body**: DeleteCollectionRequest - Contains the embedding model ID and additional request details.

### Usage
- **Purpose**: Removes a categories-specific collection from a specialized vector database.

#### Example
```python
request = DeleteCollectionRequest(embedding_model_id="model_x")
response = delete_categories_collection(request)
```

## Documentation for `list_collections`

### Functionality
Lists all available vector collections in the database. It returns metadata for each collection including its state, index status, and embedding model details. If no collections exist, an empty list is returned.

### Parameters
- None

### Usage
- **Purpose**: To obtain an overview of all vector storage collections and their current configurations.

#### Example
```python
collections_info = list_collections()
print(collections_info)
```

## Documentation for `list_query_collections`

### Functionality
This function retrieves metadata for all query-optimized collections in the database. It helps monitor the query optimization infrastructure by returning details on each collection.

### Parameters
- None

### Usage
- **Purpose**: Obtain a list of query collections from the vector database.
- **Return Value**: Returns a ListCollectionsResponse containing metadata for each collection. If no collections exist, an empty list is returned.

#### Example
Request the GET endpoint '/collections/queries/list' to invoke this function and receive the optimized query collections.

## Documentation for `list_category_collections`

### Functionality
Lists all category-specific collections in the database. It interacts with the specialized categories vector database to retrieve metadata for every category collection.

### Parameters
- None

### Usage
- **Purpose**: Retrieve and display metadata for category collections.

#### Example
```python
response = list_category_collections()
print(response)
```

## Documentation for `get_collection_info`

### Functionality
Retrieves detailed information about a specific collection. It returns comprehensive state and configuration details for a given embedding model. If the collection does not exist, the function raises a 404 error.

### Parameters
- **body (GetCollectionInfoRequest)**: Request containing the identifier of the embedding model to retrieve the collection info.

### Usage
- **Purpose**: To fetch current state and configuration details for a specified embedding model collection.

#### Example
```python
response = get_collection_info(request)
```

## Documentation for `get_categories_collection_info`

### Functionality
Retrieves detailed information about a category-specific collection. It queries the specialized categories vector database to obtain the current state and configuration of the collection. If the collection is not found, a 404 error is returned with an appropriate error message.

### Parameters
- **body**: A GetCollectionInfoRequest object containing the embedding model parameters. The `embedding_model` field must include an `id` that uniquely identifies the target collection.

### Usage
- **Purpose**: To fetch the current state information for a category collection, useful for debugging or displaying configuration details.

#### Example
```python
response = get_categories_collection_info(
    body=GetCollectionInfoRequest(
        embedding_model=Model(id="model_id")
    )
)
if response:
    print(response)
```

## Documentation for `set_blue_collection`

### Functionality
Promotes a collection to "blue" (active/primary) status using a blue-green deployment pattern. This strategy enables zero-downtime updates by designating one collection as the primary one for operations.

### Parameters
- **body (SetBlueCollectionRequest)**: Contains the embedding model ID and relevant data needed to identify the collection to be promoted.

### Usage
- **Purpose**: Switches an existing collection to blue, making it the primary collection for serving requests.

#### Example
```python
set_blue_collection({
    "embedding_model_id": "model_xyz"
})
```

## Documentation for `set_blue_categories_collection`

### Functionality
Promotes a category collection to "blue" (active) status. This marks the collection as the primary one in the categories vector database, which facilitates zero-downtime updates and seamless switching between collection versions.

### Parameters
- **body (SetBlueCollectionRequest)**: An object that holds the `embedding_model_id` and other parameters required for setting the blue collection status.

### Usage
- **Purpose**: Update the blue collection status to ensure the primary category collection is active.

#### Example
```python
response = set_blue_categories_collection(request_body)
```

## Documentation for `get_blue_collection_info`

### Functionality
Retrieves detailed state information about the blue (active) collection that serves production traffic. Returns a 404 error if no blue collection is designated.

### Parameters
- None

### Usage
- **Purpose**: To provide current status information about the active collection.
- **HTTP Endpoint**: GET `/collections/get-blue-info`.

#### Example
```bash
curl -X GET http://localhost:8000/collections/get-blue-info
```

## Documentation for `get_blue_query_collection_info`

### Functionality
Retrieves state information about the active query collection. This endpoint monitors the query optimization infrastructure. If no blue query collection exists, it returns a 404 error with a detail message.

### Parameters
- None

### Usage
Use this endpoint to check the status of the blue query collection, which is the primary query collection used in search operations.

#### Example
GET `/collections/get-blue-query-info`
Response:
```json
{
  "collection_status": "active",
  "details": { ... }
}
```

## Documentation for `get_blue_category_collection_info`

### Functionality
Retrieves information about the active blue category collection from the categories-specific vector database. If no blue category collection exists, a 404 error is returned.

### Parameters
- None

### Usage
This endpoint is used to monitor the state of the blue category collection. It returns relevant state information for troubleshooting and monitoring.

#### Example
GET `/collections/categories/get-blue-info`

## Documentation for `insert_objects`

### Functionality
Adds new vector objects to a collection. Optimized for adding objects that do not exist yet. Existing objects with matching IDs are not updated. Suitable for initial data loading.

### Parameters
- **body**: An InsertObjectsRequest instance containing:
  - **objects**: List of vector objects.
  - **embedding_model_id**: Identifier for the embedding model.

### Usage
- **Purpose**: Insert new objects into a vector collection.

#### Example
Send a POST request with:
```json
{
    "embedding_model_id": "model123",
    "objects": [
        {"id": "obj1", "vector": [ ... ]},
        {"id": "obj2", "vector": [ ... ]}
    ]
}
```

## Documentation for `insert_categories_objects`

### Functionality
Inserts new category-specific vector objects into a specialized vector collection. This endpoint leverages the categories vector database to efficiently add new category objects. It follows a pattern similar to regular vector object insertion but is optimized for category data.

### Parameters
- **embedding_model_id**: Identifier for the embedding model.
- **objects**: List of category vector objects to be inserted.

### Usage
- **Purpose**: To add category-specific vectors into the related collection using the specialized database.

#### Example
Request body example:
```json
{
  "embedding_model_id": "model123",
  "objects": [ { ... }, { ... } ]
}
```

## Documentation for `delete_objects`

### Functionality
Removes specific vector objects from a collection. This method allows targeted cleanup by efficiently removing data using object IDs, avoiding full collection scans.

### Parameters
- **body (DeleteObjectRequest)**: Contains:
  - **embedding_model_id (str)**: ID of the embedding model.
  - **object_ids (List[str])**: IDs of objects to remove.

### Usage
- **Purpose**: Remove obsolete or unwanted vector data.

#### Example
```python
request = {
    "embedding_model_id": "model_123",
    "object_ids": ["obj1", "obj2"]
}
# Call delete_objects(request) to remove objects.
```

## Documentation for `delete_categories_objects`

### Functionality
Removes specific category vector objects from a collection. This endpoint leverages the specialized categories vector database to efficiently delete entries by their unique IDs, ensuring that obsolete category data is promptly removed.

### Parameters
- **embedding_model_id (str)**: Identifier of the embedding model.
- **object_ids (list)**: List of category object IDs to delete.

### Usage
- **Purpose**: Remove outdated category vectors safely.

#### Example
```python
payload = {
    "embedding_model_id": "model_X",
    "object_ids": [101, 102, 103]
}
response = client.post(
    "/collections/categories/objects/delete", json=payload
)
print(response.json())
```

## Documentation for `find_objects_by_ids`

### Functionality
Retrieves vector objects by their identifiers with an exact match. No vector search is performed, returning complete objects including vectors and metadata.

### Parameters
- **embedding_model_id**: The model ID to select the correct collection.
- **object_ids**: A list of object IDs to retrieve.

### Usage
- **Purpose**: To fetch object data exactly without similarity measures.
- Uses the find_by_ids method on the vector database.

#### Example
Request payload:
```json
{
  "embedding_model_id": "model_001",
  "object_ids": ["id1", "id2"]
}
```

## Documentation for `find_categories_objects_by_ids`

### Functionality
Retrieves category vector objects by their identifiers. Uses the specialized categories vector database for exact lookup. Returns complete category object data including vectors and metadata.

### Parameters
- **body**: Request object of type FindObjectsByIdsRequest.
  - **embedding_model_id**: Identifier for the embedding model.
  - **object_ids**: List of vector object identifiers.

### Usage
- **Purpose**: Retrieve categories for management operations.
- **Returns**: Complete object data for the requested categories.

#### Example
Request payload:
```json
{
  "embedding_model_id": "model123",
  "object_ids": ["cat1", "cat2"]
}
```

## Documentation for `find_similar_objects`

### Functionality
Performs a similarity search on a vector collection. It requires a query vector and optional parameters for pagination and distance filtering, returning objects sorted from the closest match.

### Parameters
- **body**: An instance of FindSimilarObjectsRequest with:
  - **embedding_model_id**: Identifier of the target collection.
  - **query_vector**: List of numbers representing the query embedding.
  - **limit**: Maximum number of results to return (optional).
  - **offset**: Number of initial results to skip (optional).
  - **max_distance**: Maximum distance threshold for filtering (optional).

### Usage
- **Purpose**: Retrieve objects similar to a given vector.

#### Example
```python
request = FindSimilarObjectsRequest(
    embedding_model_id='model123',
    query_vector=[0.1, 0.2, 0.3],
    limit=10,
    offset=0,
    max_distance=0.5
)
results = find_similar_objects(request)
print(results)
```

## Documentation for `find_similar_categories_objects`

### Functionality
Performs similarity search on category vectors in the specialized vector database. It uses the same parameters as a regular vector search to locate and return categories that are most similar to the query.

### Parameters
- **body**: An instance of FindSimilarObjectsRequest containing:
  - **query_vector**: A list representing the query vector for comparison.
  - **limit**: The maximum number of similar objects to return.
  - **offset**: The offset for paginated results.
  - **max_distance**: (Optional) The maximum allowed distance for matching.

### Usage
- **Purpose**: Retrieve similar category objects from the vector database based on a provided query vector. This facilitates semantic searches within category data.
- **Return**: A list of category objects sorted by similarity (closest first).

#### Example
Send a POST request to `/collections/categories/objects/find-similar` with a JSON body such as:
```json
{
  "embedding_model_id": "your_model_id",
  "query_vector": [0.1, 0.2, 0.3],
  "limit": 10,
  "offset": 0,
  "max_distance": 0.5
}
```