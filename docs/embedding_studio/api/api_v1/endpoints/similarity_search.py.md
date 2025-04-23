## Documentation for Embedding Studio Methods

### `_find_by_payload_fiter`

#### Functionality
This function performs a similarity search on the embeddings collection. It uses parameters from a `SimilaritySearchRequest` to filter objects by payload. If the embeddings collection is not initialized, it raises an HTTP 404 error.

#### Parameters
- **body (SimilaritySearchRequest)**: Contains search parameters.
  - **offset (int)**: The starting index for search results.
  - **limit (int)**: The maximum number of results to return.
  - **filter**: Optional payload filter applied via `PayloadFilter.model_validate`.
  - **sort_by**: Sorting preference for the results.

#### Usage
- **Purpose**: Retrieve objects matching payload criteria from the database.

#### Example
```python
from embedding_studio.api.api_v1.endpoints.similarity_search import _find_by_payload_fiter
from your_module import SimilaritySearchRequest

request = SimilaritySearchRequest(offset=0, limit=10, filter=None, sort_by="date")
results = _find_by_payload_fiter(request)
print(results)
```

---

### `_count_by_payload_filter`

#### Functionality
This function counts the number of objects matching a payload filter in the embeddings collection. It first verifies that the collection exists; if not, it raises an HTTPException. Then it applies the filter and returns the count of matching objects.

#### Parameters
- **body (PayloadCountRequest)**: An instance containing a filter attribute (`filter`) that specifies filter criteria. It is optional.

#### Returns
- **int**: The total number of objects that match the filter criteria.

#### Usage
- **Purpose**: Use this function to get a count of records in the collection according to a certain payload filter.

#### Example
```python
result = _count_by_payload_filter(body)
```

---

### `_find_similars`

#### Functionality
Performs a similarity search on an embeddings collection using a query. It retrieves a query vector via an inference client and finds similar objects from the collection. It also schedules an asynchronous task that inserts the query vector into a separate query collection.

#### Parameters
- **body (SimilaritySearchRequest)**: Contains the search query, offset, limit, max_distance, an optional filter, session_id, and user_id.
- **background_tasks**: FastAPI BackgroundTasks instance used to schedule the asynchronous query vector insertion task.

#### Returns
- **SearchResults**: Object containing similar objects found from the collection.

#### Usage
- **Purpose**: To perform a similarity search on an embeddings collection and optionally add a background task for query vector insertion.

#### Example
```python
result = _find_similars(request_body, background_tasks)
```

---

### `_create_session_object`

#### Functionality
Creates a new session placeholder using request and search information. It returns a Session object with a creation timestamp.

#### Parameters
- **body**: Request body with session and search info.
- **session_id**: Unique identifier for the session.
- **is_payload_search**: Boolean flag indicating payload search.
- **payload_filter**: Optional filter for payload search.
- **sort_by**: Optional criteria to sort search results.

#### Usage
- **Purpose**: Begin a session to track similarity search queries and store results.

#### Example
```python
session = _create_session_object(request_body, 'session-uuid', is_payload_search=True, payload_filter=payload, sort_by=sort_options)
```

---

### `_register_session_with_results`

#### Functionality
This function updates the session with search results and registers the session using the clickstream DAO. It converts found objects into search result items and assigns the resulting session ID after registration.

#### Parameters
- **session**: The session object to be updated and registered.
- **search_results**: The results from the similarity search containing found objects, each converted into a search result item.

#### Usage
- **Purpose**: Update the session with new search results and register it to ensure idempotency and consistency.

#### Example
```python
session = _create_session_object(...)
_register_session_with_results(session, search_results)
```

---

### `similarity_search`

#### Functionality
This endpoint conducts a similarity search on stored embeddings. It processes both query-based and payload-based searches. Additionally, it creates a session when requested and registers search results with the session.

#### Parameters
- **body (SimilaritySearchRequest)**: Contains search query, payload filter, session details, offset, limit, and other options.
- **background_tasks**: A FastAPI BackgroundTasks instance used to schedule asynchronous tasks during the search operation.

#### Usage
- **Purpose**: To retrieve objects similar to a given query or payload filter. The endpoint returns a response including session ID, search results, next page offset, and additional meta information.

#### Example
Request:
```json
{
  "search_query": "example query",
  "offset": 0,
  "limit": 10,
  "create_session": true
}
```

Response:
```json
{
  "session_id": "<session_id>",
  "search_results": [
      {"object_id": "<id>", "distance": 0.85, "payload": {}, "meta": {}}
  ],
  "next_page_offset": 10,
  "meta_info": {}
}
```

---

### `payload_search`

#### Functionality
This endpoint searches for similar objects using a payload filter. It queries the embeddings collection and may create a new session to track search results.

#### Parameters
- **body (PayloadSearchRequest)**: Contains:
  - **create_session**: bool indicating if a new session should be created.
  - **filter**: Optional payload filter criteria.
  - **session_id**: Optional session identifier.
  - **sort_by**: Sorting criteria.
  - **offset**: Pagination offset.
  - **limit**: Pagination limit.

#### Usage
- **Purpose**: To search for objects based on payload criteria.

#### Example
Send a POST request to `/payload-search` with a JSON body:
```json
{
  "create_session": true,
  "filter": { "key": "value" },
  "session_id": "optional-id",
  "sort_by": "criteria",
  "offset": 0,
  "limit": 10
}
```

---

### `count_payload`

#### Functionality
Counts objects that match a given payload filter. The endpoint uses the filtering criteria in the request body and returns the total number of matching objects.

#### Parameters
- **body (PayloadCountRequest)**: An instance containing filter details, specifying the payload filter criteria for the count.

#### Usage
- **Purpose**: Use this endpoint to obtain the total count of objects that match a specific payload filter.

#### Example
```python
from embedding_studio.api.api_v1.schemas.similarity_search import PayloadCountRequest
from embedding_studio.api.api_v1.endpoints.similarity_search import count_payload

# Create a request with the desired payload filter
request = PayloadCountRequest(filter={"key": "value"})

# Call the endpoint to get the count
response = count_payload(request)
print(response.total_count)
```