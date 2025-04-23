# Documentation for Similarity Search API Methods

---

## `/embeddings/similarity-search`

### Functionality
Endpoint to search for similar objects using a query and/or payload filter. It accepts similarity search parameters and returns matching results along with a session ID if a session is created.

---

### Request Parameters
- `search_query` *(any)*: A string or structured query object to find similar objects.
- `filter` *(object, optional)*: Optional payload filter for refining search results.
- `offset` *(int, optional)*: Pagination offset for search results.
- `limit` *(int)*: Maximum number of search results to return.
- `max_distance` *(float, optional)*: Maximum allowed distance for similarity matching.
- `create_session` *(bool)*: Boolean flag to create a session if one does not exist.
- `user_id` *(string, optional)*: Optional user identifier for tracking.
- `session_id` *(string, optional)*: Optional session identifier.
- `sort_by` *(object, optional)*: Sort configuration, e.g., by `"field": "distance", "order": "asc"`.
- `similarity_first` *(bool)*: Prioritize similarity ranking before applying sort rules.
- `meta_info` *(object, optional)*: Additional metadata to store or log with the search.

---

### Request JSON Example
```json
{
  "search_query": "machine learning",
  "limit": 10,
  "offset": 0,
  "max_distance": 0.5,
  "filter": {
    "category": "AI",
    "published_year": { "$gte": 2020 }
  },
  "create_session": true,
  "user_id": "user-123",
  "session_id": null,
  "sort_by": {
    "field": "distance",
    "order": "asc"
  },
  "similarity_first": true,
  "meta_info": {
    "note": "production test run"
  }
}
```

---

### Response JSON Example
```json
{
  "session_id": "session-456",
  "next_page_offset": 10,
  "total_count": 189,
  "search_results": [
    {
      "object_id": "object-001",
      "distance": 0.172,
      "payload": {
        "title": "Deep Learning",
        "tags": ["AI", "ML"]
      },
      "meta": {
        "created_at": "2023-12-01T10:00:00Z"
      }
    }
  ],
  "meta_info": {
    "retrieval_time_ms": 312
  }
}
```

---

## `/embeddings/payload-search`

### Functionality
This endpoint performs a search using only payload filters (no embedding similarity required). Ideal for structured lookups using JSON attributes.

---

### Request Parameters
- `search_query`: Can be a string or structured payload (e.g., metadata object).
- `limit`: Max number of results to return.
- `offset`: Optional offset for paginated results.
- `max_distance`: Optional threshold if similarity is still relevant.
- `filter`: Optional filter conditions for payload fields.
- `create_session`: Whether to start a tracking session.
- `user_id`: Optional user performing the request.
- `session_id`: Reuse an existing session if provided.
- `sort_by`: Optional sorting configuration.
- `similarity_first`: Prioritize similarity when sorting, if enabled.
- `meta_info`: Optional metadata for analytics or context.

---

### Request JSON Example
```json
{
  "search_query": {
    "tags": ["transformers", "vision"]
  },
  "limit": 15,
  "offset": 0,
  "filter": {
    "domain": "research",
    "stars": { "$gte": 100 }
  },
  "create_session": false,
  "user_id": "user-abc",
  "session_id": null,
  "sort_by": {
    "field": "created_at",
    "order": "desc"
  },
  "similarity_first": false,
  "meta_info": {
    "note": "payload search only"
  }
}
```

---

### Response JSON Example
```json
{
  "session_id": "session-789",
  "next_page_offset": 15,
  "total_count": 65,
  "search_results": [
    {
      "object_id": "object-xyz",
      "distance": 0.24,
      "payload": {
        "title": "Efficient AI Systems",
        "domain": "research"
      },
      "meta": {
        "language": "Python"
      }
    }
  ],
  "meta_info": {
    "execution_time_ms": 289
  }
}
```

---

## `/embeddings/payload-count`

### Functionality
Returns the count of documents matching a specified payload filter. Used to estimate result volume or support faceted filtering in UI.

---

### Request Parameters
- `search_query`: Optional object for filtering or search context.
- `filter`: Optional structured object for attribute filtering.

---

### Request JSON Example
```json
{
  "search_query": {
    "domain": "research"
  },
  "filter": {
    "category": "opensource",
    "downloads": { "$gte": 1000 }
  }
}
```

---

### Response JSON Example
```json
{
  "total_count": 423
}
```

---

### Example Curl
```bash
curl -X POST "http://<server>/embeddings/payload-count" \
     -H "Content-Type: application/json" \
     -d '{"filter": {"category": "opensource"}}'
```

---
