# Documentation for VectorDB Collection Management API

---

## `POST /collections/create` — create_collection

### Description
Creates a new vector collection for an embedding model. Also initializes a query-optimized collection.

### Request
```json
{
  "embedding_model_id": "model-001"
}
```

### Response
```json
{
  "embedding_model": { "id": "model-001" },
  "index_status": "empty",
  "collection_size": 0,
  "last_updated": "2024-05-20T00:00:00Z"
}
```

---

## `POST /collections/create-index` — create_index

### Description
Creates HNSW indexes on both the main and query collections for the specified model.

### Request
```json
{
  "embedding_model_id": "model-001"
}
```

---

## `POST /collections/categories/create-index` — create_categories_index

### Description
Same as `create-index`, but for category collections.

---

## `POST /collections/delete` — delete_collection

### Request
```json
{
  "embedding_model_id": "model-001"
}
```

---

## `POST /collections/categories/delete` — delete_categories_collection

### Request
```json
{
  "embedding_model_id": "model-001"
}
```

---

## `GET /collections/list` — list_collections  
## `GET /collections/queries/list` — list_query_collections  
## `GET /collections/categories/list` — list_category_collections

### Description
Returns metadata of all collections for general, query, or category types.

### Response
```json
{
  "collections": [
    {
      "embedding_model": { "id": "model-001" },
      "index_status": "built",
      "collection_size": 12000,
      "last_updated": "2024-05-20T00:00:00Z"
    }
  ]
}
```

---

## `GET /collections/get-info`  
## `GET /collections/categories/get-info`

### Request
```json
{
  "embedding_model_id": "model-001"
}
```

---

## `POST /collections/set-blue`  
## `POST /collections/categories/set-blue`

### Description
Promotes the specified collection to "blue" (primary) status.

### Request
```json
{
  "embedding_model_id": "model-001"
}
```

---

## `GET /collections/get-blue-info`  
## `GET /collections/get-blue-query-info`  
## `GET /collections/categories/get-blue-info`

### Description
Returns state of the active ("blue") collection for model/query/category.

---

## `POST /collections/objects/insert`  
## `POST /collections/categories/objects/insert`

### Request
```json
{
  "embedding_model_id": "model-001",
  "objects": [
    { "object_id": "obj-1", "vector": [0.1, 0.2], "metadata": {}, "payload": {} }
  ]
}
```

---

## `POST /collections/objects/upsert`  
## `POST /collections/categories/objects/upsert`

### Request
```json
{
  "embedding_model_id": "model-001",
  "shrink_parts": true,
  "objects": [
    { "object_id": "obj-1", "vector": [0.1, 0.2], "metadata": {}, "payload": {} }
  ]
}
```

---

## `POST /collections/objects/delete`  
## `POST /collections/categories/objects/delete`

### Request
```json
{
  "embedding_model_id": "model-001",
  "object_ids": ["obj-1", "obj-2"]
}
```

---

## `POST /collections/objects/find-by-ids`  
## `POST /collections/categories/objects/find-by-ids`

### Request
```json
{
  "embedding_model_id": "model-001",
  "object_ids": ["obj-1", "obj-2"]
}
```

### Response
```json
[
  {
    "object_id": "obj-1",
    "vector": [0.1, 0.2],
    "metadata": {}
  }
]
```

---

## `POST /collections/objects/find-similar`  
## `POST /collections/categories/objects/find-similar`

### Request
```json
{
  "embedding_model_id": "model-001",
  "query_vector": [0.1, 0.2],
  "limit": 5,
  "offset": 0,
  "max_distance": 0.5
}
```

### Response
```json
[
  {
    "object_id": "obj-1",
    "distance": 0.123,
    "vector": [0.1, 0.2],
    "metadata": {}
  }
]
```

---