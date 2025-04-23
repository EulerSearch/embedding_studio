# Configuring and Managing Vector Databases with the Embedding Studio API

This tutorial provides a comprehensive guide to interacting with vector databases in Embedding Studio through its REST API endpoints. You'll learn how to create, configure, and manage collections, as well as perform vector operations through API calls.

## 1. API Overview

Embedding Studio provides a rich set of API endpoints for vector database management, organized into the following categories:

- **Public endpoints**: For general search and vector operations
- **Internal endpoints**: For system configuration and maintenance
- **Collection management**: For creating and managing vector collections
- **Vector operations**: For adding, updating, and querying vectors

## 2. Environment Setup

Before using the API, you'll need to configure your environment. This can be done by setting environment variables or using a `.env` file:

```ini
# PostgreSQL connection (for vector storage)
POSTGRES_HOST=your_postgres_host
POSTGRES_PORT=5432
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
POSTGRES_DB=your_database

# MongoDB connection (for metadata storage)
EMBEDDINGS_MONGO_HOST=your_mongo_host
EMBEDDINGS_MONGO_PORT=27017
EMBEDDINGS_MONGO_DB_NAME=embedding_studio
EMBEDDINGS_MONGO_USERNAME=your_mongo_user
EMBEDDINGS_MONGO_PASSWORD=your_mongo_password

# Redis connection (for suggesting service)
REDIS_HOST=your_redis_host
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# Internal API access
OPEN_INTERNAL_ENDPOINTS=1
```

## 3. Collection Management API

### 3.1 Creating a Collection

To create a new vector collection, use the internal endpoint:

```bash
POST /internal/vectordb/collections/create
```

Request body:
```json
{
  "embedding_model_id": "text-embedding-ada-002"
}
```

This will create both a main collection and a query collection for the specified embedding model. The collection names are derived from the embedding model ID.

### 3.2 Creating Indexes

After creating a collection, you need to build indexes for efficient similarity search:

```bash
POST /internal/vectordb/collections/create-index
```

Request body:
```json
{
  "embedding_model_id": "text-embedding-ada-002"
}
```

This creates HNSW indexes on both the main collection and the query collection.

### 3.3 Listing Collections

To retrieve all available collections:

```bash
GET /internal/vectordb/collections/list
```

Response example:
```json
{
  "collections": [
    {
      "collection_id": "text-embedding-ada-002",
      "embedding_model": {
        "id": "text-embedding-ada-002",
        "dimensions": 1536,
        "metric_type": "COSINE",
        "metric_aggregation_type": "AVG",
        "hnsw": {
          "m": 16,
          "ef_construction": 64
        }
      },
      "work_state": "GREEN",
      "index_created": true,
      "applied_optimizations": ["CreateOrderingIndexesOptimization"]
    }
  ]
}
```

### 3.4 Setting the Blue Collection

The "blue" collection is the primary active collection used for serving requests. To designate a collection as blue:

```bash
POST /internal/vectordb/collections/set-blue
```

Request body:
```json
{
  "embedding_model_id": "text-embedding-ada-002"
}
```

This implements the blue-green deployment pattern for zero-downtime updates.

### 3.5 Getting the Blue Collection Info

To check which collection is currently active (blue):

```bash
GET /internal/vectordb/collections/get-blue-info
```

### 3.6 Deleting a Collection

To remove a collection and all its data:

```bash
POST /internal/vectordb/collections/delete
```

Request body:
```json
{
  "embedding_model_id": "text-embedding-ada-002"
}
```

Note: You cannot delete a collection that is currently designated as "blue".

## 4. Vector Operations API

### 4.1 Inserting Vectors

To add new vector objects to a collection:

```bash
POST /internal/vectordb/collections/objects/insert
```

Request body:
```json
{
  "embedding_model_id": "text-embedding-ada-002",
  "objects": [
    {
      "object_id": "product_123",
      "parts": [
        {
          "part_id": "product_123_title",
          "vector": [0.1, 0.2, ...],
          "is_average": false
        },
        {
          "part_id": "product_123_description",
          "vector": [0.3, 0.4, ...],
          "is_average": false
        }
      ],
      "payload": {
        "name": "Ergonomic Chair",
        "category": "Office Furniture"
      }
    }
  ]
}
```

### 4.2 Upserting Vectors

To add or update vector objects (insert if not exists, update if exists):

```bash
POST /internal/vectordb/collections/objects/upsert
```

Request body:
```json
{
  "embedding_model_id": "text-embedding-ada-002",
  "objects": [...],
  "shrink_parts": true
}
```

The `shrink_parts` parameter controls whether to optimize storage by deleting all existing parts before adding new ones.

### 4.3 Deleting Vectors

To remove specific vector objects:

```bash
POST /internal/vectordb/collections/objects/delete
```

Request body:
```json
{
  "embedding_model_id": "text-embedding-ada-002",
  "object_ids": ["product_123", "product_456"]
}
```

### 4.4 Finding Vectors by ID

To retrieve specific vectors by their IDs:

```bash
POST /internal/vectordb/collections/objects/find-by-ids
```

Request body:
```json
{
  "embedding_model_id": "text-embedding-ada-002",
  "object_ids": ["product_123", "product_456"]
}
```

### 4.5 Similarity Search (Internal API)

For internal system use, you can perform vector similarity search:

```bash
POST /internal/vectordb/collections/objects/find-similar
```

Request body:
```json
{
  "embedding_model_id": "text-embedding-ada-002",
  "query_vector": [0.1, 0.2, ...],
  "limit": 10,
  "offset": 0,
  "max_distance": 0.3
}
```

## 5. Public Search API

### 5.1 Similarity Search

The public API provides a more feature-rich similarity search endpoint:

```bash
POST /embeddings/similarity-search
```

Request body:
```json
{
  "search_query": "ergonomic office chair",
  "limit": 10,
  "offset": 0,
  "max_distance": 0.3,
  "filter": {
    "query": {
      "term": {
        "field": "category",
        "value": "Office Furniture"
      }
    }
  },
  "create_session": true,
  "user_id": "user_123",
  "sort_by": {
    "field": "popularity",
    "order": "desc"
  },
  "similarity_first": true
}
```

This endpoint:
1. Converts the text query to a vector using the blue collection's model
2. Performs a similarity search with optional payload filtering
3. Optionally creates a session for tracking user interactions
4. Returns ranked results with distance scores

Response example:
```json
{
  "session_id": "6f9d7a8c-5e12-4b3b-b6e7-1a2b3c4d5e6f",
  "search_results": [
    {
      "object_id": "product_123",
      "distance": 0.15,
      "payload": {
        "name": "Ergonomic Chair",
        "category": "Office Furniture",
        "price": 299.99
      },
      "meta": {
        "source": "product_catalog"
      }
    },
    ...
  ],
  "next_page_offset": 10
}
```

### 5.2 Payload-Based Filtering

To search by payload contents without vector similarity:

```bash
POST /embeddings/payload-search
```

Request body:
```json
{
  "limit": 10,
  "offset": 0,
  "filter": {
    "query": {
      "bool": {
        "must": [
          {
            "term": {
              "field": "category",
              "value": "Office Furniture"
            }
          },
          {
            "range": {
              "field": "price",
              "range": {
                "gte": 100,
                "lte": 500
              }
            }
          }
        ]
      }
    }
  },
  "sort_by": {
    "field": "popularity",
    "order": "desc"
  }
}
```

### 5.3 Counting Records

To count records matching certain criteria:

```bash
POST /embeddings/payload-count
```

Request body:
```json
{
  "filter": {
    "query": {
      "term": {
        "field": "category",
        "value": "Office Furniture"
      }
    }
  }
}
```

Response:
```json
{
  "total_count": 143
}
```

## 6. Batch Vector Operations with Tasks

For large-scale operations, Embedding Studio provides task-based APIs that process vectors asynchronously.

### 6.1 Batch Upsertion

To upsert a large batch of vectors:

```bash
POST /embeddings/upsertion-tasks/run
```

Request body:
```json
{
  "task_id": "optional-custom-id",
  "items": [
    {
      "object_id": "product_123",
      "payload": {
        "name": "Ergonomic Chair",
        "category": "Office Furniture"
      },
      "item_info": {
        "source_url": "https://example.com/products/123.txt"
      }
    },
    ...
  ]
}
```

This creates an asynchronous task that:
1. Downloads content from the source if needed
2. Generates vectors using the blue collection's model
3. Upserts the vectors into the collection
4. Returns a task ID for status tracking

### 6.2 Batch Deletion

For deleting large numbers of vectors:

```bash
POST /embeddings/deletion-tasks/run
```

Request body:
```json
{
  "task_id": "optional-custom-id",
  "object_ids": ["product_123", "product_456", ...]
}
```

### 6.3 Checking Task Status

To monitor the progress of a task:

```bash
GET /embeddings/upsertion-tasks/{task_id}
```

or

```bash
GET /embeddings/deletion-tasks/{task_id}
```

Response example:
```json
{
  "id": "task_12345",
  "status": "processing",
  "created_at": "2023-08-01T14:30:00Z",
  "updated_at": "2023-08-01T14:32:15Z",
  "failed_items": []
}
```

## 7. Categories Vector Database

Embedding Studio maintains a separate vector database for categories. The API endpoints are similar but with `/categories` in the path:

### 7.1 Creating a Categories Collection

```bash
POST /internal/vectordb/collections/categories/create
```

### 7.2 Setting Blue Categories Collection

```bash
POST /internal/vectordb/collections/categories/set-blue
```

### 7.3 Categories Similarity Search

```bash
POST /parse-query/categories
```

Request body:
```json
{
  "search_query": "office furniture"
}
```

Response:
```json
{
  "categories": [
    {
      "object_id": "cat_123",
      "distance": 0.12,
      "payload": {
        "category_name": "Office Furniture",
        "category_path": "Home/Office/Furniture"
      }
    },
    ...
  ]
}
```

## 8. API-Based Blue-Green Deployment Workflow

Here's a typical workflow for updating embedding models with zero downtime:

1. Create a new collection with the updated embedding model:
```bash
POST /internal/vectordb/collections/create
{
  "embedding_model_id": "text-embedding-ada-003"
}
```

2. Create indexes for the new collection:
```bash
POST /internal/vectordb/collections/create-index
{
  "embedding_model_id": "text-embedding-ada-003"
}
```

3. Reindex all data from the old collection to the new one:
```bash
POST /internal/reindex-tasks/run
{
  "source": {"embedding_model_id": "text-embedding-ada-002"},
  "dest": {"embedding_model_id": "text-embedding-ada-003"},
  "deploy_as_blue": false,
  "wait_on_conflict": true
}
```

4. Once reindexing is complete, set the new collection as blue:
```bash
POST /internal/vectordb/collections/set-blue
{
  "embedding_model_id": "text-embedding-ada-003"
}
```

5. After confirming everything works, delete the old collection:
```bash
POST /internal/vectordb/collections/delete
{
  "embedding_model_id": "text-embedding-ada-002"
}
```

## 9. API Error Handling

The API uses standard HTTP status codes for error reporting:

- **400 Bad Request**: Invalid input parameters
- **404 Not Found**: Resource not found (collection, object, etc.)
- **500 Internal Server Error**: Server-side processing error

Error responses include a detailed message:

```json
{
  "detail": "Collection with id=text-embedding-ada-002 does not exist"
}
```

## 10. Advanced API Features

### 10.1 Session Tracking

The API supports session tracking for analytics and improvement:

```bash
POST /clickstream/session
```

Request body:
```json
{
  "session_id": "session_123",
  "search_query": "ergonomic chair",
  "search_results": [
    {
      "object_id": "product_123",
      "rank": 0.15
    },
    ...
  ],
  "user_id": "user_123"
}
```

### 10.2 User Interaction Tracking

To track user clicks or other interactions:

```bash
POST /clickstream/session/events
```

Request body:
```json
{
  "session_id": "session_123",
  "events": [
    {
      "event_id": "event_456",
      "object_id": "product_123",
      "event_type": "click",
      "created_at": 1627484400
    }
  ]
}
```

### 10.3 Model Improvement from Sessions

To use session data for model improvement:

```bash
POST /clickstream/internal/session/use-for-improvement
```

Request body:
```json
{
  "session_id": "session_123"
}
```

## Conclusion

The Embedding Studio API provides a comprehensive set of endpoints for configuring and managing vector databases. By using these endpoints, you can:

- Create and manage vector collections
- Perform vector operations and similarity searches
- Implement blue-green deployment for zero-downtime updates
- Track user interactions and improve models based on feedback

For more information, refer to the complete API documentation or explore the Embedding Studio SDK libraries for your preferred programming language.