# Vector DB Integration in Embedding Studio

This tutorial explains how Embedding Studio integrates with vector databases, specifically focusing on the PostgreSQL/pgvector implementation. You'll learn how vectors are stored, indexed, and queried to power semantic search capabilities.

## 1. Introduction to Vector Storage in Embedding Studio

Embedding Studio uses a flexible plugin-based architecture for storing and retrieving vector embeddings. At its core, the system is designed around these key concepts:

- **Collections**: Logical groups of vectors with common dimensionality and metric type
- **Objects**: Items that contain metadata and one or more vector parts
- **Object Parts**: Individual vector embeddings that represent a piece of an object
- **Metrics**: Different distance measurements (cosine, dot product, Euclidean)

The main implementation of the vector database in Embedding Studio uses PostgreSQL with the pgvector extension, providing an enterprise-grade foundation for vector operations.

## 2. Vector DB Architecture

The vector database system follows a layered architecture:

```
┌─────────────────────────────────────┐
│           Vector Database            │
├─────────────────────────────────────┤
│ ┌─────────────┐     ┌─────────────┐ │
│ │ Collections │     │ Query       │ │
│ │             │     │ Collections │ │
│ └─────────────┘     └─────────────┘ │
├─────────────────────────────────────┤
│         Collection Info Cache       │
├─────────────────────────────────────┤
│ ┌───────────────┐ ┌───────────────┐ │
│ │ PostgreSQL    │ │ MongoDB       │ │
│ │ (pgvector)    │ │ (metadata)    │ │
│ └───────────────┘ └───────────────┘ │
└─────────────────────────────────────┘
```

### Key Components:

1. **VectorDb Interface**: Abstract interface defining vector database operations
2. **PgvectorDb Implementation**: PostgreSQL implementation using pgvector
3. **Collection**: Interface for storing and querying vectors
4. **PgvectorCollection**: PostgreSQL-specific collection implementation
5. **CollectionInfoCache**: Manages metadata about collections

## 3. Data Model

### Objects and Vector Parts

In Embedding Studio, each object can have multiple vector parts, allowing for fine-grained representation:

```
┌─────────────────────────────────────┐
│ Object                              │
│  ID: "product_123"                  │
│  Payload: {                         │
│    "name": "Ergonomic Chair",       │
│    "category": "Office Furniture"   │
│  }                                  │
├─────────────────────────────────────┤
│ ┌─────────────┐   ┌───────────────┐ │
│ │ Part 1      │   │ Part 2        │ │
│ │ ID: "p1"    │   │ ID: "p2"      │ │
│ │ Vector: [...│   │ Vector: [...]  │ │
│ └─────────────┘   └───────────────┘ │
└─────────────────────────────────────┘
```

This design enables:
- Storage of chunked text embeddings
- Representation of multi-modal objects
- Part-level similarity search
- Aggregation across parts

### Database Schema

The vector database uses two primary tables for each collection:

1. **Object Table** (`dbo_{collection_id}`):
   - `object_id`: Primary key
   - `payload`: JSONB for metadata
   - `storage_meta`: JSONB for system metadata
   - `user_id`: Owner identification
   - `original_id`: Reference to origin (for derived objects)
   - `session_id`: Session tracking for queries

2. **Object Part Table** (`dbop_{collection_id}`):
   - `part_id`: Primary key
   - `object_id`: Foreign key to object
   - `vector`: pgvector type storing the embedding
   - `is_average`: Flag for aggregated vectors

## 4. Setting Up Vector DB Integration

### Prerequisites

- PostgreSQL 14+ with pgvector extension
- MongoDB for metadata storage

### Configuration

```python
# Example configuration in app context
from embedding_studio.vectordb.pgvector.vectordb import PgvectorDb
from embedding_studio.db import postgres, mongo

# Initialize the vector database
vectordb = PgvectorDb(
    pg_database=postgres.pg_database,
    embeddings_mongo_database=mongo.embeddings_mongo_database,
    prefix="basic",
)

# Initialize a separate database for categories if needed
categories_vectordb = PgvectorDb(
    pg_database=postgres.pg_database,
    embeddings_mongo_database=mongo.embeddings_mongo_database,
    prefix="categories",
)
```

## 5. Working with Collections

Collections are created based on embedding models. Each collection is optimized for a specific vector dimensionality and distance metric.

### Creating Collections

```python
from embedding_studio.models.embeddings.models import EmbeddingModelInfo, MetricType

# Define an embedding model
embedding_model = EmbeddingModelInfo(
    id="text-embeddings-ada-002",
    dimensions=1536,
    metric_type=MetricType.COSINE,
)

# Create or get a collection
collection = vectordb.get_or_create_collection(embedding_model)

# Create a query collection for storing user queries
query_collection = vectordb.get_or_create_query_collection(embedding_model)
```

### Managing Collections

```python
# List all collections
collections = vectordb.list_collections()

# Set active "blue" collection
vectordb.set_blue_collection("text-embeddings-ada-002")

# Get active collections
blue_collection = vectordb.get_blue_collection()
blue_query_collection = vectordb.get_blue_query_collection()

# Delete a collection
vectordb.delete_collection("old-embedding-model")
```

## 6. Storing and Retrieving Vectors

### Inserting Objects

```python
from embedding_studio.models.embeddings.objects import Object, ObjectPart

# Create an object with multiple vector parts
object = Object(
    object_id="product_123",
    parts=[
        ObjectPart(part_id="product_123_title", vector=[0.1, 0.2, ...], is_average=False),
        ObjectPart(part_id="product_123_description", vector=[0.3, 0.4, ...], is_average=False),
    ],
    payload={"name": "Ergonomic Chair", "category": "Office Furniture"},
)

# Insert into collection
collection.insert([object])
```

### Vector Search

```python
# Simple vector similarity search
results = collection.find_similarities(
    query_vector=[0.1, 0.2, ...],
    limit=10,
    max_distance=0.3,
)

# Search with payload filtering
from embedding_studio.models.payload.models import PayloadFilter, TermQuery

results = collection.find_similarities(
    query_vector=[0.1, 0.2, ...],
    limit=10,
    payload_filter=PayloadFilter(
        query=TermQuery(term={"field": "category", "value": "Office Furniture"})
    ),
)

# Advanced search with sorting
from embedding_studio.models.sort_by.models import SortByOptions

results = collection.find_similarities(
    query_vector=[0.1, 0.2, ...],
    limit=10,
    sort_by=SortByOptions(field="payload.popularity", order="desc"),
)
```

## 7. Vector Indexing

Embedding Studio uses HNSW (Hierarchical Navigable Small World) indexes for efficient similarity search.

### Default Index Configuration

```python
from embedding_studio.models.embeddings.models import HnswParameters, SearchIndexInfo, MetricType, MetricAggregationType

# Example index configuration from a plugin
def get_search_index_info(self) -> SearchIndexInfo:
    return SearchIndexInfo(
        dimensions=384,
        metric_type=MetricType.COSINE,
        metric_aggregation_type=MetricAggregationType.AVG,
        hnsw=HnswParameters(m=16, ef_construction=96),
    )
```

### Creating Indexes

```python
# Create an index for a collection
collection.create_index()
```

### Customizing Indexes

You can customize index parameters in your fine-tuning plugins:

```python
# In your custom fine-tuning method
def get_search_index_info(self) -> SearchIndexInfo:
    return SearchIndexInfo(
        dimensions=1024,  # For E5-large
        metric_type=MetricType.DOT,  # Use dot product
        metric_aggregation_type=MetricAggregationType.MIN,  # Take minimum distance
        hnsw=HnswParameters(
            m=16,  # Graph connections
            ef_construction=128,  # Build-time precision
        ),
    )
```

## 8. Vector DB Optimization

Embedding Studio provides an optimization framework for vector databases. Optimization strategies can be applied to collections to enhance performance.

### Built-in Optimizations

```python
from plugins.custom.optimizations.indexes import CreateOrderingIndexesOptimization
from embedding_studio.vectordb.pgvector.optimization import PgvectorObjectsOptimization

# Apply optimizations to collections
vectordb.add_optimization(CreateOrderingIndexesOptimization())
vectordb.apply_optimizations()
```

### Custom Optimizations

You can create custom optimizations by subclassing `PgvectorObjectsOptimization`:

```python
from sqlalchemy import text
from embedding_studio.vectordb.pgvector.optimization import PgvectorObjectsOptimization

class VacuumAnalyzeOptimization(PgvectorObjectsOptimization):
    def __init__(self):
        super().__init__(name="VacuumAnalyzeOptimization")

    def _get_statement(self, tablename: str):
        return text(f"VACUUM ANALYZE {tablename}")
```

## 9. Advanced Vector Queries

### Range Queries

```python
from embedding_studio.models.payload.models import PayloadFilter, RangeQuery

# Find objects with popularity score between 4.5 and 5.0
filter = PayloadFilter(
    query=RangeQuery(
        field="popularity",
        range={"gte": 4.5, "lte": 5.0}
    )
)

results = collection.find_by_payload_filter(
    payload_filter=filter,
    limit=10
)
```

### Boolean Queries

```python
from embedding_studio.models.payload.models import PayloadFilter, BoolQuery, TermQuery, MatchQuery

# Complex boolean query
filter = PayloadFilter(
    query=BoolQuery(
        must=[
            TermQuery(term={"field": "category", "value": "Office Furniture"}),
        ],
        should=[
            MatchQuery(match={"field": "description", "value": "ergonomic"}),
            MatchQuery(match={"field": "description", "value": "comfortable"}),
        ],
        must_not=[
            TermQuery(term={"field": "in_stock", "value": False}),
        ]
    )
)

results = collection.find_by_payload_filter(
    payload_filter=filter,
    limit=10
)
```

## 10. Performance Considerations

### Memory Management

PgvectorCollection uses connection pooling and prepared statements to maintain performance:

```python
# Configuration parameters (from settings.py)
SQLALCHEMY_POOL_SIZE = 10  # Number of connections in the pool
SQLALCHEMY_MAX_OVERFLOW = 20  # Max extra connections
SQLALCHEMY_POOL_TIMEOUT = 30  # Seconds to wait for connection
```

### Batch Operations

For large-scale operations, use batching:

```python
import itertools

def batch_upsert(collection, objects, batch_size=1000):
    for i in range(0, len(objects), batch_size):
        batch = objects[i:i+batch_size]
        collection.upsert(batch)
```

### Query Optimization

- Use `similarity_first=True` for pure similarity-based ranking
- Apply `max_distance` thresholds to limit search scope
- Use appropriate `MetricAggregationType` (AVG/MIN) based on your use case

## 11. Integration with Fine-Tuning

Vector DB integration is a key component of the fine-tuning process in Embedding Studio:

1. **Before Fine-Tuning**: Original vectors are stored and indexed
2. **During Fine-Tuning**: Clickstream data is analyzed for query-item pairs
3. **After Fine-Tuning**: Updated vectors are stored with the same object IDs

Example from a fine-tuning plugin:

```python
def get_vectordb_optimizations(self) -> List[Optimization]:
    """
    Return a list of vector DB optimizations to apply.
    In this case: index ordering by similarity or freshness.
    """
    return [CreateOrderingIndexesOptimization()]
```

## 12. Troubleshooting

Common issues and solutions:

### Connection Issues

If you encounter database connection problems:

```python
# Check connection health
try:
    with vectordb._pg_database.connect() as conn:
        result = conn.execute(text("SELECT 1")).scalar()
        print(f"Connection test: {result == 1}")
except Exception as e:
    print(f"Connection error: {e}")
```

### Index Performance

If similarity searches are slow:

1. Check that indexes are created: `collection.get_state_info().index_created`
2. Verify index parameters: `collection.get_info().embedding_model.hnsw`
3. Consider rebuilding the index: `collection.create_index()`

### Query Debugging

To debug complex queries:

```python
# Enable PostgreSQL query logging
with vectordb._pg_database.connect() as conn:
    conn.execute(text("SET log_statement = 'all'"))
    # Run your query
    # Check PostgreSQL logs
```

## Conclusion

Vector database integration is a core feature of Embedding Studio, providing the foundation for semantic search and recommendation capabilities. Understanding the PostgreSQL/pgvector implementation helps you optimize and extend these capabilities for your specific use cases.

For more information, refer to:
- The `vectordb` module documentation
- The pgvector extension documentation
- PostgreSQL performance tuning guides