# Configuring and Querying Vector Databases in Embedding Studio

This tutorial provides in-depth guidance on configuring your own vector database connection, managing collections, and executing advanced queries in Embedding Studio.

## 1. Configuring Your Vector Database Connection

Embedding Studio primarily uses PostgreSQL with pgvector extension for vector storage and MongoDB for metadata management. Here's how to set up your own connection.

### Environment Configuration

Create or modify your `.env` file with the following database connection settings:

```ini
# PostgreSQL (for vector storage)
POSTGRES_HOST=your_postgres_host
POSTGRES_PORT=5432
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
POSTGRES_DB=your_database

# MongoDB (for collection metadata)
EMBEDDINGS_MONGO_HOST=your_mongo_host
EMBEDDINGS_MONGO_PORT=27017
EMBEDDINGS_MONGO_DB_NAME=embedding_studio
EMBEDDINGS_MONGO_USERNAME=your_mongo_user
EMBEDDINGS_MONGO_PASSWORD=your_mongo_password
```

### PostgreSQL with pgvector Setup

1. Ensure you have PostgreSQL 14+ installed
2. Install the pgvector extension:

```sql
CREATE EXTENSION vector;
```

3. Grant appropriate permissions to your database user:

```sql
GRANT ALL PRIVILEGES ON DATABASE your_database TO your_user;
```

4. Configure connection pooling:

```ini
# Add to your .env file
SQLALCHEMY_POOL_SIZE=10
SQLALCHEMY_MAX_OVERFLOW=20
SQLALCHEMY_POOL_TIMEOUT=30
```

### Custom Database Initialization

If you need to programmatically initialize your vector database connection:

```python
import sqlalchemy
from sqlalchemy import create_engine
import pymongo
from embedding_studio.vectordb.pgvector.vectordb import PgvectorDb

# Connect to PostgreSQL
pg_connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
pg_engine = create_engine(pg_connection_string)

# Connect to MongoDB for metadata
mongo_client = pymongo.MongoClient(
    host=mongo_host,
    port=mongo_port,
    username=mongo_username,
    password=mongo_password,
)
mongo_db = mongo_client[mongo_database]

# Initialize vector database
vectordb = PgvectorDb(
    pg_database=pg_engine,
    embeddings_mongo_database=mongo_db,
    prefix="my_custom_prefix",  # Use a unique prefix for your application
)
```

## 2. Understanding Collection Management

Collections are the fundamental organizational units in Embedding Studio's vector database. Each collection corresponds to a specific embedding model and contains both objects and their vector parts.

### Collection Information Structure

The `CollectionInfo` object contains essential metadata about a collection:

```python
from embedding_studio.models.embeddings.collections import CollectionInfo
from embedding_studio.models.embeddings.models import EmbeddingModelInfo, MetricType, HnswParameters

# Collection information structure
collection_info = CollectionInfo(
    collection_id="my-embedding-model",  # Usually the embedding model ID
    embedding_model=EmbeddingModelInfo(
        id="my-embedding-model",
        dimensions=768,  # Vector size
        metric_type=MetricType.COSINE,  # Distance metric
        hnsw=HnswParameters(  # Index parameters
            m=16,
            ef_construction=128,
        ),
    ),
    applied_optimizations=[]  # List of applied optimization names
)
```

### Collection State

Collections have a state that includes work status:

```python
from embedding_studio.models.embeddings.collections import CollectionStateInfo, CollectionWorkState

# Collection state with operational status
collection_state = CollectionStateInfo(
    collection_id="my-embedding-model",
    embedding_model=my_embedding_model,
    work_state=CollectionWorkState.GREEN,  # GREEN, BLUE, RED status
    index_created=True,
    applied_optimizations=["CreateOrderingIndexesOptimization"]
)
```

### Blue Collections

The "blue" collection is the primary active collection used for serving requests. Only one collection can be designated as "blue" at a time.

```python
# Set a collection as the active "blue" collection
vectordb.set_blue_collection("text-embedding-ada-002")

# Get the current blue collection
blue_collection = vectordb.get_blue_collection()

# Get the blue query collection (for storing user queries)
blue_query_collection = vectordb.get_blue_query_collection()
```

This design supports continuous deployment with zero downtime:

1. Create a new collection with updated embeddings
2. Populate it with vectors
3. Switch the "blue" designation to the new collection
4. Clients automatically use the new collection

### Collection Info Cache

The `CollectionInfoCache` manages metadata about collections in MongoDB:

```python
# Access collection metadata
collection_infos = vectordb._collection_info_cache.list_collections()
query_collection_infos = vectordb._collection_info_cache.list_query_collections()

# Update collection metadata
vectordb._collection_info_cache.update_collection(updated_collection_info)

# Manually refresh the cache
vectordb._collection_info_cache.invalidate_cache()
```

## 3. Advanced Query Capabilities

Embedding Studio provides sophisticated query capabilities that combine vector similarity with metadata filtering.

### Query Structure

The query system is built around these components:

1. **Query Vector**: The embedding to search for similarity
2. **Payload Filter**: Optional filtering based on metadata
3. **Sort Options**: Control result ordering
4. **Distance Threshold**: Maximum distance for similarity matches

### Vector Similarity Search

```python
# Basic similarity search
results = collection.find_similarities(
    query_vector=[0.1, 0.2, ...],  # Your embedding vector
    limit=10,  # Number of results
    max_distance=0.3,  # Maximum cosine distance (0-1)
)

# Search with user-specific results
results = collection.find_similarities(
    query_vector=[0.1, 0.2, ...],
    limit=10,
    user_id="user_123",  # For personalized results
)
```

### Complex Payload Filtering

Embedding Studio supports Elasticsearch-inspired query filtering:

```python
from embedding_studio.models.payload.models import (
    PayloadFilter, BoolQuery, TermQuery, RangeQuery, MatchQuery
)

# Complex nested filter
filter = PayloadFilter(
    query=BoolQuery(
        must=[
            TermQuery(term={"field": "category", "value": "electronics"}),
            RangeQuery(
                field="price",
                range={"gte": 100, "lte": 500}
            ),
        ],
        should=[
            MatchQuery(match={"field": "description", "value": "wireless"}),
            MatchQuery(match={"field": "description", "value": "bluetooth"}),
        ],
        must_not=[
            TermQuery(term={"field": "in_stock", "value": False}),
        ]
    )
)

# Apply filter to similarity search
results = collection.find_similarities(
    query_vector=[0.1, 0.2, ...],
    limit=10,
    payload_filter=filter,
)
```

### Field-Based Sorting

Control the order of results:

```python
from embedding_studio.models.sort_by.models import SortByOptions

# Sort by popularity or other field
results = collection.find_similarities(
    query_vector=[0.1, 0.2, ...],
    limit=10,
    sort_by=SortByOptions(
        field="popularity",
        order="desc",  # 'asc' or 'desc'
        force_not_payload=False,  # True if it's a column, not JSON field
    ),
    similarity_first=True,  # True: sort by similarity then field, False: field only
)
```

### Hybrid Search

Combine exact metadata matching with semantic similarity:

```python
# Hybrid search with both filters and similarity
results = collection.find_similarities(
    query_vector=[0.1, 0.2, ...],
    limit=10,
    payload_filter=PayloadFilter(
        query=TermQuery(term={"field": "category", "value": "laptops"})
    ),
    max_distance=0.4,  # Semantic similarity threshold
)
```

### Metadata-Only Search

Search purely by metadata:

```python
# Search without vector similarity
results = collection.find_by_payload_filter(
    payload_filter=PayloadFilter(
        query=MatchQuery(match={"field": "description", "value": "gaming"})
    ),
    limit=10,
)

# Count matching records
count = collection.count_by_payload_filter(
    payload_filter=PayloadFilter(
        query=TermQuery(term={"field": "brand", "value": "Apple"})
    )
)
```

## 4. Collection Optimization

Optimizations improve vector database performance for specific workloads.

### Built-in Optimizations

Embedding Studio includes several optimization strategies:

```python
from plugins.custom.optimizations.indexes import CreateOrderingIndexesOptimization
from embedding_studio.vectordb.pgvector.optimization import PgvectorObjectsOptimization

# Add optimizations to vector database
vectordb.add_optimization(CreateOrderingIndexesOptimization())

# Apply optimizations to all collections
vectordb.apply_optimizations()
```

The `CreateOrderingIndexesOptimization` adds PostgreSQL indexes for common sorting fields:
- `likes` (Integer)
- `downloads` (Integer)
- `popularity_pace` (Float)
- `created_at` (Timestamp)
- `modified_at` (Timestamp)
- `source_name` (Text)
- `name` (Text)

### Custom Optimization Example

Create a custom optimization that vacuums and analyzes tables:

```python
from sqlalchemy import text
from embedding_studio.vectordb.pgvector.optimization import PgvectorObjectsOptimization

class VacuumAnalyzeOptimization(PgvectorObjectsOptimization):
    def __init__(self):
        super().__init__(name="VacuumAnalyzeOptimization")

    def _get_statement(self, tablename: str):
        return text(f"VACUUM ANALYZE {tablename}")

# Add to vector database
vectordb.add_optimization(VacuumAnalyzeOptimization())
```

### Optimization in Fine-Tuning Plugins

Specify optimizations in your fine-tuning plugins:

```python
def get_vectordb_optimizations(self) -> List[Optimization]:
    """
    Return a list of vector DB optimizations to apply after training.
    """
    return [
        CreateOrderingIndexesOptimization(),
        VacuumAnalyzeOptimization(),
    ]
```

## 5. Query Execution Flow

Understanding the internal query execution helps with debugging and optimization:

1. **Preparation**:
   - Vector validation (dimensions check)
   - Payload filter conversion to SQL
   - Sort options preparation

2. **SQL Generation**:
   - For similarity queries, Embedding Studio dynamically selects the appropriate SQL function:
     - `simple_so_*` for simple similarity-ordered queries
     - `advanced_*` for queries with payload filters
     - `*_v_*` variants when vectors need to be returned

3. **Execution**:
   - Query runs against the PostgreSQL database using pgvector operators
   - Results are grouped by object with appropriate distance calculation
   - Pagination is applied (limit/offset)

4. **Result Processing**:
   - Results are converted from database rows to `Object` instances
   - Distance scores are included in `ObjectWithDistance` instances
   - Original IDs are substituted where appropriate

## 6. Complete Configuration Example

Here's a comprehensive example for setting up and configuring a vector database:

```python
import os
from dotenv import load_dotenv
import sqlalchemy
from sqlalchemy import create_engine
import pymongo
from embedding_studio.vectordb.pgvector.vectordb import PgvectorDb
from embedding_studio.models.embeddings.models import EmbeddingModelInfo, MetricType
from plugins.custom.optimizations.indexes import CreateOrderingIndexesOptimization

# Load environment variables
load_dotenv()

# PostgreSQL connection
pg_host = os.getenv("POSTGRES_HOST", "localhost")
pg_port = os.getenv("POSTGRES_PORT", "5432")
pg_user = os.getenv("POSTGRES_USER", "embedding_studio")
pg_password = os.getenv("POSTGRES_PASSWORD", "password")
pg_database = os.getenv("POSTGRES_DB", "embedding_studio")

pg_connection_string = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"
pg_engine = create_engine(pg_connection_string)

# MongoDB connection
mongo_host = os.getenv("EMBEDDINGS_MONGO_HOST", "localhost")
mongo_port = int(os.getenv("EMBEDDINGS_MONGO_PORT", "27017"))
mongo_username = os.getenv("EMBEDDINGS_MONGO_USERNAME", "root")
mongo_password = os.getenv("EMBEDDINGS_MONGO_PASSWORD", "mongopassword")
mongo_database = os.getenv("EMBEDDINGS_MONGO_DB_NAME", "embedding_studio")

mongo_client = pymongo.MongoClient(
    host=mongo_host,
    port=mongo_port,
    username=mongo_username,
    password=mongo_password,
)
mongo_db = mongo_client[mongo_database]

# Initialize vector database
vectordb = PgvectorDb(
    pg_database=pg_engine,
    embeddings_mongo_database=mongo_db,
    prefix="my_app",
)

# Add optimizations
vectordb.add_optimization(CreateOrderingIndexesOptimization())

# Define an embedding model
embedding_model = EmbeddingModelInfo(
    id="text-embedding-ada-002",
    dimensions=1536,
    metric_type=MetricType.COSINE,
)

# Create or get a collection
collection = vectordb.get_or_create_collection(embedding_model)

# Create a query collection for storing user queries
query_collection = vectordb.get_or_create_query_collection(embedding_model)

# Set as active "blue" collection
vectordb.set_blue_collection("text-embedding-ada-002")

# Apply optimizations
vectordb.apply_optimizations()
```

By following this guide, you should be able to fully configure, optimize, and utilize the vector database capabilities of Embedding Studio for your specific use case.