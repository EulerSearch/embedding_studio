# Data Sources Integration Tutorial for Embedding Studio

This tutorial will guide you through integrating various data sources with Embedding Studio. You'll learn how to load data from PostgreSQL databases, cloud storage providers like Amazon S3 and Google Cloud Storage, and how to combine multiple data sources in a unified architecture.

## 1. Introduction to Data Loaders

Data loaders are the foundation of Embedding Studio's data integration layer. They handle:

- Connecting to data sources
- Fetching raw data
- Transforming data into a format suitable for embedding
- Managing pagination and batching for large datasets

All data loaders implement the `DataLoader` abstract base class, which defines a standard interface:

```python
class DataLoader(ABC):
    @property
    @abstractmethod
    def item_meta_cls(self) -> Type[ItemMeta]:
        """Returns the ItemMeta class used by this loader."""
        
    @abstractmethod
    def load(self, items_data: List[ItemMeta]) -> Dataset:
        """Loads items as a Hugging Face Dataset."""
        
    @abstractmethod
    def load_items(self, items_data: List[ItemMeta]) -> List[DownloadedItem]:
        """Loads specific items by their metadata."""
        
    @abstractmethod
    def _load_batch_with_offset(self, offset: int, batch_size: int, **kwargs) -> List[DownloadedItem]:
        """Loads a batch of items with pagination."""
        
    @abstractmethod
    def total_count(self, **kwargs) -> Optional[int]:
        """Returns the total count of available items."""
        
    def load_all(self, batch_size: int, **kwargs) -> Generator[List[DownloadedItem], None, None]:
        """Generator that loads all items in batches."""
```

## 2. Core Data Loader Components

### ItemMeta Classes

Each data loader has a corresponding `ItemMeta` class that stores metadata about items:

- `S3FileMeta`: Metadata for files in Amazon S3 buckets
- `GCPFileMeta`: Metadata for files in Google Cloud Storage
- `PgsqlFileMeta`: Metadata for rows in PostgreSQL tables

These classes provide a consistent way to identify and reference items across different data sources:

```python
class BucketFileMeta(ItemMeta):
    bucket: str             # Storage bucket name
    file: str               # File path within the bucket
    index: Optional[int]    # Optional index for sub-items
    
    @property
    def derived_id(self) -> str:
        """Creates a unique identifier from bucket and file."""
        if self.index is None:
            return f"{self.bucket}/{self.file}"
        else:
            return f"{self.bucket}/{self.file}:{self.index}"
```

### Query Generators

SQL-based loaders use query generators to build database queries:

```python
class QueryGenerator:
    def __init__(self, table_name: str, engine: Engine) -> None:
        """Initialize with a table name and database engine."""
        self.table_name = table_name
        self.engine = engine
        self.metadata = MetaData()

    def fetch_all(self, row_ids: List[str]) -> Select:
        """Generate a query to fetch multiple rows by ID."""
        
    def one(self, row_id: str) -> Select:
        """Generate a query to fetch a single row by ID."""
        
    def all(self, offset: int, batch_size: int) -> Select:
        """Generate a paginated query."""
        
    def count(self) -> Select:
        """Generate a count query."""
```

## 3. PostgreSQL Integration

PostgreSQL integration allows you to load data directly from relational databases.

### Basic Setup

```python
from embedding_studio.data_storage.loaders.sql.pgsql.pgsql_text_loader import PgsqlTextLoader
from embedding_studio.data_storage.loaders.sql.query_generator import QueryGenerator

# Define database connection
connection_string = "postgresql://username:password@hostname:5432/database"

# Create loader
data_loader = PgsqlTextLoader(
    connection_string=connection_string,
    query_generator=QueryGenerator,  # Use default query generator
    text_column="content"            # Column containing text data
)
```

### Custom Query Generator

For more advanced database schemas, create a custom query generator:

```python
from sqlalchemy import Engine, MetaData, Select, Table, func, select
from embedding_studio.data_storage.loaders.sql.query_generator import QueryGenerator

class ArticleQueryGenerator(QueryGenerator):
    def __init__(self, engine: Engine) -> None:
        super().__init__("articles", engine)
        self.metadata = MetaData()
        self.articles_table = None
        self.authors_table = None
        self.categories_table = None
    
    def _init_tables(self):
        if self.articles_table is None:
            # Load tables with reflection
            self.articles_table = Table(self.table_name, self.metadata, autoload_with=self.engine)
            self.authors_table = Table("authors", self.metadata, autoload_with=self.engine)
            self.categories_table = Table("categories", self.metadata, autoload_with=self.engine)
    
    def all(self, offset: int, batch_size: int) -> Select:
        """Join articles with authors and categories."""
        self._init_tables()
        return (
            select(
                self.articles_table,
                self.authors_table.c.name.label("author_name"),
                self.categories_table.c.name.label("category_name"),
                # Create a rich text field combining multiple columns
                func.concat(
                    "Title: ", self.articles_table.c.title, "\n",
                    "Author: ", self.authors_table.c.name, "\n",
                    "Category: ", self.categories_table.c.name, "\n",
                    "Content: ", self.articles_table.c.content
                ).label("rich_text")
            )
            .join(
                self.authors_table,
                self.articles_table.c.author_id == self.authors_table.c.id
            )
            .join(
                self.categories_table,
                self.articles_table.c.category_id == self.categories_table.c.id
            )
            .order_by(self.articles_table.c.id)
            .limit(batch_size)
            .offset(offset)
        )
```

### Working with Multiple Text Columns

When your data has multiple text fields, use `PgsqlMultiTextColumnLoader`:

```python
from embedding_studio.data_storage.loaders.sql.pgsql.pgsql_multi_text_column_loader import PgsqlMultiTextColumnLoader

# Load multiple text columns
data_loader = PgsqlMultiTextColumnLoader(
    connection_string=connection_string,
    query_generator=QueryGenerator,
    text_columns=["title", "summary", "content"]  # Process all these columns
)
```

### Loading JSON Data

For JSON data stored in PostgreSQL:

```python
from embedding_studio.data_storage.loaders.sql.pgsql.pgsql_jsonb_loader import PgsqlJSONBLoader

# Load JSONB data
data_loader = PgsqlJSONBLoader(
    connection_string=connection_string,
    query_generator=QueryGenerator,
    jsonb_column="data",              # Column containing JSONB
    fields_to_keep=["title", "tags"]  # Extract only these fields
)
```

## 4. Amazon S3 Integration

Amazon S3 integration allows you to work with files stored in S3 buckets.

### Authentication

```python
from embedding_studio.data_storage.loaders.cloud_storage.s3.s3_text_loader import AwsS3TextLoader

# With explicit credentials
data_loader = AwsS3TextLoader(
    aws_access_key_id="YOUR_ACCESS_KEY",
    aws_secret_access_key="YOUR_SECRET_KEY"
)

# With IAM role (for EC2/ECS/Lambda)
data_loader = AwsS3TextLoader(
    role_arn="arn:aws:iam::123456789012:role/your-role",
    external_id="optional-external-id"  # If required by the role
)

# With anonymous access (for public buckets)
data_loader = AwsS3TextLoader(
    use_system_info=True  # Use credentials from environment or instance profile
)
```

### Loading Text Files

```python
from embedding_studio.data_storage.loaders.cloud_storage.s3.item_meta import S3FileMeta

# Create loader
text_loader = AwsS3TextLoader(
    aws_access_key_id="YOUR_ACCESS_KEY",
    aws_secret_access_key="YOUR_SECRET_KEY",
    encoding="utf-8"  # Specify text encoding
)

# Load specific text files
items = text_loader.load_items([
    S3FileMeta(bucket="my-data-bucket", file="documents/doc1.txt"),
    S3FileMeta(bucket="my-data-bucket", file="documents/doc2.txt")
])

# Work with loaded items
for item in items:
    print(f"ID: {item.id}")
    print(f"Content: {item.data[:100]}...")  # First 100 chars
```

### Loading JSON Files

```python
from embedding_studio.data_storage.loaders.cloud_storage.s3.s3_json_loader import AwsS3JSONLoader

# Create loader with field filtering
json_loader = AwsS3JSONLoader(
    aws_access_key_id="YOUR_ACCESS_KEY",
    aws_secret_access_key="YOUR_SECRET_KEY",
    fields_to_keep=["id", "title", "description", "tags"],
    encoding="utf-8"
)

# Load JSON files
items = json_loader.load_items([
    S3FileMeta(bucket="my-data-bucket", file="data/item1.json"),
    S3FileMeta(bucket="my-data-bucket", file="data/item2.json")
])
```

### Loading Images

```python
from embedding_studio.data_storage.loaders.cloud_storage.s3.s3_image_loader import AwsS3ImageLoader
from PIL import Image

# Create image loader
image_loader = AwsS3ImageLoader(
    aws_access_key_id="YOUR_ACCESS_KEY",
    aws_secret_access_key="YOUR_SECRET_KEY"
)

# Load image files (returns PIL Image objects)
items = image_loader.load_items([
    S3FileMeta(bucket="my-data-bucket", file="images/img1.jpg"),
    S3FileMeta(bucket="my-data-bucket", file="images/img2.png")
])

# Process images
for item in items:
    img = item.data  # PIL Image object
    width, height = img.size
    print(f"Image {item.id}: {width}x{height}")
```

### Loading Entire Buckets

```python
# Load all text files from a bucket in batches
for batch in text_loader.load_all(batch_size=100, buckets=["my-data-bucket"]):
    for item in batch:
        # Process each item in the batch
        process_text(item.data)
```

## 5. Google Cloud Storage Integration

Google Cloud Storage integration is similar to Amazon S3 but uses GCP-specific authentication.

### Authentication

```python
from embedding_studio.data_storage.loaders.cloud_storage.gcp.gcp_text_loader import GCPTextLoader

# With service account credentials file
data_loader = GCPTextLoader(
    credentials_path="./path/to/service-account.json",
    use_system_info=False
)

# With application default credentials
data_loader = GCPTextLoader(
    use_system_info=True  # Use ADC from environment
)
```

### Loading Text Files

```python
from embedding_studio.data_storage.loaders.cloud_storage.gcp.item_meta import GCPFileMeta

# Create loader
text_loader = GCPTextLoader(
    use_system_info=True,
    encoding="utf-8"
)

# Load specific text files
items = text_loader.load_items([
    GCPFileMeta(bucket="my-gcp-bucket", file="documents/doc1.txt"),
    GCPFileMeta(bucket="my-gcp-bucket", file="documents/doc2.txt")
])
```

### Loading JSON and Images

The pattern is similar to S3 loaders:

```python
from embedding_studio.data_storage.loaders.cloud_storage.gcp.gcp_json_loader import GCPJSONLoader
from embedding_studio.data_storage.loaders.cloud_storage.gcp.gcp_image_loader import GCPImageLoader

# JSON loader
json_loader = GCPJSONLoader(
    use_system_info=True,
    fields_to_keep=["name", "description", "metadata"],
    encoding="utf-8"
)

# Image loader
image_loader = GCPImageLoader(
    use_system_info=True
)
```

## 6. Aggregated Data Loading

The `AggregatedDataLoader` is a powerful component that lets you combine multiple data loaders into a single unified interface. This is particularly useful when your data is distributed across different systems or when you want to create embeddings from heterogeneous data sources.

### Basic Usage

Here's how to combine multiple data sources:

```python
from embedding_studio.data_storage.loaders.aggregated_data_loader import AggregatedDataLoader
from embedding_studio.data_storage.loaders.item_meta import ItemMetaWithSourceInfo

class CustomItemMeta(ItemMetaWithSourceInfo):
    """Custom item metadata with source tracking."""
    pass

# Create specialized loaders for each source
postgres_loader = PgsqlTextLoader(
    connection_string="postgresql://user:pass@host:5432/db",
    query_generator=QueryGenerator,
    text_column="content"
)

s3_loader = AwsS3TextLoader(
    aws_access_key_id="YOUR_ACCESS_KEY",
    aws_secret_access_key="YOUR_SECRET_KEY",
    encoding="utf-8"
)

gcp_loader = GCPTextLoader(
    use_system_info=True,
    encoding="utf-8"
)

# Combine loaders into a single interface
aggregated_loader = AggregatedDataLoader(
    {
        "postgres": postgres_loader,
        "s3": s3_loader,
        "gcp": gcp_loader
    },
    item_meta_cls=CustomItemMeta
)

# Load items from different sources through one interface
items = aggregated_loader.load_items([
    CustomItemMeta(source_name="postgres", object_id="row_123"),
    CustomItemMeta(source_name="s3", bucket="my-bucket", file="doc.txt"),
    CustomItemMeta(source_name="gcp", bucket="gcp-bucket", file="file.txt")
])
```

### Real-World Example with PostgreSQL Sources

A common scenario is loading data from multiple database tables using specialized query generators. Here's a real-world example that loads data from both models and datasets tables in PostgreSQL:

```python
from embedding_studio.data_storage.loaders.aggregated_data_loader import AggregatedDataLoader
from embedding_studio.data_storage.loaders.sql.pgsql.pgsql_multi_text_column_loader import PgsqlMultiTextColumnLoader
from plugins.custom.data_storage.loaders.hf_models_query_generator import HugsearchModelsQueryGenerator
from plugins.custom.data_storage.loaders.hf_datasets_query_generator import HugsearchDatasetsQueryGenerator
from plugins.custom.data_storage.loaders.item_meta import PgsqlItemMetaWithSourceInfo

# Database connection string
connection_string = "postgresql://username:password@hostname:5432/database"

# Create an aggregated loader that combines models and datasets sources
self.data_loader = AggregatedDataLoader(
    {
        "hf_models": PgsqlMultiTextColumnLoader(
            connection_string=connection_string,
            query_generator=HugsearchModelsQueryGenerator,
            text_columns=["id", "readme", "tags"],
        ),
        "hf_datasets": PgsqlMultiTextColumnLoader(
            connection_string=connection_string,
            query_generator=HugsearchDatasetsQueryGenerator,
            text_columns=["id", "readme", "tags"],
        ),
    },
    item_meta_cls=PgsqlItemMetaWithSourceInfo,
)
```

In this example:

1. Two specialized query generators (`HugsearchModelsQueryGenerator` and `HugsearchDatasetsQueryGenerator`) are used to build tailored SQL queries for different tables
2. Both loaders use the `PgsqlMultiTextColumnLoader` to load multiple text columns from each table
3. The `PgsqlItemMetaWithSourceInfo` class tracks which source each item came from
4. All these loaders are combined into a single `AggregatedDataLoader` with source name keys

### How AggregatedDataLoader Works

The `AggregatedDataLoader` routes requests to the appropriate loader based on the `source_name` property in each item's metadata:

1. When you call `load_items()`, it groups items by their `source_name`
2. It then dispatches each group to the corresponding loader
3. Results from all loaders are combined and returned as a single list
4. The same process happens for all other loader methods like `load()` and `load_all()`

This allows transparent access to multiple data sources while maintaining the standard `DataLoader` interface.

### Custom ItemMeta with Source Information

To use `AggregatedDataLoader`, your ItemMeta class must extend `ItemMetaWithSourceInfo`, which adds a `source_name` field:

```python
class PgsqlItemMetaWithSourceInfo(ItemMetaWithSourceInfo):
    """Metadata for a PostgreSQL item with source tracking."""
    
    @property
    def derived_id(self) -> str:
        """
        Return unique ID combining source name and object ID.
        Format: source_name:object_id
        """
        return f"{self.source_name}:{self.object_id}"
```

The `source_name` field tells the aggregated loader which sub-loader should handle each item.

### Benefits of Using AggregatedDataLoader

- **Unified Interface**: Access multiple data sources through a single consistent API
- **Source Tracking**: Each item maintains information about its origin
- **Scalability**: Easily add new data sources without changing client code
- **Heterogeneous Data**: Combine different types of data (text, images, structured) in one pipeline

## 7. Using Data Loaders in Plugins

To integrate a data loader into your Embedding Studio plugin:

```python
from embedding_studio.core.plugin import FineTuningMethod
from embedding_studio.data_storage.loaders.data_loader import DataLoader

class MyPlugin(FineTuningMethod):
    def __init__(self):
        # Configure your data loader in the plugin initialization
        self.data_loader = PgsqlTextLoader(
            connection_string="postgresql://user:pass@host:5432/db",
            query_generator=QueryGenerator,
            text_column="content"
        )
        # ... other initializations
    
    def get_data_loader(self) -> DataLoader:
        """Return the configured data loader."""
        return self.data_loader
```

## 8. Best Practices

### Error Handling

Data loaders include built-in retry logic for transient failures:

```python
# Configure custom retry settings
from embedding_studio.workers.fine_tuning.utils.config import RetryConfig, RetryParams

retry_config = RetryConfig(
    default_params=RetryParams(
        max_attempts=5,
        wait_time_seconds=2.0
    )
)

data_loader = PgsqlTextLoader(
    connection_string=connection_string,
    query_generator=QueryGenerator,
    retry_config=retry_config  # Custom retry configuration
)
```

### Credentials Management

Always store credentials securely:

1. Use environment variables when possible
2. Consider AWS Secrets Manager, GCP Secret Manager, or HashiCorp Vault
3. For development, use `.env` files (but exclude from version control)
4. For production, use IAM roles/service accounts with minimal permissions

### Performance Optimization

For large datasets:

1. Use appropriate batch sizes (typically 100-1000 items)
2. Consider implementing pagination in custom query generators
3. Monitor memory usage when processing large files
4. For very large datasets, use streaming approaches rather than loading all at once

```python
# Process a large bucket in manageable batches
for batch in loader.load_all(batch_size=500, buckets=["large-data-bucket"]):
    # Process each batch incrementally
    process_batch(batch)
    # Explicitly free memory if needed
    del batch
    gc.collect()
```

### Caching Strategies

For infrequently changing data:

1. Implement a local cache of downloaded files
2. Use ETags or last-modified timestamps to detect changes
3. Consider Redis or a similar service for distributed caching
4. For cloud storage, check if CDN caching is appropriate

```python
# Simple caching example (for illustration)
class CachedLoader:
    def __init__(self, loader, cache_dir):
        self.loader = loader
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def load_items(self, items_data):
        results = []
        for item in items_data:
            cache_path = os.path.join(self.cache_dir, f"{item.id}.cache")
            if os.path.exists(cache_path):
                # Load from cache
                with open(cache_path, 'rb') as f:
                    results.append(pickle.load(f))
            else:
                # Load from source
                loaded_item = self.loader.load_items([item])[0]
                # Cache for next time
                with open(cache_path, 'wb') as f:
                    pickle.dump(loaded_item, f)
                results.append(loaded_item)
        return results
```
