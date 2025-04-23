# Items Data Flow: From Initiation to DB Storage

## Introduction

Embedding Studio relies on a sophisticated data pipeline to transform raw content into searchable vector embeddings. This tutorial explains the complete data flow from initiating a data upload to the final storage in the vector database. Understanding this pipeline is crucial for effective system integration and optimization.

## Architecture Overview

The items data flow in Embedding Studio follows these key stages:

1. **Initiation**: Creating upsertion tasks with item metadata
2. **Downloading**: Retrieving item content from source systems
3. **Splitting**: Breaking content into appropriate chunks
4. **Preprocessing**: Normalizing and enriching text for embedding
5. **Inference**: Generating vector embeddings using models
6. **Upsertion**: Storing vectors and metadata in the vector database

```
┌───────────┐     ┌────────────┐     ┌───────────┐     ┌──────────────┐     ┌────────────┐     ┌─────────────┐
│ Initiate  │────▶│  Download  │────▶│   Split   │────▶│  Preprocess  │────▶│  Inference │────▶│   Upsert    │
└───────────┘     └────────────┘     └───────────┘     └──────────────┘     └────────────┘     └─────────────┘
```

## 1. Initiation

The data flow begins by creating an upsertion task. This is done through the API endpoints defined in `embedding_studio/api/api_v1/endpoints/upsert.py`.

### API Endpoints

#### Create Upsertion Task

```python
@router.post("/run", response_model=UpsertionTaskResponse)
def create_upsertion_task(body: UpsertionTaskRunRequest) -> Any:
    """Create a new upsertion task."""
```

This endpoint creates a task to process and index a batch of items.

### Example Request

```python
import requests

response = requests.post(
    "https://api.embeddingstudio.com/api/v1/embeddings/upsertion-tasks/run",
    json={
        "task_id": "upsert_task_001",  # Optional custom ID
        "items": [
            {
                "object_id": "product_123",
                "payload": {
                    "title": "Ergonomic Office Chair",
                    "description": "Adjustable height and lumbar support",
                    "category": "furniture",
                    "price": 299.99
                },
                "item_info": {
                    "source_name": "product_catalog",
                    "file_path": "products/office/chairs.json"
                }
            }
        ]
    }
)

task = response.json()
print(f"Task ID: {task['id']}, Status: {task['status']}")
```

### Key Components

- **object_id**: Unique identifier for the item
- **payload**: Content and metadata that will be stored and made searchable
- **item_info**: Information about where to find the actual content

## 2. Downloading

Once the task is created, the upsertion worker processes it by first downloading the items from their source location. This is handled by the `download_items` function in `embedding_studio/workers/upsertion/utils/upsertion_stages.py`.

```python
@retry_function(max_attempts=10, wait_time_seconds=30)
def download_items(items: List[DataItem], data_loader: DataLoader) -> List[DownloadedItem]:
    """Download a list of items using the specified DataLoader."""
```

### Data Loaders

Embedding Studio supports various data sources through specialized loaders:

- **S3 Loaders**: For content stored in AWS S3 buckets
- **GCP Loaders**: For content stored in Google Cloud Storage
- **PostgreSQL Loaders**: For data stored in PostgreSQL databases
- **Custom Loaders**: For integration with proprietary data stores

Each loader implements the `DataLoader` abstract base class from `embedding_studio/data_storage/loaders/data_loader.py`.

### Example: S3 Text Loader

```python
# Configure an AWS S3 text loader
s3_loader = AwsS3TextLoader(
    aws_access_key_id="YOUR_ACCESS_KEY",
    aws_secret_access_key="YOUR_SECRET_KEY",
    encoding="utf-8"
)

# Download items
downloaded_items = s3_loader.load_items([
    S3FileMeta(bucket="my-content", file="products/descriptions/123.txt")
])
```

## 3. Splitting

After downloading, content is split into appropriate chunks using an `ItemSplitter`. This is handled by the `split_items` function in `embedding_studio/workers/upsertion/utils/upsertion_stages.py`.

```python
def split_items(
    items: List[DownloadedItem],
    item_splitter: ItemSplitter,
    preprocessor: ItemsDatasetDictPreprocessor,
) -> Tuple[List[Any], Dict[str, List[int]], List[Tuple[DownloadedItem, str]]]:
    """Split each item into parts using the specified ItemSplitter."""
```

### Splitting Strategies

Different content types require different splitting strategies:

- **Text Splitting**: Divide long texts into semantic chunks (paragraphs, sentences)
- **Document Splitting**: Process structured documents (PDF, Word) into logical sections
- **Image Splitting**: Extract regions or features from images

### Example: Text Splitter

```python
# Configure a text splitter with custom settings
text_splitter = TextItemSplitter(
    chunk_size=512,      # Maximum tokens per chunk
    chunk_overlap=50,    # Overlap between chunks
    separator="\n\n"     # Split on paragraph breaks
)

# Split downloaded content
parts, object_to_parts, failed = split_items(downloaded_items, text_splitter, preprocessor)
```

## 4. Preprocessing

Before generating embeddings, content is preprocessed to normalize and enhance it. This is typically done by the `ItemsDatasetDictPreprocessor`.

### Preprocessing Operations

- **Text Normalization**: Lowercase, remove special characters, expand contractions
- **Tokenization**: Split text into tokens or words
- **Stop Word Removal**: Filter out common words with low semantic value
- **Stemming/Lemmatization**: Reduce words to their base forms
- **Entity Recognition**: Identify and highlight named entities

### Example: Text Preprocessor

```python
# Configure a text preprocessor
preprocessor = TextItemsPreprocessor(
    normalize_case=True,
    remove_punctuation=True,
    expand_contractions=True
)

# Apply preprocessing to split items
preprocessed_items = [preprocessor(item) for item in parts]
```

## 5. Inference

With preprocessed content ready, the system generates vector embeddings using the specified embedding model. This is handled by the `run_inference` function in `embedding_studio/workers/upsertion/utils/upsertion_stages.py`.

```python
@retry_function(max_attempts=10, wait_time_seconds=2)
def run_inference(
    items_data: List[Any],
    inference_client: TritonClient,
) -> np.ndarray:
    """Run inference on the given items data using the specified TritonClient."""
```

### Inference Process

1. Items are batched for efficient processing
2. Each batch is sent to the inference service (Triton)
3. The service returns vector embeddings
4. Vectors are collected and combined into a single array

### Example: Running Inference

```python
# Configure inference client
inference_client = plugin.get_inference_client_factory().get_client(
    embedding_model_id="text-embedding-v1"
)

# Generate embeddings
vectors = run_inference(preprocessed_items, inference_client)
print(f"Generated {len(vectors)} vectors with dimension {vectors.shape[1]}")
```

## 6. Upsertion

Finally, the generated vectors are stored in the vector database along with metadata. This is handled by the `upload_vectors` function in `embedding_studio/workers/upsertion/utils/upsertion_stages.py`.

```python
@retry_function(max_attempts=10, wait_time_seconds=30)
def upload_vectors(
    items: List[DownloadedItem],
    vectors: np.ndarray,
    object_to_parts: Dict[str, List[int]],
    collection: Collection,
):
    """Upload vectors to the specified collection."""
```

### Upsertion Process

1. Create `Object` instances with their associated vector parts
2. For each object, include an average vector for overall similarity search
3. Store payload and metadata alongside vectors
4. Optionally delete previous versions of the vectors

### Vector Database Structure

In the pgvector implementation:

- **DbObject**: Table storing object metadata and payload
- **DbObjectPart**: Table storing individual vector parts with their IDs
- **Indexes**: HNSW indexes for efficient similarity search

### Example: Upserting to Vector DB

```python
# Create objects with vectors
objects = []
for item in items:
    parts = []
    for part_index in object_to_parts[item.meta.object_id]:
        parts.append(
            ObjectPart(
                vector=vectors[part_index].tolist(),
                part_id=f"{item.meta.object_id}:{part_index}"
            )
        )
    
    # Add an average vector for overall similarity
    average_vector = np.mean([vectors[i] for i in object_to_parts[item.meta.object_id]], axis=0)
    parts.append(
        ObjectPart(
            vector=average_vector.tolist(),
            part_id=f"{item.meta.object_id}:average",
            is_average=True
        )
    )
    
    objects.append(
        Object(
            object_id=item.meta.object_id,
            parts=parts,
            payload=item.meta.payload,
            storage_meta=item.meta.dict()
        )
    )

# Upsert to the collection
collection.upsert(objects)
```

## Error Handling and Retries

The upsertion pipeline includes robust error handling and retry logic:

- **Downloading**: Retries up to 10 times with 30-second delays
- **Inference**: Retries up to 10 times with 2-second delays
- **Upsertion**: Retries up to 10 times with 30-second delays

Failed items are tracked and reported in the task status, allowing for monitoring and debugging.

## Monitoring Task Progress

You can monitor the progress of an upsertion task by querying its status:

```python
import requests

response = requests.get(
    f"https://api.embeddingstudio.com/api/v1/embeddings/upsertion-tasks/{task_id}"
)

task_status = response.json()
print(f"Task ID: {task_status['id']}")
print(f"Status: {task_status['status']}")
print(f"Processed: {task_status['processed_count']} items")
print(f"Failed: {len(task_status['failed_items'])} items")
```

## Handling Large Datasets

For large datasets, the upsertion process uses batching:

1. Items are processed in batches defined by `UPSERTION_BATCH_SIZE`
2. Inference uses sub-batching with `UPSERTION_INFERENCE_BATCH_SIZE`
3. Each batch is processed separately to manage memory usage
4. Progress is tracked at the individual item level
