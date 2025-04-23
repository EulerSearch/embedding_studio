# Data Management in Embedding Studio: Upsert, Delete, and Reindex

Embedding Studio provides powerful APIs for managing vector data throughout its lifecycle. This guide covers the three primary data management operations:

1. **Upserting**: Adding or updating items in your vector database
2. **Deleting**: Removing items from your vector database
3. **Reindexing**: Transferring data between embedding models

## Prerequisites

Before using these APIs, ensure you have:

- An Embedding Studio instance up and running
- Knowledge of your embedding model IDs
- Appropriate permissions to modify collections

## API Base URL

All examples in this guide use the following base URL:

```
https://api.embeddingstudio.com/api/v1
```

Replace this with your actual Embedding Studio API endpoint.

## 1. Upserting Data

Upsertion is the process of adding new items or updating existing items in your vector database. 

### Upsertion API Endpoint

```
POST /embeddings/upsertion-tasks/run
```

### Request Format

```json
{
  "task_id": "optional_custom_task_id",
  "items": [
    {
      "object_id": "unique_item_id",
      "payload": {
        "field1": "value1",
        "field2": "value2",
        "nested": {
          "field3": "value3"
        }
      },
      "item_info": {
        "source_name": "data_source",
        "additional_metadata": "value"
      }
    }
  ]
}
```

### Key Fields

- **task_id**: (Optional) Custom identifier for the task
- **items**: Array of items to upsert
  - **object_id**: Unique identifier for each item
  - **payload**: Content and metadata that will be stored and made searchable
  - **item_info**: Information about the data source

### Example: Upserting Product Data

```python
import requests
import json

url = "https://api.embeddingstudio.com/api/v1/embeddings/upsertion-tasks/run"

payload = {
    "task_id": "product_upsert_20250422",
    "items": [
        {
            "object_id": "product-12345",
            "payload": {
                "title": "Ergonomic Office Chair",
                "description": "Adjustable height with lumbar support for comfortable all-day use. Features breathable mesh back and cushioned seat.",
                "category": "furniture",
                "price": 299.99,
                "tags": ["office", "furniture", "ergonomic", "chair"],
                "attributes": {
                    "color": "black",
                    "material": "mesh",
                    "weight_capacity": "300lbs"
                }
            },
            "item_info": {
                "source_name": "product_catalog",
                "last_updated": "2025-04-20T14:30:00Z"
            }
        },
        {
            "object_id": "product-12346",
            "payload": {
                "title": "Adjustable Standing Desk",
                "description": "Electric height-adjustable desk with memory settings. Smooth transition between sitting and standing positions.",
                "category": "furniture",
                "price": 549.99,
                "tags": ["office", "furniture", "desk", "standing desk"],
                "attributes": {
                    "color": "walnut",
                    "material": "engineered wood",
                    "weight_capacity": "200lbs"
                }
            },
            "item_info": {
                "source_name": "product_catalog",
                "last_updated": "2025-04-21T10:15:00Z"
            }
        }
    ]
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, headers=headers, data=json.dumps(payload))
print(response.json())
```

### Response

```json
{
  "id": "product_upsert_20250422",
  "status": "pending",
  "created_at": "2025-04-22T15:30:00Z",
  "updated_at": "2025-04-22T15:30:00Z",
  "processed_count": 0,
  "failed_items": [],
  "embedding_model_id": "text-embedding-ada-002",
  "fine_tuning_method": "openai"
}
```

### Monitoring Upsertion Progress

To check the status of your upsertion task:

```python
import requests

task_id = "product_upsert_20250422"
url = f"https://api.embeddingstudio.com/api/v1/embeddings/upsertion-tasks/{task_id}"

response = requests.get(url)
task_status = response.json()

print(f"Task ID: {task_status['id']}")
print(f"Status: {task_status['status']}")
print(f"Processed: {task_status['processed_count']} items")
print(f"Failed: {len(task_status['failed_items'])} items")
```

### Upserting Categories

For category data, use the categories-specific endpoint:

```
POST /embeddings/upsertion-tasks/categories/run
```

The request format is identical to regular upsertion, but it operates on the categories collection.

## 2. Deleting Data

Deletion removes items from your vector database.

### Deletion API Endpoint

```
POST /embeddings/deletion-tasks/run
```

### Request Format

```json
{
  "task_id": "optional_custom_task_id",
  "object_ids": ["id1", "id2", "id3"]
}
```

### Key Fields

- **task_id**: (Optional) Custom identifier for the task
- **object_ids**: Array of object IDs to delete

### Example: Deleting Products

```python
import requests
import json

url = "https://api.embeddingstudio.com/api/v1/embeddings/deletion-tasks/run"

payload = {
    "task_id": "product_delete_20250422",
    "object_ids": ["product-12345", "product-12346"]
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, headers=headers, data=json.dumps(payload))
print(response.json())
```

### Response

```json
{
  "id": "product_delete_20250422",
  "status": "pending",
  "created_at": "2025-04-22T16:45:00Z",
  "updated_at": "2025-04-22T16:45:00Z",
  "embedding_model_id": "text-embedding-ada-002",
  "fine_tuning_method": "openai",
  "object_ids": ["product-12345", "product-12346"],
  "failed_item_ids": []
}
```

### Monitoring Deletion Progress

To check the status of your deletion task:

```python
import requests

task_id = "product_delete_20250422"
url = f"https://api.embeddingstudio.com/api/v1/embeddings/deletion-tasks/{task_id}"

response = requests.get(url)
task_status = response.json()

print(f"Task ID: {task_status['id']}")
print(f"Status: {task_status['status']}")
print(f"Failed: {len(task_status['failed_item_ids'])} items")
```

### Deleting Categories

For category data, use the categories-specific endpoint:

```
POST /embeddings/deletion-tasks/categories/run
```

The request format is identical to regular deletion, but it operates on the categories collection.

## 3. Reindexing Data

Reindexing transfers data between embedding models. This is useful when:

- Upgrading to a new embedding model
- Creating specialized collections for different purposes
- Migrating between systems

### Reindex API Endpoint

```
POST /internal/reindex-tasks/run
```

### Request Format

```json
{
  "task_id": "optional_custom_task_id",
  "source": {
    "embedding_model_id": "source_model_id"
  },
  "dest": {
    "embedding_model_id": "destination_model_id"
  },
  "deploy_as_blue": false,
  "wait_on_conflict": false
}
```

### Key Fields

- **task_id**: (Optional) Custom identifier for the task
- **source**: Information about the source embedding model
  - **embedding_model_id**: ID of the source model
- **dest**: Information about the destination embedding model
  - **embedding_model_id**: ID of the destination model
- **deploy_as_blue**: Whether to set the destination model as the active (blue) model after reindexing
- **wait_on_conflict**: Whether to wait and retry if there's a conflict with another reindexing task

### Example: Reindexing to a New Model

```python
import requests
import json

url = "https://api.embeddingstudio.com/api/v1/internal/reindex-tasks/run"

payload = {
    "task_id": "reindex_20250422_v1_to_v2",
    "source": {
        "embedding_model_id": "text-embedding-ada-002"
    },
    "dest": {
        "embedding_model_id": "text-embedding-v2"
    },
    "deploy_as_blue": true,
    "wait_on_conflict": true
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, headers=headers, data=json.dumps(payload))
print(response.json())
```

### Response

```json
{
  "id": "reindex_20250422_v1_to_v2",
  "status": "pending",
  "created_at": "2025-04-22T17:30:00Z",
  "updated_at": "2025-04-22T17:30:00Z",
  "source": {
    "embedding_model_id": "text-embedding-ada-002"
  },
  "dest": {
    "embedding_model_id": "text-embedding-v2"
  },
  "deploy_as_blue": true,
  "wait_on_conflict": true,
  "children": [],
  "failed_items": []
}
```

### Monitoring Reindexing Progress

To check the status of your reindexing task:

```python
import requests

task_id = "reindex_20250422_v1_to_v2"
url = f"https://api.embeddingstudio.com/api/v1/internal/reindex-tasks/{task_id}"

response = requests.get(url)
task_status = response.json()

print(f"Task ID: {task_status['id']}")
print(f"Status: {task_status['status']}")
print(f"Source Model: {task_status['source']['embedding_model_id']}")
print(f"Destination Model: {task_status['dest']['embedding_model_id']}")
print(f"Child Tasks: {len(task_status['children'])}")
print(f"Failed Items: {len(task_status['failed_items'])}")
```

## Advanced Operations

### Batch Processing

For large datasets, consider breaking your operations into smaller batches:

```python
import requests
import json
import time

base_url = "https://api.embeddingstudio.com/api/v1"
headers = {
    "Content-Type": "application/json"
}

# Function to process a batch
def upsert_batch(items, batch_id):
    payload = {
        "task_id": f"batch_upsert_{batch_id}",
        "items": items
    }
    
    response = requests.post(
        f"{base_url}/embeddings/upsertion-tasks/run",
        headers=headers,
        data=json.dumps(payload)
    )
    
    return response.json()["id"]

# Load your large dataset
all_items = load_items_from_source()  # Your function to load items

# Process in batches of 100
batch_size = 100
batch_tasks = []

for i in range(0, len(all_items), batch_size):
    batch = all_items[i:i+batch_size]
    task_id = upsert_batch(batch, i // batch_size)
    batch_tasks.append(task_id)
    
    # Optional: Add delay between batches to avoid overloading
    time.sleep(2)

print(f"Created {len(batch_tasks)} batch tasks")
```

### Error Handling and Retries

Implement error handling and retries for robust data operations:

```python
import requests
import json
import time
from requests.exceptions import RequestException

def upsert_with_retry(items, max_retries=3, retry_delay=5):
    payload = {
        "task_id": f"upsert_{int(time.time())}",
        "items": items
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.embeddingstudio.com/api/v1/embeddings/upsertion-tasks/run",
                headers=headers,
                data=json.dumps(payload),
                timeout=30  # Set a reasonable timeout
            )
            
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            return response.json()
            
        except RequestException as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"All {max_retries} attempts failed. Last error: {str(e)}")
                raise
```

### Switching to a Different Embedding Model

To make a specific embedding model the active (blue) collection:

```python
import requests
import json

url = "https://api.embeddingstudio.com/api/v1/internal/collections/blue"

payload = {
    "embedding_model_id": "text-embedding-v2"
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, headers=headers, data=json.dumps(payload))
print(response.json())
```

## Best Practices

### Optimizing Data Management

1. **Batch Operations**:
   - Group operations in batches of 100-1000 items
   - Add short delays between batches to avoid rate limits

2. **Task Management**:
   - Use meaningful task_id values for better tracking
   - Monitor task progress, especially for large operations
   - Store task IDs for potential rollback operations

3. **Data Quality**:
   - Validate data before upserting
   - Include rich metadata in payloads
   - Use consistent object IDs across operations

4. **Error Handling**:
   - Implement retries with exponential backoff
   - Log failed items for manual review
   - Consider separate error recovery processes for critical data

### Reindexing Strategy

When planning a reindexing operation:

1. **Test First**: 
   - Test reindexing with a small subset of data
   - Verify search quality with the new model

2. **Schedule Wisely**:
   - Plan for downtime or reduced performance during reindexing
   - Schedule during low-traffic periods

3. **Phased Rollout**:
   - Don't immediately set deploy_as_blue=true for critical systems
   - Test the new index before switching

4. **Backup**:
   - Consider keeping the old model available as a fallback
