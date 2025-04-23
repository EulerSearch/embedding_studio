# Categories Data Flow in Embedding Studio

## Introduction

Categories play a crucial role in organizing and enhancing search in Embedding Studio. They provide structure to your data and enable faceted navigation, filters, and recommendations. This tutorial covers the complete workflow for managing category data, from creation to storage and utilization.

## Understanding Categories in Embedding Studio

Categories in Embedding Studio are specialized vector collections that:

1. Represent classification or taxonomy nodes
2. Have their own vector embeddings
3. Can be used for automatic classification
4. Enable semantic navigation of content
5. Are stored in a separate vector database collection

## Architecture Overview

The categories data flow follows a similar path to regular items but with dedicated endpoints and collections:

```
┌───────────────┐     ┌────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Category API  │────▶│ Categories │────▶│ Category Vectors │────▶│ Categories DB   │
└───────────────┘     └────────────┘     └──────────────────┘     └─────────────────┘
```

## Category-Specific Endpoints

Embedding Studio provides dedicated endpoints for managing categories in `embedding_studio/api/api_v1/endpoints/upsert.py` and `embedding_studio/api/api_v1/endpoints/delete.py`.

### Upserting Categories

```python
@router.post(
    "/categories/run",
    response_model=UpsertionTaskResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def create_categories_upsertion_task(
    body: UpsertionTaskRunRequest,
) -> Any:
    """Create a new categories upsertion task."""
```

### Deleting Categories

```python
@router.post(
    "/categories/run",
    response_model=DeletionTaskResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def create_categories_deletion_task(
    body: DeletionTaskRunRequest,
) -> Any:
    """Create a new categories deletion task."""
```

## Creating Categories

To create or update categories, you'll use the categories upsertion endpoint:

```python
import requests

response = requests.post(
    "https://api.embeddingstudio.com/api/v1/embeddings/upsertion-tasks/categories/run",
    json={
        "task_id": "category_upsert_001",  # Optional custom ID
        "items": [
            {
                "object_id": "category_electronics",
                "payload": {
                    "name": "Electronics",
                    "description": "Electronic devices and accessories",
                    "parent_id": "category_root",
                    "level": 1,
                    "properties": {
                        "display_order": 1,
                        "is_visible": True
                    }
                },
                "item_info": {
                    "source_name": "taxonomy",
                    "file_path": "categories/main.json"
                }
            },
            {
                "object_id": "category_smartphones",
                "payload": {
                    "name": "Smartphones",
                    "description": "Mobile phones with advanced computing capability",
                    "parent_id": "category_electronics",
                    "level": 2,
                    "properties": {
                        "display_order": 1,
                        "is_visible": True
                    }
                },
                "item_info": {
                    "source_name": "taxonomy",
                    "file_path": "categories/main.json"
                }
            }
        ]
    }
)

task = response.json()
print(f"Category Upsertion Task ID: {task['id']}, Status: {task['status']}")
```

## Category Data Structure

Categories typically include:

1. **Hierarchical Information**:
   - Parent-child relationships
   - Level in the hierarchy
   - Path from root

2. **Descriptive Content**:
   - Name
   - Description
   - Attributes or properties

3. **Metadata**:
   - Display information
   - Visibility flags
   - Sort order

## Differences from Regular Items

While categories follow a similar data flow to regular items, there are important differences:

1. **Separate Collection**: Categories are stored in a dedicated vector collection
2. **Hierarchical Relationships**: Categories maintain parent-child references
3. **Special Indexes**: Categories often have optimized indexes for traversal
4. **Different Usage Patterns**: Categories are used for classification and navigation

## Behind the Scenes: Categories VectorDB

When you create categories, Embedding Studio:

1. Uses a separate vector database collection:
   ```python
   collection = context.categories_vectordb.get_blue_collection()
   ```

2. Creates or retrieves the category collection:
   ```python
   collection_info = collection.get_info()
   ```

3. Processes and embeds category content similar to regular items

## Internal Implementation

The categories upsertion process leverages the same worker infrastructure as regular items:

```python
# From embedding_studio/api/api_v1/endpoints/upsert.py
task = context.upsertion_task.create(
    schema=InternalUpsertionTaskRunRequest(
        embedding_model_id=collection_info.embedding_model.id,
        fine_tuning_method=collection_info.embedding_model.name,
        items=body.items,
    ),
    return_obj=True,
    id=body.task_id,
)

updated_task = create_and_send_task(
    upsertion_worker, task, context.upsertion_task
)
```

The key difference is that it operates on the categories vector database.

## Category Relationships and Hierarchies

Categories in Embedding Studio can form complex hierarchies:

```
Root
├── Electronics
│   ├── Smartphones
│   ├── Laptops
│   └── Audio
│       ├── Headphones
│       └── Speakers
└── Clothing
    ├── Men's
    └── Women's
```

These relationships are maintained through parent-child references in the payload:

```python
{
    "object_id": "category_headphones",
    "payload": {
        "name": "Headphones",
        "parent_id": "category_audio",
        "level": 3,
        "path": ["category_root", "category_electronics", "category_audio"]
    }
}
```

## Using Categories for Classification

One powerful application of categories is automatic item classification:

```python
# Get the category collection
category_collection = context.categories_vectordb.get_blue_collection()

# Generate embedding for an item
item_vector = inference_client.forward_items([item_text])[0]

# Find the most similar categories
results = category_collection.find_similarities(
    query_vector=item_vector.tolist(),
    limit=5
)

# Extract category matches
category_matches = [
    {
        "category_id": obj.object_id,
        "name": obj.payload.get("name"),
        "confidence": 1.0 - distance
    }
    for obj, distance in zip(results.objects, results.distances)
]

print(f"Top categories for item: {category_matches}")
```

## Faceted Navigation with Categories

Categories enable faceted navigation in search interfaces:

```python
# Get item search results
search_results = collection.find_similarities(
    query_vector=query_vector,
    limit=20
)

# Extract category information from payloads
category_facets = {}
for obj in search_results.objects:
    categories = obj.payload.get("categories", [])
    for category in categories:
        if category not in category_facets:
            category_facets[category] = 0
        category_facets[category] += 1

# Return facets with counts
facets = [
    {"id": cat_id, "count": count}
    for cat_id, count in category_facets.items()
]

print(f"Category facets: {facets}")
```

## Deleting Categories

To delete categories, use the categories deletion endpoint:

```python
import requests

response = requests.post(
    "https://api.embeddingstudio.com/api/v1/embeddings/deletion-tasks/categories/run",
    json={
        "task_id": "category_delete_001",  # Optional custom ID
        "object_ids": ["category_smartphones", "category_tablets"]
    }
)

task = response.json()
print(f"Category Deletion Task ID: {task['id']}, Status: {task['status']}")
```

## Updating Category Hierarchies

When restructuring your category hierarchy, follow these steps:

1. **Update Parent References**:
   ```python
   # Update a category's parent
   updated_category = {
       "object_id": "category_tablets",
       "payload": {
           "name": "Tablets",
           "description": "Portable computing devices with touchscreens",
           "parent_id": "category_computers",  # Previously under electronics
           "level": 2,
           "path": ["category_root", "category_computers"]
       }
   }
   ```

2. **Update Child Categories**:
   - If you move a parent, you may need to update all children
   - Update the `level` and `path` values accordingly

3. **Rebuild Category Indexes**:
   - After major hierarchy changes, consider reindexing categories

## Best Practices for Category Management

### Structure Design

1. **Limit Hierarchy Depth**: Keep hierarchies to 3-5 levels for usability
2. **Balanced Categories**: Aim for balanced distribution of items
3. **Consistent Naming**: Use consistent naming conventions
4. **Descriptive Content**: Include rich descriptions for better embeddings

### Technical Optimization

1. **Batch Updates**: Group category changes in batches
2. **Hierarchy Updates**: Update entire branches at once
3. **Precompute Paths**: Include full paths in category data
4. **Index Management**: Create appropriate indexes for hierarchy traversal

### Monitoring and Maintenance

1. **Audit Categories**: Regularly check for orphaned categories
2. **Measure Usage**: Track which categories are most utilized
3. **Update Descriptions**: Keep category descriptions current
4. **Validate Relationships**: Ensure parent-child integrity
