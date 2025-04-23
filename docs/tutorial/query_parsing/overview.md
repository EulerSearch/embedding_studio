# Understanding Query Parsing in Embedding Studio

Query parsing is a powerful feature in Embedding Studio that helps understand and categorize user search queries by mapping them to relevant categories. This tutorial explains how the query parsing system works, its architecture, and the underlying algorithms that make it effective.

## Core Concepts

Query parsing in Embedding Studio is built around a few fundamental concepts:

### 1. Vector-Based Category Matching

Rather than using traditional keyword matching, Embedding Studio uses vector embeddings to semantically match search queries to categories:

- **Embedding-Based Matching**: Converts the user's search query into a vector representation
- **Semantic Understanding**: Captures the meaning of the query instead of just matching keywords
- **Category Vectors**: Categories are also represented as vectors for accurate matching

### 2. Category Selection

The system uses sophisticated selection strategies to determine which categories best match a query:

- **Similarity Scoring**: Calculates how similar a query is to each potential category
- **Threshold-Based Selection**: Uses distance/similarity thresholds to identify relevant matches
- **Multiple Selection Strategies**: Different selector implementations for various use cases

### 3. Distance-Based Selectors

The system includes various selector implementations that filter results based on distance metrics:

- **DistBasedSelector**: Abstract base class for selectors that work with distances
- **ProbsDistBasedSelector**: Uses probability calculations for nuanced selection
- **VectorsBasedSelector**: Works directly with vector representations for advanced matching

## How Query Parsing Works: The Algorithm

When a user submits a search query, here's how Embedding Studio processes it:

### Step 1: Query Vectorization

```python
# Retrieve the query retriever and inference client
query_retriever = plugin.get_query_retriever()
inference_client = plugin.get_inference_client_factory().get_client(
    collection_info.embedding_model.id
)

# Convert the search query to vector format
search_query = query_retriever(search_query)
query_vector = inference_client.forward_query(search_query)[0]
```

1. The raw text query is processed by a query retriever specific to the model
2. The processed query is then converted to a vector using the inference client
3. This vector representation captures the semantic meaning of the search query

### Step 2: Similar Category Search

```python
# Search for similar categories in the vector database
found_objects, _ = collection.find_similar_objects(
    query_vector=query_vector.tolist(),
    offset=0,
    limit=plugin.get_max_similar_categories(),
    max_distance=plugin.get_max_margin(),
    with_vectors=categories_selector.vectors_are_needed
)
```

1. The query vector is compared against category vectors in the database
2. A similarity search returns categories that are semantically similar
3. Results are limited by the maximum number of categories and distance threshold

### Step 3: Category Selection

```python
# Apply the category selector to filter the results
final_indexes = categories_selector.select(found_objects, query_vector)
results = []
for index in final_indexes:
    results.append(found_objects[index])
```

1. A category selector is applied to the candidate categories
2. The selector implements a strategy to filter for the most relevant matches
3. Only categories meeting the selection criteria are returned to the user

## Selector Types and Algorithms

### Distance-Based Selector

The foundational selector works with pre-calculated distance values:

```python
def select(self, categories, query_vector=None):
    # Convert distance values to a normalized tensor
    values = self._convert_values(categories)
    
    # Apply margin threshold
    positive_threshold_min = 1 - self._margin if self._is_similarity else self._margin
    corrected_values = values - positive_threshold_min
    
    # Calculate binary selection labels (implemented by subclasses)
    bin_labels = self._calculate_binary_labels(corrected_values)
    
    # Return indices of selected objects
    return torch.nonzero(bin_labels).T[0].tolist()
```

This selector:
1. Normalizes distance values based on the metric type
2. Applies the configured margin threshold
3. Delegates the final decision logic to subclasses

### Probability-Based Selector

The `ProbsDistBasedSelector` extends the base selector with probability calculations:

```python
def _calculate_binary_labels(self, corrected_values):
    return (
        torch.sigmoid(corrected_values * self._scale)
        > self._prob_threshold
    )
```

This selector:
1. Applies a sigmoid function to convert distances to probabilities (0-1 range)
2. Uses a probability threshold to determine which categories to select
3. Allows for more nuanced selection with configurable scaling and thresholds

### Vector-Based Selector

For advanced matching scenarios, the `VectorsBasedSelector` works directly with embedding vectors:

```python
def select(self, categories, query_vector):
    # Get tensor representation of categories
    category_vectors = self._get_categories_tensor(categories)
    
    # Calculate distances between query and category vectors
    values = self._calculate_distance(
        query_vector,
        category_vectors,
        self._softmin_temperature,
        self._is_similarity
    )
    
    # Apply threshold and selection
    positive_threshold_min = 1 - self._margin if self._is_similarity else self._margin
    corrected_values = values - positive_threshold_min
    bin_labels = self._calculate_binary_labels(corrected_values)
    
    return torch.nonzero(bin_labels).T[1].tolist()
```

This selector:
1. Works with the raw vectors rather than pre-calculated distances
2. Can implement more complex distance calculations between query and categories
3. Supports various metrics (cosine, dot product, Euclidean) and aggregation methods

## Distance Metrics and Selection Strategies

The system supports multiple distance metrics for comparing vectors:

### Metric Types

- **Cosine Similarity**: Measures the cosine of the angle between vectors (value between -1 and 1)
- **Euclidean Distance**: Measures the straight-line distance between vectors
- **Dot Product**: Measures vector similarity through their dot product

### Selection Strategies

Different selection strategies can be employed depending on your needs:

- **Threshold-Based**: Select categories with distances below (or similarities above) a threshold
- **Probability-Based**: Convert distances to probabilities and select based on probability threshold
- **Top-K**: Select the top K most similar categories regardless of absolute distance

## Using the Query Parsing API

The query parsing functionality is exposed through a REST API endpoint:

```http
POST /parse-query/categories
{
    "search_query": "wireless headphones with noise cancellation"
}
```

Response:
```json
{
    "categories": [
        {
            "object_id": "headphones",
            "distance": 0.15,
            "payload": {
                "name": "Headphones",
                "parent_category": "Audio Equipment"
            }
        },
        {
            "object_id": "noise_cancellation",
            "distance": 0.22,
            "payload": {
                "name": "Noise Cancellation",
                "parent_category": "Audio Features"
            }
        }
    ]
}
```

## Behind the Scenes: pgvector Integration

Embedding Studio's query parsing leverages pgvector, a PostgreSQL extension for storing and searching vector embeddings:

### Collection and Vector Database

The system uses:
- **PgvectorDb**: Handles high-level database operations
- **PgvectorCollection**: Represents a collection of vectors
- **Collection**: Interface for vector collections

The search process relies on SQL functions that implement efficient vector search algorithms:

```sql
SELECT ... FROM vector_search_function(
    '{query_vector}'::vector, 
    limit, 
    offset, 
    max_distance,
    '{metadata}'
);
```

## Performance Considerations

Several optimizations enhance query parsing performance:

- **HNSW Indexes**: Uses Hierarchical Navigable Small World (HNSW) graphs for efficient approximate nearest neighbor search
- **Batch Processing**: Processes categories in batches for memory efficiency
- **Parallel Search**: Implements concurrent search strategies where appropriate

## Customization and Extension

The query parsing system is designed to be customizable:

- **Custom Selectors**: Create new selector implementations by extending the base classes
- **Embedding Models**: Change the embedding model to match your specific domain
- **Distance Metrics**: Configure the distance metric and threshold to control match sensitivity
- **Category Structure**: Define your own category hierarchy to match your business needs
