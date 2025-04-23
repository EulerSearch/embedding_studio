# What is the Payload Filter and its Capabilities

## Overview

Embedding Studio's Payload Filter system provides a powerful and flexible way to query vector embeddings based on their associated metadata. This functionality allows you to combine vector similarity search with metadata filtering, creating more precise and context-aware search experiences.

## Core Concepts

The payload filter system is designed to be familiar to users of Elasticsearch or other document search systems. It works with JSON payloads attached to your vector embeddings and allows complex boolean queries against this metadata.

Each embedding in Embedding Studio can have an associated payload, which is a JSON object containing arbitrary metadata. The payload filter lets you query these JSON objects using a variety of operators.

## Query Types

Embedding Studio supports the following query types:

### Basic Query Types

| Query Type | Description | Example |
|------------|-------------|---------|
| `MatchQuery` | Full-text search that uses PostgreSQL's text search capabilities | Match documents containing specific words |
| `TermQuery` | Exact value matching for a single value | Find products with exactly "red" color |
| `TermsQuery` | Matches a field against multiple values (similar to SQL's IN operator) | Find products in multiple categories |
| `ExistsQuery` | Checks if a field exists in the data | Find all products that have a "discount" field |
| `WildcardQuery` | Supports pattern matching with wildcards | Find products with names starting with "smart*" |

### Array Query Types

| Query Type | Description | Example |
|------------|-------------|---------|
| `ListHasAllQuery` | Checks if an array field contains all of the provided values | Find products with all specified tags |
| `ListHasAnyQuery` | Checks if an array field contains any of the provided values | Find products with any of the specified tags |

### Range Query Type

| Query Type | Description | Example |
|------------|-------------|---------|
| `RangeQuery` | Queries for fields with values in a specified numeric range | Find products in a specific price range |

### Boolean Query Type

| Query Type | Description | Example |
|------------|-------------|---------|
| `BoolQuery` | Combines multiple query conditions with boolean logic | Complex combinations of the above queries |

## Boolean Operators

The `BoolQuery` type supports the following boolean operators:

- `must`: All conditions must match (AND logic)
- `should`: At least one condition should match (OR logic)
- `filter`: All filter conditions must match, similar to must but doesn't affect scoring
- `must_not`: Conditions must not match (NOT logic)

## Example Usage

Here's a complete example of a complex payload filter:

```python
from embedding_studio.models.payload.models import (
    PayloadFilter, BoolQuery, TermQuery, RangeQuery, MatchQuery
)

# Create a complex filter for products
filter = PayloadFilter(
    query=BoolQuery(
        must=[
            # Must be in the "electronics" category
            TermQuery(term=SingleValueQuery(field="category", value="electronics")),
            
            # Must contain the word "smartphone" in description
            MatchQuery(match=SingleTextValueQuery(field="description", value="smartphone")),
        ],
        filter=[
            # Price between $200 and $800
            RangeQuery(
                field="price",
                range=RangeCondition(gte=200, lte=800)
            )
        ],
        should=[
            # Preferably made by "Apple" or "Samsung"
            TermQuery(term=SingleValueQuery(field="brand", value="Apple")),
            TermQuery(term=SingleValueQuery(field="brand", value="Samsung")),
        ],
        must_not=[
            # Not refurbished
            TermQuery(term=SingleValueQuery(field="condition", value="refurbished")),
        ]
    )
)
```

This filter would find electronic products that:
- Have "smartphone" in their description
- Cost between $200 and $800
- Are not refurbished
- With preference for Apple or Samsung brands

## Direct vs. Payload Fields

All query types have a `force_not_payload` flag that determines whether the query field is in the payload JSON or is a direct database column:

- When `force_not_payload=False` (default): The field is looked up in the payload JSON.
- When `force_not_payload=True`: The field is treated as a direct column name in the database.

## Combining with Vector Search

The real power of Embedding Studio's payload filter comes when combining it with vector similarity search:

```python
search_results = collection.find_similarities(
    query_vector=my_query_vector,
    limit=10,
    payload_filter=my_complex_filter,
    sort_by=SortByOptions(field="price", order="asc")
)
```

This enables semantic search combined with structured metadata filtering - for example, finding products that are semantically similar to a query while still matching specific criteria like price range, category, or brand.

## Performance Considerations

1. **Indexing**: For optimal performance, consider which fields you'll frequently query and ensure they're properly indexed.

2. **Complexity**: While the payload filter system is powerful, excessively complex queries may impact performance.

3. **Balance**: Find the right balance between vector similarity and metadata filtering for your specific use case.

## Conclusion

Embedding Studio's payload filter system provides a sophisticated query language for your vector embeddings' metadata. By combining traditional filtering capabilities with vector similarity search, you can create powerful, flexible search experiences that leverage both semantic understanding and structured metadata.