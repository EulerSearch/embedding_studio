# How Payload Filters Convert to SQL for pgvector

## Introduction

Embedding Studio uses pgvector as its vector database backend, which means payload filters need to be translated into efficient SQL queries. This document explains how various payload filter queries are converted to SQL for execution against the PostgreSQL database with pgvector extension.

## The Translation Process

When you submit a payload filter to Embedding Studio's API, it undergoes a two-step translation process:

1. The payload filter structure (JSON/Python objects) is parsed and validated
2. The filter is translated into SQL expressions through one of two mechanisms:
   - ORM-based filters (using SQLAlchemy)
   - Raw SQL string generation

This translation happens primarily in the `translate_query_to_orm_filters` and `translate_query_to_sql_filters` functions in the `query_to_sql.py` module.

## Basic Query Translation Examples

Let's examine how each type of filter gets translated to SQL:

### MatchQuery (Text Search)

**Payload Filter:**
```python
MatchQuery(match=SingleTextValueQuery(field="description", value="smartphone"))
```

**Generated SQL:**
```sql
to_tsvector('simple', jsonb_extract_path_text(payload, 'description')) @@ to_tsquery('simple', 'smartphone')
```

This uses PostgreSQL's full-text search capabilities with the `to_tsvector` and `to_tsquery` functions.

### TermQuery (Exact Match)

**Payload Filter:**
```python
TermQuery(term=SingleValueQuery(field="category", value="electronics"))
```

**Generated SQL:**
```sql
(payload ->> 'category') = 'electronics'
```

For numeric values, the SQL includes casting:
```sql
((payload ->> 'price')::numeric) = 299.99
```

### TermsQuery (Multiple Values)

**Payload Filter:**
```python
TermsQuery(terms=MultipleValuesQuery(field="category", values=["electronics", "gadgets"]))
```

**Generated SQL:**
```sql
(payload ->> 'category') IN ('electronics', 'gadgets')
```

### ListHasAllQuery (Array Contains All)

**Payload Filter:**
```python
ListHasAllQuery(all=MultipleValuesQuery(field="tags", values=["waterproof", "bluetooth"]))
```

**Generated SQL:**
```sql
payload -> 'tags' ?& array['waterproof', 'bluetooth']
```

This uses PostgreSQL's `?&` operator, which checks if the JSON array contains all specified elements.

### ListHasAnyQuery (Array Contains Any)

**Payload Filter:**
```python
ListHasAnyQuery(any=MultipleValuesQuery(field="tags", values=["discount", "sale"]))
```

**Generated SQL:**
```sql
payload -> 'tags' ?| array['discount', 'sale']
```

This uses the `?|` operator to check if the JSON array contains any of the specified elements.

### ExistsQuery (Field Exists)

**Payload Filter:**
```python
ExistsQuery(field="discount_percentage")
```

**Generated SQL:**
```sql
payload ? 'discount_percentage'
```

The `?` operator checks if the JSON object contains the specified key.

### WildcardQuery (Pattern Matching)

**Payload Filter:**
```python
WildcardQuery(wildcard=SingleTextValueQuery(field="model", value="iphone*"))
```

**Generated SQL:**
```sql
to_tsvector('simple', jsonb_extract_path_text(payload, 'model')) @@ to_tsquery('simple', 'iphone:*')
```

Notice how the asterisk is converted to PostgreSQL's `:*` syntax for prefix matching.

### RangeQuery (Numeric Range)

**Payload Filter:**
```python
RangeQuery(field="price", range=RangeCondition(gte=100, lte=500))
```

**Generated SQL:**
```sql
((payload ->> 'price')::numeric) >= 100 AND ((payload ->> 'price')::numeric) <= 500
```

This casts the JSON field to a numeric type and applies the range conditions.

## Boolean Query Translation

The `BoolQuery` type allows combining multiple conditions with boolean logic. Here's how it translates:

**Payload Filter:**
```python
BoolQuery(
    must=[
        TermQuery(term=SingleValueQuery(field="category", value="electronics")),
        MatchQuery(match=SingleTextValueQuery(field="description", value="smartphone"))
    ],
    filter=[
        RangeQuery(field="price", range=RangeCondition(gte=200, lte=800))
    ],
    should=[
        TermQuery(term=SingleValueQuery(field="brand", value="Apple")),
        TermQuery(term=SingleValueQuery(field="brand", value="Samsung"))
    ],
    must_not=[
        TermQuery(term=SingleValueQuery(field="condition", value="refurbished"))
    ]
)
```

**Generated SQL:**
```sql
(
  (
    ((payload ->> 'category') = 'electronics')
    AND
    (to_tsvector('simple', jsonb_extract_path_text(payload, 'description')) @@ to_tsquery('simple', 'smartphone'))
  )
  AND
  (
    ((payload ->> 'price')::numeric) >= 200 AND ((payload ->> 'price')::numeric) <= 800
  )
  AND
  (
    ((payload ->> 'brand') = 'Apple') OR ((payload ->> 'brand') = 'Samsung')
  )
  AND
  NOT (
    ((payload ->> 'condition') = 'refurbished')
  )
)
```

The boolean logic is translated as:
- `must` clauses are combined with `AND`
- `should` clauses are combined with `OR`
- `filter` clauses are functionally similar to `must` and use `AND`
- `must_not` clauses are wrapped in `NOT ()`

## Direct Database Column vs. Payload Field

Each query type supports a `force_not_payload` flag that determines whether the field is in the JSON payload or a direct database column:

**With `force_not_payload=True`:**

```python
TermQuery(term=SingleValueQuery(field="user_id", value="123", force_not_payload=True))
```

**Generated SQL:**
```sql
user_id = '123'
```

Instead of accessing the payload JSON, this targets the actual `user_id` column in the database.

## Combining with Vector Search

The real power comes when combining payload filtering with vector similarity search. This happens through specialized SQL functions in pgvector.

For example, a combined query might generate a SQL statement like:

```sql
WITH filtered_objects AS (
  SELECT 
    object_id, payload, storage_meta, user_id, original_id
  FROM dbo_model_name
  WHERE (payload ->> 'category') = 'electronics'
), 
prefiltered_vectors AS (
  SELECT
    op.object_id, op.part_id, o.payload, o.storage_meta, o.user_id, o.original_id,
    (op.vector <=> '[0.1, 0.2, ...]'::vector) AS distance
  FROM dbop_model_name op
  INNER JOIN filtered_objects o ON op.object_id = o.object_id 
  WHERE op.vector <=> '[0.1, 0.2, ...]'::vector <= 0.5
)
SELECT 
  object_id, payload, storage_meta, user_id, original_id,
  ARRAY_AGG(part_id) AS part_ids,
  MIN(distance) AS distance
FROM prefiltered_vectors
GROUP BY object_id, payload, storage_meta, user_id, original_id
ORDER BY distance ASC
LIMIT 10 OFFSET 0;
```

This SQL:
1. First filters objects by payload metadata
2. Then computes vector distances only for objects that passed the filter
3. Finally aggregates and returns the results

## Advanced SQL Functions

Embedding Studio uses a set of specialized SQL functions for efficient vector search:

1. **Simple search functions** (`simple_*`): For basic vector similarity search without complex filters
2. **Advanced search functions** (`advanced_*`): For combining vector search with payload filtering
3. **Similarity-ordered functions** (`*_so_*`): Prioritize similarity in result ordering
4. **With-vectors functions** (`*_v_*`): Include vector values in results

These functions are created when a collection is initialized and are optimized for different search scenarios.

## The Translation Pipeline

The complete translation pipeline works as follows:

1. API receives a payload filter
2. The filter is parsed into Python objects
3. Based on the search type and parameters, an appropriate search strategy is selected:
   - Simple search (vector only)
   - Advanced search (vector + filter)
   - Similarity-ordered search
   - With or without returning vectors
4. The appropriate SQL function is called with parameters based on the filter
5. Results are processed and returned

## SQL Optimization Techniques

Several optimization techniques are used when translating payload filters to SQL:

### 1. Pre-filtering

Where possible, Embedding Studio first filters objects by metadata before computing vector distances, which is more computationally expensive.

### 2. Pagination Management

The system uses SQL's `LIMIT` and `OFFSET` for efficient pagination.

### 3. Specialized Indexes

PostgreSQL's JSONB indexes are used for payload fields that are frequently queried.

### 4. Custom SQL Functions

Pre-compiled SQL functions optimize common query patterns.

### 5. Materialized CTEs

Common Table Expressions with the `materialized` hint force PostgreSQL to materialize intermediate results, improving performance for complex queries.

## Example: Full Translation Process

Let's trace a complete translation example:

1. **Original API request:**
```json
{
  "search_query": "smartphone",
  "filter": {
    "query": {
      "bool": {
        "must": [
          {"term": {"field": "category", "value": "electronics"}}
        ],
        "filter": [
          {"field": "price", "range": {"gte": 300, "lte": 1000}}
        ]
      }
    }
  },
  "limit": 10
}
```

2. **Parsed into Python objects:**
```python
SimilaritySearchRequest(
    search_query="smartphone",
    filter=PayloadFilter(
        query=BoolQuery(
            must=[TermQuery(term=SingleValueQuery(field="category", value="electronics"))],
            filter=[RangeQuery(field="price", range=RangeCondition(gte=300, lte=1000))]
        )
    ),
    limit=10
)
```

3. **Payload filter translated to SQL:**
```sql
(
  ((payload ->> 'category') = 'electronics')
  AND
  (((payload ->> 'price')::numeric) >= 300 AND ((payload ->> 'price')::numeric) <= 1000)
)
```

4. **Embedded in a vector search SQL function call:**
```sql
SELECT
  result_object_id, result_payload, result_storage_meta, result_user_id, 
  result_original_id, result_part_ids, result_distance, subset_count
FROM advanced_so_model_name_cosine(
  '[0.1, 0.2, ...]'::vector,   -- Vectorized search query
  NULL,                        -- user_id parameter
  $FILTER$(                    -- The translated filter
    ((payload ->> 'category') = 'electronics')
    AND
    (((payload ->> 'price')::numeric) >= 300 AND ((payload ->> 'price')::numeric) <= 1000)
  )$FILTER$,
  10,                          -- limit parameter
  0,                           -- offset parameter
  NULL,                        -- max_distance parameter
  50,                          -- enlarged_limit parameter
  0,                           -- enlarged_offset parameter
  FALSE                        -- average_only parameter
);
```

5. **Results returned:**
The query executes and returns the appropriate records, which are then processed into the API response format.

## Conclusion

The translation of payload filters to SQL in Embedding Studio is a sophisticated process that bridges the gap between user-friendly filter definitions and efficient database queries. By leveraging PostgreSQL's JSONB capabilities and pgvector's similarity search functions, Embedding Studio provides a powerful yet performant way to filter vector search results by metadata.

Understanding this translation process can help you:

1. Design more efficient payload filters
2. Troubleshoot unexpected search results
3. Optimize database indexes for your specific query patterns
4. Balance vector similarity with metadata filtering for optimal search experiences

Whether you're building a semantic product search, document retrieval system, or recommendation engine, the payload filter to SQL translation is a key part of making your vector search both powerful and precise.