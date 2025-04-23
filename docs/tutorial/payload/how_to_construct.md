# How to Construct Payload for API

## Introduction

This guide explains how to construct payload filters for Embedding Studio's API endpoints. Effective payload construction allows you to perform sophisticated filtering on vector search results, combining metadata conditions with semantic similarity.

## API Endpoints That Accept Payload Filters

Embedding Studio provides several endpoints that accept payload filters:

- `POST /embeddings/similarity-search`: Search for vectors similar to a query vector with optional payload filtering
- `POST /embeddings/payload-search`: Search using only payload filtering (no vector similarity)
- `POST /embeddings/payload-count`: Count objects matching a payload filter

## Payload Structure

When constructing API requests, the payload filter should be structured in JSON format according to the query type you want to use. Let's look at how to form each type of query:

### Basic Request Structure

```json
{
  "search_query": "optional search text",
  "filter": {
    "query": {
      // Your filter query goes here
    }
  },
  "limit": 10,
  "offset": 0,
  "create_session": true
}
```

## Query Types in JSON

### MatchQuery - Text Search

```json
{
  "filter": {
    "query": {
      "match": {
        "field": "description",
        "value": "smartphone",
        "force_not_payload": false
      }
    }
  }
}
```

### TermQuery - Exact Value Match

```json
{
  "filter": {
    "query": {
      "term": {
        "field": "category",
        "value": "electronics",
        "force_not_payload": false
      }
    }
  }
}
```

### TermsQuery - Multiple Values (OR)

```json
{
  "filter": {
    "query": {
      "terms": {
        "field": "category",
        "values": ["electronics", "gadgets", "accessories"],
        "force_not_payload": false
      }
    }
  }
}
```

### ListHasAllQuery - Array Contains All Values

```json
{
  "filter": {
    "query": {
      "all": {
        "field": "tags",
        "values": ["waterproof", "wireless", "bluetooth"],
        "force_not_payload": false
      }
    }
  }
}
```

### ListHasAnyQuery - Array Contains Any Value

```json
{
  "filter": {
    "query": {
      "any": {
        "field": "tags",
        "values": ["discount", "sale", "clearance"],
        "force_not_payload": false
      }
    }
  }
}
```

### MatchPhraseQuery - Phrase Match

```json
{
  "filter": {
    "query": {
      "match_phrase": {
        "field": "description",
        "value": "high resolution display",
        "force_not_payload": false
      }
    }
  }
}
```

### ExistsQuery - Field Exists

```json
{
  "filter": {
    "query": {
      "exists": {
        "field": "discount_percentage",
        "force_not_payload": false
      }
    }
  }
}
```

### WildcardQuery - Pattern Matching

```json
{
  "filter": {
    "query": {
      "wildcard": {
        "field": "model_name",
        "value": "iphone*",
        "force_not_payload": false
      }
    }
  }
}
```

### RangeQuery - Numeric Range

```json
{
  "filter": {
    "query": {
      "field": "price",
      "range": {
        "gte": 100,
        "lte": 500,
        "gt": null,
        "lt": null,
        "eq": null
      },
      "force_not_payload": false
    }
  }
}
```

### BoolQuery - Combining Multiple Queries

```json
{
  "filter": {
    "query": {
      "must": [
        {
          "term": {
            "field": "category",
            "value": "electronics",
            "force_not_payload": false
          }
        },
        {
          "match": {
            "field": "description",
            "value": "smartphone",
            "force_not_payload": false
          }
        }
      ],
      "should": [
        {
          "term": {
            "field": "brand",
            "value": "Apple",
            "force_not_payload": false
          }
        },
        {
          "term": {
            "field": "brand",
            "value": "Samsung",
            "force_not_payload": false
          }
        }
      ],
      "filter": [
        {
          "field": "price",
          "range": {
            "gte": 200,
            "lte": 800
          },
          "force_not_payload": false
        }
      ],
      "must_not": [
        {
          "term": {
            "field": "condition",
            "value": "refurbished",
            "force_not_payload": false
          }
        }
      ]
    }
  }
}
```

## Complete API Examples

### Example 1: Similarity Search with Payload Filter

```json
{
  "search_query": "smartphone with good camera",
  "filter": {
    "query": {
      "term": {
        "field": "category",
        "value": "electronics"
      }
    }
  },
  "limit": 10,
  "offset": 0,
  "create_session": true
}
```

### Example 2: Pure Payload Search (No Vector Similarity)

```json
{
  "filter": {
    "query": {
      "bool": {
        "must": [
          {
            "term": {
              "field": "category",
              "value": "electronics"
            }
          }
        ],
        "filter": [
          {
            "field": "price",
            "range": {
              "gte": 300,
              "lte": 1000
            }
          }
        ]
      }
    }
  },
  "limit": 20,
  "offset": 0,
  "sort_by": {
    "field": "price",
    "order": "asc"
  }
}
```

### Example 3: Count Objects Matching Payload Filter

```json
{
  "filter": {
    "query": {
      "bool": {
        "must": [
          {
            "term": {
              "field": "in_stock",
              "value": true
            }
          },
          {
            "field": "price",
            "range": {
              "lt": 500
            }
          }
        ]
      }
    }
  }
}
```

## Additional Options

### Sorting Results

You can sort results using the `sort_by` parameter:

```json
{
  "filter": { /* Your filter here */ },
  "sort_by": {
    "field": "price",
    "order": "asc",
    "force_not_payload": false
  }
}
```

Options for `order` are `"asc"` (ascending) or `"desc"` (descending).

### Pagination

Use the `limit` and `offset` parameters for pagination:

```json
{
  "filter": { /* Your filter here */ },
  "limit": 10,
  "offset": 20
}
```

This would fetch items 21-30 (using zero-based indexing).

## Combining Vector Search with Payload Filtering

To get the most powerful search results, combine semantic vector search with payload filtering:

```json
{
  "search_query": "modern minimalist furniture",
  "filter": {
    "query": {
      "bool": {
        "must": [
          {
            "term": {
              "field": "category",
              "value": "furniture"
            }
          }
        ],
        "filter": [
          {
            "field": "price",
            "range": {
              "lte": 1200
            }
          }
        ],
        "must_not": [
          {
            "term": {
              "field": "out_of_stock",
              "value": true
            }
          }
        ]
      }
    }
  },
  "limit": 10,
  "create_session": true
}
```

This will find items that are semantically similar to "modern minimalist furniture" while ensuring they:
- Are in the "furniture" category
- Cost less than $1,200
- Are not out of stock

## Best Practices

1. **Start Simple**: Begin with basic filters and gradually add complexity as needed.

2. **Test Thoroughly**: Test your payload filters with smaller datasets first to ensure they work as expected.

3. **Consider Performance**: Complex nested boolean queries may impact performance.

4. **Use Sessions**: Set `create_session: true` when you want to track user interactions with the search results.

5. **Balance Filtering & Similarity**: Too strict filtering might exclude semantically relevant results.

## Conclusion

Embedding Studio's payload filter API provides a flexible way to combine structured filtering with vector similarity search. By constructing appropriate payload filters, you can create rich, contextual search experiences that combine the best of semantic search and traditional filtering.