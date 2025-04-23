# Documentation for Query Parsing API Methods

---

## `/categories/parse_categories`

### Functionality
The `parse_categories` endpoint parses a natural-language query and returns a list of similar categories based on vector similarity matching.

---

### Request Parameters
- `search_query` *(Any)*: The search query text or structure to be parsed and matched against known categories.

---

### Request JSON Example
```json
{
  "search_query": "deep learning optimization"
}
```

- `search_query`: Can be a plain text string or structured object; it's transformed into an embedding and matched against category vectors.

---

### Response JSON Example
```json
{
  "categories": [
    {
      "object_id": "cat-ml-001",
      "distance": 0.087,
      "payload": {
        "name": "Machine Learning",
        "tags": ["AI", "Modeling"]
      },
      "meta": {
        "source_table": "categories_dataset",
        "row_pointer": 42
      }
    }
  ]
}
```

- `object_id`: ID of the matched category.
- `distance`: Cosine or L2 similarity score between query and category.
- `payload`: Structured metadata describing the matched category (e.g., name, tags).
- `meta`: Storage metadata indicating where the original category data resides (e.g., table name, row reference).

---

### Usage
- **Purpose**: To semantically parse user queries and recommend relevant category labels from the database using vector-based matching.

#### Example cURL
```bash
curl -X POST "http://<server>/categories" \
     -H "Content-Type: application/json" \
     -d '{
           "search_query": "example search text"
         }'
```