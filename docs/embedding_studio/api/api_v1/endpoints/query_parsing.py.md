# Merged Documentation for `_get_similar_categories` and `parse_categories`

## Method: `_get_similar_categories`

### Functionality
This function retrieves a list of similar category objects based on the vectorized search query. It first processes the query using a query retriever, then embeds the query and searches for similar objects in the vector database.

### Parameters
- `search_query (Any)`: The input query to search for similar categories. It is processed to generate a vector representation.

### Returns
- List[SearchResults]: A list of similar category objects. Returns an empty list if no matches are found.

### Usage
- **Purpose:** Find categories that match a given search query via vector embedding and similarity search.

#### Example
```python
results = _get_similar_categories("example query")
if results:
    for category in results:
        process(category)
else:
    print("No similar categories found.")
```

---

## Method: `parse_categories`

### Functionality
This endpoint handles POST requests to retrieve similar categories based on a provided search query. It vectorizes the query, performs a similarity search, and returns matching categories.

### Parameters
- `body`: A QueryParsingRequest object that contains the search query to be parsed.

### Usage
- **Purpose:** To search and return categories relevant to a text query.

#### Example
```python
from embedding_studio.api.api_v1.endpoints.query_parsing import parse_categories
from embedding_studio.api.api_v1.schemas.query_parsing import QueryParsingRequest

req = QueryParsingRequest(search_query="sample query")
response = parse_categories(req)
print(response.categories)
```