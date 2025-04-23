## Documentation for `RedisSuggester`

### Functionality
RedisSuggester provides a Redis-based implementation for generating suggestions from a pre-indexed phrase store. It leverages tokenization to split phrases into searchable chunks and supports both exact and fuzzy matching.

### Motivation
This class was developed to deliver rapid suggestion retrieval by utilizing Redis and Redisearch for efficient data storage and query processing. It overcomes performance challenges in generating suggestions.

### Inheritance
RedisSuggester inherits from AbstractSuggester, ensuring a consistent interface across different suggestion modules and promoting code reuse.

### Usage Example
An instance can be created by supplying a Redis URL, a tokenizer instance, and optional parameters like index name and maximum chunks. The associated phrase manager encapsulates the logic for storing and querying phrase data.

---

## Documentation for `RedisSuggester.phrases_manager`

### Functionality
Returns the suggestion phrase manager used by RedisSuggester. This property provides access to a Redis-based implementation that manages suggestion phrases, allowing for interactions with the underlying Redis datastore.

### Parameters
None.

### Usage
- **Purpose** - To retrieve the phrase manager instance which handles the storage and retrieval of suggestion phrases.

#### Example
```python
from embedding_studio.suggesting.redis.suggester import RedisSuggester

# Initialize the suggester with required parameters
suggester = RedisSuggester(
    redis_url='redis://localhost:6379',
    tokenizer=your_tokenizer_instance,
    index_name='suggestion_phrases',
    max_chunks=20
)

# Retrieve the phrase manager
phrase_manager = suggester.phrases_manager
```

---

## Documentation for `RedisSuggester._generate_query`

### Functionality
Constructs a Redis query string used to fetch suggestions from the Redis search index. As an abstract method, it must be overridden by subclasses to define custom query logic.

### Parameters
- `request`: A SuggestingRequest object containing input data and context for formulating the query.
- `top_k`: An integer specifying the maximum number of suggestions to retrieve (default is 10).
- `soft_match`: A boolean flag that enables fuzzy matching of query tokens when set to True.

### Usage
- **Purpose** - To generate a query string for retrieving suggestion results from a Redis index.

#### Example
Suppose a subclass implements the method as follows:
```python
query_string = instance._generate_query(request, top_k=5, soft_match=True)
```

---

## Documentation for `RedisSuggester._find_match_position_prefix`

### Functionality
This method searches a list of chunk strings for the first occurrence where a chunk starts with a given prefix. The match is case-sensitive and returns the index of the first match, or -1 if no match is found.

### Parameters
- `raw_chunks`: A list of strings representing chunk texts. The method scans these sequentially.
- `found_chunk`: A string representing the prefix to search for. Comparison is case-sensitive.

### Usage
- **Purpose**: Locate the first chunk that starts with a specified prefix in a list of strings.

#### Example
```python
raw_chunks = ["alpha", "beta", "alphabet", "gamma"]
pos = suggester._find_match_position_prefix(raw_chunks, "alpha")
# pos is 0
```

---

## Documentation for `RedisSuggester._find_match_position_soft`

### Functionality
Finds the chunk that is closest to the provided 'found_chunk' using fuzzy matching. It leverages difflib.get_close_matches with an 80% similarity cutoff. The 'max_distance' parameter is accepted but not directly used for limiting edit distance.

### Parameters
- `raw_chunks`: A list of chunk strings to search through.
- `found_chunk`: The string to find a close match for.
- `max_distance`: Maximum allowed edit distance (parameter accepted for interface consistency).

### Usage
- **Purpose**: Determine the best fuzzy match from a list of chunks.

#### Example
```python
index = suggester._find_match_position_soft(
    ["chunk1", "chunk2", "chunk3"], "chunk"
)
```

---

## Documentation for `RedisSuggester._find_match_position_exact`

### Functionality
Returns the index of a given chunk that exactly matches an element in a list of chunks. If the chunk is not found, it returns -1.

### Parameters
- `raw_chunks`: List[str] - List of chunk strings to search through.
- `found_chunk`: str - The chunk to locate with an exact match.

### Usage
- **Purpose** - To determine the exact match position in a list for further processing.

#### Example
```python
raw_chunks = ["chunk1", "chunk2", "chunk3"]
position = suggester._find_match_position_exact(raw_chunks, "chunk2")
# position is 1
```

---

## Documentation for `RedisSuggester._doc_to_suggest`

### Functionality
Converts a Redis document into a Suggest object. It extracts raw chunks from the document, tries to match the last found chunk using exact, prefix, or fuzzy matching, and constructs the final suggestion with separate prefix and matched chunks.

### Parameters
- `doc`: A dictionary representing a Redis document with keys like "chunk_i" and optional entries "prob" and "labels".
- `request`: A SuggestingRequest object containing found chunks for matching.

### Usage
- **Purpose** - To convert raw Redis data into a structured Suggest object for further processing and suggestion delivery.

#### Example
Suppose you have a Redis document:
```python
doc = {
  "chunk_0": "hello",
  "chunk_1": "world",
  "prob": 0.95,
  "labels": "greeting\nexample"
}
```
and a SuggestingRequest with found chunks:
```python
request = SuggestingRequest(chunks=SomeChunks(found_chunks=["hello"]))
```
Then, calling _doc_to_suggest yields:
```python
suggest = redis_suggester._doc_to_suggest(doc, request)
```

---

## Documentation for `RedisSuggester._search_docs`

### Functionality
Executes the provided RediSearch query string against the Redis search index and returns matching documents. It sorts results by the "prob" field in descending order and fetches results using paging.

### Parameters
- `text_query`: The RediSearch query string to execute.
- `top_k`: The maximum number of results to return. The number of documents fetched is top_k multiplied by 100.

### Usage
- **Purpose**: Retrieve matching document objects from the Redis search index.

#### Example
Assume a RedisSuggester instance called `suggester`:
```python
results = suggester._search_docs("example query", 10)
for doc in results:
    print(doc)
```

---

## Documentation for `RedisSuggester.get_topk_suggestions`

### Functionality
Retrieves the top-k suggestion results for a given request by running strict and soft queries concurrently. It deduplicates and prioritizes results based on labels and scores.

### Parameters
- `request`: An instance of SuggestingRequest containing the context for suggestions.
- `top_k`: The maximum number of suggestions to return (default is 10).

### Usage
- **Purpose:** Merges concurrent query results to generate suggestions.

#### Example
```python
request = SuggestingRequest(...)
results = redis_suggester.get_topk_suggestions(request, top_k=5)
```