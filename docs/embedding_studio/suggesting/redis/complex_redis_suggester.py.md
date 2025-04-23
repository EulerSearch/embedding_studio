# Documentation for `ComplexRedisSuggester`

## Functionality
The `ComplexRedisSuggester` class combines multiple query generators to provide a flexible way to generate suggestions from a Redis store. It adapts the query selection based on the state of the suggestion request, using strategies like simple, prefix, combined, or most probable queries. This class was created to simplify the process of handling various dynamic suggestion requirements. By consolidating multiple query generation strategies into a single class, it facilitates easier maintenance and extension of suggestion functionalities.

## Inheritance
`ComplexRedisSuggester` inherits from `RedisSuggester`, inheriting core methods for interacting with Redis, such as storing and retrieving document suggestions.

## Parameters
- `redis_url`: The Redis client URL for data storage and retrieval.
- `tokenizer`: An instance of `SuggestingTokenizer` used for splitting text into meaningful chunks.
- `index_name` (optional): Name of the index where suggestion data is stored.
- `max_chunks` (optional): Maximum number of chunks allowed per document.

## Method: `ComplexRedisSuggester._generate_query`

### Functionality
This method selects and returns an appropriate query string based on the contents of a SuggestingRequest object. It dynamically chooses between different query generators (simple, prefix, combined, or most probable) depending on the presence of found_chunks, next_chunk, and the domain attribute.

### Parameters
- `request`: A SuggestingRequest object containing text chunks and a domain for context.
- `top_k`: An integer specifying the number of suggestions to return. Default is 10.
- `soft_match`: A boolean flag indicating whether to apply soft matching. Default is False.

### Usage
The method is used internally to generate a Redis query string based on the state of the request.

#### Example
```python
request = SuggestingRequest(
    chunks=Chunks(found_chunks=["example"], next_chunk=["ex"]),
    domain="example_domain"
)
query = complex_redis_suggester._generate_query(request, top_k=5)
print(query)
```

## Usage of `ComplexRedisSuggester`
Instantiate `ComplexRedisSuggester` with the required parameters to enable its suggestion functionality. Depending on the content of the request, the class selects the appropriate internal query generator.

#### Example
```python
suggester = ComplexRedisSuggester(
    redis_url="redis://localhost:6379/0",
    tokenizer=your_tokenizer,
    index_name="suggestion_phrases",
    max_chunks=20
)
query = suggester._generate_query(request, top_k=10, soft_match=False)
```