## Documentation for `RedisSuggestionPhraseManager`

### Functionality

This class manages suggestion phrases in a Redis database. It tokenizes phrases into chunks, creates search indexes for fast retrieval, and offers CRUD operations for suggestion data. It also handles weight adjustments for relevance scoring and efficient searching by organizing data into multiple chunks.

### Motivation

The primary goal is to offer a robust solution for storing and retrieving suggestion phrases in Redis. By breaking down phrases into chunks and creating specialized indexes, this class improves the efficiency of search operations and enables better result sorting, which is essential for applications that rely on rapid access to dynamic suggestion data.

### Inheritance

RedisSuggestionPhraseManager extends AbstractSuggestionPhraseManager, ensuring a consistent interface while implementing Redis-specific logic for phrase management. This inheritance supports modularity and ease of maintenance in the overall system design.

### Parameters

- `redis_url`: URL for connecting to the Redis database.
- `tokenizer`: An instance of SuggestingTokenizer for splitting phrases.
- `index_name`: The name of the search index in Redis.
- `max_chunks`: The maximum number of tokenized chunks (capped at 20).

### Usage Example

Below is a short example demonstrating how to instantiate and use RedisSuggestionPhraseManager:

```python
from embedding_studio.suggesting.redis.phrases_manager import RedisSuggestionPhraseManager
from embedding_studio.suggesting.tokenizer import SuggestingTokenizer

tokenizer = SuggestingTokenizer()
manager = RedisSuggestionPhraseManager(
    "redis://localhost", tokenizer, index_name="suggestion_phrases",
    max_chunks=20
)

# Use the manager to interact with Redis
result = manager.redis_client.get("some_key")
```

### Method Documentation

#### `redis_client`

- **Functionality**: Returns the active Redis client instance used by the RedisSuggestionPhraseManager to perform Redis operations.
- **Parameters**: None.
- **Usage**: Access the underlying Redis client for executing Redis commands and queries.
- **Example**:
  ```python
  redis_client = manager.redis_client
  ```

#### `search_client`

- **Functionality**: This property returns the initialized Redisearch Client for executing search queries on suggestion phrases.
- **Parameters**: None (read-only property).
- **Usage**: Access the Redisearch Client instance for query operations on suggestion phrases.
- **Example**:
  ```python
  client = manager.search_client
  result = client.search('example phrase')
  ```

#### `domains_search_client`

- **Functionality**: This method returns the client instance used for domain-specific search operations in Redis.
- **Parameters**: None.
- **Usage**: Retrieve the configured domains search client for executing domain-based search queries.
- **Example**:
  ```python
  domains_client = redis_manager.domains_search_client
  ```

#### `_create_main_index`

- **Functionality**: Creates the primary search index in Redis for storing phrase data.
- **Parameters**: 
  - `index_name`: The name of the Redisearch index to create.
- **Usage**: Set up the Redisearch index and client for suggestion phrase document retrieval.
- **Example**:
  ```python
  manager._create_main_index(index_name='suggestion_phrases')
  ```

#### `convert_phrase_to_request`

- **Functionality**: Converts a given phrase into a structured SuggestingRequest.
- **Parameters**:
  - `phrase`: A non-empty string representing the phrase to be converted.
  - `domain`: An optional string specifying the domain for the suggestion request.
- **Return Value**: Returns a `SuggestingRequest` object with structured data.
- **Usage**: Structure and process text input to generate a formatted request for suggestions.
- **Example**:
  ```python
  request = convert_phrase_to_request("hello world", "en")
  ```

#### `_convert_phrase`

- **Functionality**: Converts a SuggestingPhrase object into a flattened dictionary suitable for Redis storage.
- **Parameters**: 
  - `suggesting_phrase`: The SuggestingPhrase object to convert.
- **Usage**: Internally convert a SuggestingPhrase into a Redis-ready dictionary.
- **Example**:
  ```python
  doc = redis_manager._convert_phrase(phrase_obj)
  ```

#### `add`

- **Functionality**: Inserts multiple SuggestingPhrase documents into a Redis data store.
- **Parameters**:
  - `phrases`: A list of SuggestingPhrase objects to be inserted.
- **Usage**: Efficiently add suggestion phrases into Redis for fast retrieval and search.
- **Example**:
  ```python
  inserted_ids = manager.add(phrases)
  ```

#### `delete`

- **Functionality**: Deletes suggestion phrase documents from a Redis database.
- **Parameters**:
  - `phrase_ids`: A list of strings representing the IDs of the phrases to be deleted.
- **Usage**: Remove one or multiple suggestion phrases from the Redis store.
- **Example**:
  ```python
  manager.delete(phrase_ids)
  ```

#### `update_probability`

- **Functionality**: Updates the probability score for a suggestion phrase document.
- **Parameters**:
  - `phrase_id`: Unique identifier for the phrase document.
  - `new_probability`: Float value (0-1) for the new probability.
- **Usage**: Updates a phrase document's probability in Redis with proper validation.
- **Example**:
  ```python
  manager.update_probability("example_phrase_id", 0.85)
  ```

#### `add_labels`

- **Functionality**: Adds new labels to an existing document in Redis, preventing duplication.
- **Parameters**:
  - `phrase_id`: The unique string identifier of the phrase document.
  - `labels`: A list of label strings to add to the document.
- **Usage**: Update a document's labels by combining them with any pre-existing labels.
- **Example**:
  ```python
  manager.add_labels("doc123", ["blue", "green"])
  ```

#### `remove_labels`

- **Functionality**: Removes specified labels from a suggestion phrase document in Redis.
- **Parameters**:
  - `phrase_id`: The ID of the phrase document to update.
  - `labels`: A list of label strings to remove from the document.
- **Usage**: Use this method to remove unwanted labels from a phrase.
- **Example**:
  ```python
  manager.remove_labels("abc123", ["urgent", "obsolete"])
  ```

#### `remove_all_label_values`

- **Functionality**: Removes given label values from all documents in Redis.
- **Parameters**:
  - `labels`: List of strings representing labels to remove.
- **Usage**: Remove specified labels from documents in Redis.
- **Example**:
  ```python
  redis_manager.remove_all_label_values(["label1", "label2"])
  ```

#### `add_domains`

- **Functionality**: Updates a phrase document in Redis by appending domain values.
- **Parameters**:
  - `phrase_id`: The identifier of the phrase document to update.
  - `domains`: A list of domain strings to add.
- **Usage**: Enhance phrase documents with additional domain metadata.
- **Example**:
  ```python
  redis_manager.add_domains("example_id", ["tech", "news"])
  ```

#### `remove_domains`

- **Functionality**: Removes the specified domains from a phrase document stored in Redis.
- **Parameters**:
  - `phrase_id`: The ID of the document whose domains are to be removed.
  - `domains`: A list of domain names to remove.
- **Usage**: Remove domains from a suggestion phrase document.
- **Example**:
  ```python
  manager.remove_domains('phrase_id', ['example-domain'])
  ```

#### `remove_all_domain_values`

- **Functionality**: Removes specified domains from all documents in the Redis database that contain them.
- **Parameters**:
  - `domains`: A list of domain strings to remove.
- **Usage**: Remove unwanted domain tags from Redis records.
- **Example**:
  ```python
  redis_manager.remove_all_domain_values(["example-domain", "sample-domain"])
  ```

#### `get_info_by_id`

- **Functionality**: Fetches the phrase document by its ID.
- **Parameters**:
  - `phrase_id`: The string ID of the document.
- **Returns**: A SearchDocument object containing the document's full details.
- **Raises**: ValueError if no document is found for the provided phrase_id.
- **Usage**: Retrieve document information from Redis using a unique ID.
- **Example**:
  ```python
  doc = manager.get_info_by_id("example_id")
  ```

#### `list_phrases`

- **Functionality**: Returns a paginated list of full phrase documents rehydrated as SearchDocument objects.
- **Parameters**:
  - `offset`: Number of documents to skip (default is 0).
  - `limit`: Maximum number of documents to return (default is 100).
- **Usage**: Retrieve paginated phrase documents from Redis.
- **Example**:
  ```python
  docs = manager.list_phrases(offset=0, limit=100)
  ```