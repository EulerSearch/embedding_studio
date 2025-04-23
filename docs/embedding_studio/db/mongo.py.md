## Documentation for `_mongo_env_client`

### Functionality

The `_mongo_env_client` function is designed to create and return a MongoDB client tailored to the running environment. If the `ES_UNIT_TESTS` environment variable is set to "1", the function will return a `mongomock.MongoClient` instance, which is suitable for testing purposes. In all other scenarios, the function provides a `pymongo.MongoClient` instance, ensuring direct connection and timezone awareness, with an unlimited pool size.

### Parameters

- `*args`: Variable length argument list passed to the client constructor, allowing for positional arguments to be forwarded.
- `**kwargs`: Arbitrary keyword arguments passed to the client constructor, which can be utilized during client initialization. 

### Usage

Use `_mongo_env_client` to initialize a MongoDB client that is adaptable to various environments, whether for testing or production.

#### Example

```python
from embedding_studio.db.mongo import _mongo_env_client

# Create a MongoDB client instance
client = _mongo_env_client(
    'your_mongo_url',
    uuidRepresentation='standard',
    tz_aware=True
)

# Alternative Example
client = _mongo_env_client("mongodb://localhost:27017")
```