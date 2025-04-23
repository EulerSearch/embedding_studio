# Configuring Redis for Embedding Studio's Suggestion System

This tutorial explains how to set up and configure Redis for Embedding Studio's suggestion system, covering installation, configuration, and optimization techniques.

## Why Redis for Suggestions?

Redis with RediSearch provides several advantages for a suggestion system:

- **Performance**: Extremely fast for prefix-based searches
- **Memory efficiency**: Optimized for in-memory operations
- **Advanced text search**: Built-in support for fuzzy matching and prefix queries
- **Scalability**: Can handle high volumes of suggestion requests

## Installation Requirements

### 1. Redis Stack Installation

You need Redis with the RediSearch module. The easiest way is to use Redis Stack, which includes RediSearch:

**Using Docker:**

```bash
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

**Manual Installation:**

Follow the [Redis Stack installation guide](https://redis.io/docs/stack/get-started/install/) for your platform.

### 2. Environment Configuration

In your Embedding Studio environment, set these configuration variables:

```bash
# Redis connection settings
REDIS_URL=redis://localhost:6379

# Suggestion system settings
SUGGESTING_REDIS_COLLECTION=suggestion_phrases
SUGGESTING_MAX_CHUNKS=20
```

These can be defined in your `.env` file or set directly in your environment.

## Basic Redis Configuration

Create a `redis.conf` file with these recommended settings for the suggestion system:

```
# Memory configuration
maxmemory 2gb
maxmemory-policy allkeys-lru

# RediSearch configuration
search.query_timeout_ms 5000
search.prefix_min_length 1
search.max_expansion 100

# Performance tuning
io-threads 4
```

Start Redis with this configuration:

```bash
redis-server /path/to/redis.conf
```

## Configuring the Suggesting Engine

### 1. Initializing the Redis Suggester

In Embedding Studio, the Redis-based suggester is configured in `app_context.py`:

```python
from embedding_studio.suggesting.redis.complex_redis_suggester import ComplexRedisSuggester
from embedding_studio.suggesting.tokenizer import SuggestingTokenizer

suggester = ComplexRedisSuggester(
    redis_url=settings.REDIS_URL,
    tokenizer=SuggestingTokenizer(),
    index_name=settings.SUGGESTING_REDIS_COLLECTION,
    max_chunks=settings.SUGGESTING_MAX_CHUNKS
)
```

### 2. Index Creation

When first used, Embedding Studio automatically creates the required Redis search index with the following fields:

- Text fields for chunks (`chunk_0`, `chunk_1`, etc.)
- Numeric fields for metadata (probability, chunk count)
- Tag fields for labels and domains

You can manually create or reset the index using Redis CLI:

```bash
# Connect to Redis
redis-cli

# Optionally, drop existing index
FT.DROPINDEX suggestion_phrases

# Exit
exit
```

The index will be automatically recreated when the application restarts.

## Advanced Configuration

### Memory Optimization

For large suggestion datasets, optimize Redis memory usage:

```
# In redis.conf
activedefrag yes
maxmemory-samples 10
```

### Fine-tuning RediSearch

Adjust RediSearch settings for your specific needs:

```
# Faster but less accurate fuzzy matching
search.max_edit_distance 1

# Optimize for prefix searches
search.prefix_min_length 1
search.prefix_max_expansion 200
```

### Persistence Configuration

Configure how Redis persists suggestion data:

```
# In redis.conf
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec
```

This balances performance with data safety:
- RDB snapshots at intervals
- Append-only file for better durability
- Syncing every second for good performance/safety balance

## Securing Redis

For production environments, secure your Redis instance:

```
# In redis.conf
bind 127.0.0.1
requirepass YourStrongPassword
```

Update your Embedding Studio connection string:

```
REDIS_URL=redis://:YourStrongPassword@localhost:6379
```

## Scaling Configuration

For high-volume production environments:

### Redis Sentinel for High Availability

```
# In sentinel.conf
sentinel monitor mymaster 127.0.0.1 6379 2
sentinel down-after-milliseconds mymaster 5000
sentinel failover-timeout mymaster 60000
```

Update your connection string:
```
REDIS_URL=redis+sentinel://mymaster/0?sentinels=localhost:26379,localhost:26380
```

### Redis Cluster for Horizontal Scaling

For extremely large suggestion databases, configure Redis Cluster:

```bash
# Create cluster
redis-cli --cluster create 127.0.0.1:7000 127.0.0.1:7001 127.0.0.1:7002 127.0.0.1:7003 127.0.0.1:7004 127.0.0.1:7005 --cluster-replicas 1
```

Adjust your connection string:
```
REDIS_URL=redis://localhost:7000,localhost:7001,localhost:7002
```

## Performance Monitoring

Monitor your Redis suggestion system performance:

```bash
# Connect to Redis
redis-cli

# Monitor memory usage
INFO memory

# Check search index stats
FT.INFO suggestion_phrases

# Exit
exit
```

Key metrics to monitor:
- Memory usage
- Search latency
- Index size
- Cache hit ratio

## Loading Initial Suggestions

Here's how to preload suggestions into your Redis instance:

```python
import requests

# Load a batch of suggestions
suggestions = [
    {
        "phrase": "how to make pasta",
        "prob": 0.8,
        "labels": ["cooking", "italian"],
        "domains": ["recipes", "food"]
    },
    # Add more suggestions...
]

# Send to API
response = requests.post(
    "http://your-embedding-studio/suggesting/phrases/add",
    json={"phrases": suggestions}
)
print(f"Added {len(response.json()['phrase_ids'])} suggestions")
```

## Common Configuration Issues

### Problem: High Memory Usage
- **Solution**: Decrease `SUGGESTING_MAX_CHUNKS` or adjust Redis `maxmemory`

### Problem: Slow Suggestion Responses
- **Solution**: Increase `io-threads` and adjust RediSearch timeout:
  ```
  search.query_timeout_ms 10000
  ```

### Problem: Missing Fuzzy Matches
- **Solution**: Increase edit distance:
  ```
  search.max_edit_distance 2
  ```