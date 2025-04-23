## Next Steps

Now that you understand how to configure Embedding Studio, you can proceed to:

1. [Setting up with Docker](docker_quickstart.md)
2. [Understanding the architecture](architecture_overview.md)
3. [Developing custom plugins](../plugins/creating_plugins.md)## Example Basic Configuration

Here's a comprehensive `.env` file example for getting started with Embedding Studio:

```env
# API and general settings
API_V1_STR=/api/v1
OPEN_INTERNAL_ENDPOINTS=True
OPEN_MOCKED_ENDPOINTS=False

# MongoDB settings
FINETUNING_MONGO_HOST=mongo
FINETUNING_MONGO_PORT=27017
FINETUNING_MONGO_DB_NAME=embedding_studio
FINETUNING_MONGO_USERNAME=root
FINETUNING_MONGO_PASSWORD=mongopassword

# Redis settings
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=redispassword

# PostgreSQL settings
POSTGRES_HOST=pgvector
POSTGRES_PORT=5432
POSTGRES_USER=embedding_studio
POSTGRES_PASSWORD=123456789
POSTGRES_DB=embedding_studio

# MinIO settings
MINIO_HOST=minio
MINIO_PORT=9000
MINIO_ROOT_USER=root
MINIO_ROOT_PASSWORD=miniopassword
MINIO_DEFAULT_BUCKETS=embeddingstudio
MINIO_ACCESS_KEY=mtGNiEvoTL6C0EXAMPLE
MINIO_SECRET_KEY=HY5JserXAaWmphNyCpQPEXAMPLEKEYEXAMPLEKEY

# MySQL settings (for MLflow)
MYSQL_HOST=mlflow_db
MYSQL_PORT=3306
MYSQL_DATABASE=mlflow
MYSQL_USER=mlflow_user
MYSQL_PASSWORD=Baxp3O5rUvpIxiD77BfZ
MYSQL_ROOT_PASSWORD=PrK5qmPTDsm2IYKvHVG8

# MLflow settings
MLFLOW_HOST=mlflow
MLFLOW_PORT=5001

# Plugins configuration
ES_PLUGINS_PATH=plugins
INFERENCE_USED_PLUGINS=["TextDefaultFineTuningMethodForText", "HFCategoriesTextFineTuningMethod"]

# Worker settings
FINE_TUNING_WORKER_MAX_RETRIES=3
FINE_TUNING_WORKER_TIME_LIMIT=18000000
UPSERTION_BATCH_SIZE=16
UPSERTION_INFERENCE_BATCH_SIZE=16
REINDEX_BATCH_SIZE=16
REINDEX_MAX_SUBTASKS_COUNT=4

# Inference settings
INFERENCE_MODEL_REPO=/models
INFERENCE_HOST=inference_worker
INFERENCE_GRPC_PORT=8001
```
### External Services Retry Settings

These settings control the retry behavior for various external service operations:

#### Global Retry Settings

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DEFAULT_MAX_ATTEMPTS` | Default retry attempts for all operations | `3` | No |
| `DEFAULT_WAIT_TIME_SECONDS` | Default wait time between retries | `3.0` | No |

#### Data Loader Retry Settings

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `S3_READ_CREDENTIALS_ATTEMPTS` | S3 credential read attempts | `3` | No |
| `S3_READ_WAIT_TIME_SECONDS` | S3 credential wait time | `3.0` | No |
| `S3_DOWNLOAD_DATA_ATTEMPTS` | S3 download attempts | `3` | No |
| `S3_DOWNLOAD_DATA_WAIT_TIME_SECONDS` | S3 download wait time | `3.0` | No |
| `GCP_READ_CREDENTIALS_ATTEMPTS` | GCP credential read attempts | `3` | No |
| `GCP_READ_WAIT_TIME_SECONDS` | GCP credential wait time | `3.0` | No |
| `GCP_DOWNLOAD_DATA_ATTEMPTS` | GCP download attempts | `3` | No |
| `GCP_DOWNLOAD_DATA_WAIT_TIME_SECONDS` | GCP download wait time | `3.0` | No |
| `PGSQL_DATA_LOADER_ATTEMPTS` | PostgreSQL data load attempts | `3` | No |
| `PGSQL_DATA_LOADER_WAIT_TIME_SECONDS` | PostgreSQL load wait time | `3.0` | No |

#### MLflow Operation Retry Settings

MLflow operations have individual retry settings with this pattern:
```
MLFLOW_<OPERATION>_ATTEMPTS
MLFLOW_<OPERATION>_WAIT_TIME_SECONDS
```

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MLFLOW_LOG_METRIC_ATTEMPTS` | MLflow metric logging attempts | `3` | No |
| `MLFLOW_LOG_METRIC_WAIT_TIME_SECONDS` | Wait time between attempts | `3.0` | No |
| `MLFLOW_LOG_PARAM_ATTEMPTS` | MLflow param logging attempts | `3` | No |
| `MLFLOW_LOG_PARAM_WAIT_TIME_SECONDS` | Wait time between attempts | `3.0` | No |
| `MLFLOW_LOG_MODEL_ATTEMPTS` | MLflow model logging attempts | `3` | No |
| `MLFLOW_LOG_MODEL_WAIT_TIME_SECONDS` | Wait time between attempts | `3.0` | No |
| `MLFLOW_LOAD_MODEL_ATTEMPTS` | MLflow model loading attempts | `3` | No |
| `MLFLOW_LOAD_MODEL_WAIT_TIME_SECONDS` | Wait time between attempts | `3.0` | No |
| `MLFLOW_DELETE_MODEL_ATTEMPTS` | MLflow model deletion attempts | `3` | No |
| `MLFLOW_DELETE_MODEL_WAIT_TIME_SECONDS` | Wait time between attempts | `3.0` | No |
| `MLFLOW_SEARCH_RUNS_ATTEMPTS` | MLflow run search attempts | `3` | No |
| `MLFLOW_SEARCH_RUNS_WAIT_TIME_SECONDS` | Wait time between attempts | `3.0` | No |
| `MLFLOW_END_RUN_ATTEMPTS` | MLflow end run attempts | `3` | No |
| `MLFLOW_END_RUN_WAIT_TIME_SECONDS` | Wait time between attempts | `3.0` | No |
| `MLFLOW_GET_RUN_ATTEMPTS` | MLflow get run attempts | `3` | No |
| `MLFLOW_GET_RUN_WAIT_TIME_SECONDS` | Wait time between attempts | `3.0` | No |
| `MLFLOW_SEARCH_EXPERIMENTS_ATTEMPTS` | MLflow experiment search attempts | `3` | No |
| `MLFLOW_SEARCH_EXPERIMENTS_WAIT_TIME_SECONDS` | Wait time between attempts | `3.0` | No |
| `MLFLOW_DELETE_EXPERIMENT_ATTEMPTS` | MLflow experiment deletion attempts | `3` | No |
| `MLFLOW_DELETE_EXPERIMENT_WAIT_TIME_SECONDS` | Wait time between attempts | `3.0` | No |
| `MLFLOW_CREATE_EXPERIMENT_ATTEMPTS` | MLflow experiment creation attempts | `3` | No |
| `MLFLOW_CREATE_EXPERIMENT_WAIT_TIME_SECONDS` | Wait time between attempts | `3.0` | No |
| `MLFLOW_GET_EXPERIMENT_ATTEMPTS` | MLflow experiment retrieval attempts | `3` | No |
| `MLFLOW_GET_EXPERIMENT_WAIT_TIME_SECONDS` | Wait time between attempts | `3.0` | No |### Feature-Specific Settings

#### Clickstream Settings

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `CLICKSTREAM_TIME_MAX_DELTA_MINUS_SEC` | Max allowed time in past (s) | `43200` | No |
| `CLICKSTREAM_TIME_MAX_DELTA_PLUS_SEC` | Max allowed time in future (s) | `300` | No |

#### Query Parsing Settings

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `QUERY_PARSING_DB_META_INFO` | Query parsing metadata | `{"enlarged_limit": 36}` | No |

#### Suggesting Settings (Redis-based)

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `SUGGESTING_MAX_CHUNKS` | Maximum suggestion chunks | `20` | No |
| `SUGGESTING_REDIS_COLLECTION` | Redis collection for suggestions | `suggestion_phrases` | No |

Redis is used primarily for the suggestion system, providing fast access to autocompletion phrases and recommendations.
# Understanding Embedding Studio Configuration

This guide explains how to configure Embedding Studio using environment variables and other configuration methods. Understanding these settings is crucial for customizing the system to your specific needs.

## Configuration Hierarchy

Embedding Studio uses a hierarchical configuration system with the following priority (highest to lowest):

1. **Environment Variables**: Direct system or container environment variables
2. **`.env` Files**: Local development configuration files
3. **Default Settings**: Built-in fallback values in the codebase

## Common Configuration Methods

### Using Environment Variables

Environment variables can be set directly in your shell or container environment:

```bash
export POSTGRES_HOST=localhost
export REDIS_HOST=redis
```

### Using .env Files

Create a `.env` file in your project root directory:

```
POSTGRES_HOST=localhost
REDIS_PORT=6379
```

### Docker Compose Configuration

When using Docker Compose, you can specify environment variables:

1. Directly in `docker-compose.yml`:
   ```yaml
   services:
     embedding_studio:
       environment:
         - POSTGRES_HOST=pgvector
         - REDIS_HOST=redis
   ```

2. Using an env_file reference:
   ```yaml
   services:
     embedding_studio:
       env_file:
         - .env
   ```

## Configuration Categories

This section outlines the full range of configuration options available in Embedding Studio, organized by functional category. Some advanced settings are particularly useful for production deployments and performance tuning.

### Service Connection Settings

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `API_V1_STR` | API endpoint prefix | `/api/v1` | No |
| `OPEN_INTERNAL_ENDPOINTS` | Enable internal API endpoints | `True` | No |
| `OPEN_MOCKED_ENDPOINTS` | Enable mocked API endpoints for testing | `False` | No |

### Database Settings

#### PostgreSQL/Vector Database

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `POSTGRES_HOST` | PostgreSQL server host | `localhost` | No |
| `POSTGRES_PORT` | PostgreSQL server port | `5432` | No |
| `POSTGRES_USER` | PostgreSQL username | `embedding_studio` | No |
| `POSTGRES_PASSWORD` | PostgreSQL password | `123456789` | No |
| `POSTGRES_DB` | PostgreSQL database name | `embedding_studio` | No |

#### MongoDB Connections (for Queue and Data Storage)

MongoDB is used for both the task queue system and storing various metadata. Each MongoDB collection has its own connection settings with the following pattern:

```
<COLLECTION_NAME>_MONGO_HOST
<COLLECTION_NAME>_MONGO_PORT
<COLLECTION_NAME>_MONGO_DB_NAME
<COLLECTION_NAME>_MONGO_USERNAME
<COLLECTION_NAME>_MONGO_PASSWORD
```

Main collections include:

- `FINETUNING_MONGO_*` - Fine-tuning task data and queue system
- `CLICKSTREAM_MONGO_*` - User interaction data
- `EMBEDDINGS_MONGO_*` - Embedding metadata
- `SESSIONS_FOR_IMPROVEMENT_MONGO_*` - Session data for model improvement
- `UPSERTION_MONGO_*` - Upsertion task data and queue
- `INFERENCE_DEPLOYMENT_MONGO_*` - Inference model deployment data and tasks

Example:
```
FINETUNING_MONGO_HOST=mongo
FINETUNING_MONGO_PORT=27017
FINETUNING_MONGO_DB_NAME=embedding_studio
FINETUNING_MONGO_USERNAME=root
FINETUNING_MONGO_PASSWORD=mongopassword
```

### Redis (for Suggestions)

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `REDIS_HOST` | Redis server hostname | `localhost` | No |
| `REDIS_PORT` | Redis server port | `6379` | No |
| `REDIS_PASSWORD` | Redis server password | `redispassword` | No |
| `REDIS_URL` | Complete Redis URL | `redis://{REDIS_HOST}:{REDIS_PORT}/0` | No |

### Storage and Tracking Systems

#### MLflow Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MLFLOW_HOST` | MLflow server hostname | `localhost` | No |
| `MLFLOW_PORT` | MLflow server port | `5001` | No |
| `MLFLOW_TRACKING_URI` | Complete MLflow URI | `http://{MLFLOW_HOST}:{MLFLOW_PORT}` | No |

#### MinIO/S3 Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MINIO_HOST` | MinIO server hostname | `localhost` | No |
| `MINIO_PORT` | MinIO server port | `9000` | No |
| `MINIO_ROOT_USER` | MinIO admin username | `root` | No |
| `MINIO_ROOT_PASSWORD` | MinIO admin password | `miniopassword` | No |
| `MINIO_DEFAULT_BUCKETS` | Default bucket names | `embeddingstudio` | No |
| `MINIO_ACCESS_KEY` | S3 compatible access key | `mtGNiEvoTL6C0EXAMPLE` | No |
| `MINIO_SECRET_KEY` | S3 compatible secret key | `HY5JserXAaWmphNyCpQPEXAMPLEKEYEXAMPLEKEY` | No |

#### MySQL (for MLflow)

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MYSQL_HOST` | MySQL server hostname | `localhost` | No |
| `MYSQL_PORT` | MySQL server port | `3306` | No |
| `MYSQL_DATABASE` | MySQL database name | `mlflow` | No |
| `MYSQL_USER` | MySQL username | `mlflow_user` | No |
| `MYSQL_PASSWORD` | MySQL password | `Baxp3O5rUvpIxiD77BfZ` | No |
| `MYSQL_ROOT_PASSWORD` | MySQL root password | `PrK5qmPTDsm2IYKvHVG8` | No |

### Worker Configuration

#### Fine-Tuning Worker

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `FINE_TUNING_WORKER_MAX_RETRIES` | Maximum retry attempts | `3` | No |
| `FINE_TUNING_WORKER_TIME_LIMIT` | Task time limit (ms) | `18000000` | No |

#### Inference Worker

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `INFERENCE_WORKER_MAX_RETRIES` | Maximum retry attempts | `3` | No |
| `INFERENCE_WORKER_TIME_LIMIT` | Task time limit (ms) | `18000000` | No |
| `INFERENCE_WORKER_MAX_DEPLOYED_MODELS` | Max concurrent models | `3` | No |
| `INFERENCE_HOST` | Inference server hostname | `localhost` | No |
| `INFERENCE_GRPC_PORT` | Inference server gRPC port | `8001` | No |
| `INFERENCE_MODEL_REPO` | Model repository path | `/models` | No |

#### Upsertion Worker

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `UPSERTION_BATCH_SIZE` | Processing batch size | `16` | No |
| `UPSERTION_INFERENCE_BATCH_SIZE` | Inference batch size | `16` | No |
| `UPSERTION_WORKER_MAX_RETRIES` | Maximum retry attempts | `3` | No |
| `UPSERTION_WORKER_TIME_LIMIT` | Task time limit (ms) | `18000000` | No |
| `UPSERTION_IGNORE_FAILED_ITEMS` | Continue despite failures | `True` | No |
| `UPSERTION_PASS_TO_REINDEXING_MODEL` | Pass updates to reindex model | `True` | No |
| `DELETE_IMPROVED_VECTORS_ON_UPSERTION` | Delete vectors on upsert | `True` | No |

#### Improvement Worker

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `IMPROVEMENT_WORKER_MAX_RETRIES` | Maximum retry attempts | `3` | No |
| `IMPROVEMENT_WORKER_TIME_LIMIT` | Task time limit (ms) | `600000` | No |
| `IMPROVEMENT_SECONDS_INTERVAL` | Interval between improvement runs | `5` | No |

#### Reindex Worker

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `REINDEX_BATCH_SIZE` | Items per batch | `16` | No |
| `REINDEX_MAX_SUBTASKS_COUNT` | Max concurrent subtasks | `4` | No |
| `REINDEX_WORKER_MAX_RETRIES` | Maximum retry attempts | `3` | No |
| `REINDEX_WORKER_TIME_LIMIT` | Task time limit (ms) | `180000000` | No |
| `REINDEX_WORKER_LOOP_WAIT_TIME` | Wait time between loops (s) | `10` | No |
| `REINDEX_WORKER_MAX_FAILED` | Max failed items before stopping | `-1` | No |
| `REINDEX_INITIATE_MODEL_DEPLOYMENT` | Deploy model after reindex | `True` | No |

#### Deletion Worker

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DELETION_WORKER_MAX_RETRIES` | Maximum retry attempts | `3` | No |
| `DELETION_WORKER_TIME_LIMIT` | Task time limit (ms) | `18000000` | No |
| `DELETION_PASS_TO_REINDEXING_MODEL` | Pass deletions to reindex | `True` | No |

### Plugin System

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ES_PLUGINS_PATH` | Path to custom plugins | `plugins` | No |
| `INFERENCE_USED_PLUGINS` | List of enabled plugins | `["HFDictTextFineTuningMethod", "HFCategoriesTextFineTuningMethod"]` | Yes |

This setting controls which plugins are available to the system. Custom plugins must be added to this list to be recognized.

### Retry Strategy Settings

#### Global Retry Settings

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DEFAULT_MAX_ATTEMPTS` | Default retry attempts | `3` | No |
| `DEFAULT_WAIT_TIME_SECONDS` | Default wait between retries | `3.0` | No |

#### Data Loader Retry Settings

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `S3_READ_CREDENTIALS_ATTEMPTS` | S3 credential read attempts | `3` | No |
| `S3_READ_WAIT_TIME_SECONDS` | S3 credential wait time | `3.0` | No |
| `S3_DOWNLOAD_DATA_ATTEMPTS` | S3 download attempts | `3` | No |
| `S3_DOWNLOAD_DATA_WAIT_TIME_SECONDS` | S3 download wait time | `3.0` | No |
| `GCP_READ_CREDENTIALS_ATTEMPTS` | GCP credential read attempts | `3` | No |
| `GCP_READ_WAIT_TIME_SECONDS` | GCP credential wait time | `3.0` | No |
| `GCP_DOWNLOAD_DATA_ATTEMPTS` | GCP download attempts | `3` | No |
| `GCP_DOWNLOAD_DATA_WAIT_TIME_SECONDS` | GCP download wait time | `3.0` | No |
| `PGSQL_DATA_LOADER_ATTEMPTS` | PostgreSQL data load attempts | `3` | No |
| `PGSQL_DATA_LOADER_WAIT_TIME_SECONDS` | PostgreSQL load wait time | `3.0` | No |

#### MLflow Operation Retry Settings

MLflow operations have individual retry settings with pattern:
```
MLFLOW_<OPERATION>_ATTEMPTS
MLFLOW_<OPERATION>_WAIT_TIME_SECONDS
```

Examples include:
- `MLFLOW_LOG_METRIC_*`
- `MLFLOW_LOG_PARAM_*`
- `MLFLOW_LOG_MODEL_*`
- `MLFLOW_LOAD_MODEL_*`

### Performance Tuning Settings

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `INFERENCE_WORKER_MAX_DEPLOYED_MODELS` | Maximum models deployed simultaneously | `3` | No |
| `REINDEX_WORKER_MAX_FAILED` | Max failed items before stopping reindex | `-1` (unlimited) | No |
| `REINDEX_MAX_TASKS_COUNT` | Maximum concurrent reindex tasks | `2` | No |
| `REINDEX_TASK_DELAY_TIME` | Delay between reindex tasks (ms) | `1200000` (20 min) | No |
| `IMPROVEMENT_SECONDS_INTERVAL` | Interval between improvement runs (s) | `5` | No |
| `REINDEX_INITIATE_MODEL_DEPLOYMENT_PENDING_TIME` | Wait time before deployment (s) | `18000` | No |

### Feature Flags

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPEN_INTERNAL_ENDPOINTS` | Enable internal API endpoints | `True` | No |
| `OPEN_MOCKED_ENDPOINTS` | Enable mocked API endpoints for testing | `False` | No |
| `UPSERTION_IGNORE_FAILED_ITEMS` | Continue despite failed items | `True` | No |
| `DELETION_PASS_TO_REINDEXING_MODEL` | Pass deletions to reindex model | `True` | No |
| `REINDEX_INITIATE_MODEL_DEPLOYMENT` | Deploy model after reindexing | `True` | No |
| `DELETE_IMPROVED_VECTORS_ON_UPSERTION` | Remove vectors when upserting | `True` | No |
| `REINDEX_IGNORE_FAILED_ITEMS` | Continue reindexing despite failures | `True` | No |

### Processing Behaviour Settings

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `INFERENCE_QUERY_EMBEDDING_ATTEMPTS` | Query embedding max attempts | `3` | No |
| `INFERENCE_QUERY_EMBEDDING_WAIT_TIME_SECONDS` | Wait time between query attempts | `3.0` | No |
| `INFERENCE_ITEMS_EMBEDDING_ATTEMPTS` | Item embedding max attempts | `3` | No |
| `INFERENCE_ITEMS_EMBEDDING_WAIT_TIME_SECONDS` | Wait time between item embedding attempts | `3.0` | No |
| `REINDEX_WORKER_LOOP_WAIT_TIME` | Wait time between reindex loops (s) | `10` | No |
| `REINDEX_SUBWORKER_LOOP_WAIT_TIME` | Wait time between subworker loops (s) | `10` | No |
| `REINDEX_INITIATE_MODEL_DEPLOYMENT_LOOP_WAIT_TIME` | Wait time for deployment checks (s) | `30` | No |


## Advanced Configurations for Production

When deploying Embedding Studio in production environments, these additional configurations can help optimize performance, resiliency, and resource utilization:

### Production Environment Optimizations

#### Memory Usage Optimization

- Adjust `UPSERTION_BATCH_SIZE` and `REINDEX_BATCH_SIZE` based on available memory
- Use `INFERENCE_WORKER_MAX_DEPLOYED_MODELS` to control model memory consumption
- Consider adjusting PostgreSQL's `work_mem` for vector operations (external to Embedding Studio)

#### CPU/GPU Utilization

- Set `REINDEX_MAX_SUBTASKS_COUNT` based on available CPU cores
- For GPU-intensive operations, ensure `FINE_TUNING_WORKER_TIME_LIMIT` is sufficient
- Adjust worker concurrency based on available CPU/GPU resources

#### Network Resilience

- Increase retry counts for cloud storage operations in high-latency environments
- Adjust `DEFAULT_WAIT_TIME_SECONDS` for backoff between retries
- Set appropriate timeouts for database connections

### Production Security Considerations

- Use environment variables rather than `.env` files
- Set strong passwords for `REDIS_PASSWORD`, `POSTGRES_PASSWORD`, etc.
- Consider using Docker secrets for sensitive values
- Disable `OPEN_INTERNAL_ENDPOINTS` and `OPEN_MOCKED_ENDPOINTS` in production

### High-Availability Configuration

For high-availability deployments:

- Configure appropriate health checks using `/api/v1/ping` endpoint
- Set up monitoring for task queues and worker health
- Configure redundant Redis and MongoDB instances (external to Embedding Studio configuration)

### Logging and Monitoring

While not directly configured through environment variables, consider:

- Setting up log aggregation for worker services
- Monitoring task queue lengths in MongoDB
- Tracking worker performance metrics

## Example Production Configuration

Here's an example `.env` file tuned for a production environment:

```env
# API and general settings
API_V1_STR=/api/v1
OPEN_INTERNAL_ENDPOINTS=False
OPEN_MOCKED_ENDPOINTS=False

# Production-optimized batch sizes
UPSERTION_BATCH_SIZE=32
UPSERTION_INFERENCE_BATCH_SIZE=32
REINDEX_BATCH_SIZE=32
REINDEX_MAX_SUBTASKS_COUNT=8

# Increased timeouts for stability
FINE_TUNING_WORKER_TIME_LIMIT=36000000
REINDEX_WORKER_TIME_LIMIT=360000000
UPSERTION_WORKER_TIME_LIMIT=36000000

# Enhanced retry settings
DEFAULT_MAX_ATTEMPTS=5
DEFAULT_WAIT_TIME_SECONDS=5.0
S3_DOWNLOAD_DATA_ATTEMPTS=5
GCP_DOWNLOAD_DATA_ATTEMPTS=5

# Resource limits
INFERENCE_WORKER_MAX_DEPLOYED_MODELS=5
REINDEX_WORKER_MAX_FAILED=100

# Performance tuning
IMPROVEMENT_SECONDS_INTERVAL=30
REINDEX_TASK_DELAY_TIME=3600000  # 60 minutes
```

These examples provide a starting point for tuning Embedding Studio for your specific production needs. Always monitor system performance and adjust configurations as necessary based on observed behavior.

## Troubleshooting Common Configuration Issues

### Connection Problems

- **MongoDB connection failures**: Verify `*_MONGO_HOST`, `*_MONGO_PORT`, and credentials
- **Redis connection issues**: Check `REDIS_HOST`, `REDIS_PORT`, and `REDIS_PASSWORD`
- **PostgreSQL errors**: Ensure `POSTGRES_HOST` and credentials are correct

### Worker Performance Issues

- **Tasks timing out**: Increase `*_WORKER_TIME_LIMIT` for the relevant worker
- **High failure rates**: Check logs and increase `*_MAX_RETRIES` if appropriate
- **Slow batch processing**: Adjust batch sizes up or down based on resource availability

### Plugin Loading Problems

- **Plugin not found**: Verify `ES_PLUGINS_PATH` and that plugin is listed in `INFERENCE_USED_PLUGINS`
- **Plugin errors**: Check plugin compatibility with your Embedding Studio version

### Memory Issues

- **Out of memory in workers**: Decrease batch sizes and limit concurrent tasks
- **Vector database memory pressure**: Consider PostgreSQL tuning (external to Embedding Studio)

If issues persist, examining worker logs often provides more detailed error messages that can help diagnose configuration problems.