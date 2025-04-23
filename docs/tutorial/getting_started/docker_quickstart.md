# Embedding Studio Docker Quick Start Guide

This guide provides instructions for running Embedding Studio using Docker Compose, with several deployment options based on your needs - from a full deployment with all services to minimal configurations and custom integrations.

## Prerequisites

Before you begin, ensure you have the following installed:

- [Docker](https://docs.docker.com/get-docker/) (version 19.03 or later)
- [Docker Compose](https://docs.docker.com/compose/install/) (version 1.27 or later)
- At least 8GB of RAM allocated to Docker
- For fine-tuning operations: NVIDIA GPU with CUDA support
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)

## System Requirements

Recommended specifications:

- **CPU**: 8+ cores
- **RAM**: 16GB minimum (32GB recommended)
- **Storage**: 100GB+ of free space
- **GPU**: NVIDIA GPU with 8GB+ VRAM

## Proper Startup Sequence

For Embedding Studio to initialize correctly, components should be started in a specific order:

### 1. Infrastructure Services First

Start the basic infrastructure services first and wait for them to be fully initialized:

```bash
# Start infrastructure services
docker-compose up -d redis mongo pgvector
docker-compose up -d minio mlflow_db mlflow

# Give them time to initialize
sleep 30
```

### 2. Inference Worker

The inference worker needs to start next, as it loads initial models and creates the initial vector collections:

```bash
# Start inference worker
docker-compose up -d inference_worker

# Allow time for model loading and collection initialization
sleep 60
```

### 3. API and Upsertion Worker

Once the inference worker has created collections, start the API and upsertion worker:

```bash
# Start API service and upsertion worker
docker-compose up -d embedding_studio upsertion_worker

# Wait for these services to fully initialize
sleep 20
```

### 4. Load Initial Data

At this point, you can load your initial data through the API:

```bash
# Example: Upload items using the API
curl -X POST http://localhost:5000/api/v1/embeddings/upsertion-tasks/run \
  -H "Content-Type: application/json" \
  -d @your-data-file.json
```

### 5. Start Additional Workers

After initial data is loaded, start the improvement and fine-tuning workers if needed:

```bash
# Start improvement worker for continuous enhancement
docker-compose up -d improvement_worker

# Start fine-tuning worker if you need to train custom models
docker-compose up -d fine_tuning_worker
```

### Automated Startup Script

You can create a startup script to handle this sequence automatically:

```bash
#!/bin/bash
echo "Starting infrastructure services..."
docker-compose up -d redis mongo pgvector minio mlflow_db mlflow
echo "Waiting for infrastructure initialization..."
sleep 30

echo "Starting inference worker..."
docker-compose up -d inference_worker
echo "Waiting for model loading and collection creation..."
sleep 60

echo "Starting API and upsertion worker..."
docker-compose up -d embedding_studio upsertion_worker
echo "Waiting for API initialization..."
sleep 20

echo "System is ready for data loading. Start additional workers with:"
echo "docker-compose up -d improvement_worker fine_tuning_worker"
```

This phased approach ensures that each component has the dependencies it needs fully initialized before starting.

## 1. Full Deployment (All Services)

This option runs all Embedding Studio services, providing a complete environment for development and testing.

### Setup Steps

1. Create a new directory for your project:

```bash
mkdir embedding-studio-project
cd embedding-studio-project
```

2. Create a basic `.env` file with required configuration:

```bash
cat > .env << 'EOL'
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

# MySQL settings (for MLflow)
MYSQL_DATABASE=mlflow
MYSQL_USER=mlflow_user
MYSQL_PASSWORD=Baxp3O5rUvpIxiD77BfZ
MYSQL_ROOT_PASSWORD=PrK5qmPTDsm2IYKvHVG8

# Inference settings
INFERENCE_HOST=inference_worker
INFERENCE_GRPC_PORT=8001
INFERENCE_USED_PLUGINS=["TextDefaultFineTuningMethodForText", "HFCategoriesTextFineTuningMethod"]
EOL
```

3. Create a `docker-compose.yml` file:

```bash
cat > docker-compose.yml << 'EOL'
version: "3.8"

services:
  embedding_studio:
    image: embeddingstudio/service:latest
    restart: always
    ports:
      - '5000:5000'
    env_file:
      - .env
    depends_on:
      - mongo
      - redis
      - pgvector
    volumes:
      - ./plugins:/embedding_studio/plugins:ro
    healthcheck:
      test: curl --fail http://localhost:5000/api/v1/ping || exit 1
      interval: 10s
      timeout: 10s
      retries: 5
      start_period: 10s

  fine_tuning_worker:
    image: embeddingstudio/fine_tuning_worker:latest
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    restart: always
    env_file:
      - .env
    depends_on:
      - mongo
      - redis
      - mlflow
    volumes:
      - ./plugins:/embedding_studio/plugins:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [ gpu ]

  inference_worker:
    image: embeddingstudio/inference_worker:latest
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    restart: always
    ports:
      - '8001:8001'
    env_file:
      - .env
    depends_on:
      - redis
    volumes:
      - ./plugins:/embedding_studio/plugins:ro
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [ gpu ]

  improvement_worker:
    image: embeddingstudio/improvement_worker:latest
    restart: always
    env_file:
      - .env
    depends_on:
      - mongo
      - redis
      - pgvector
    volumes:
      - ./plugins:/embedding_studio/plugins:ro

  upsertion_worker:
    image: embeddingstudio/upsertion_worker:latest
    restart: always
    env_file:
      - .env
    depends_on:
      - mongo
      - redis
      - pgvector
    volumes:
      - ./plugins:/embedding_studio/plugins:ro

  redis:
    image: redis:latest
    restart: always
    ports:
      - '6379:6379'
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - ./data/redis:/data
    healthcheck:
      test: redis-cli -a ${REDIS_PASSWORD} ping
      interval: 10s
      timeout: 5s
      retries: 10

  mongo:
    image: mongo:4
    restart: always
    ports:
      - '27017:27017'
    environment:
      - MONGO_INITDB_DATABASE=${FINETUNING_MONGO_DB_NAME}
      - MONGO_INITDB_ROOT_USERNAME=${FINETUNING_MONGO_USERNAME}
      - MONGO_INITDB_ROOT_PASSWORD=${FINETUNING_MONGO_PASSWORD}
    volumes:
      - ./data/mongo:/data/db
    healthcheck:
      test: echo 'db.runCommand("ping").ok' | mongosh --quiet
      interval: 10s
      timeout: 10s
      retries: 5
      start_period: 40s

  pgvector:
    image: pgvector/pgvector:pg16
    restart: always
    env_file:
      - .env
    ports:
      - "5432:5432"
    volumes:
      - ./data/pgvector:/var/lib/postgresql/data
    healthcheck:
      test: pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 10s

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.9.2
    restart: always
    ports:
      - "5001:5000"
    environment:
      - MLFLOW_TRACKING_URI=mysql+pymysql://${MYSQL_USER}:${MYSQL_PASSWORD}@mlflow_db:3306/${MYSQL_DATABASE}
      - AWS_ACCESS_KEY_ID=${MINIO_ROOT_USER}
      - AWS_SECRET_ACCESS_KEY=${MINIO_ROOT_PASSWORD}
      - MLFLOW_S3_ENDPOINT_URL=http://${MINIO_HOST}:${MINIO_PORT}
    command: mlflow server --host 0.0.0.0 --backend-store-uri mysql+pymysql://${MYSQL_USER}:${MYSQL_PASSWORD}@mlflow_db:3306/${MYSQL_DATABASE} --default-artifact-root s3://${MINIO_DEFAULT_BUCKETS}/
    depends_on:
      - mlflow_db
      - minio

  mlflow_db:
    image: mysql:8
    restart: always
    environment:
      - MYSQL_DATABASE=${MYSQL_DATABASE}
      - MYSQL_USER=${MYSQL_USER}
      - MYSQL_PASSWORD=${MYSQL_PASSWORD}
      - MYSQL_ROOT_PASSWORD=${MYSQL_ROOT_PASSWORD}
    volumes:
      - ./data/mysql:/var/lib/mysql
    healthcheck:
      test: mysqladmin ping -h localhost -u root -p${MYSQL_ROOT_PASSWORD}
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s

  minio:
    image: minio/minio:latest
    restart: always
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - MINIO_ROOT_USER=${MINIO_ROOT_USER}
      - MINIO_ROOT_PASSWORD=${MINIO_ROOT_PASSWORD}
    command: server /data --console-address ":9001"
    volumes:
      - ./data/minio:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 10s
EOL
```

4. Create required directories:

```bash
mkdir -p data/redis data/mongo data/pgvector data/mysql data/minio plugins models
```

5. Start services in the correct order using the startup sequence described above.

## 2. Minimal Deployment (API + Inference Only)

This lightweight configuration runs only the essential services for serving embeddings, without fine-tuning capabilities.

```bash
cat > docker-compose.minimal.yml << 'EOL'
version: "3.8"

services:
  embedding_studio:
    image: embeddingstudio/service:latest
    restart: always
    ports:
      - '5000:5000'
    env_file:
      - .env
    depends_on:
      - redis
      - pgvector
    volumes:
      - ./plugins:/embedding_studio/plugins:ro

  inference_worker:
    image: embeddingstudio/inference_worker:latest
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    restart: always
    ports:
      - '8001:8001'
    env_file:
      - .env
    depends_on:
      - redis
    volumes:
      - ./plugins:/embedding_studio/plugins:ro
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [ gpu ]

  redis:
    image: redis:latest
    restart: always
    ports:
      - '6379:6379'
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - ./data/redis:/data

  pgvector:
    image: pgvector/pgvector:pg16
    restart: always
    env_file:
      - .env
    ports:
      - "5432:5432"
    volumes:
      - ./data/pgvector:/var/lib/postgresql/data
EOL
```

Run this minimal setup in the following order:
```bash
# Start infrastructure
docker-compose -f docker-compose.minimal.yml up -d redis pgvector
sleep 20

# Start inference worker and wait for it to initialize
docker-compose -f docker-compose.minimal.yml up -d inference_worker
sleep 60

# Start API
docker-compose -f docker-compose.minimal.yml up -d embedding_studio
```

## 3. Improvement Setup (API + Inference + Improvement Worker)

This configuration adds the improvement worker to enable continuous vector quality enhancement.

```bash
cat > docker-compose.improvement.yml << 'EOL'
version: "3.8"

services:
  embedding_studio:
    image: embeddingstudio/service:latest
    restart: always
    ports:
      - '5000:5000'
    env_file:
      - .env
    depends_on:
      - mongo
      - redis
      - pgvector
    volumes:
      - ./plugins:/embedding_studio/plugins:ro

  inference_worker:
    image: embeddingstudio/inference_worker:latest
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    restart: always
    ports:
      - '8001:8001'
    env_file:
      - .env
    depends_on:
      - redis
    volumes:
      - ./plugins:/embedding_studio/plugins:ro
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [ gpu ]

  improvement_worker:
    image: embeddingstudio/improvement_worker:latest
    restart: always
    env_file:
      - .env
    depends_on:
      - mongo
      - redis
      - pgvector
    volumes:
      - ./plugins:/embedding_studio/plugins:ro

  redis:
    image: redis:latest
    restart: always
    ports:
      - '6379:6379'
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - ./data/redis:/data

  mongo:
    image: mongo:4
    restart: always
    ports:
      - '27017:27017'
    environment:
      - MONGO_INITDB_DATABASE=${FINETUNING_MONGO_DB_NAME}
      - MONGO_INITDB_ROOT_USERNAME=${FINETUNING_MONGO_USERNAME}
      - MONGO_INITDB_ROOT_PASSWORD=${FINETUNING_MONGO_PASSWORD}
    volumes:
      - ./data/mongo:/data/db

  pgvector:
    image: pgvector/pgvector:pg16
    restart: always
    env_file:
      - .env
    ports:
      - "5432:5432"
    volumes:
      - ./data/pgvector:/var/lib/postgresql/data
EOL
```

Run the improvement setup in the correct order:
```bash
# Start infrastructure
docker-compose -f docker-compose.improvement.yml up -d redis mongo pgvector
sleep 30

# Start inference worker
docker-compose -f docker-compose.improvement.yml up -d inference_worker
sleep 60

# Start API and upsertion
docker-compose -f docker-compose.improvement.yml up -d embedding_studio
sleep 20

# After loading some data, start improvement worker
docker-compose -f docker-compose.improvement.yml up -d improvement_worker
```

## 4. Fine-tuning and Upsertion Setup

This configuration focuses on fine-tuning and upsertion capabilities, adding the necessary services to train and update models.

```bash
cat > docker-compose.finetuning.yml << 'EOL'
version: "3.8"

services:
  embedding_studio:
    image: embeddingstudio/service:latest
    restart: always
    ports:
      - '5000:5000'
    env_file:
      - .env
    depends_on:
      - mongo
      - redis
      - pgvector
    volumes:
      - ./plugins:/embedding_studio/plugins:ro

  fine_tuning_worker:
    image: embeddingstudio/fine_tuning_worker:latest
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    restart: always
    env_file:
      - .env
    depends_on:
      - mongo
      - redis
      - mlflow
    volumes:
      - ./plugins:/embedding_studio/plugins:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [ gpu ]

  inference_worker:
    image: embeddingstudio/inference_worker:latest
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    restart: always
    ports:
      - '8001:8001'
    env_file:
      - .env
    depends_on:
      - redis
    volumes:
      - ./plugins:/embedding_studio/plugins:ro
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [ gpu ]

  upsertion_worker:
    image: embeddingstudio/upsertion_worker:latest
    restart: always
    env_file:
      - .env
    depends_on:
      - mongo
      - redis
      - pgvector
    volumes:
      - ./plugins:/embedding_studio/plugins:ro

  redis:
    image: redis:latest
    restart: always
    ports:
      - '6379:6379'
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - ./data/redis:/data

  mongo:
    image: mongo:4
    restart: always
    ports:
      - '27017:27017'
    environment:
      - MONGO_INITDB_DATABASE=${FINETUNING_MONGO_DB_NAME}
      - MONGO_INITDB_ROOT_USERNAME=${FINETUNING_MONGO_USERNAME}
      - MONGO_INITDB_ROOT_PASSWORD=${FINETUNING_MONGO_PASSWORD}
    volumes:
      - ./data/mongo:/data/db

  pgvector:
    image: pgvector/pgvector:pg16
    restart: always
    env_file:
      - .env
    ports:
      - "5432:5432"
    volumes:
      - ./data/pgvector:/var/lib/postgresql/data

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.9.2
    restart: always
    ports:
      - "5001:5000"
    environment:
      - MLFLOW_TRACKING_URI=mysql+pymysql://${MYSQL_USER}:${MYSQL_PASSWORD}@mlflow_db:3306/${MYSQL_DATABASE}
      - AWS_ACCESS_KEY_ID=${MINIO_ROOT_USER}
      - AWS_SECRET_ACCESS_KEY=${MINIO_ROOT_PASSWORD}
      - MLFLOW_S3_ENDPOINT_URL=http://${MINIO_HOST}:${MINIO_PORT}
    command: mlflow server --host 0.0.0.0 --backend-store-uri mysql+pymysql://${MYSQL_USER}:${MYSQL_PASSWORD}@mlflow_db:3306/${MYSQL_DATABASE} --default-artifact-root s3://${MINIO_DEFAULT_BUCKETS}/
    depends_on:
      - mlflow_db
      - minio

  mlflow_db:
    image: mysql:8
    restart: always
    environment:
      - MYSQL_DATABASE=${MYSQL_DATABASE}
      - MYSQL_USER=${MYSQL_USER}
      - MYSQL_PASSWORD=${MYSQL_PASSWORD}
      - MYSQL_ROOT_PASSWORD=${MYSQL_ROOT_PASSWORD}
    volumes:
      - ./data/mysql:/var/lib/mysql

  minio:
    image: minio/minio:latest
    restart: always
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - MINIO_ROOT_USER=${MINIO_ROOT_USER}
      - MINIO_ROOT_PASSWORD=${MINIO_ROOT_PASSWORD}
    command: server /data --console-address ":9001"
    volumes:
      - ./data/minio:/data
EOL
```

Run the fine-tuning and upsertion setup in the correct order:
```bash
# Start infrastructure
docker-compose -f docker-compose.finetuning.yml up -d redis mongo pgvector minio mlflow_db mlflow
sleep 30

# Start inference worker
docker-compose -f docker-compose.finetuning.yml up -d inference_worker
sleep 60

# Start API and upsertion worker
docker-compose -f docker-compose.finetuning.yml up -d embedding_studio upsertion_worker
sleep 20

# After loading some data and collecting user feedback, start fine-tuning
docker-compose -f docker-compose.finetuning.yml up -d fine_tuning_worker
```

## 5. Custom Integration with External Services

This configuration connects Embedding Studio to your existing infrastructure. Use this when you already have PostgreSQL, Redis, MongoDB, and MLflow running elsewhere.

First, modify your `.env` file to point to your existing services:

```bash
cat > .env.external << 'EOL'
# MongoDB settings (external)
FINETUNING_MONGO_HOST=your-mongo-host
FINETUNING_MONGO_PORT=27017
FINETUNING_MONGO_DB_NAME=embedding_studio
FINETUNING_MONGO_USERNAME=your-mongo-user
FINETUNING_MONGO_PASSWORD=your-mongo-password

# Redis settings (external)
REDIS_HOST=your-redis-host
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password

# PostgreSQL settings (external)
POSTGRES_HOST=your-pgvector-host
POSTGRES_PORT=5432
POSTGRES_USER=your-pg-user
POSTGRES_PASSWORD=your-pg-password
POSTGRES_DB=embedding_studio

# MLflow settings (external)
MLFLOW_HOST=your-mlflow-host
MLFLOW_PORT=5001

# Inference settings
INFERENCE_HOST=inference_worker
INFERENCE_GRPC_PORT=8001
INFERENCE_USED_PLUGINS=["TextDefaultFineTuningMethodForText", "HFCategoriesTextFineTuningMethod"]
EOL
```

Then create a Docker Compose file that uses external services:

```bash
cat > docker-compose.external.yml << 'EOL'
version: "3.8"

services:
  embedding_studio:
    image: embeddingstudio/service:latest
    restart: always
    ports:
      - '5000:5000'
    env_file:
      - .env.external
    volumes:
      - ./plugins:/embedding_studio/plugins:ro

  fine_tuning_worker:
    image: embeddingstudio/fine_tuning_worker:latest
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    restart: always
    env_file:
      - .env.external
    volumes:
      - ./plugins:/embedding_studio/plugins:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [ gpu ]

  inference_worker:
    image: embeddingstudio/inference_worker:latest
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    restart: always
    ports:
      - '8001:8001'
    env_file:
      - .env.external
    volumes:
      - ./plugins:/embedding_studio/plugins:ro
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [ gpu ]

  improvement_worker:
    image: embeddingstudio/improvement_worker:latest
    restart: always
    env_file:
      - .env.external
    volumes:
      - ./plugins:/embedding_studio/plugins:ro

  upsertion_worker:
    image: embeddingstudio/upsertion_worker:latest
    restart: always
    env_file:
      - .env.external
    volumes:
      - ./plugins:/embedding_studio/plugins:ro
EOL
```

Run the setup with external services in the correct order:
```bash
# Start inference worker first
docker-compose -f docker-compose.external.yml up -d inference_worker
sleep 60

# Start API and upsertion worker
docker-compose -f docker-compose.external.yml up -d embedding_studio upsertion_worker
sleep 20

# After loading some data, start improvement and fine-tuning workers
docker-compose -f docker-compose.external.yml up -d improvement_worker fine_tuning_worker
```

## Verifying Your Installation

To verify that Embedding Studio is running correctly:

1. Check the API status:
   ```bash
   curl http://localhost:5000/api/v1/ping
   ```

2. Access the API documentation at:
   ```
   http://localhost:5000/docs
   ```

3. View MLflow UI (if running) at:
   ```
   http://localhost:5001
   ```

## Important Configuration Notes

### Using GPU Support

For GPU-accelerated operations (required for fine-tuning and recommended for inference):

1. Ensure the NVIDIA Container Toolkit is installed
2. Verify GPU access with: `docker run --gpus all nvidia/cuda:11.0-base nvidia-smi`
3. Make sure the `deploy` section with GPU reservations is included in your docker-compose file

### Data Persistence

All examples mount local directories to store data:

- `./data/mongo` for MongoDB data
- `./data/pgvector` for PostgreSQL/pgvector data
- `./data/redis` for Redis data
- `./data/mysql` for MySQL (MLflow) data
- `./data/minio` for MinIO (model artifacts) data
- `./models` for model files used by the inference worker

### Plugin Management

Custom plugins should be placed in the `./plugins` directory, which is mounted to all worker containers. Make sure to update the `INFERENCE_USED_PLUGINS` environment variable to include your plugins.

## Troubleshooting

### Common Issues

1. **GPU not detected**: Verify NVIDIA drivers and Container Toolkit are properly installed
2. **Services failing to start**: Check logs with `docker-compose logs <service_name>`
3. **Connection issues between services**: Ensure network names match and services can resolve each other
4. **Collection initialization errors**: Make sure you start the inference worker before the API and upsertion worker
5. **Missing models**: Ensure the models directory is properly mounted to the inference worker

### Viewing Logs

```bash
# View logs for a specific service
docker-compose logs -f embedding_studio

# View logs for multiple services
docker-compose logs -f embedding_studio inference_worker
```

### Resource Issues

If containers are crashing due to resource constraints:

1. Increase Docker memory allocation
2. Reduce batch sizes via environment variables (e.g., `UPSERTION_BATCH_SIZE=8`)
3. Consider running with fewer services (using the minimal configuration)

## Next Steps

- [Understand Embedding Studio configuration](configurations.md)
- [Explore the architecture](architecture_overview.md)
- [Create custom plugins](../plugins/creating_plugins.md)