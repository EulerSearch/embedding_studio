# System Architecture Overview

This document provides a comprehensive overview of the Embedding Studio architecture, explaining how different components work together to create, fine-tune, and serve embedding models.

## High-Level Architecture

### Core API Service

The central API service (`embedding_studio` container) provides:

- REST API endpoints for application integration
- Plugin management and discovery
- Session and clickstream data collection
- Task scheduling and coordination

This service acts as the entry point for applications using Embedding Studio and orchestrates the workflow between components.

### Worker Services

#### Fine-Tuning Worker

The `fine_tuning_worker` container:

- Handles model fine-tuning tasks
- Runs training jobs for embedding models
- Integrates with MLflow for experiment tracking
- Requires GPU acceleration for efficient training
- Uses the selected plugin's fine-tuning method

#### Inference Worker

The `inference_worker` container:

- Serves embedding models via Triton Inference Server
- Handles real-time embedding generation
- Supports model versioning and A/B testing
- Provides gRPC and HTTP endpoints
- Manages model deployment lifecycle

#### Improvement Worker

The `improvement_worker` container:

- Processes incremental vector adjustments
- Applies post-training optimizations to embeddings
- Handles small improvements without full fine-tuning
- Works on embedding quality enhancement

#### Upsertion Worker

The `upsertion_worker` container:

- Manages embedding generation for new content
- Handles batch processing of items
- Updates vector database with new embeddings
- Processes deletion and reindexing tasks

### Data Storage

#### Vector Database

Embedding Studio uses PostgreSQL with the pgvector extension as its primary vector store:

- Stores embedding vectors with metadata
- Provides fast approximate nearest neighbor search
- Supports various distance metrics (cosine, dot product, Euclidean)
- Handles index optimization for performance

#### Document Storage

MongoDB is used for storing:

- Fine-tuning task metadata
- Session and clickstream data
- Improvement and upsertion task tracking
- Reindexing task management

#### Model Storage

MLflow, backed by MinIO and MySQL, manages:

- Model versioning and artifacts
- Training metrics and parameters
- Experiment tracking
- Model registry for deployment

#### Queue System

Redis serves as the task queue and provides:

- Distributed task scheduling
- Worker coordination
- Job priority management
- Failure handling and retries

## Data Flow

The typical data flow in Embedding Studio follows these stages:

1. **Content Ingestion**:
   - Content is loaded via data loaders from S3, GCP, or databases
   - Documents are preprocessed and split into appropriate chunks
   - Initial embeddings are generated using base models

2. **User Interaction**:
   - Users search or interact with content
   - Clickstream data is collected via API endpoints
   - Sessions are processed and converted to training signals

3. **Fine-Tuning Process**:
   - Training data is prepared from user interactions
   - Models are fine-tuned using the specified method
   - Experiments are tracked in MLflow
   - The best model version is selected for deployment

4. **Model Deployment**:
   - The fine-tuned model is packaged for Triton
   - The inference service is updated with the new model
   - Content is reindexed with the improved model
   - A/B testing may be performed to validate improvements

5. **Search and Retrieval**:
   - Queries are embedded using the fine-tuned model
   - Vector similarity search is performed
   - Results are ranked and returned to users
   - The cycle continues with new interactions

## Plugin Integration Points

Embedding Studio's architecture is highly extensible through plugins that can customize:

1. **Data Ingestion**: Custom data loaders for specific sources
2. **Text Processing**: Specialized text processors and tokenizers
3. **Image Processing**: Custom image transformations and models
4. **Fine-Tuning Methods**: Application-specific training approaches
5. **Vector Adjustments**: Custom embedding improvement techniques
6. **Query Processing**: Specialized query understanding and expansion
7. **Search Optimization**: Custom ranking and filtering logic

## Resource Requirements

The system has different resource needs for different components:

- **Fine-Tuning Worker**: Requires GPU acceleration (NVIDIA CUDA)
- **Inference Worker**: Benefits from GPU for high throughput
- **Vector Database**: Needs sufficient memory for index performance
- **API and Other Workers**: CPU-bound, moderate memory requirements

In the next section, we'll explore the environment variables and configuration options that control this architecture.