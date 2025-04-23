# Understanding the Embedding Model Lifecycle in Embedding Studio

This tutorial provides a comprehensive walkthrough of the embedding model lifecycle within Embedding Studio, from initial fine-tuning to deployment and continuous improvement.

## Overview

Embedding Studio manages the full lifecycle of embedding models:

1. **Fine-tuning**: Improving embedding models using feedback data
2. **Deployment**: Making models available for inference
3. **Upsertion**: Adding or updating vectors in the database
4. **Improvement**: Adjusting vectors based on user feedback
5. **Reindexing**: Migrating data between models

Let's explore each phase in detail.

## Fine-Tuning Pipeline

Fine-tuning takes an existing embedding model and improves it using clickstream data (user interactions with search results).

### Key Components

- **Fine-Tuning Tasks**: Managed via the `/fine-tuning/task` endpoint
- **Clickstream Data**: User sessions and interactions used for training
- **MLflow Tracking**: Records experiments, parameters, and model metrics

### Fine-Tuning Process

```
                    ┌───────────────────┐
                    │ Clickstream Data  │
                    └─────────┬─────────┘
                              │
                              ▼
┌───────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Initial Model │───▶│ Fine-Tuning Job │───▶│ Improved Model  │
└───────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ MLflow Tracking   │
                    └───────────────────┘
```

### Implementation Details

1. **Preparing Data**:
   - User search sessions are collected via the clickstream API
   - Sessions are converted to training examples with positive/negative pairs
   - Data is split into training and evaluation sets

2. **Hyperparameter Optimization**:
   - Multiple parameter configurations are tested
   - Performance is evaluated using metrics like relevance improvement
   - The best performing model is selected

3. **Model Storage**:
   - Trained models are stored in MLflow
   - Models include both query and item encoders
   - Metadata tracks lineage and performance metrics

## Deployment Pipeline

Once a model is fine-tuned, it needs to be deployed to the inference service to be used for vector creation.

### Key Components

- **Triton Inference Server**: Handles efficient model serving
- **Deployment Worker**: Manages the deployment process
- **Blue-Green Deployment**: Enables zero-downtime updates

### Deployment Process

```
┌───────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ MLflow Model  │───▶│ Model Converter │───▶│ Triton Model    │
└───────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Inference Service │
                    └───────────────────┘
```

### Implementation Details

1. **Model Retrieval**:
   - The model is downloaded from MLflow storage
   - Both query and item models are extracted

2. **Conversion for Triton**:
   - Models are traced using PyTorch's JIT compiler
   - Configuration files are generated for Triton
   - Models are organized in the model repository

3. **Deployment Strategy**:
   - Models are versioned in the repository
   - Triton handles model loading and GPU allocation
   - Blue-green deployment ensures zero-downtime updates

## Vector Management

Once models are deployed, Embedding Studio manages vector creation, storage, and querying.

### Key Components

- **Upsertion Worker**: Handles adding or updating vectors
- **Deletion Worker**: Removes vectors from the database
- **Vector Database**: Stores and indexes vectors (based on pgvector)

### Upsertion Process

```
┌───────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Content Data  │───▶│ Item Splitter   │───▶│ Inference       │
└───────────────┘    └─────────────────┘    └─────────────────┘
                                                     │
                                                     ▼
                                            ┌─────────────────┐
                                            │ Vector Database │
                                            └─────────────────┘
```

### Implementation Details

1. **Data Processing**:
   - Content is loaded using data loaders
   - Items are split into manageable chunks
   - Each chunk is processed through preprocessing pipeline

2. **Vector Creation**:
   - Chunks are sent to the inference service
   - Resulting vectors are assembled
   - Average vectors may be created for consolidated representation

3. **Storage Management**:
   - Vectors are stored with metadata and payload
   - Indexes are maintained for efficient similarity search
   - User-specific vectors can be stored for personalized results

## Continuous Improvement

Embedding Studio enables continuous improvement through user feedback and incremental model updates.

### Key Components

- **Clickstream Collection**: Captures user interactions
- **Improvement Worker**: Adjusts vectors based on feedback
- **Reindexing Worker**: Migrates data between model versions

### Improvement Process

```
┌───────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ User Sessions │───▶│ Vector Adjuster │───▶│ Improved Vectors│
└───────────────┘    └─────────────────┘    └─────────────────┘
       │                                             │
       │                                             │
       ▼                                             ▼
┌───────────────┐                         ┌─────────────────┐
│ Fine-Tuning   │                         │ Personalization │
└───────────────┘                         └─────────────────┘
```

### Implementation Details

1. **Feedback Collection**:
   - User clicks and interactions are recorded
   - Sessions are analyzed for relevance patterns
   - Irrelevant sessions can be marked and excluded

2. **Vector Adjustment**:
   - Clicked items' vectors are pulled closer to query vectors
   - Non-clicked items' vectors are pushed away
   - User-specific vector adjustments enable personalization

3. **Model Evolution**:
   - New models are fine-tuned based on collected feedback
   - Data is migrated between model versions via reindexing
   - Blue-green deployment ensures smooth transitions

## Reindexing Between Models

When a new model version is created, data needs to be migrated from the old model to the new one.

### Key Components

- **Reindex Worker**: Manages the overall reindexing process
- **Reindex Subtasks**: Process batches of data in parallel
- **Blue Collection Switch**: Changes which model serves production traffic

### Reindexing Process

```
┌───────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Source Model  │───▶│ Reindex Worker  │───▶│ Destination Model│
└───────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Blue-Green Switch │
                    └───────────────────┘
```

### Implementation Details

1. **Task Creation**:
   - Reindexing task specifies source and destination models
   - Locking prevents concurrent operations on the same models
   - Configuration controls batch size and concurrency

2. **Parallel Processing**:
   - Data is processed in batches for efficiency
   - Multiple subtasks run concurrently
   - Progress is tracked and failures are recorded

3. **Deployment Coordination**:
   - Optional model deployment if needed
   - Blue collection switch changes active model
   - Source model cleanup can be performed after successful migration

## Complete Workflow

Here's a step-by-step workflow of the entire embedding model lifecycle in Embedding Studio:

1. **Initial Model Deploy and Collection Creation**
   - Upload initial model or use an existing one
   - Deploy model to Triton Inference Server
   - Create vector collection in the database for this model
   - Set as "blue" (active) collection for serving traffic

2. **Upsertion**
   - Send content items to the upsertion endpoint
   - Content is split into chunks
   - Chunks are transformed into vectors via the inference service
   - Vectors are stored in the database with metadata

3. **Search and Clickstream Collection**
   - Users perform searches via similarity search endpoints
   - Search queries are vectorized and compared against stored vectors
   - User interactions with results are captured via clickstream API
   - Sessions track queries, results, and user actions

4. **Vector Improvement via Feedback**
   - Clickstream data is analyzed for feedback signals
   - Improvement worker processes feedback sessions
   - Vectors are adjusted based on user interactions
   - Personalized vectors maintain user-specific adjustments

5. **Fine-tuning via Feedback**
   - Sufficient feedback triggers fine-tuning job (via API)
   - Clickstream data is converted to training examples
   - Model undergoes hyperparameter optimization
   - New model version is created and evaluated

6. **New Model Deployment**
   - If quality improvement is sufficient, deploy new model
   - Create new vector collection for the improved model
   - New collection initially doesn't serve production traffic

7. **Reindexing**
   - Data is migrated from old model to new model
   - Process runs in batches with parallel workers
   - New items/updates go directly to the new model during migration
   - Personalized vectors are removed or recreated

8. **Switch Active Model**
   - New model and collection are set as "blue" (active)
   - All new search traffic uses the improved model
   - Switch happens with zero downtime

9. **Cleanup**
   - Previous collection is deleted after successful switch
   - Old model is removed from the inference service
   - System is ready for next improvement cycle

This cycle continues iteratively, with each round potentially delivering better search quality based on real user feedback.
