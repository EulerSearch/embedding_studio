# Core Concepts of Embedding Studio

This guide introduces the fundamental concepts and terminology used throughout Embedding Studio. Understanding these core concepts will help you navigate the system more effectively.

## Vector Embeddings

At the heart of Embedding Studio are vector embeddings - numerical representations of data items (text, images, etc.) in high-dimensional space, where semantic similarity is captured by vector proximity.

### Key Embedding Concepts:

- **Embedding Vector**: A fixed-length numerical array (e.g., 384, 768, or 1024 dimensions) representing the semantic content of an item
- **Embedding Model**: Neural network that transforms raw data into embedding vectors
- **Metric Type**: Method to measure similarity between vectors (cosine similarity, dot product, Euclidean distance)
- **Metric Aggregation**: How to combine multiple similarity scores (MIN, AVG, etc.)
- **Vector Collection**: A database table storing embedding vectors and their metadata

## Search and Retrieval System

The search system combines vector similarity with traditional filtering:

- **Similarity Search**: Finding content similar to a query by comparing vector embeddings
- **Payload Filtering**: Limiting results based on structured data attributes
- **Hybrid Ranking**: Combining vector similarity with other factors like recency or popularity
- **Category Prediction**: Using embeddings to identify relevant categories for queries

The similarity search functionality supports both pure semantic search and hybrid approaches with filtering and sorting.

## Clickstream and Session Tracking

The clickstream system captures and analyzes user interactions:

- **Sessions**: Groups of related user actions starting with a search query
- **Events**: Individual user actions like clicks, views, or conversions
- **Relevance Signals**: Implicit feedback derived from user behaviors
- **Irrelevance Marking**: Explicit mechanisms to flag unhelpful sessions or results

This user interaction data forms the foundation of the continuous learning loop.

## Plugin Architecture

Embedding Studio uses a plugin-based architecture for extensibility:

```
FineTuningMethod
├── get_data_loader() → Returns loader for training data
├── get_items_preprocessor() → Returns data preprocessor
├── get_query_retriever() → Extracts queries from sessions
├── get_inference_client_factory() → Creates clients for inference
├── get_manager() → Returns experiment manager 
├── get_search_index_info() → Defines vector DB schema
├── get_vectors_adjuster() → Handles vector improvements
├── get_fine_tuning_builder() → Creates training pipeline
└── upload_initial_model() → Uploads base model
```

The architecture supports two specialized plugin types:
- **FineTuningMethod**: Base plugin for general embedding models
- **CategoriesFineTuningMethod**: Specialized plugin for category prediction models

Each plugin encapsulates the entire workflow from data loading to model deployment, allowing for customized embedding solutions.

## Data Management Components

Embedding Studio includes several components for data handling:

### Data Loaders
Data loaders fetch content from various sources:
- **S3 Loaders**: For AWS cloud storage
- **GCP Loaders**: For Google Cloud Platform
- **PostgreSQL Loaders**: For database-stored content
- **Aggregated Loaders**: Combine multiple sources into a unified interface

### Content Processors
These components transform raw content for embedding:
- **Preprocessors**: Clean and normalize input data
- **Splitters**: Break content into appropriate chunks
- **Augmentation**: Create variations of data for robust training
- **Tokenization**: Prepare text for model input

### Vector Database Operations
Embedding Studio provides a complete lifecycle for vector data:
- **Upsertion**: Add or update items with their vectors
- **Deletion**: Remove items from the vector database
- **Reindexing**: Rebuild vector indices after model updates
- **Collection Management**: Create, optimize, and switch between vector collections

## Task-Based Processing

Embedding Studio uses a task-based approach for asynchronous operations:

- **Tasks**: Self-contained units of work with tracking metadata
- **Workers**: Specialized services that process specific task types
- **Status Tracking**: Monitoring task progress and outcomes
- **Idempotency**: Safe retry mechanisms for failed operations

Tasks are used for fine-tuning, upsertion, deletion, and other long-running processes.

## Fine-Tuning System

The fine-tuning system improves embedding models based on user feedback:

### Core Fine-Tuning Components:

- **MLflow Integration**: Tracks experiments, metrics, and model artifacts
- **Hyperparameter Optimization**: Finds optimal model settings
- **Specialized Loss Functions**: Improve embedding quality for search
- **Progressive Evaluation**: Monitors improvements using test datasets

### Fine-Tuning Workflow:

1. User interactions from the clickstream system form training data
2. Training data is preprocessed into positive/negative examples
3. Models are fine-tuned with specialized ranking loss functions
4. Multiple hyperparameter combinations are evaluated
5. The best model version is selected based on performance metrics
6. The model is registered in MLflow for deployment

## Blue-Green Deployment

Embedding Studio uses a blue-green deployment pattern for zero-downtime updates:

- **Blue Collection**: The currently active vector collection serving requests
- **Green Collection**: A new collection being prepared for deployment
- **Deployment Switching**: Process of transitioning traffic from blue to green
- **Rollback Capability**: Ability to revert to previous collection if needed

This pattern ensures reliability and continuity during model improvements.

## Suggestion System

The suggestion system provides query autocompletion and assistance:

- **Suggestion Phrases**: Managed pool of possible suggestions
- **Domain-Specific Suggestions**: Context-aware suggestion filtering
- **Probability Weighting**: Controls suggestion prominence
- **Matching Types**: Various matching strategies (exact, prefix, fuzzy)

## Worker Architecture

Embedding Studio operates through specialized worker services:

- **Fine-Tuning Worker**: Executes model training (GPU-accelerated)
- **Inference Worker**: Manages Triton Inference Server for embedding generation
- **Improvement Worker**: Applies incremental vector adjustments
- **Upsertion Worker**: Processes content updates and database operations
- **Reindex Worker**: Handles complete database rebuilds after model changes

Workers use MongoDB and Dramatiq for reliable task queuing and execution.

## Continuous Improvement Loop

The core improvement loop in Embedding Studio:

1. **Capture**: User interactions are recorded through the clickstream system
2. **Convert**: Sessions are transformed into training examples
3. **Train**: Embedding models are fine-tuned with this feedback
4. **Deploy**: Improved models are deployed using blue-green pattern
5. **Embed**: Content is re-embedded with the new model
6. **Serve**: Users receive improved search results
7. **Repeat**: The cycle continues, progressively enhancing quality

This feedback loop creates a self-improving system that gets better over time, adapting to your specific domain and user behaviors.
