# Core Concepts in Fine-Tuning

Fine-tuning is a key capability in Embedding Studio that allows you to optimize embedding models based on user feedback and clickstream data. The system is designed with a plugin-based architecture that makes it flexible and extensible. This document explores the core concepts that form the foundation of fine-tuning in Embedding Studio.

## Key Components

### 1. Experiments and Iterations

Fine-tuning happens in **iterations**, which are tracked and managed by the `ExperimentsManager`. Each iteration has:

- A batch ID that groups related clickstream data
- A run ID that identifies the starting model
- A plugin name that determines which fine-tuning method is used

The `ExperimentsManager` tracks metrics, parameters, and models using MLflow as a backend. It maintains:

- An initial model as a reference point
- A history of fine-tuning runs with their parameters and results
- The best model from each iteration based on configured metrics

### 2. Fine-Tuning Methods

Fine-tuning methods are implemented as plugins that define how models are fine-tuned. Each method:

- Inherits from a base class like `FineTuningMethod` or `CategoriesFineTuningMethod`
- Specifies how to load and process data
- Defines the fine-tuning configuration and hyperparameters
- Implements the training loop and model evaluation

Embedding Studio includes several built-in methods:
- `DefaultFineTuningMethod` - A general-purpose method for text-to-image models
- `TextDefaultFineTuningMethod` - Specialized for text embeddings
- `DictDefaultMethodForObjectsTextOnly` - Focused on dictionary-based text items
- `CategoriesTextFineTuningMethod` - Optimized for category prediction

### 3. Data Representation

Fine-tuning relies on several key data structures:

- **FineTuningInput** - Represents a user query and the results shown to the user, including which results received clicks
- **RankingData** - Combines clickstream data (user interactions) with the actual items being ranked
- **FineTuningFeatures** - Extracted features used for training, including positive/negative ranks and confidences
- **ItemsSet** - A dataset of items with their embeddings

### 4. Training Process

The `EmbeddingsFineTuner` class is a PyTorch Lightning module that manages the training process:

- It implements training_step, validation_step, and other required methods
- Handles optimizers and learning rate schedulers for both query and item models
- Tracks metrics during training and validation
- Supports early stopping through callbacks
- Can freeze specific layers of the model during training

### 5. Hyperparameter Optimization

The system includes built-in hyperparameter optimization:

- Uses Hyperopt with Tree of Parzen Estimators (TPE) for initial model tuning
- Stores and reuses the best parameters from previous iterations
- Can evaluate a configurable number of parameter combinations (controlled by initial_max_evals)
- Supports both continuous and discrete hyperparameters

### 6. Loss Functions

The system uses specialized loss functions for embedding fine-tuning:

- `ProbMarginRankingLoss` - A probabilistic version of margin ranking loss
- `CosineProbMarginRankingLoss` - Optimized for cosine similarity metrics

These loss functions incorporate confidence values to weigh different examples and can adjust the margin based on similarity metrics.

### 7. Worker System

Fine-tuning is executed asynchronously through a robust worker system:

- Dramatiq for task queue management and processing
- Support for task retry logic and error handling
- Progress tracking and status updates
- Memory management for large models (using garbage collection and CUDA cache clearing)
- Automatic model deployment capability 

### 8. GPU Acceleration

The system is designed to leverage GPU acceleration when available:

- Automatically detects CUDA availability
- Moves models and data to the appropriate device (GPU or CPU)
- Manages memory carefully to avoid out-of-memory errors
- Supports moving models back to CPU after training

## Process Flow

The high-level flow of a fine-tuning task is:

1. A fine-tuning task is created via the API
2. The `fine_tuning_worker` picks up the task
3. It loads the appropriate fine-tuning plugin based on the model's plugin name
4. If needed, it releases a batch of clickstream data
5. It builds the fine-tuning pipeline with the plugin's builder
6. The fine-tuning process runs, either with hyperparameter optimization or using best previous params
7. Results are tracked in MLflow
8. The best model is selected and can be automatically deployed
9. Task status is updated to reflect completion or failure
