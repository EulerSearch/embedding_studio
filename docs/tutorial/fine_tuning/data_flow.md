# Data Flow in Fine-Tuning

The fine-tuning process in Embedding Studio involves a sophisticated flow of data from raw user interactions to optimized embedding models. This document traces the journey of data through the system, highlighting the transformations and operations that occur at each stage.

## Data Sources and Ingestion

### 1. Clickstream Collection

The fine-tuning process begins with user interactions, collected as clickstream data:

- User queries and the results shown for each query
- Click events indicating which results users found relevant
- Timestamps and metadata about the interaction

Clickstream data is organized into batches through the `clickstream_dao` interface, with each batch representing a collection of related user sessions.

```python
# Retrieve a specific batch of clickstream data
clickstream = context.clickstream_dao.get_batch_sessions(task.batch_id)
```

### 2. Item Retrieval

Along with clickstream data, the system needs access to the actual items being ranked:

- Items are loaded through various `DataLoader` implementations:
  - `AwsS3ImageLoader` - For images stored in S3
  - `GCPTextLoader` - For text stored in Google Cloud
  - `PgsqlTextLoader` - For text stored in PostgreSQL
  - `PgsqlMultiTextColumnLoader` - For structured text data in PostgreSQL

The appropriate loader is selected based on the fine-tuning method's configuration.

```python
# Identify which items need to be loaded
files_to_load: Set[ItemMeta] = set()
for obj in input_with_items:
    files_to_load.update(set(obj.items))

# Download the items using the configured loader
downloaded: List[DownloadedItem] = loader.load(files_to_load)
```

## Data Preparation

### 1. Clickstream Conversion

Raw clickstream data is converted into a structured format for fine-tuning:

```python
# Convert raw sessions to FineTuningInputWithItems objects
input_with_items: List[FineTuningInputWithItems] = [
    converter.convert(session) for session in fine_tuning_inputs
]

# Extract just the input portion (without items metadata)
inputs = [obj.input for obj in input_with_items]
```

Each `FineTuningInput` contains:
- A query that initiated the search
- Results that were shown to the user
- Events (clicks) indicating which results were selected
- Rank information for each result

### 2. Query Retrieval

Queries are extracted from the clickstream data:

```python
# Extract query information from inputs
query_retriever.get_queries(inputs)
```

The `QueryRetriever` handles the extraction of query terms or vectors that will be used during the fine-tuning process.

### 3. Train/Test Splitting

The data is split into training and validation sets:

```python
# Split inputs into train and test sets
training_dataset = clickstream_splitter.split(inputs)
```

The `TrainTestSplitter` ensures:
- A consistent train/test ratio (configurable, default is 80/20)
- Related items stay together in the same split
- Both splits maintain a balance of relevant vs. irrelevant examples

### 4. Item Processing

Items are processed to prepare them for the embedding model:

```python
# Process the items and clickstream data to create the final RankingData
dataset, clickstream_dataset = items_set_manager(downloaded, training_dataset)
```

The `ItemSetManager` handles:
- Field normalization using `DatasetFieldsNormalizer`
- Item splitting when content is too long
- Optional data augmentation
- Preprocessing specific to the item type (text, image, etc.)

## Fine-Tuning Process

### 1. Model Preparation

The initial model is downloaded and prepared for fine-tuning:

```python
# Download the initial model
initial_model = tracker.download_model_by_run_id(iteration.run_id)

# Save to temporary file for hyperopt to use
torch.save(initial_model, initial_model_path)
```

The model is saved to a temporary file to allow multiple hyperparameter optimization runs without re-downloading.

### 2. Parameter Selection

The system uses one of two approaches for parameter selection:

#### a. Hyperparameter Optimization (for initial runs)

```python
# Set up hyperparameter space
initial_hyper_params: Dict[str, Any] = dict()
for key, value in initial_params.items():
    initial_hyper_params[key] = hp.choice(key, value)

# Run hyperopt optimization
fmin(
    lambda params: _finetune_embedding_model_one_step_hyperopt(...),
    initial_hyper_params,
    algo=tpe.suggest,
    max_evals=initial_max_evals,
    trials=trials,
    verbose=False,
)
```

#### b. Best Previous Parameters (for subsequent runs)

```python
# Get best parameters from previous iteration
best_params = [starting_run_param] + tracker.get_top_params_by_experiment_id(starting_run_experiment_id)

# Run with each set of parameters
for index, finetuning_params in enumerate(best_params):
    _finetune_embedding_model_one_step(
        initial_model_path,
        settings,
        ranking_data,
        query_retriever,
        finetuning_params,
        tracker,
    )
```

### 3. Training Loop

For each parameter set, the following steps occur:

```python
# Initialize the fine-tuner
fine_tuner = EmbeddingsFineTuner.create(
    initial_model,
    settings,
    ranking_data.items,
    query_retriever,
    fine_tuning_params,
    tracker,
)

# Move to GPU if available
fine_tuner.to(device)

# Preprocess inputs
fine_tuner.preprocess_inputs(ranking_data.clickstream)

# Set up data loaders
train_dataloader = DataLoader(
    ranking_data.clickstream["train"],
    batch_size=settings.batch_size,
    collate_fn=CustomDataCollator(),
    shuffle=True,
)
test_dataloader = DataLoader(
    ranking_data.clickstream["test"],
    batch_size=1,
    collate_fn=CustomDataCollator(),
    shuffle=False,
)

# Set up early stopping
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=3,
    strict=False,
    verbose=False,
    mode="min",
)

# Run training
trainer = Trainer(
    max_epochs=settings.num_epochs,
    callbacks=[early_stop_callback],
    val_check_interval=int(settings.test_each_n_inputs),
)
trainer.fit(fine_tuner, train_dataloader, test_dataloader)
```

During each training step:
1. Batches of data are passed through the feature extractor
2. Loss is computed with the configured loss function
3. Gradients are computed and applied
4. Metrics are logged to track progress

### 4. Model Selection and Storage

Once training is complete, the model is evaluated and saved:

```python
# Read current embedding quality
quality = tracker.get_quality()

# Save the model if it's the best so far
tracker.save_model(initial_model, True)
```

The `ExperimentsManager` handles saving the model to MLflow and tracks metrics to identify the best model.

## Post-Training Processing

### 1. Model Deployment

If configured, the best model is automatically deployed:

```python
if task.deploy_as_blue:
    # Create reindex task to deploy the model
    reindex_task = context.reindex_task.create(
        schema=ReindexTaskCreateSchema(
            source=ModelParams(embedding_model_id=task.embedding_model_id),
            dest=ModelParams(embedding_model_id=task.best_run_id),
            deploy_as_blue=True,
            wait_on_conflict=task.wait_on_conflict,
            parent_id=task.id,
        ),
        return_obj=True,
    )
    
    # Execute the reindex task
    reindex_task = create_and_send_task(
        reindex_worker, reindex_task, context.reindex_task
    )
```

### 2. Task Completion

The task status is updated to reflect completion:

```python
task.status = TaskStatus.done
task.best_run_id = best_run_id
task.best_model_url = best_model_url
context.fine_tuning_task.update(obj=task)
```

## Memory Management

The system carefully manages memory throughout the process:

```python
# Release memory after using the model
del initial_model
gc.collect()
torch.cuda.empty_cache()
```

This is particularly important for large models that might otherwise cause out-of-memory errors.

## Error Handling

Robust error handling ensures that issues are properly tracked:

```python
except Exception:
    try:
        task.status = TaskStatus.failed
        context.fine_tuning_task.update(obj=task)
    except Exception as exc:
        logger.exception(f"Failed to update task status: {exc}")
    raise
```

## Data Flow Diagram

The end-to-end data flow can be summarized as:

```
Clickstream Data + Item Data
       ↓
Data Preparation (Conversion, Splitting, Normalization)
       ↓
Feature Extraction
       ↓
Loss Calculation
       ↓
Model Optimization
       ↓
Metric Collection and Evaluation
       ↓
Model Versioning and Deployment
```

This pipeline is orchestrated by the `fine_tuning_worker`, which executes each step asynchronously and handles task state management.