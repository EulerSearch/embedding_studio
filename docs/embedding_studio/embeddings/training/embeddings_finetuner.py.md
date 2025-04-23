## Documentation for `EmbeddingsFineTuner`

### Functionality
The `EmbeddingsFineTuner` class fine-tunes embedding models using PyTorch Lightning. It integrates training logic for ranking loss-based tasks in retrieval scenarios.

### Motivation
This class is designed to adjust embedding models based on query retrieval performance. It simplifies experiments and supports tracking metrics during the fine-tuning process.

### Inheritance
`EmbeddingsFineTuner` extends `pl.LightningModule`, inheriting its flexible training workflows and optimization capabilities.

### Parameters
- `model`: Instance of `EmbeddingsModelInterface` representing the embedding model to be fine-tuned.
- `items_sets`: `DatasetDict` containing training and testing items.
- `query_retriever`: `QueryRetriever` object to fetch relevant items for queries.
- `loss_func`: Ranking loss object implementing `RankingLossInterface`.
- `fine_tuning_params`: `FineTuningParams` holding task hyperparameters.
- `tracker`: `ExperimentsManager` for experiment and metric tracking.
- `metric_calculators`: Optional list of `MetricCalculator` for detailed metric computation (default uses `DistanceShift`).
- `ranker`: Callable ranking function, defaulting to cosine similarity.
- `is_similarity`: Boolean flag indicating if the ranker is similarity-based (True) or distance-based (False).
- `confidence_calculator`: Function to compute confidence scores (default is `dummy_confidences`).
- `step_size`: Scheduler step size (default is 500).
- `gamma`: Learning rate scheduler gamma (default is 0.9).

### Usage
Instantiate `EmbeddingsFineTuner` with the appropriate model, data, loss function, and parameters to begin fine-tuning for improved retrieval performance.

#### Example
```python
model = ...  # An instance of EmbeddingsModelInterface
items_sets = ...  # DatasetDict with train and test splits
query_retriever = QueryRetriever(...)
loss_func = ...
params = FineTuningParams(...)
tracker = ExperimentsManager(...)

tuner = EmbeddingsFineTuner(model, items_sets, query_retriever, loss_func,
                              params, tracker)
```

## Method Documentation

### `EmbeddingsFineTuner.preprocess_inputs`

#### Functionality
Preprocesses fine-tuning inputs in a given `DatasetDict` by calculating rank values for both positive and negative examples. If an input has empty ranks or ranks with None values, the method updates them using the `features_extractor`. This ensures all inputs are valid prior to the training process.

#### Parameters
- `clickstream_dataset`: A `DatasetDict` containing fine-tuning inputs to be preprocessed. Each key (e.g., train, test) should have inputs with attributes like 'not_irrelevant' and 'irrelevant'.

#### Usage
**Purpose**: Prepares the dataset by confirming that every fine-tuning input has correctly calculated ranks.

#### Example
```python
# Assuming clickstream_dataset is prepared and finetuner is an
# instance of EmbeddingsFineTuner:
finetuner.preprocess_inputs(clickstream_dataset)
```

### `EmbeddingsFineTuner.configure_optimizers`

#### Functionality
Configures optimizers and learning rate schedulers for the fine-tuning process in a PyTorch Lightning setup. It validates that `step_size` is a positive integer and that `gamma` is a float in the range (0, 1). Based on model configuration, it returns either one or two optimizers with their corresponding schedulers.

#### Parameters
- `self`: Instance of `EmbeddingsFineTuner`. No additional parameters are required.

#### Usage
**Purpose**: Set up optimizers and learning rate schedulers for fine-tuning the embedding model.

#### Example
```python
optimizers, schedulers = finetuner.configure_optimizers()
```

### `EmbeddingsFineTuner.training_step`

#### Functionality
Executes a single training step for fine-tuning. This method computes features using a feature extractor, calculates the loss, performs a backward pass, updates optimizers, and logs training metrics.

#### Parameters
- `batch`: A list of tuples, each containing two `FineTuningInput` instances.
- `batch_idx`: The index of the current batch.

#### Usage
**Purpose**: Run one training step for model fine-tuning, updating parameters and tracking performance.

#### Example
```python
loss = model.training_step(batch, batch_idx)
print("Training loss:", loss.item())
```

### `EmbeddingsFineTuner.validation_step`

#### Functionality
Performs a single validation step during the validation phase of training. It accepts a batch of data, computes features using the feature extractor, calculates the loss via the loss function, and accumulates validation metrics for later aggregation.

#### Parameters
- `batch`: A list of tuples, where each tuple contains two `FineTuningInput` objects. A single tuple input is also accepted and converted to a list.
- `batch_idx`: The index of the current batch in the validation run.

#### Usage
**Purpose**: Validate one batch, compute the loss, and store metrics for aggregation at the end of an epoch.

#### Example
```python
loss = model.validation_step(batch, batch_idx)
print("Validation Loss:", loss.item())
```

### `EmbeddingsFineTuner.on_validation_epoch_end`

#### Functionality
This method aggregates validation metrics at the end of an epoch. It computes and logs the mean loss along with other validation metrics, then resets the internal metrics accumulator for the next epoch.

#### Parameters
This method does not accept any parameters.

#### Returns
- A float representing the mean validation loss.

#### Usage
**Purpose:** Finalize epoch validation by aggregating the logged metrics. Automatically invoked by the PyTorch Lightning Trainer at the end of an epoch.

#### Example
```python
loss = fine_tuner.on_validation_epoch_end()
```

### `EmbeddingsFineTuner.create`

#### Functionality
Creates a fine-tuner instance for the embeddings model using the provided fine-tuning settings and parameters.

#### Parameters
- `model`: An instance of `EmbeddingsModelInterface` representing the embeddings model.
- `settings`: A `FineTuningSettings` object containing loss function, metric calculators, ranker, and other fine-tuning settings.
- `items_sets`: A `DatasetDict` with train and test keys that holds the data items for fine tuning.
- `query_retriever`: An object to retrieve items related to queries.
- `fine_tuning_params`: Hyperparameters for the fine-tuning task.
- `tracker`: An `ExperimentsManager` for tracking training metrics.

#### Usage
**Purpose**: To create and configure an embeddings fine-tuner with appropriate settings.

#### Example
```python
fine_tuner = EmbeddingsFineTuner.create(
    model, settings, items_sets, query_retriever,
    fine_tuning_params, tracker
)
```