## Documentation for Fine-Tuning Embedding Models

### Overview
This documentation encompasses three functions related to the fine-tuning of embedding models: `_finetune_embedding_model_one_step`, `_finetune_embedding_model_one_step_hyperopt`, and `finetune_embedding_model`. 

---

### 1. `_finetune_embedding_model_one_step`

#### Functionality
This function performs a single fine-tuning step for an embeddings model. It loads a model from disk, applies parameter tuning, cleans up, and returns a quality metric.

#### Parameters
- `initial_model_path` (str): Path to the initial model file.
- `settings` (FineTuningSettings): Fine-tuning configuration.
- `ranking_data` (RankingData): Ranking and clickstream data.
- `query_retriever` (QueryRetriever): Retrieves query-related items.
- `fine_tuning_params` (FineTuningParams): Tuning parameters.
- `tracker` (ExperimentsManager): Tracks experiment details.

#### Usage
**Purpose**: Run a single fine-tuning step on an embeddings model.

#### Example
Assuming valid objects are provided:
```python
quality = _finetune_embedding_model_one_step(
    "model.pt", settings, ranking_data, query_retriever,
    fine_tuning_params, tracker
)
```

---

### 2. `_finetune_embedding_model_one_step_hyperopt`

#### Functionality
This function fine-tunes an embedding model using hyperopt. It acts as a wrapper around a one-step fine-tuning process and handles exceptions by logging errors.

#### Parameters
- `initial_model_path` (str): Local path of the pre-trained model.
- `settings` (FineTuningSettings): Instance with tuning settings.
- `ranking_data` (RankingData): Instance for ranking data.
- `query_retriever`: Object to retrieve queries.
- `hyperopt_params` (dict): Dictionary of hyperparameter values.
- `tracker` (ExperimentsManager): Tracks tuning progress.

#### Usage
**Purpose**: Optimize an embedding model through a hyperopt search, with quality adjusted according to tracker configuration.

#### Example
```python
quality = _finetune_embedding_model_one_step_hyperopt(
    "model/path", settings, ranking_data, query_retriever, 
    {"lr": 0.01, "dropout": 0.3}, tracker
)
```

---

### 3. `finetune_embedding_model`

#### Functionality
Starts an embedding fine-tuning iteration by downloading and preparing an embeddings model. It performs hyperparameter selection using a provided hyperopt strategy and manages the experiment run via a tracker. The function handles both initial and subsequent runs by applying tuned parameters when available.

#### Parameters
- `iteration`: Information on the fine-tuning iteration.
- `settings`: Settings that govern the fine-tuning process.
- `ranking_data`: Data with clickstream and item details.
- `query_retriever`: Object to retrieve items based on queries.
- `tracker`: Manager for tracking experiment runs.
- `initial_params` (dict): Dictionary mapping hyperparameter names to lists of candidate values.
- `initial_max_evals` (int): Positive integer for maximum initial hyperparameter evaluations (default: 100).

#### Usage
Call this function to initiate a fine-tuning iteration. It downloads an initial model, saves it locally, and executes a hyperparameter search using provided or previously tuned parameters. It returns a quality measure indicating tuning performance.

#### Example
```python
finetune_embedding_model(iteration, settings, ranking_data,
    query_retriever, tracker, initial_params,
    initial_max_evals=100)
```