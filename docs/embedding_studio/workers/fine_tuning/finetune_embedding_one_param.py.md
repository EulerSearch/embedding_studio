# Documentation for `fine_tune_embedding_model_one_param`

## Functionality
The `fine_tune_embedding_model_one_param` function runs embedding fine tuning with a single hyperparameter set. It executes the training for an embedding model while checking for previous runs to avoid redundant work.

## Parameters

- `initial_model`: The embedding model to fine-tune.
- `settings`: Settings for the fine-tuning process.
- `ranking_data`: Data containing clickstream and item information.
- `query_retriever`: Component responsible for retrieving items for queries.
- `fine_tuning_params`: Hyperparameters designated for training.
- `tracker`: Manages experiment tracking.

## Usage
- **Purpose**: The function is utilized to fine-tune an embedding model when necessary.

### Example
Here is a simple usage example:

```python
quality = fine_tune_embedding_model_one_param(
    model, settings, data, retriever, tuning_params, tracker
)
if quality > 0:
    print('Fine tuning completed.')
```