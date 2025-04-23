# How to Configure Fine-Tuning

Embedding Studio provides a highly configurable fine-tuning system that can be adapted to various use cases and data types. This document explains how to configure the fine-tuning process to achieve optimal results for your specific needs.

## Fine-Tuning Parameters

The `FineTuningParams` class defines the core parameters that control the fine-tuning process:

```python
class FineTuningParams(BaseModel):
    num_fixed_layers: int
    query_lr: float
    items_lr: float
    query_weight_decay: float
    items_weight_decay: float
    margin: float
    not_irrelevant_only: bool
    negative_downsampling: float
    min_abs_difference_threshold: float = 0.0
    max_abs_difference_threshold: float = 1.0
    examples_order: List[ExamplesType] = [ExamplesType.all_examples]
```

### Key Parameter Explanation

- **num_fixed_layers**: Number of embedding model layers to freeze during fine-tuning
- **query_lr**: Learning rate for the query model optimizer
- **items_lr**: Learning rate for the items model optimizer
- **query_weight_decay**: Weight decay for the query model optimizer
- **items_weight_decay**: Weight decay for the items model optimizer
- **margin**: Margin parameter for the ranking loss function
- **not_irrelevant_only**: Whether to use only non-irrelevant inputs (True for triplet loss)
- **negative_downsampling**: Ratio of negative samples to use (for balancing)
- **min_abs_difference_threshold**: Filter out "soft" pairs with small rank differences
- **max_abs_difference_threshold**: Filter out "hard" pairs with large rank differences
- **examples_order**: Order of examples to use in training

## Fine-Tuning Settings

While `FineTuningParams` controls the hyperparameters, the `FineTuningSettings` class configures the overall training process:

```python
class FineTuningSettings(BaseModel):
    loss_func: RankingLossInterface
    metric_calculators: Optional[List[MetricCalculator]] = None
    ranker: Optional[Callable[[FloatTensor, FloatTensor], FloatTensor]] = COSINE_SIMILARITY
    is_similarity: Optional[bool] = True
    confidence_calculator: Optional[Callable] = dummy_confidences
    step_size: Optional[int] = 500
    gamma: Optional[float] = 0.9
    num_epochs: Optional[int] = 10
    batch_size: Optional[int] = 1
    test_each_n_inputs: Optional[Union[float, int]] = -1
```

### Key Setting Explanation

- **loss_func**: Loss object for ranking task (e.g., `CosineProbMarginRankingLoss`)
- **metric_calculators**: List of metric calculators to track performance
- **ranker**: Function for calculating ranking scores between queries and items
- **is_similarity**: Whether the ranking function is similarity-based (True) or distance-based (False)
- **confidence_calculator**: Function to calculate confidence scores for examples
- **step_size**: Step size for learning rate scheduler
- **gamma**: Gamma value for learning rate scheduler
- **num_epochs**: Number of training epochs
- **batch_size**: Batch size for training
- **test_each_n_inputs**: Frequency of validation (can be ratio if between 0-1)

## PyTorch Lightning Training Configuration

Fine-tuning uses PyTorch Lightning's `Trainer` class, which can be configured with various options:

```python
# Configure early stopping to prevent overfitting
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=3,
    strict=False,
    verbose=False,
    mode="min",
)

# Create and configure the trainer
trainer = Trainer(
    max_epochs=settings.num_epochs,
    callbacks=[early_stop_callback],
    val_check_interval=int(
        settings.test_each_n_inputs
        if settings.test_each_n_inputs > 0
        else len(train_dataloader)
    ),
)

# Start the training process
trainer.fit(fine_tuner, train_dataloader, test_dataloader)
```

## Hyperparameter Optimization

Embedding Studio uses Hyperopt for hyperparameter optimization:

```python
# Define hyperparameter search space
initial_hyper_params = dict()
for key, value in initial_params.items():
    initial_hyper_params[key] = hp.choice(key, value)

# Run the optimization
fmin(
    lambda params: _finetune_embedding_model_one_step_hyperopt(
        initial_model_path,
        settings,
        ranking_data,
        query_retriever,
        params,
        tracker,
    ),
    initial_hyper_params,
    algo=tpe.suggest,
    max_evals=initial_max_evals,
    trials=trials,
    verbose=False,
)
```

A typical search space might look like:

```python
INITIAL_PARAMS = {
    "num_fixed_layers": [0, 1, 2],
    "query_lr": [1e-4, 5e-4, 1e-3],
    "items_lr": [1e-4, 5e-4, 1e-3],
    "query_weight_decay": [0.0, 1e-5, 1e-4],
    "items_weight_decay": [0.0, 1e-5, 1e-4],
    "margin": [0.1, 0.2, 0.5],
    "not_irrelevant_only": [True, False],
    "negative_downsampling": [0.1, 0.3, 0.5],
}
```

The `initial_max_evals` parameter controls how many combinations to try:

```python
fine_tuning_builder = FineTuningBuilder(
    # ... other parameters
    initial_params=self.initial_params,
    initial_max_evals=10,  # Try 10 hyperparameter combinations
)
```

## Creating a Fine-Tuning Plugin

To create a custom fine-tuning method, you'll need to implement a plugin class that provides all the necessary components for fine-tuning:

```python
class CustomFineTuningMethod(FineTuningMethod):
    meta = PluginMeta(
        name="CustomFineTuningMethod",
        version="0.0.1",
        description="A custom fine-tuning plugin"
    )

    def __init__(self):
        # Configure components needed for fine-tuning
        self.model_name = "your-model-name"
        self.data_loader = YourDataLoader()
        self.retriever = YourQueryRetriever()
        self.sessions_converter = ClickstreamSessionConverter(item_type=YourItemMeta)
        self.splitter = TrainTestSplitter()
        self.normalizer = DatasetFieldsNormalizer("item", "item_id")
        self.items_set_manager = YourItemSetManager(...)
        self.accumulators = [...]
        self.manager = ExperimentsManager.from_wrapper(...)
        self.initial_params = YOUR_INITIAL_PARAMS
        self.settings = FineTuningSettings(...)

    def upload_initial_model(self) -> None:
        # Upload initial model to experiment manager
        model = context.model_downloader.download_model(...)
        self.manager.upload_initial_model(model)
        
    def get_fine_tuning_builder(self, clickstream: List[SessionWithEvents]) -> FineTuningBuilder:
        # Prepare data and return builder
        ranking_dataset = prepare_data(...)
        return FineTuningBuilder(
            data_loader=self.data_loader,
            query_retriever=self.retriever,
            clickstream_sessions_converter=self.sessions_converter,
            clickstream_sessions_splitter=self.splitter,
            dataset_fields_normalizer=self.normalizer,
            items_set_manager=self.items_set_manager,
            accumulators=self.accumulators,
            experiments_manager=self.manager,
            fine_tuning_settings=self.settings,
            initial_params=self.initial_params,
            ranking_data=ranking_dataset,
            initial_max_evals=2,
        )
    
    # Implement other required methods...
```

## Task API Configuration

Fine-tuning tasks can be configured through the API:

```python
# Create a fine-tuning task
task = FineTuningTaskRunRequest(
    embedding_model_id="your-model-id",  # The model to fine-tune
    batch_id="your-batch-id",            # Optional: specific batch of clickstream data
    idempotency_key="your-key",          # Optional: for idempotent requests
    deploy_as_blue=True,                 # Whether to deploy as production after fine-tuning
    wait_on_conflict=True                # Whether to wait if another task is using the same resources
)
```

## GPU and Memory Configuration

The system automatically detects and uses available GPU resources:

```python
# Check for CUDA availability
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Move model to appropriate device
fine_tuner.to(device)

# Clean up memory after use
del model
gc.collect()
torch.cuda.empty_cache()
```

## Early Stopping Configuration

To prevent overfitting, you can configure early stopping:

```python
early_stop_callback = EarlyStopping(
    monitor="val_loss",     # Metric to monitor
    patience=3,             # Number of epochs with no improvement before stopping
    strict=False,           # Whether to error if monitor is not found in validation metrics
    verbose=False,          # Whether to log when early stopping is triggered
    mode="min",             # Whether to minimize or maximize the monitored quantity
)
```

## Example: Text Model Fine-Tuning

Here's an example configuration for text embedding models:

```python
# Define hyperparameters
initial_params = {
    "num_fixed_layers": [0, 1],
    "query_lr": [1e-4, 5e-4],
    "items_lr": [1e-4, 5e-4],
    "query_weight_decay": [0.0, 1e-5],
    "items_weight_decay": [0.0, 1e-5],
    "margin": [0.2, 0.5],
    "not_irrelevant_only": [True],
    "negative_downsampling": [0.5],
    "examples_order": [[11]],  # All examples
}

# Configure fine-tuning settings
settings = FineTuningSettings(
    loss_func=CosineProbMarginRankingLoss(),
    step_size=35,
    test_each_n_inputs=0.5,  # Check validation every half epoch
    num_epochs=3,
    batch_size=1,
)

# Set up the item splitter for text
splitter = TokenGroupTextSplitter(
    tokenizer=AutoTokenizer.from_pretrained(model_name),
    blocks_splitter=DummySentenceSplitter(),
    max_tokens=512,
)

# Configure item manager with data augmentation
items_set_manager = TextItemSetManager(
    field_normalizer=DatasetFieldsNormalizer("item", "item_id"),
    items_set_splitter=ItemsSetSplitter(splitter),
    augmenter=ItemsSetAugmentationApplier(
        AugmentationsComposition([ChangeCases(5), Misspellings(5)])
    ),
    do_augment_test=False,
)
```

## Best Practices

1. **Start with sensible defaults**:
   - Learning rates: 1e-4 to 5e-4
   - Weight decay: 0 to 1e-5
   - Margin: 0.2 to 0.5
   - Fixed layers: 0 to 2

2. **Monitor key metrics**:
   - `train_loss` and `test_loss` for overall performance
   - `train_not_irrelevant_dist_shift` and `test_not_irrelevant_dist_shift` for how well positive examples are ranked
   - `train_irrelevant_dist_shift` and `test_irrelevant_dist_shift` for how well negative examples are handled

3. **Use early stopping**:
   - Configure patience based on your dataset size (larger datasets can use smaller patience)
   - Monitor validation loss to prevent overfitting

4. **Layer freezing considerations**:
   - For small datasets, freeze more layers (higher `num_fixed_layers`)
   - For large datasets, unfreeze more layers (lower `num_fixed_layers`)

5. **Optimize hyperparameter search**:
   - Start with a broad search on a small subset of data
   - Refine with narrower ranges on the full dataset
   - Consider the computational cost when setting `initial_max_evals`

6. **Resource management**:
   - Ensure proper GPU memory cleanup with `gc.collect()` and `torch.cuda.empty_cache()`
   - Consider batch size based on available GPU memory
   - For very large models, use gradient accumulation (smaller batch size with multiple steps)
