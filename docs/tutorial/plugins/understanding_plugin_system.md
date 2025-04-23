# Deep Dive: Understanding the Embedding Studio Plugin System

Embedding Studio's plugin architecture is the backbone of its extensibility and customizability. This deep dive will explore how the plugin system works, its key components, and the design patterns that make it powerful.

## 1. Plugin System Architecture Overview

At its core, the plugin system follows a modular architecture pattern with several key design principles:

- **Interface-based design**: All plugins implement abstract base classes
- **Dependency injection**: The application context provides shared resources
- **Runtime discovery**: Plugins are discovered and registered dynamically
- **Configuration-driven selection**: Active plugins are controlled through configuration

The central component is the `PluginManager`, which discovers, validates, and registers plugins during system startup.

## 2. Base Plugin Interfaces

All plugins inherit from abstract base classes that define the required contract:

### FineTuningMethod

```python
class FineTuningMethod(ABC):
    """Base class for fine-tuning methods."""
    
    meta: PluginMeta  # Plugin metadata
    
    @abstractmethod
    def upload_initial_model(self) -> None:
        """Upload the initial model to start from."""
        
    @abstractmethod
    def get_query_retriever(self) -> QueryRetriever:
        """Return the retriever for extracting queries from sessions."""
        
    @abstractmethod
    def get_items_preprocessor(self) -> ItemsDatasetDictPreprocessor:
        """Return the preprocessor for items."""
        
    @abstractmethod
    def get_data_loader(self) -> DataLoader:
        """Return the data loader implementation."""
        
    @abstractmethod
    def get_manager(self) -> ExperimentsManager:
        """Return the experiment manager."""
        
    @abstractmethod
    def get_inference_client_factory(self) -> TritonClientFactory:
        """Return the inference client factory."""
        
    @abstractmethod
    def get_fine_tuning_builder(self, clickstream: List[SessionWithEvents]) -> FineTuningBuilder:
        """Return a fine-tuning builder configured with the given clickstream."""
        
    @abstractmethod
    def get_search_index_info(self) -> SearchIndexInfo:
        """Return search index configuration."""
        
    @abstractmethod
    def get_vectors_adjuster(self) -> VectorsAdjuster:
        """Return the vector adjustment logic."""
        
    @abstractmethod
    def get_vectordb_optimizations(self) -> List[Optimization]:
        """Return vector database optimizations."""
```

### CategoriesFineTuningMethod

```python
class CategoriesFineTuningMethod(FineTuningMethod):
    """Extension of FineTuningMethod for category prediction."""
    
    @abstractmethod
    def get_category_selector(self) -> AbstractSelector:
        """Return the selector for categories."""
        
    @abstractmethod
    def get_max_similar_categories(self) -> int:
        """Return max number of similar categories to retrieve."""
        
    @abstractmethod
    def get_max_margin(self) -> float:
        """Return maximum margin for category similarity."""
```

The abstract methods in these interfaces define the "contract" that any plugin must fulfill. This approach ensures consistency across plugins while allowing for diverse implementations.

## 3. Plugin Metadata and Registration

Each plugin must define a `meta` attribute with identifying information:

```python
class PluginMeta(BaseModel):
    """Metadata for plugin registration and versioning."""
    name: str
    version: str
    description: str
```

The `PluginManager` handles discovery and registration:

```python
class PluginManager:
    """Manages the loading and return of plugins."""
    
    def __init__(self):
        self._plugins: Dict[str, FineTuningMethod] = {}
    
    @property
    def plugin_names(self) -> List[str]:
        return list(self._plugins.keys())
    
    def get_plugin(self, name: str) -> Optional[FineTuningMethod]:
        """Return a plugin by name."""
        return self._plugins.get(name)
    
    def discover_plugins(self, directory: str) -> None:
        """Discover and load plugins from the specified directory."""
        # Dynamically import modules and find plugin classes
        # Register plugins that are in INFERENCE_USED_PLUGINS
```

The manager scans the specified directory, imports Python modules, and registers plugin classes that:
1. Inherit from `FineTuningMethod` or its subclasses
2. Have a valid `meta` attribute
3. Are listed in the `INFERENCE_USED_PLUGINS` configuration

## 4. The App Context and Dependency Injection

The `context` object serves as a dependency injection container, providing access to shared resources:

```python
@dataclasses.dataclass
class AppContext:
    """Application context with shared resources."""
    clickstream_dao: ClickstreamDao
    fine_tuning_task: CRUDFineTuning
    vectordb: VectorDb
    categories_vectordb: VectorDb
    plugin_manager: PluginManager
    model_downloader: ModelDownloader
    mlflow_client: MLflowClientWrapper
    # ... other shared resources
```

Plugins can access these shared resources:

```python
from embedding_studio.context.app_context import context

class MyPlugin(FineTuningMethod):
    def __init__(self):
        # Access shared experiment tracker
        self.manager = ExperimentsManager.from_wrapper(
            wrapper=context.mlflow_client,
            main_metric="test_not_irrelevant_dist_shift",
            plugin_name=self.meta.name,
            accumulators=self.accumulators,
        )
```

This approach:
- Avoids the need for global variables
- Makes dependencies explicit and testable
- Centralizes resource management

## 5. Plugin Component Breakdown

Let's examine each component that plugins must implement:

### Data Loading

The `get_data_loader()` method returns a `DataLoader` implementation that defines how data is loaded:

```python
def get_data_loader(self) -> DataLoader:
    """Return the PostgreSQL loader with custom query generator."""
    return PgsqlTextLoader(
        connection_string="postgresql://user:pass@host:5432/db",
        query_generator=ArticleQueryGenerator,
        text_column="content"
    )
```

**Purpose**: Connects to data sources (PostgreSQL, S3, GCP) and extracts content for embedding.

### Query Retrieval

The `get_query_retriever()` method returns a strategy for extracting search queries from user sessions:

```python
def get_query_retriever(self) -> QueryRetriever:
    """Return the text query retriever."""
    return TextQueryRetriever()
```

**Purpose**: Extracts search terms from user sessions to create training examples.

### Preprocessing

The `get_items_preprocessor()` method returns a preprocessor that prepares raw data for embedding:

```python
def get_items_preprocessor(self) -> ItemsDatasetDictPreprocessor:
    """Return the text preprocessor with sentence splitting."""
    return self.items_set_manager.preprocessor
```

**Purpose**: Handles tokenization, normalization, and preparation of raw data.

The preprocessing pipeline is crucial for effective embeddings. It typically consists of multiple components:

```python
# Set up text tokenization and augmentation
self.items_set_manager = TextItemSetManager(
    # Field normalization ensures consistent field names
    self.normalizer,
    
    # Splitting pipeline for handling text that exceeds token limits
    items_set_splitter=ItemsSetSplitter(
        # Tokenized splitting respects model token limits
        TokenGroupTextSplitter(
            # Use the model's own tokenizer for accurate token counting
            tokenizer=AutoTokenizer.from_pretrained(self.model_name),
            # Sentence splitter creates semantic chunks
            blocks_splitter=DummySentenceSplitter(),
            # Maximum tokens per chunk
            max_tokens=512
        )
    ),
    
    # Data augmentation pipeline for improved robustness
    augmenter=ItemsSetAugmentationApplier(
        # Compose multiple augmentation strategies
        AugmentationsComposition([
            # Vary text case (creates 5 variations)
            ChangeCases(5),
            # Introduce deliberate misspellings (creates 5 variations)
            Misspellings(5),
            # Could add more augmentations here
        ])
    ),
    
    # Apply augmentation only to training data, not test data
    do_augment_test=False,
)
```

For dictionary-style data (JSON objects), you can use specialized components:

```python
# For structured data with multiple fields
self.items_set_manager = DictItemSetManager(
    self.normalizer,
    items_set_splitter=ItemsSetSplitter(
        TokenGroupTextSplitter(
            tokenizer=AutoTokenizer.from_pretrained(self.model_name),
            # Use JSON-aware splitter that respects field boundaries
            blocks_splitter=JSONSplitter(field_names=["title", "description", "tags"]),
            max_tokens=512,
        )
    ),
    # Custom transformation for converting dict to text
    transform=get_json_line_from_dict,
)
```

The preprocessing architecture allows for specialized handling of different data types while maintaining a consistent interface.

### Model Initialization

The `upload_initial_model()` method prepares the base embedding model:

```python
def upload_initial_model(self) -> None:
    """Download and prepare the E5 model."""
    model = context.model_downloader.download_model(
        model_name=self.model_name,
        download_fn=lambda mn: SentenceTransformer(mn),
    )
    model = TextToTextE5Model(model)
    self.manager.upload_initial_model(model)
```

**Purpose**: Establishes the starting point for fine-tuning.

### Vector Database Integration

The `get_search_index_info()` method configures how vectors are stored:

```python
def get_search_index_info(self) -> SearchIndexInfo:
    """Define vector database parameters."""
    return SearchIndexInfo(
        dimensions=768,  # Vector dimensions
        metric_type=MetricType.COSINE,  # Distance metric
        metric_aggregation_type=MetricAggregationType.AVG,  # Aggregation strategy
    )
```

**Purpose**: Ensures vectors are stored with the correct dimensionality and similarity metrics.

### Inference Integration

The `get_inference_client_factory()` method creates a factory for connecting to inference servers:

```python
def get_inference_client_factory(self) -> TritonClientFactory:
    """Create the inference client for Triton server."""
    if self.inference_client_factory is None:
        self.inference_client_factory = TextToTextE5TritonClientFactory(
            f"{settings.INFERENCE_HOST}:{settings.INFERENCE_GRPC_PORT}",
            plugin_name=self.meta.name,
            preprocessor=self.items_set_manager.preprocessor,
            model_name=self.model_name,
        )
    return self.inference_client_factory
```

**Purpose**: Enables efficient model serving via Triton Inference Server.

Inference integration involves several key aspects:

1. **Client Factories**: The factory pattern allows for dynamic creation of inference clients tailored to specific model types:

```python
# For text-to-text E5 models
class TextToTextE5TritonClientFactory(TritonClientFactory):
    def __init__(
        self,
        url: str,
        plugin_name: str,
        preprocessor: Callable[[Union[str, dict]], str] = None,
        model_name: str = "intfloat/multilingual-e5-large",
        retry_config: Optional[RetryConfig] = None,
    ):
        super(TextToTextE5TritonClientFactory, self).__init__(
            url=url,
            plugin_name=plugin_name,
            same_query_and_items=True,  # E5 uses same model for queries and items
            retry_config=retry_config,
        )
        self.preprocessor = preprocessor
        self.model_name = model_name
        self.tokenizer = context.model_downloader.download_model(
            model_name=model_name,
            download_fn=lambda m: AutoTokenizer.from_pretrained(m, use_fast=False),
        )
```

2. **Model-Specific Clients**: Each model type has a specialized client implementation:

```python
# Client for BERT-type models
class TextToTextBERTTritonClient(TritonClient):
    def _prepare_query(self, query: str) -> List[InferInput]:
        """Convert query text to model inputs."""
        inputs = self.tokenizer(
            [query], return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_length,
        )
        # Convert to Triton input format
        infer_inputs = []
        for key, value in inputs.items():
            if key not in ["attention_mask", "input_ids"]:
                continue
            tensor_np = value.numpy().astype(np.int64)
            infer_input = InferInput(key, tensor_np.shape, "INT64")
            infer_input.set_data_from_numpy(tensor_np)
            infer_inputs.append(infer_input)
        return infer_inputs
```

3. **Deployment Metadata**: Each model deployment includes tracking information:

```python
# Model deployment tracking
self.query_model_info = DeployedModelInfo(
    plugin_name=plugin_name,
    embedding_model_id=embedding_model_id,
    model_type="query",
)
```

4. **Retry Logic**: Built-in resilience for transient inference failures:

```python
@retry_method(name="query_inference")
def _send_query_request(self, inputs: List[InferInput]) -> np.ndarray:
    """Send request to Triton with automatic retries."""
    try:
        response = self.client.infer(
            self.query_model_info.name,
            inputs=inputs,
            model_version="1",
            priority=0,
        )
        return response.as_numpy("output")
    except InferenceServerException as e:
        logger.exception(f"Request failed: {e}")
```

The inference integration allows Embedding Studio to decouple the embedding model implementation from the serving infrastructure, enabling easy switching between different model architectures while maintaining a consistent API.

### Fine-Tuning Configuration

The `get_fine_tuning_builder()` method creates a builder with the complete training configuration:

```python
def get_fine_tuning_builder(
    self, clickstream: List[SessionWithEvents]
) -> FineTuningBuilder:
    """Prepare training data and configuration."""
    ranking_dataset = prepare_data(
        clickstream,
        self.sessions_converter,
        self.splitter,
        self.retriever,
        self.data_loader,
        self.items_set_manager,
    )
    
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
```

**Purpose**: Orchestrates the entire training process.

The fine-tuning configuration integrates several components:

1. **Data Preparation**: Transforms clickstream data into training examples:

```python
# Transform clickstream sessions into a format suitable for training
ranking_dataset = prepare_data(
    clickstream,  # Raw user session data
    self.sessions_converter,  # Extracts item information
    self.splitter,  # Divides into train/test sets
    self.retriever,  # Extracts search queries
    self.data_loader,  # Loads actual item content
    self.items_set_manager,  # Processes loaded items
)
```

2. **Training Settings**: Configures the learning process:

```python
# Define the training process parameters
self.settings = FineTuningSettings(
    # Loss function determines how model learns from examples
    loss_func=CosineProbMarginRankingLoss(),
    # How often to perform validation steps
    step_size=35,
    # Percentage of samples to use for testing
    test_each_n_sessions=0.5,
    # Number of complete passes through the data
    num_epochs=3,
)
```

3. **Hyperparameter Search**: Defines the space for optimization:

```python
# Define the hyperparameter search space
self.initial_params = INITIAL_PARAMS  # Base parameters
self.initial_params.update({
    # Whether to only use positive examples
    "not_irrelevant_only": [True],
    # Control the ratio of negative examples
    "negative_downsampling": [0.5],
    # Ordering strategy for examples
    "examples_order": [[11]],
    # Learning rate options
    "learning_rate": [1e-5, 5e-5],
    # Batch size options
    "batch_size": [16, 32],
})
```

4. **Metric Tracking**: Configures performance measurement:

```python
# Define metrics to track during training
self.accumulators = [
    # Training loss with various statistics
    MetricsAccumulator(
        "train_loss",
        calc_mean=True,   # Average loss
        calc_sliding=True,  # Moving average
        calc_min=True,    # Best performance
        calc_max=True,    # Worst performance
    ),
    # How much positive examples improved
    MetricsAccumulator(
        "train_not_irrelevant_dist_shift",
        calc_mean=True,
        calc_sliding=True,
        calc_min=True,
        calc_max=True,
    ),
    # How much negative examples worsened
    MetricsAccumulator(
        "train_irrelevant_dist_shift",
        calc_mean=True,
        calc_sliding=True,
        calc_min=True,
        calc_max=True,
    ),
    # Evaluation metrics (simpler, just final values)
    MetricsAccumulator("test_loss"),
    MetricsAccumulator("test_not_irrelevant_dist_shift"),
    MetricsAccumulator("test_irrelevant_dist_shift"),
]
```

5. **Experiments Manager**: Tracks experiments and manages model versions:

```python
# Configure the experiment tracking system
self.manager = ExperimentsManager.from_wrapper(
    # Use MLflow for experiment tracking
    wrapper=context.mlflow_client,
    # Main metric for comparing model versions
    main_metric="test_not_irrelevant_dist_shift",
    # Tag experiments with plugin name
    plugin_name=self.meta.name,
    # Metrics to track
    accumulators=self.accumulators,
    # Optional experiment name
    experiment_name="text_embeddings_fine_tuning",
    # Optional run name pattern
    run_name_pattern="fine_tuning_{timestamp}",
    # Whether to archive previous runs
    archive_previous_runs=True,
)
```

The fine-tuning configuration brings together all aspects of the training pipeline, from data preparation to model evaluation, in a cohesive and configurable way. This modular approach allows for extensive customization while maintaining a consistent interface.

### Vector Adjustment

The `get_vectors_adjuster()` method defines post-training vector optimization:

```python
def get_vectors_adjuster(self) -> VectorsAdjuster:
    """Define vector adjustment logic."""
    return TorchBasedAdjuster(
        adjustment_rate=0.1, 
        search_index_info=self.get_search_index_info()
    )
```

**Purpose**: Performs incremental adjustments to vectors based on user feedback.

Vector adjustment is a crucial feature that allows embeddings to improve continuously based on user feedback, without requiring full retraining. The implementation involves:

1. **Abstract Adjuster Interface**: Defines the contract for vector adjusters:

```python
class VectorsAdjuster(ABC):
    @abstractmethod
    def adjust_vectors(
        self, data_for_improvement: List[ImprovementInput]
    ) -> List[ImprovementInput]:
        """
        Adjust embedding vectors based on provided improvement data.
        
        :param data_for_improvement: List of ImprovementInput objects with
                                    queries and corresponding clicked/non-clicked items
        :return: Updated list with adjusted vectors
        """
```

2. **Torch-Based Implementation**: Uses PyTorch for gradient-based optimization:

```python
class TorchBasedAdjuster(VectorsAdjuster):
    def __init__(
        self,
        search_index_info: SearchIndexInfo,
        adjustment_rate: float = 0.1,
        num_iterations: int = 10,
        softmin_temperature: float = 1.0,
    ):
        """
        Initialize the TorchBasedAdjuster.
        
        :param search_index_info: Vector database configuration
        :param adjustment_rate: Learning rate for optimization (default: 0.1)
        :param num_iterations: Number of optimization steps (default: 10)
        :param softmin_temperature: Temperature for softmin approximation (default: 1.0)
        """
        self.search_index_info = search_index_info
        self.adjustment_rate = adjustment_rate
        self.num_iterations = num_iterations
        self.softmin_temperature = softmin_temperature
```

3. **Optimization Process**: Implements gradient-based adjustments:

```python
def adjust_vectors(self, data_for_improvement: List[ImprovementInput]) -> List[ImprovementInput]:
    """Adjust vectors to improve search relevance."""
    # Stack query vectors [B, N1, D]
    queries = torch.stack([inp.query.vector for inp in data_for_improvement])
    
    # Stack clicked vectors [B, N2, M, D]
    clicked_vectors = torch.stack([
        torch.stack([ce.vector for ce in inp.clicked_elements], dim=1)
        for inp in data_for_improvement
    ]).transpose(1, 2)
    
    # Stack non-clicked vectors [B, N2, M, D]
    non_clicked_vectors = torch.stack([
        torch.stack([nce.vector for nce in inp.non_clicked_elements], dim=1)
        for inp in data_for_improvement
    ]).transpose(1, 2)
    
    # Enable gradient tracking
    clicked_vectors.requires_grad_(True)
    non_clicked_vectors.requires_grad_(True)
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        [clicked_vectors, non_clicked_vectors], 
        lr=self.adjustment_rate
    )
    
    # Run optimization for several iterations
    for _ in range(self.num_iterations):
        # Reset gradients
        optimizer.zero_grad()
        
        # Compute similarity between queries and clicked items
        clicked_similarity = self.compute_similarity(
            queries, clicked_vectors, self.softmin_temperature
        )
        
        # Compute similarity between queries and non-clicked items
        non_clicked_similarity = self.compute_similarity(
            queries, non_clicked_vectors, self.softmin_temperature
        )
        
        # Define loss: maximize clicked similarity, minimize non-clicked similarity
        loss = -torch.mean(clicked_similarity**3) + torch.mean(non_clicked_similarity**3)
        
        # Compute gradients
        loss.backward()
        
        # Update vectors
        optimizer.step()
    
    # Update the original data structure with optimized vectors
    for batch_idx, inp in enumerate(data_for_improvement):
        for n2_idx, ce in enumerate(inp.clicked_elements):
            ce.vector = clicked_vectors[batch_idx, n2_idx].detach()
        
        for n2_idx, nce in enumerate(inp.non_clicked_elements):
            nce.vector = non_clicked_vectors[batch_idx, n2_idx].detach()
    
    return data_for_improvement
```

4. **Integration with Feedback Loop**: The system collects user feedback and periodically runs adjustments:

```python
# In the feedback processing worker
async def process_feedback_session(session_id: str):
    # Fetch session with user feedback
    session = await context.clickstream_dao.get_session(session_id)
    
    # Convert to improvement input format
    improvement_inputs = converter.convert_to_improvement_inputs(session)
    
    # Get the appropriate plugin
    plugin = context.plugin_manager.get_plugin(plugin_name)
    
    # Get the vector adjuster
    adjuster = plugin.get_vectors_adjuster()
    
    # Apply vector adjustments
    updated_inputs = adjuster.adjust_vectors(improvement_inputs)
    
    # Save adjusted vectors back to database
    await context.vectordb.update_vectors(updated_inputs)
```

Vector adjustment provides a lightweight alternative to full retraining, allowing the system to continuously improve based on user feedback. This is especially valuable in production systems where collecting feedback is ongoing and full retraining is expensive.

## 6. Plugin Lifecycle

The plugin lifecycle follows several distinct phases:

### Discovery and Registration

1. The application starts and initializes the `PluginManager`
2. The manager scans the plugins directory for Python files
3. For each file, it imports the module and finds classes that inherit from `FineTuningMethod`
4. It validates that each class has a `meta` attribute
5. It registers plugins that are listed in `INFERENCE_USED_PLUGINS`

```python
# In app startup code:
context.plugin_manager.discover_plugins(settings.ES_PLUGINS_PATH)
```

### Plugin Initialization

When a plugin is needed:

1. The system gets the plugin by name from the manager
2. The plugin is instantiated, calling its `__init__()` method
3. The plugin initializes its components and resources

```python
plugin = context.plugin_manager.get_plugin("MyPlugin")
```

### Training Workflow

During fine-tuning:

1. The system calls `get_fine_tuning_builder()` with clickstream data
2. The builder prepares training data and configuration
3. The training process runs with the configured parameters
4. Results are logged via the experiment manager

### Inference Workflow

During inference:

1. The system calls `get_inference_client_factory()` to create an inference client
2. The client connects to the Triton server
3. Data is preprocessed using the plugin's preprocessor
4. The model generates embeddings
5. Vector search is performed using the configured index parameters

## 7. Categories Plugin Specialization

The `CategoriesFineTuningMethod` extends the base interface with category-specific functionality:

```python
def get_category_selector(self) -> AbstractSelector:
    """Return selector logic for categories."""
    return ProbsDistBasedSelector(
        search_index_info=SearchIndexInfo(
            dimensions=384,
            metric_type=MetricType.DOT,
            metric_aggregation_type=MetricAggregationType.MIN,
        ),
        is_similarity=True,
        margin=0.2,
        scale=10.0,
        scale_to_one=False,
        prob_threshold=0.985,
    )

def get_max_similar_categories(self) -> int:
    """Return the maximum number of similar categories."""
    return 36

def get_max_margin(self) -> float:
    """Return the maximum similarity margin."""
    return -1.0
```

This specialization allows for category prediction tasks that have different requirements from general embedding tasks.

Category prediction has unique characteristics compared to general embedding:

1. **Category Selectors**: The core of category prediction is the selector that determines which categories match a query:

```python
class AbstractSelector(ABC):
    """Abstract base class for selector algorithms that filter embedding search results."""
    
    @abstractmethod
    def select(
        self,
        categories: List[ObjectWithDistance],
        query_vector: Optional[torch.Tensor] = None,
    ) -> List[int]:
        """
        Selects indices of objects that meet the selection criteria.
        
        :param categories: List of objects with distance metrics
        :param query_vector: Optional query embedding vector
        :return: List of indices of selected objects
        """
    
    @property
    @abstractmethod
    def vectors_are_needed(self) -> bool:
        """Whether this selector requires access to the actual vectors."""
```

2. **Probability-Based Selection**: The `ProbsDistBasedSelector` uses a sigmoid function to convert distances to probabilities:

```python
class ProbsDistBasedSelector(DistBasedSelector):
    """Probability-based selector using sigmoid scaling."""
    
    def __init__(
        self,
        search_index_info: SearchIndexInfo,
        is_similarity: bool = False,
        margin: float = 0.2,
        scale: float = 10.0,
        prob_threshold: float = 0.5,
        scale_to_one: bool = False,
    ):
        """
        Initialize with selection parameters.
        
        :param search_index_info: Vector DB configuration
        :param is_similarity: Whether higher values are better
        :param margin: Base threshold for selection
        :param scale: Controls steepness of sigmoid curve
        :param prob_threshold: Probability cutoff (0.0-1.0)
        :param scale_to_one: Whether to normalize to [0,1]
        """
        super().__init__(
            search_index_info, is_similarity, margin,
            softmin_temperature, scale_to_one
        )
        self._scale = scale
        self._prob_threshold = prob_threshold
    
    def _calculate_binary_labels(self, corrected_values: torch.Tensor) -> torch.Tensor:
        """Convert distances to probabilities and apply threshold."""
        return torch.sigmoid(corrected_values * self._scale) > self._prob_threshold
```

3. **Hierarchical Categories**: Some implementations support hierarchical category structures:

```python
class HierarchicalCategorySelector(AbstractSelector):
    """Selector that respects category hierarchy."""
    
    def __init__(
        self, 
        base_selector: AbstractSelector,
        category_graph: CategoryGraph
    ):
        """
        Initialize with base selector and category hierarchy.
        
        :param base_selector: Core selection logic
        :param category_graph: Directed graph of category relationships
        """
        self.base_selector = base_selector
        self.category_graph = category_graph
    
    def select(
        self,
        categories: List[ObjectWithDistance],
        query_vector: Optional[torch.Tensor] = None,
    ) -> List[int]:
        """Select categories and their ancestors in the hierarchy."""
        # Get direct matches using base selector
        base_indices = self.base_selector.select(categories, query_vector)
        
        # Add parent categories from the hierarchy
        selected_indices = set(base_indices)
        for idx in base_indices:
            category_id = categories[idx].id
            # Add all ancestors in the category hierarchy
            for ancestor in self.category_graph.get_ancestors(category_id):
                # Find the index of this ancestor in our list
                for i, cat in enumerate(categories):
                    if cat.id == ancestor:
                        selected_indices.add(i)
                        break
        
        return sorted(list(selected_indices))
```

4. **Category-Specific Inference**: The inference flow for categories is specialized:

```python
async def get_categories(query: str, plugin_name: str, top_k: int = 10) -> List[CategoryWithConfidence]:
    """Get predicted categories for a query."""
    # Get the plugin
    plugin = context.plugin_manager.get_plugin(plugin_name)
    if not isinstance(plugin, CategoriesFineTuningMethod):
        raise ValueError(f"Plugin {plugin_name} is not a CategoriesFineTuningMethod")
    
    # Get inference client
    client = plugin.get_inference_client_factory().get_client(plugin.model_id)
    
    # Generate query embedding
    query_embedding = client.forward_query(query)
    
    # Get category selector
    selector = plugin.get_category_selector()
    
    # Get max parameters
    max_categories = plugin.get_max_similar_categories()
    max_margin = plugin.get_max_margin()
    
    # Search for similar categories
    categories = await context.categories_vectordb.search(
        plugin_name,
        query_embedding,
        top_k=max_categories,
        max_margin=max_margin
    )
    
    # Apply selector to filter categories
    selected_indices = selector.select(categories)
    selected_categories = [categories[i] for i in selected_indices]
    
    # Convert to response format with confidence scores
    return [
        CategoryWithConfidence(
            id=cat.id,
            name=cat.metadata.get("name", ""),
            confidence=calculate_confidence(cat.distance, is_similarity=plugin.is_similarity)
        )
        for cat in selected_categories
    ]
```

Category plugins enable semantic classification, allowing the system to suggest categories, tags, or labels for queries or content. This is particularly useful for content organization, recommendation systems, and automated tagging.

## 8. Plugin Configuration

Plugins are enabled through configuration in `settings.py`:

```python
# In embedding_studio/core/config.py
INFERENCE_USED_PLUGINS: List[str] = [
    "HFDictTextFineTuningMethod",
    "HFCategoriesTextFineTuningMethod",
    # Add your custom plugins here
]
```

Only plugins listed in this configuration will be registered and available for use.

## 9. Plugin Design Patterns

The plugin system leverages several design patterns:

### Abstract Factory

The `TritonClientFactory` interface and its implementations create families of inference clients:

```python
class TritonClientFactory:
    @abstractmethod
    def get_client(self, embedding_model_id: str, **kwargs):
        """Create and return an inference client."""
```

### Strategy Pattern

Various components (data loaders, query retrievers, selectors) implement strategy interfaces:

```python
class QueryRetriever(ABC):
    @abstractmethod
    def retrieve_query(self, session: SessionWithEvents) -> str:
        """Extract query from a session using a specific strategy."""
```

### Builder Pattern

The `FineTuningBuilder` encapsulates the complex process of constructing a fine-tuning pipeline:

```python
class FineTuningBuilder:
    """Builder for fine-tuning processes."""
    
    def build(self) -> FineTuningProcess:
        """Construct and return a fine-tuning process."""
```

### Dependency Injection

The `AppContext` acts as a service locator and dependency container:

```python
# Configuring dependencies
context = AppContext(
    clickstream_dao=MongoClickstreamDao(mongo_database=mongo.clickstream_mongo_database),
    vectordb=PgvectorDb(...),
    plugin_manager=PluginManager(),
    # ...other dependencies
)

# Accessing dependencies in plugins
self.manager = ExperimentsManager.from_wrapper(
    wrapper=context.mlflow_client,
    # ...other parameters
)
```

## 10. Creating Custom Plugins

To create your own plugin:

1. Create a new Python file in the plugins directory
2. Subclass `FineTuningMethod` or `CategoriesFineTuningMethod`
3. Implement the required abstract methods
4. Define the `meta` attribute with your plugin's information
5. Add your plugin name to `INFERENCE_USED_PLUGINS` in settings

A minimal example:

```python
class MyCustomPlugin(FineTuningMethod):
    meta = PluginMeta(
        name="MyCustomPlugin",
        version="0.1.0",
        description="My custom plugin implementation"
    )
    
    def __init__(self):
        # Initialize your plugin
        # ...
    
    # Implement all abstract methods
    def upload_initial_model(self) -> None:
        # ...
    
    def get_query_retriever(self) -> QueryRetriever:
        # ...
    
    # ... and so on
```

## 11. Advanced Plugin Techniques

### Composition of Components

Plugins often compose multiple smaller components to create sophisticated processing pipelines. This composition pattern allows for high flexibility and reusability:

```python
# Text processing pipeline with multiple components
self.items_set_manager = TextItemSetManager(
    # Field normalizer ensures consistent field names
    self.normalizer,
    
    # Text splitting pipeline (handles token limits)
    items_set_splitter=ItemsSetSplitter(
        # TokenGroupTextSplitter handles tokenization and chunking
        TokenGroupTextSplitter(
            # Use model's own tokenizer for accurate token counting
            tokenizer=AutoTokenizer.from_pretrained(self.model_name),
            # DummySentenceSplitter breaks text into semantic units
            blocks_splitter=DummySentenceSplitter(),
            # Maximum tokens per chunk
            max_tokens=512,
        )
    ),
    
    # Data augmentation pipeline (improves robustness)
    augmenter=ItemsSetAugmentationApplier(
        # Combine multiple augmentation strategies
        AugmentationsComposition([
            # Case variation augmentations
            ChangeCases(5),
            # Misspelling augmentations
            Misspellings(5),
            # Could add more: synonym replacement, word dropping, etc.
        ])
    ),
    
    # Only augment training data, not test data
    do_augment_test=False,
)
```

Each component has a single responsibility, following the single responsibility principle:

1. **Field Normalizer**: Ensures consistent field names across different data sources
2. **Splitters**: Break down text into manageable chunks that respect token limits
3

### Plugin Extensibility

Plugins themselves can be extended through inheritance:

```python
class ImprovedTextPlugin(DefaultTextFineTuningMethod):
    """Extends the default text plugin with improved functionality."""
    
    meta = PluginMeta(
        name="ImprovedTextPlugin",
        version="0.1.0",
        description="Improved version of the default text plugin"
    )
    
    def __init__(self):
        super().__init__()  # Initialize the base plugin
        # Override specific components
        self.splitter = ImprovedTrainTestSplitter()
```

### Resource Management

Plugins should manage resources properly, especially for large models:

```python
def upload_initial_model(self) -> None:
    model = context.model_downloader.download_model(
        model_name=self.model_name,
        download_fn=lambda mn: SentenceTransformer(mn),
    )
    model = TextToTextE5Model(model)
    self.manager.upload_initial_model(model)
    
    # Free memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
```

## 12. Testing Plugins

To test your plugins properly:

1. **Unit Testing**: Test individual components in isolation
2. **Integration Testing**: Test the plugin's interaction with other components
3. **End-to-End Testing**: Test the complete workflow from data loading to inference

Example of a unit test for a custom query retriever:

```python
def test_custom_query_retriever():
    # Arrange
    retriever = CustomQueryRetriever()
    test_session = create_test_session()
    
    # Act
    query = retriever.retrieve_query(test_session)
    
    # Assert
    assert query == "expected query"
```

Example of mocking the app context for testing:

```python
@pytest.fixture
def mock_context(monkeypatch):
    # Create mock components
    mock_loader = MagicMock(spec=DataLoader)
    mock_vectordb = MagicMock(spec=VectorDb)
    
    # Create mock context
    mock_app_context = MagicMock(spec=AppContext)
    mock_app_context.vectordb = mock_vectordb
    
    # Patch the global context
    monkeypatch.setattr(
        "embedding_studio.context.app_context.context",
        mock_app_context
    )
    
    return mock_app_context
```

## Conclusion

The plugin system in Embedding Studio is designed for flexibility and extensibility while maintaining a consistent interface. By understanding the core components and their interactions, you can create custom plugins that integrate seamlessly with the platform.

The key takeaways:

- Plugins implement a consistent interface defined by abstract base classes
- Each plugin handles a specific aspect of the embedding workflow
- The app context provides dependency injection for shared resources
- Plugins are discovered and registered dynamically at startup
- Active plugins are controlled through configuration

With this understanding, you can now create your own custom plugins that extend Embedding Studio's functionality to match your specific requirements.
