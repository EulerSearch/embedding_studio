# Implementing Your Own Embedding Studio Plugin: A Complete Guide

This tutorial will walk you through the complete process of creating a custom plugin for Embedding Studio. We'll build a plugin step-by-step, explaining each component's purpose, motivation, and how they fit together.

## Prerequisites

- Basic understanding of Python and object-oriented programming
- Familiarity with embedding models and vector databases
- Embedding Studio installed and configured
- Understanding of the plugin system architecture (refer to the "Understanding the Plugin System" tutorial)

## Step 1: Planning Your Plugin

**Motivation**: A well-planned plugin saves development time and ensures you address your specific requirements. Without proper planning, you might create a plugin that doesn't fully solve your problem or requires significant refactoring later.

Before writing any code, define what your plugin will do:

1. **Purpose**: What problem will your plugin solve?
   - Are you building a domain-specific search solution?
   - Do you need to embed specialized content types?
   - Are there specific performance requirements?

2. **Data Source**: Where will your plugin get data from?
   - What type of database or storage system contains your data?
   - What's the schema or structure of your data?
   - Are there authentication or access considerations?

3. **Model Selection**: Which embedding model will your plugin use?
   - What dimensionality do you need?
   - Is multilingual support required?
   - Are there domain-specific models that might perform better?

4. **Processing Requirements**: Any special data processing needs?
   - Do you need custom text splitting for long content?
   - Is specialized tokenization required for your domain?
   - Would data augmentation improve your results?

For this tutorial, we'll create a plugin that:
- Loads text data from PostgreSQL
- Uses an E5 model for embeddings (good multilingual performance)
- Includes sentence-level splitting and text augmentation
- Supports continuous learning from user feedback

**Why these choices?** PostgreSQL is widely used for structured data storage, E5 offers excellent multilingual embedding quality, sentence splitting helps maintain semantic coherence, and continuous learning allows the system to improve over time without full retraining.

## Step 2: Set Up Your Plugin File

**Motivation**: Properly organizing your code makes it discoverable by the plugin system and maintainable by developers. Embedding Studio uses a convention-based discovery system that expects plugins in specific locations.

Create a new Python file in the `plugins/custom` directory. Let's call it `custom_text_plugin.py`.

**Why this location?** The `plugins/custom` directory is specifically designed for user-created plugins, keeping them separate from the built-in plugins. This separation makes it easier to maintain your custom code during upgrades.

## Step 3: Import Required Dependencies

**Motivation**: Embedding Studio provides a rich ecosystem of components you can leverage. Importing the right dependencies ensures your plugin integrates smoothly with the platform and avoids reinventing existing functionality.

Start by importing all the necessary components:

```python
from typing import List

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from embedding_studio.clickstream_storage.converters.converter import ClickstreamSessionConverter
from embedding_studio.clickstream_storage.query_retriever import QueryRetriever
from embedding_studio.clickstream_storage.text_query_retriever import TextQueryRetriever
from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.core.plugin import FineTuningMethod
from embedding_studio.data_storage.loaders.sql.pgsql.item_meta import PgsqlFileMeta
from embedding_studio.data_storage.loaders.sql.pgsql.pgsql_text_loader import PgsqlTextLoader
from embedding_studio.data_storage.loaders.sql.query_generator import QueryGenerator
from embedding_studio.data_storage.loaders.data_loader import DataLoader
from embedding_studio.embeddings.augmentations.compose import AugmentationsComposition
from embedding_studio.embeddings.augmentations.items_set_augmentation_applier import ItemsSetAugmentationApplier
from embedding_studio.embeddings.augmentations.text.cases import ChangeCases
from embedding_studio.embeddings.augmentations.text.misspellings import Misspellings
from embedding_studio.embeddings.data.clickstream.train_test_splitter import TrainTestSplitter
from embedding_studio.embeddings.data.items.managers.text import TextItemSetManager
from embedding_studio.embeddings.data.preprocessors.preprocessor import ItemsDatasetDictPreprocessor
from embedding_studio.embeddings.data.utils.fields_normalizer import DatasetFieldsNormalizer
from embedding_studio.embeddings.improvement.torch_based_adjuster import TorchBasedAdjuster
from embedding_studio.embeddings.inference.triton.client import TritonClientFactory
from embedding_studio.embeddings.inference.triton.text_to_text.e5 import TextToTextE5TritonClientFactory
from embedding_studio.embeddings.losses.prob_cosine_margin_ranking_loss import CosineProbMarginRankingLoss
from embedding_studio.embeddings.models.text_to_text.e5 import TextToTextE5Model
from embedding_studio.embeddings.splitters.dataset_splitter import ItemsSetSplitter
from embedding_studio.embeddings.splitters.text.dummy_sentence_splitter import DummySentenceSplitter
from embedding_studio.embeddings.splitters.text.tokenized_grouped_splitter import TokenGroupTextSplitter
from embedding_studio.experiments.experiments_tracker import ExperimentsManager
from embedding_studio.experiments.finetuning_settings import FineTuningSettings
from embedding_studio.experiments.initial_params.clip import INITIAL_PARAMS
from embedding_studio.experiments.metrics_accumulator import MetricsAccumulator
from embedding_studio.models.clickstream.sessions import SessionWithEvents
from embedding_studio.models.embeddings.models import MetricAggregationType, MetricType, SearchIndexInfo
from embedding_studio.models.plugin import FineTuningBuilder, PluginMeta
from embedding_studio.vectordb.optimization import Optimization
from embedding_studio.workers.fine_tuning.prepare_data import prepare_data
```

**Why these imports?** Each import serves a specific purpose:
- **SentenceTransformer/AutoTokenizer**: For loading and tokenizing with the E5 model
- **ClickstreamStorage components**: For processing user interactions
- **DataStorage components**: For PostgreSQL integration
- **Embeddings components**: For text processing, augmentation, and vector operations
- **Experiments components**: For tracking training metrics and runs
- **Models components**: For defining how embeddings are structured and stored

## Step 4: Define Your Plugin Class

**Motivation**: The class definition establishes your plugin's identity within the system. The metadata helps Embedding Studio identify, register, and manage your plugin, while inheritance from FineTuningMethod ensures your plugin provides all required functionality.

Create your plugin class by inheriting from `FineTuningMethod` and define its metadata:

```python
class CustomTextPlugin(FineTuningMethod):
    """
    Custom plugin for text embeddings using PostgreSQL data.
    
    This plugin integrates with PostgreSQL data sources and uses E5 models
    for text embedding with sentence-level tokenization.
    """
    
    meta = PluginMeta(
        name="CustomTextPlugin",  # Used in configuration and registration
        version="0.1.0",          # For versioning and tracking
        description="Custom text embedding plugin using PostgreSQL data sources"
    )
```

**Why this structure?** 
- **Inheritance**: Inheriting from `FineTuningMethod` ensures your plugin implements all required interfaces
- **Metadata**: The `meta` attribute provides essential information for plugin registration and management
- **Docstring**: Comprehensive documentation helps other developers understand your plugin's purpose and functionality

## Step 5: Implement Plugin Initialization

**Motivation**: The initialization phase is where you configure all components your plugin will use. This step determines how your plugin will integrate with data sources, process content, train models, and serve embeddings. Each component serves a specific purpose in the embedding pipeline.

The `__init__` method configures all components your plugin will use:

```python
def __init__(self):
    """Initialize the plugin with all required components."""
    
    # Step 1: Define embedding model and tokenizer
    self.model_name = "intfloat/multilingual-e5-base"
    self.inference_client_factory = None
    
    # Step 2: Configure PostgreSQL connection
    connection_string = "postgresql://username:password@hostname:5432/database"
    
    # Step 3: Set up data loader
    self.data_loader = PgsqlTextLoader(
        connection_string=connection_string,
        query_generator=QueryGenerator,  # Default query generator for single table
        text_column="content"            # Column containing text data
    )
    
    # Step 4: Configure query retriever and session converter
    self.retriever = TextQueryRetriever()
    self.sessions_converter = ClickstreamSessionConverter(
        item_type=PgsqlFileMeta
    )
    
    # Step 5: Set up train/test splitter
    self.splitter = TrainTestSplitter()
    
    # Step 6: Configure field normalization
    self.normalizer = DatasetFieldsNormalizer("item", "item_id")
    
    # Step 7: Set up text processing pipeline
    self.items_set_manager = TextItemSetManager(
        self.normalizer,
        items_set_splitter=ItemsSetSplitter(
            TokenGroupTextSplitter(
                tokenizer=AutoTokenizer.from_pretrained(self.model_name),
                blocks_splitter=DummySentenceSplitter(),
                max_tokens=512,
            )
        ),
        augmenter=ItemsSetAugmentationApplier(
            AugmentationsComposition([
                ChangeCases(5),      # Case variation augmentations
                Misspellings(5),     # Misspelling augmentations
            ])
        ),
        do_augment_test=False,  # Don't augment test data
    )
    
    # Step 8: Configure training metrics
    self.accumulators = [
        MetricsAccumulator(
            "train_loss",
            calc_mean=True,
            calc_sliding=True,
            calc_min=True,
            calc_max=True,
        ),
        MetricsAccumulator(
            "train_not_irrelevant_dist_shift",
            calc_mean=True,
            calc_sliding=True,
            calc_min=True,
            calc_max=True,
        ),
        MetricsAccumulator(
            "train_irrelevant_dist_shift",
            calc_mean=True,
            calc_sliding=True,
            calc_min=True,
            calc_max=True,
        ),
        MetricsAccumulator("test_loss"),
        MetricsAccumulator("test_not_irrelevant_dist_shift"),
        MetricsAccumulator("test_irrelevant_dist_shift"),
    ]
    
    # Step 9: Set up experiment tracking
    self.manager = ExperimentsManager.from_wrapper(
        wrapper=context.mlflow_client,
        main_metric="test_not_irrelevant_dist_shift",
        plugin_name=self.meta.name,
        accumulators=self.accumulators,
    )
    
    # Step 10: Configure hyperparameter search
    self.initial_params = INITIAL_PARAMS
    self.initial_params.update({
        "not_irrelevant_only": [True],
        "negative_downsampling": [0.5],
        "examples_order": [[11]],
        "batch_size": [16, 32],  # Try both batch sizes
        "learning_rate": [1e-5, 3e-5],  # Try both learning rates
    })
    
    # Step 11: Set up training settings
    self.settings = FineTuningSettings(
        loss_func=CosineProbMarginRankingLoss(),
        step_size=35,
        test_each_n_sessions=0.5,
        num_epochs=3,
    )
```

**Why these components?**

1. **Model Selection**: The E5 model provides state-of-the-art multilingual performance and is well-suited for text-to-text embedding tasks. The base version (768 dimensions) offers a good balance between quality and computational efficiency.

2. **PostgreSQL Loader**: This component handles the connection to your database and extraction of text content. Using a database allows for structured filtering and rich metadata.

3. **Query Retriever**: The `TextQueryRetriever` extracts search queries from user sessions, which are essential for fine-tuning the model to improve search relevance based on real user behavior.

4. **Train/Test Splitter**: Separating data into training and testing sets is crucial for evaluating model performance and preventing overfitting.

5. **Text Processing Pipeline**: This multi-component system handles:
   - **Tokenization**: Breaking text into tokens the model can understand
   - **Sentence Splitting**: Preserving semantic units when chunking long texts
   - **Token Limits**: Ensuring text chunks fit within model constraints (512 tokens)
   - **Augmentation**: Creating text variations to improve model robustness

6. **Metrics Accumulators**: These track various performance metrics during training, enabling you to:
   - Monitor loss trends (mean, min, max)
   - Track improvement in similarity for relevant items
   - Track decreases in similarity for irrelevant items
   - Compare training and test metrics to detect overfitting

7. **Experiment Tracking**: Integration with MLflow provides:
   - Systematic tracking of training runs
   - Version control for models
   - Visualization of training metrics
   - Comparison between different runs

8. **Hyperparameter Search**: Trying multiple configurations helps find optimal training settings:
   - Whether to use only positive examples
   - How much to downsample negative examples
   - Different batch sizes and learning rates

9. **Training Settings**: These control the fine-tuning process:
   - **Loss Function**: Determines how the model learns from examples
   - **Step Size**: Controls how frequently to evaluate during training
   - **Epochs**: Number of complete passes through the training data

## Step 6: Implement Required Methods

**Motivation**: The abstract methods defined by the `FineTuningMethod` interface are crucial for the plugin system to interact with your plugin. Each method serves a specific purpose in the embedding pipeline, from data loading to model training to inference. By implementing these methods, you enable your plugin to participate in all aspects of the embedding workflow.

Now implement all abstract methods required by the `FineTuningMethod` interface:

```python
def upload_initial_model(self) -> None:
    """
    Download and prepare the E5 model for training.
    This is the base model that will be fine-tuned.
    """
    model = context.model_downloader.download_model(
        model_name=self.model_name,
        download_fn=lambda mn: SentenceTransformer(mn),
    )
    model = TextToTextE5Model(model)
    self.manager.upload_initial_model(model)
    
    # Free memory
    import gc
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()

def get_query_retriever(self) -> QueryRetriever:
    """
    Return the query retriever for extracting search queries from session data.
    """
    return self.retriever

def get_items_preprocessor(self) -> ItemsDatasetDictPreprocessor:
    """
    Return the preprocessor that handles text tokenization and preparation.
    """
    return self.items_set_manager.preprocessor

def get_data_loader(self) -> DataLoader:
    """
    Return the PostgreSQL loader that fetches text content.
    """
    return self.data_loader

def get_manager(self) -> ExperimentsManager:
    """
    Return the experiment manager for tracking training runs.
    """
    return self.manager

def get_inference_client_factory(self) -> TritonClientFactory:
    """
    Return the Triton client factory for model inference.
    """
    if self.inference_client_factory is None:
        self.inference_client_factory = TextToTextE5TritonClientFactory(
            f"{settings.INFERENCE_HOST}:{settings.INFERENCE_GRPC_PORT}",
            plugin_name=self.meta.name,
            preprocessor=self.items_set_manager.preprocessor,
            model_name=self.model_name,
        )
    return self.inference_client_factory

def get_fine_tuning_builder(
    self, clickstream: List[SessionWithEvents]
) -> FineTuningBuilder:
    """
    Prepare data and configure the training process.
    
    This method transforms raw clickstream data into a format suitable for
    training and creates a builder with the complete training configuration.
    """
    # Transform clickstream data into training examples
    ranking_dataset = prepare_data(
        clickstream,
        self.sessions_converter,
        self.splitter,
        self.retriever,
        self.data_loader,
        self.items_set_manager,
    )
    
    # Create and return the builder with complete configuration
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
        initial_max_evals=2,  # Number of hyperparameter combinations to try
    )

def get_search_index_info(self) -> SearchIndexInfo:
    """
    Define the vector database configuration.
    
    This method specifies how vectors will be stored in the database
    and how similarity will be calculated during search.
    """
    return SearchIndexInfo(
        dimensions=768,  # E5-base embedding size
        metric_type=MetricType.COSINE,  # Use cosine similarity
        metric_aggregation_type=MetricAggregationType.AVG,  # Average chunks
    )

def get_vectors_adjuster(self) -> VectorsAdjuster:
    """
    Define the vector adjustment logic for post-training improvements.
    
    This method configures how vectors will be adjusted based on user feedback
    without requiring full retraining.
    """
    return TorchBasedAdjuster(
        adjustment_rate=0.1,  # Learning rate for adjustments
        search_index_info=self.get_search_index_info(),
    )

def get_vectordb_optimizations(self) -> List[Optimization]:
    """
    Define any vector database optimizations.
    
    This method can specify optimizations like index creation or
    vacuum operations. We're returning an empty list for simplicity.
    """
    return []
```

**Why each method matters:**

1. **upload_initial_model()**: This method initializes the base model that will be fine-tuned. It's crucial because:
   - It downloads and prepares the specific model architecture (E5 in this case)
   - It wraps the model in a compatible interface (TextToTextE5Model)
   - It registers the model with the experiment manager for versioning
   - It properly cleans up memory to prevent resource leaks

2. **get_query_retriever()**: This method defines how search queries are extracted from user sessions, which:
   - Enables learning from real user behavior
   - Converts natural language queries into training examples
   - Helps align embeddings with actual search intent

3. **get_items_preprocessor()**: This method returns the component that prepares text for embedding, which:
   - Ensures consistent tokenization across training and inference
   - Handles text normalization and cleaning
   - Applies transformation rules specific to your data domain

4. **get_data_loader()**: This method provides access to your data source, which:
   - Connects to your PostgreSQL database
   - Retrieves content using the configured query generator
   - Converts database rows into a format suitable for embedding

5. **get_manager()**: This method returns the experiment tracking manager, which:
   - Logs training metrics and hyperparameters
   - Stores model versions and artifacts
   - Facilitates comparison between training runs

6. **get_inference_client_factory()**: This method creates a client for model inference, which:
   - Connects to the Triton inference server
   - Handles preprocessing of queries and items
   - Ensures consistent embedding generation

7. **get_fine_tuning_builder()**: This method orchestrates the training process by:
   - Transforming clickstream data into training examples
   - Configuring the complete training pipeline
   - Setting up hyperparameter optimization
   - Managing training/test split and evaluation

8. **get_search_index_info()**: This method defines how vectors are stored and compared, which:
   - Specifies the vector dimensionality (768 for E5-base)
   - Sets the similarity metric (cosine similarity)
   - Configures how multiple vectors per item are aggregated (average)

9. **get_vectors_adjuster()**: This method enables continuous learning by:
   - Providing incremental vector adjustments based on user feedback
   - Setting the learning rate for adjustments
   - Maintaining consistency with the vector database configuration

10. **get_vectordb_optimizations()**: This method can improve database performance by:
    - Adding specialized indexes
    - Configuring vacuum operations
    - Implementing custom SQL optimizationsor

def get_data_loader(self) -> DataLoader:
    """
    Return the PostgreSQL loader that fetches text content.
    """
    return self.data_loader

def get_manager(self) -> ExperimentsManager:
    """
    Return the experiment manager for tracking training runs.
    """
    return self.manager

def get_inference_client_factory(self) -> TritonClientFactory:
    """
    Return the Triton client factory for model inference.
    """
    if self.inference_client_factory is None:
        self.inference_client_factory = TextToTextE5TritonClientFactory(
            f"{settings.INFERENCE_HOST}:{settings.INFERENCE_GRPC_PORT}",
            plugin_name=self.meta.name,
            preprocessor=self.items_set_manager.preprocessor,
            model_name=self.model_name,
        )
    return self.inference_client_factory

def get_fine_tuning_builder(
    self, clickstream: List[SessionWithEvents]
) -> FineTuningBuilder:
    """
    Prepare data and configure the training process.
    
    This method transforms raw clickstream data into a format suitable for
    training and creates a builder with the complete training configuration.
    """
    # Transform clickstream data into training examples
    ranking_dataset = prepare_data(
        clickstream,
        self.sessions_converter,
        self.splitter,
        self.retriever,
        self.data_loader,
        self.items_set_manager,
    )
    
    # Create and return the builder with complete configuration
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
        initial_max_evals=2,  # Number of hyperparameter combinations to try
    )

def get_search_index_info(self) -> SearchIndexInfo:
    """
    Define the vector database configuration.
    
    This method specifies how vectors will be stored in the database
    and how similarity will be calculated during search.
    """
    return SearchIndexInfo(
        dimensions=768,  # E5-base embedding size
        metric_type=MetricType.COSINE,  # Use cosine similarity
        metric_aggregation_type=MetricAggregationType.AVG,  # Average chunks
    )

def get_vectors_adjuster(self) -> VectorsAdjuster:
    """
    Define the vector adjustment logic for post-training improvements.
    
    This method configures how vectors will be adjusted based on user feedback
    without requiring full retraining.
    """
    return TorchBasedAdjuster(
        adjustment_rate=0.1,  # Learning rate for adjustments
        search_index_info=self.get_search_index_info(),
    )

def get_vectordb_optimizations(self) -> List[Optimization]:
    """
    Define any vector database optimizations.
    
    This method can specify optimizations like index creation or
    vacuum operations. We're returning an empty list for simplicity.
    """
    return []
```

## Step 7: Create a Custom Query Generator (Optional)

For more advanced database querying, you can create a custom query generator:

```python
from sqlalchemy import Engine, MetaData, Select, Table, func, select
from embedding_studio.data_storage.loaders.sql.query_generator import QueryGenerator

class ArticleQueryGenerator(QueryGenerator):
    """Custom query generator for articles with joined author and category data."""
    
    def __init__(self, engine: Engine) -> None:
        super().__init__("articles", engine)
        self.metadata = MetaData()
        self.articles_table = None
        self.authors_table = None
        self.categories_table = None
    
    def _init_tables(self):
        """Initialize table references."""
        if self.articles_table is None:
            # Load tables with reflection
            self.articles_table = Table(self.table_name, self.metadata, autoload_with=self.engine)
            self.authors_table = Table("authors", self.metadata, autoload_with=self.engine)
            self.categories_table = Table("categories", self.metadata, autoload_with=self.engine)
    
    def all(self, offset: int, batch_size: int) -> Select:
        """Join articles with authors and categories."""
        self._init_tables()
        return (
            select(
                self.articles_table,
                self.authors_table.c.name.label("author_name"),
                self.categories_table.c.name.label("category_name"),
                # Create a rich text field combining multiple columns
                func.concat(
                    "Title: ", self.articles_table.c.title, "\n",
                    "Author: ", self.authors_table.c.name, "\n",
                    "Category: ", self.categories_table.c.name, "\n",
                    "Content: ", self.articles_table.c.content
                ).label("rich_text")
            )
            .join(
                self.authors_table,
                self.articles_table.c.author_id == self.authors_table.c.id
            )
            .join(
                self.categories_table,
                self.articles_table.c.category_id == self.categories_table.c.id
            )
            .order_by(self.articles_table.c.id)
            .limit(batch_size)
            .offset(offset)
        )
    
    # Override other methods as needed...
```

To use this custom query generator in your plugin:

```python
# In your __init__ method:
self.data_loader = PgsqlTextLoader(
    connection_string=connection_string,
    query_generator=ArticleQueryGenerator,  # Use custom query generator
    text_column="rich_text"  # Use the concatenated text column
)
```

## Step 8: Register Your Plugin

To make your plugin available to Embedding Studio, add it to the configuration. Edit the `settings.py` file to include your plugin in the `INFERENCE_USED_PLUGINS` list:

```python
# In embedding_studio/core/config.py
INFERENCE_USED_PLUGINS: List[str] = [
    # Other plugins...
    "CustomTextPlugin",  # Add your plugin name here
]
```

## Step 9: Test Your Plugin

### 9.1 Basic Validation

First, verify that your plugin is properly registered and accessible:

```python
from embedding_studio.context.app_context import context

# Get all registered plugins
plugin_names = context.plugin_manager.plugin_names
print(f"Registered plugins: {plugin_names}")

# Get your specific plugin
plugin = context.plugin_manager.get_plugin("CustomTextPlugin")
if plugin:
    print(f"Plugin found: {plugin.meta.name} v{plugin.meta.version}")
    print(f"Description: {plugin.meta.description}")
else:
    print("Plugin not registered!")
```

### 9.2 Data Loading Test

Test that your plugin can successfully load data:

```python
# Get your plugin
plugin = context.plugin_manager.get_plugin("CustomTextPlugin")

# Get the data loader
loader = plugin.get_data_loader()

# Load a sample batch
from embedding_studio.data_storage.loaders.sql.pgsql.item_meta import PgsqlFileMeta
sample_items = [
    PgsqlFileMeta(object_id="1"),
    PgsqlFileMeta(object_id="2"),
]
loaded_items = loader.load_items(sample_items)

# Print the results
for item in loaded_items:
    print(f"ID: {item.id}")
    print(f"Content preview: {item.data[:100]}...")
    print("-" * 50)
```

### 9.3 Full Fine-Tuning Test

To test the complete fine-tuning process:

1. Create a new embedding index using your plugin
2. Upload sample clickstream data
3. Initiate fine-tuning
4. Monitor the training metrics in MLflow

## Step 10: Customizing Your Plugin

Once your basic plugin is working, you can enhance it:

### 10.1 Add Custom Augmentations

Create custom text augmentations:

```python
from embedding_studio.embeddings.augmentations.augmentation_with_random_selection import AugmentationWithRandomSelection
from typing import List

class SynonymReplacement(AugmentationWithRandomSelection):
    """Replace words with synonyms."""
    
    def __init__(self, selection_size: float = 1.0):
        super(SynonymReplacement, self).__init__(selection_size)
        # Initialize synonym dictionary
        self.synonyms = {
            "good": ["great", "excellent", "fine", "positive"],
            "bad": ["poor", "terrible", "awful", "negative"],
            # Add more synonyms...
        }
    
    def _raw_transform(self, object: str) -> List[str]:
        """Apply synonym replacement to create variations."""
        import random
        
        # Original text is always included
        results = [object]
        
        # Create 3 variations with different synonym replacements
        for _ in range(3):
            text = object
            # Apply replacements with a certain probability
            for word, synonyms in self.synonyms.items():
                if word in text and random.random() < 0.7:
                    text = text.replace(word, random.choice(synonyms))
            results.append(text)
        
        return results
```

Then use it in your plugin:

```python
# In your __init__ method:
augmenter=ItemsSetAugmentationApplier(
    AugmentationsComposition([
        ChangeCases(5),            # Case variation augmentations
        Misspellings(5),           # Misspelling augmentations
        SynonymReplacement(5),     # Add your custom augmentation
    ])
)
```

### 10.2 Implement Advanced Vector Adjustment

You can implement a custom vector adjuster:

```python
from embedding_studio.embeddings.improvement.vectors_adjuster import VectorsAdjuster
from embedding_studio.embeddings.data.clickstream.improvement_input import ImprovementInput

class AdvancedVectorAdjuster(VectorsAdjuster):
    """Custom vector adjuster with advanced features."""
    
    def __init__(self, learning_rate: float = 0.1, decay_factor: float = 0.9):
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
    
    def adjust_vectors(
        self, data_for_improvement: List[ImprovementInput]
    ) -> List[ImprovementInput]:
        """Adjust vectors with a custom logic."""
        for improvement_input in data_for_improvement:
            query_vector = improvement_input.query.vector
            
            # Adjust clicked items to be closer to query
            for clicked in improvement_input.clicked_elements:
                # Move vector toward query with learning rate
                clicked.vector = clicked.vector + self.learning_rate * (
                    query_vector - clicked.vector
                )
            
            # Adjust non-clicked items to be further from query
            for non_clicked in improvement_input.non_clicked_elements:
                # Move vector away from query with decay factor
                non_clicked.vector = non_clicked.vector - self.learning_rate * self.decay_factor * (
                    query_vector - non_clicked.vector
                )
        
        return data_for_improvement
```

Then use it in your plugin:

```python
def get_vectors_adjuster(self) -> VectorsAdjuster:
    """Return a custom vector adjuster."""
    return AdvancedVectorAdjuster(
        learning_rate=0.1,
        decay_factor=0.8
    )
```

## Step 11: Documentation and Best Practices

### 11.1 Document Your Plugin

Add comprehensive documentation to your plugin:

```python
class CustomTextPlugin(FineTuningMethod):
    """
    Custom plugin for text embeddings using PostgreSQL data.
    
    This plugin integrates with PostgreSQL data sources and uses E5 models
    for text embedding with sentence-level tokenization. It includes data
    augmentation for improved robustness and supports continuous learning
    from user feedback.
    
    Features:
    - PostgreSQL data loading with rich text support
    - E5-based text embeddings (768 dimensions)
    - Sentence-level tokenization and chunking
    - Text augmentation (case variation, misspellings)
    - Gradient-based vector adjustment for continuous learning
    
    Usage:
    1. Configure the database connection in __init__
    2. Add the plugin to INFERENCE_USED_PLUGINS in settings
    3. Create an embedding index using this plugin
    4. Upload clickstream data for fine-tuning
    
    Example:
    ```
    # Create an embedding index
    curl -X POST "http://localhost:8000/api/v1/embeddings/" \
      -H "Content-Type: application/json" \
      -d '{"name": "custom_text_index", "plugin_name": "CustomTextPlugin"}'
    ```
    """
```

### 11.2 Follow Best Practices

1. **Error Handling**: Add robust error handling to your plugin

```python
def get_data_loader(self) -> DataLoader:
    """Return the PostgreSQL loader with error handling."""
    try:
        return self.data_loader
    except Exception as e:
        logger.error(f"Failed to get data loader: {e}")
        # Fall back to a simpler loader or raise a more informative error
        raise RuntimeError(f"Data loader initialization failed: {e}")
```

2. **Resource Management**: Ensure proper cleanup of large resources

```python
def upload_initial_model(self) -> None:
    """Download and upload the model with proper resource management."""
    try:
        model = context.model_downloader.download_model(
            model_name=self.model_name,
            download_fn=lambda mn: SentenceTransformer(mn),
        )
        model = TextToTextE5Model(model)
        self.manager.upload_initial_model(model)
    finally:
        # Always clean up, even if there was an error
        import gc
        import torch
        try:
            del model
        except NameError:
            pass  # model wasn't created
        gc.collect()
        torch.cuda.empty_cache()
```

3. **Configuration Management**: Use environment variables for sensitive data

```python
import os

# In your __init__ method:
connection_string = os.environ.get(
    "POSTGRES_CONNECTION_STRING",
    "postgresql://username:password@hostname:5432/database"
)
```

## Conclusion

You've now created a complete, production-ready plugin for Embedding Studio! This plugin:

1. Connects to PostgreSQL data sources
2. Uses E5 models for text embedding
3. Implements sentence-level tokenization and text augmentation
4. Supports continuous learning from user feedback
5. Follows best practices for error handling and resource management

Your plugin can now be used to create embedding indexes, fine-tune models on your specific data, and power semantic search applications.

The plugin architecture allows for extensive customization while maintaining a consistent interface, making it easy to adapt to different use cases and data sources.