## Documentation for FineTuningMethod

### Functionality
FineTuningMethod is an abstract base class that acts as a plugin for fine-tuning methods. It defines several abstract methods for preprocessing, data loading, query retrieval, experiment management, and inference client creation.

### Motivation
This class provides a unified structure for implementing fine-tuning strategies. By forcing subclasses to implement core functionality, it ensures consistency and ease of extension for custom tuning methods.

### Inheritance
FineTuningMethod inherits from Python's ABC, requiring subclasses to override methods such as get_items_preprocessor, get_data_loader, get_query_retriever, get_manager, and get_inference_client_factory.

### Example
A subclass of FineTuningMethod must implement all abstract methods to provide concrete behavior for specific fine-tuning tasks.

---

## Documentation for FineTuningMethod.get_items_splitter

### Functionality
Return an ItemSplitter instance. By default, it returns an instance of NoSplitter, which means no splitting operation is applied on items.

### Parameters
None.

### Returns
- Instance of ItemSplitter: Normally, a NoSplitter instance is returned.

### Usage
- Purpose: Retrieve the default item splitter for fine-tuning methods.

#### Example
```python
splitter = fine_tuning_instance.get_items_splitter()
if isinstance(splitter, NoSplitter):
    print("No splitter is applied.")
```

---

## Documentation for FineTuningMethod.get_items_preprocessor

### Functionality
This method returns an instance of ItemsDatasetDictPreprocessor. Subclasses must implement this method to provide custom preprocessing logic for fine-tuning methods.

### Parameters
This method does not take any parameters.

### Usage
- **Purpose** - Provides a custom items preprocessor for fine-tuning. The returned preprocessor is used to transform datasets before training.

#### Example
```python
class CustomFineTuning(FineTuningMethod):
    def get_items_preprocessor(self):
        # Implement custom preprocessing
        return CustomItemsPreprocessor()
```

---

## Documentation for FineTuningMethod.get_data_loader

### Functionality
This method is intended to return a DataLoader instance that is used to load and preprocess data for fine-tuning tasks. Subclasses should implement it to provide a data loading mechanism appropriate for their context.

### Parameters
This method does not take any parameters.

### Usage
- Purpose: Retrieve a DataLoader instance to manage dataset loading during fine-tuning.

#### Example
```python
class MyFineTuningMethod(FineTuningMethod):
    def get_data_loader(self):
        # Initialize and return a DataLoader instance
        return MyDataLoader()
```

---

## Documentation for FineTuningMethod.get_query_retriever

### Functionality
Return a QueryRetriever instance that retrieves query data for fine-tuning. Subclasses must override this method to supply a custom QueryRetriever.

### Parameters
- self: The instance of FineTuningMethod.

### Usage
- Purpose: To obtain a query retriever for managing query operations during fine-tuning.

#### Example
```python
class MyFineTuningMethod(FineTuningMethod):
    def get_query_retriever(self) -> QueryRetriever:
        return MyQueryRetriever(config)
```

---

## Documentation for FineTuningMethod.get_manager

### Functionality
This method is designed to return an instance of the ExperimentsManager, which handles experiment tracking and management during fine-tuning. It ensures that any subclass provides a mechanism to log and track experiment details.

### Parameters
This method does not require any parameters.

### Usage
Subclasses of FineTuningMethod must implement this method to return a valid ExperimentsManager instance. This instance will handle tasks such as logging experiment metrics and managing experiment configurations.

#### Example
```python
class MyFineTuningMethod(FineTuningMethod):
    def get_manager(self) -> ExperimentsManager:
        from embedding_studio.experiments.experiments_tracker import ExperimentsManager
        return ExperimentsManager()
```

---

## Documentation for FineTuningMethod.get_inference_client_factory

### Functionality
This method returns a TritonClientFactory instance that is used to create clients for inference operations in the fine-tuning workflow. Subclasses of FineTuningMethod must implement this method to provide a valid TritonClientFactory instance.

### Parameters
None.

### Usage
- **Purpose**: To obtain an inference client factory for starting inference operations during fine-tuning.

#### Example
```python
class MyFineTuningMethod(FineTuningMethod):
    def get_inference_client_factory(self) -> TritonClientFactory:
        return MyTritonClientFactory()
```

---

## Documentation for FineTuningMethod.upload_initial_model

### Functionality
Uploads the initial model to the items_set. This abstract method must be implemented by subclasses, ensuring that the initial model is provided for further fine-tuning processes.

### Parameters
This method does not accept any parameters.

### Usage
- **Purpose** - Define the logic for uploading the initial model prior to the fine-tuning process.

#### Example
```python
class MyFineTuningMethod(FineTuningMethod):
    def upload_initial_model(self) -> None:
        # Example: Implement model upload logic here
```

---

## Documentation for FineTuningMethod.get_fine_tuning_builder

### Functionality
Returns a FineTuningBuilder instance that is used to set up and launch the fine-tuning process. This process leverages user feedback collected in the clickstream sessions to enhance the model.

### Parameters
- `clickstream`: A list of SessionWithEvents containing user feedback. This data is used to adjust the model during fine-tuning.

### Usage
Use this method to obtain a FineTuningBuilder for initiating the fine-tuning process. The builder can then be configured with additional options as required.

#### Example
```python
builder = plugin_instance.get_fine_tuning_builder(clickstream)
builder.configure(...)
```

---

## Documentation for FineTuningMethod.get_embedding_model_info

### Functionality
This method returns an instance of EmbeddingModelInfo for a given model ID. It uses the plugin's meta information and search index details to populate the EmbeddingModelInfo object.

### Parameters
- `id`: A string representing the unique identifier for an embedding model in the model storage system.

### Usage
- **Purpose** - Generates the embedding model info including its name, dimensions, metric type, aggregation type, and vector database settings.

#### Example
For example, given a fine-tuning method instance and a valid model ID:
```python
model_info = fine_tuning_method.get_embedding_model_info("model_id")
```
This call returns a fully populated EmbeddingModelInfo instance with relevant parameters.

---

## Documentation for FineTuningMethod.get_search_index_info

### Functionality
Returns a SearchIndexInfo instance that defines the parameters of the vectordb index used for search operations. Subclasses should implement this method to provide the correct configuration for the index.

### Parameters
- None

### Usage
- **Purpose:** Configure search index parameters for an embedding model.

#### Example
```python
def get_search_index_info(self) -> SearchIndexInfo:
    return SearchIndexInfo(
        dimensions=128,
        metric_type="cosine",
        metric_aggregation_type="avg",
        hnsw={"ef_construction": 200}
    )
```

---

## Documentation for FineTuningMethod.get_vectors_adjuster

### Functionality
Returns a VectorsAdjuster instance. This method defines a procedure for adjusting or transforming vectors generated during fine-tuning. Subclasses must implement this method to provide a custom vector adjustment logic.

### Parameters
- None

### Usage
- **Purpose** - To specify how vectors are modified after fine-tuning.

#### Example
In a subclass, implement the method as follows:
```python
class MyFineTuningMethod(FineTuningMethod):
    def get_vectors_adjuster(self) -> VectorsAdjuster:
        return MyCustomVectorsAdjuster()
```

---

## Documentation for FineTuningMethod.get_vectordb_optimizations

### Functionality
Returns a list of vectordb optimizations that are pending or applicable. This method should be implemented by subclasses to provide custom optimization strategies for vector databases.

### Parameters
None

### Usage
- **Purpose** - To retrieve optimization settings for a vector database. Subclasses implement this method to return a list of outstanding optimizations.

#### Example
```python
class MyFineTuningMethod(FineTuningMethod):
    def get_vectordb_optimizations(self) -> List[Optimization]:
        return [Optimization('optimize_index')]
```

---

## Documentation for CategoriesFineTuningMethod

### Functionality
CategoriesFineTuningMethod is an abstract fine-tuning class for categories. It defines methods for selecting categories and setting parameters like maximum similar categories and margin. This design ensures a standard approach for category-based tuning in plugins.

### Motivation
This class enforces a uniform interface for category fine-tuning, aiding consistent development of category optimization within the project. It clarifies responsibilities and reduces implementation errors.

### Inheritance
CategoriesFineTuningMethod extends FineTuningMethod. It inherits common methods for preprocessing, data loading, query retrieval, manager handling, and inference client creation. Subclasses must implement category-specific methods.

### Example
Below is an outline of a subclass implementation:
```python
class MyCategoriesTuning(CategoriesFineTuningMethod):
    meta = PluginMeta(...)
    
    def get_items_preprocessor(self):
        # custom preprocessor code
        pass
    
    def get_data_loader(self):
        # load data here
        pass
    
    def get_query_retriever(self):
        # retrieve queries
        pass
    
    def get_manager(self):
        # return manager
        pass
    
    def get_inference_client_factory(self):
        # return inference factory
        pass
    
    def get_category_selector(self):
        # return category selector
        pass
    
    def get_max_similar_categories(self):
        return 10
    
    def get_max_margin(self):
        return 0.5
```

### Usage
Extend CategoriesFineTuningMethod to create new plugins that require category-based fine-tuning. Ensure all abstract methods are implemented.

---

## Documentation for CategoriesFineTuningMethod.get_category_selector

### Functionality
This method returns an instance of AbstractSelector. It allows implementers to define custom category selection logic for fine-tuning methods. Subclasses must implement this method to provide the logic needed to select appropriate categories.

### Parameters
- (None) The method does not take parameters apart from self.

### Usage
Override this abstract method in your subclass to return an instance of AbstractSelector. This instance is used during fine-tuning to choose relevant categories.

#### Example
```python
class MyFineTuningMethod(CategoriesFineTuningMethod):
    def get_category_selector(self) -> AbstractSelector:
        # Return an instance of a custom selector
        return MyCategorySelector()
```

---

## Documentation for CategoriesFineTuningMethod.get_max_similar_categories

### Functionality
Returns the maximum number of similar categories to be retrieved. This abstract method should be implemented by concrete classes to control the retrieval process in a fine-tuning setting.

### Parameters
This method does not accept any parameters.

### Usage
**Purpose** - Limit and control the number of similar categories retrieved for further fine-tuning.

#### Example
```python
class CustomCategoriesMethod(CategoriesFineTuningMethod):
    def get_category_selector(self):
        # implementation details
        pass

    def get_max_similar_categories(self) -> int:
        return 5

    def get_max_margin(self) -> float:
        return 0.3
```

---

## Documentation for CategoriesFineTuningMethod.get_max_margin

### Functionality
Returns a float that defines the maximum margin or distance/similarity threshold for category retrieval.

### Parameters
This method takes no parameters.

### Usage
Implement this method in your subclass to return a float that signifies the allowed maximum margin when comparing categories.

#### Example
```python
class MyFineTuningMethod(CategoriesFineTuningMethod):
    def get_category_selector(self):
        # provide implementation

    def get_max_similar_categories(self):
        return 5

    def get_max_margin(self):
        return 0.75
```

---

## Documentation for PluginManager

### Functionality
PluginManager manages plugin discovery, registration, and retrieval for fine-tuning methods. It loads all plugins that extend the FineTuningMethod interface and makes them available through a simple API. This design enables dynamic extensibility for the application.

### Motivation
This class supports dynamic loading of alternative fine-tuning methods. It allows developers to extend system capabilities without altering core logic, promoting modularity and ease of integration.

### Inheritance
PluginManager does not inherit from any other class. However, the plugins registered must be subclasses of FineTuningMethod, ensuring a consistent interface.

### Usage
1. Instantiate PluginManager.
2. Call discover_plugins(directory) to load plugins from a given folder.
3. Retrieve a plugin using get_plugin(name) or inspect available plugins via the plugin_names property.

#### Example
```python
pm = PluginManager()
pm.discover_plugins("plugins")
plugin = pm.get_plugin("example_plugin")
if plugin:
    # Use plugin methods as needed
    pass
```

---

## Documentation for PluginManager.plugin_names

### Functionality
Returns a list of registered plugin names from the internal manager. It retrieves the keys from the plugins dictionary.

### Parameters
- None

### Usage
- Purpose: Get a list of plugin names for discovery and usage.

#### Example
```python
plugin_manager = PluginManager()
names = plugin_manager.plugin_names
print(names)
```

---

## Documentation for PluginManager.get_plugin

### Functionality
Returns a plugin instance for a given plugin name from the internal registry. If the plugin is not registered, it returns None.

### Parameters
- `name`: A string identifier for selecting a plugin.

### Usage
- Purpose: Retrieve and use a plugin based on its name.

#### Example
```python
plugin = plugin_manager.get_plugin('example_plugin')
if plugin:
    plugin.do_something()
```

---

## Documentation for PluginManager.discover_plugins

### Functionality
This method dynamically discovers and loads plugin classes extending FineTuningMethod from a given directory. It imports Python modules, checks for valid plugin meta information, and registers plugins configured in the settings.

### Parameters
- `directory`: A string specifying the path to search for plugin modules. Only files ending with ".py" (except "__init__.py") are considered.

### Usage
- **Purpose:** Load and register plugin classes for fine-tuning.

#### Example
Assuming plugins are located in a folder named "plugins":
```python
from embedding_studio.core.plugin import PluginManager

manager = PluginManager()
manager.discover_plugins("plugins")
```

---

## Documentation for PluginManager._register_plugin

### Functionality
Registers a plugin in the manager. It performs a type check to ensure the provided plugin is an instance of FineTuningMethod and stores it in the plugins dictionary using plugin.meta.name as the key. If the plugin is not a valid instance, a ValueError is raised.

### Parameters
- `plugin`: The plugin instance to register. It must be an instance of FineTuningMethod and contain a meta attribute.

### Usage
- **Purpose** - To register a plugin so it can be later retrieved by its name using get_plugin.

#### Example
```python
manager = PluginManager()
manager._register_plugin(plugin_instance)
```