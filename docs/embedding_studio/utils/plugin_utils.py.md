# Merged Documentation

## Functionality Overview

This documentation covers several methods used to manage fine-tuning plugins and their associated vector databases.

### `is_basic_plugin`

#### Functionality
Determines if the given plugin is a basic fine-tuning method, excluding those with categorization capabilities.

#### Parameters
- `plugin`: An instance of `FineTuningMethod` to be checked.

#### Returns
- `bool`: Returns `True` if the plugin is a basic fine-tuning method; otherwise, `False`.

#### Usage
- **Purpose**: To verify a plugin's type for proper vector database selection.

#### Example
```python
if is_basic_plugin(my_plugin):
    vector_db = context.vectordb
```

---

### `is_categories_plugin`

#### Functionality
Determines if a given plugin supports category-based fine-tuning by checking if it is an instance of `CategoriesFineTuningMethod`.

#### Parameters
- `plugin`: The plugin to check (instance of `FineTuningMethod`).

#### Returns
- `bool`: True if the plugin is a `CategoriesFineTuningMethod`; False otherwise.

#### Usage
- **Purpose**: Identify plugins that offer category tuning capabilities for specialized vector database selection.

#### Example
```python
if is_categories_plugin(plugin):
    print("Plugin supports category tuning.")
```

---

### `get_vectordb`

#### Functionality
Retrieves the appropriate vector database for a given plugin. Basic plugins use the standard vector database, while categories plugins use the categories vector database.

#### Parameters
- `plugin`: An instance of `FineTuningMethod` representing the plugin.

#### Usage
- **Purpose**: Retrieve a vector database that matches the plugin's type, ensuring the correct database is used for standard or categorized plugins.

#### Example
```python
vector_db = get_vectordb(plugin)
```

---

### `get_vectordb_by_fine_tuning_name`

#### Functionality
Retrieves the vector database for a plugin by its name. The function looks up the plugin using the context's `plugin_manager` and returns the vector database based on the plugin type: standard database for basic plugins and a categories database for category plugins.

#### Parameters
- `name` (str): The name of the fine-tuning plugin.

#### Returns
- `VectorDb`: The vector database associated with the plugin. For basic plugins, returns the standard vector database; for category plugins, returns the categories database.

#### Usage
- **Purpose**: To obtain the correct vector database for a plugin based on its fine-tuning configuration.

#### Example
```python
vector_db = get_vectordb_by_fine_tuning_name("my_plugin")
# Use vector_db for vector data operations.
```