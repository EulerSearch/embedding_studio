## Documentation for `PluginMeta` and `PluginMeta.validate_name`

### `PluginMeta` Class

#### Functionality
`PluginMeta` represents metadata for a plugin. It stores the name, version, and description of the plugin, and validates the name using a regular expression. The class inherits from `pydantic.BaseModel`, ensuring reliable data validation and parsing.

#### Parameters
- `name`: A string; the unique identifier for the plugin. Must be a valid Python identifier.
- `version`: A string; the plugin version. Defaults to "1.0.0".
- `description`: An optional string that provides extra details about the plugin.

#### Usage
- **Purpose** - Encapsulates plugin identity and version information, aiding in plugin management and validation.

#### Example
```python
pm = PluginMeta(name="MyPlugin", version="1.2.0", description="A plugin for XYZ")
print(pm)
```

---

### `PluginMeta.validate_name` Method

#### Functionality
Validates that the plugin name is a valid Python identifier. It ensures that the name starts with a letter or underscore and only contains letters, digits, and underscores. If the name fails this check, a `ValueError` is raised.

#### Parameters
- `value`: The plugin name string to validate.
- `info`: A `FieldValidationInfo` instance providing context.

#### Usage
- **Purpose** - Ensure plugin names conform to Python identifier rules for consistency across the system.

#### Example
A sample usage in a Pydantic model:
```python
from pydantic import BaseModel
from embedding_studio.models.plugin import PluginMeta

class MyPlugin(PluginMeta):
    pass

# Creating an instance with a valid name
plugin = MyPlugin(name="_plugin123")
```