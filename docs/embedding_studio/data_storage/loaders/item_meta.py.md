# Documentation for `ItemMeta` and its Methods

## Class: `ItemMeta`

### Functionality

The `ItemMeta` class serves as a base class for metadata about data items. It provides a mechanism for unique identification by utilizing either an explicit `object_id` or an abstract `derived_id` that must be implemented in subclasses.

### Parameters

- `object_id`: An optional string that explicitly identifies an item.
- `payload`: An optional dictionary containing additional metadata.

### Usage

- **Purpose**: To offer a consistent way to reference items with unique identifiers. When an `object_id` is absent, the `derived_id` computed by the subclass ensures uniqueness.

### Inheritance

`ItemMeta` inherits from `BaseModel` in the Pydantic library, enabling data validation and extensibility.

#### Example

Below is a sample subclass implementation of `ItemMeta`:

```python
class MyItem(ItemMeta):
    @property
    def derived_id(self) -> str:
        return "unique_identifier_based_on_attributes"
```

---

## Method: `id`

### Functionality

The `id` method returns a unique identifier for an item. It first checks for an explicit `object_id` and uses it. If an `object_id` is not provided, it falls back on the `derived_id` defined by the subclass, ensuring a unique identifier in all cases.

### Parameters

This method does not accept additional parameters.

### Usage

- **Purpose**: To provide a universal way to retrieve a unique identifier for data items in the system.

#### Example

```python
# Example when object_id is provided
item = SomeItemMetaSubclass(object_id='1234', payload={'key': 'value'})
print(item.id)  # Outputs: 1234

# Example when object_id is not provided, derived_id is used
item = SomeItemMetaSubclass(payload={'key': 'value'})
print(item.id)  
```

In the second example, the subclass implementation of `derived_id` will generate the identifier.

---

## Property: `derived_id`

### Functionality

The abstract property `derived_id` computes a unique identifier for an item when an `object_id` is not provided. It should be implemented by subclasses to derive the ID from intrinsic properties, such as file paths or timestamps.

### Parameters

This property does not accept parameters. Subclasses implement their own logic using instance attributes to generate a unique identifier.

### Usage

- **Purpose**: To ensure every item has a unique identifier when an explicit `object_id` is absent. Subclasses must override this property to define their own unique generation strategy.

#### Example

A typical implementation in a subclass might be:

```python
@property
def derived_id(self) -> str:
    return f"{self.file_path}:{self.creation_timestamp}"
```

---

## Class: `ItemMetaWithSourceInfo`

### Functionality

`ItemMetaWithSourceInfo` extends `ItemMeta` by including a source identifier. It ensures unique identification across multiple sources by integrating the source name into the derived ID. This is particularly useful when items originate from various sources, aiding in tracking and logging.

### Parameters

- `source_name`: The name of the source from which the item originates.

### Usage

- **Purpose**: To distinguish item metadata by merging the source name with a derived ID.
- **Motivation**: In multi-source systems, items may clash in ID values.

#### Example

```python
class FileItemMeta(ItemMetaWithSourceInfo):
    file_path: str

    @property
    def derived_id(self) -> str:
        return f"{self.source_name}:{self.file_path}"
```

---

## Property: `derived_id` in `ItemMetaWithSourceInfo`

### Functionality

This derived ID is created by including the source name. It enhances the base class implementation by prefixing the identifier with the source name to maintain uniqueness across items from different sources.

### Parameters

None, as this property method computes the value internally.

### Usage

- **Purpose**: To provide a unique identifier when no explicit `object_id` is set.
- It combines class-specific attributes with source information to guarantee uniqueness.