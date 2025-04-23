# Documentation for CollectionBase

## Functionality
CollectionBase is a foundational implementation of the Collection interface. It provides common functionalities required by various collection types and stores basic collection information. 

## Parameters
- `collection_info`: An instance of CollectionInfo that holds details about the collection, including metadata and configuration.

## Usage
**Purpose** - To define a common behavior for collections in the vector database. It serves as a base class for specialized collections.

### Example
Below is an example of extending CollectionBase:

```python
class CustomCollection(CollectionBase):
    def __init__(self, collection_info):
        super().__init__(collection_info)
        # add custom initialization here
```

### Inheritance
CollectionBase inherits from the Collection interface, ensuring that any subclass adheres to the expected API for collections.

---

## Documentation for CollectionBase.get_info

### Functionality
Returns the collection information stored in the object. This method provides a simple accessor for the underlying CollectionInfo data.

### Parameters
None

### Usage
- **Purpose** - Retrieve basic details about the collection.

#### Example
```python
data = collection.get_info()
```