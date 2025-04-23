# Documentation for `TextQueryRetriever`

## Overview

TextQueryRetriever is a specialized implementation of QueryRetriever that handles text-based queries. It converts query objects into a text format suitable for retrieval and uses the text content as a unique identifier.

### Inheritance

TextQueryRetriever inherits from QueryRetriever. It extends the base retrieval functionality to handle text-based queries. This design allows it to process TextQueryItem objects effectively and sets the foundation for handling additional query types in the future.

### Parameters

This class does not take initialization parameters. It processes TextQueryItem instances representing queries.

### Usage

**Purpose** - The module simplifies query processing by leveraging a text representation. Inheriting from QueryRetriever, it focuses on converting and retrieving text queries effectively.

### Example

```python
retriever = TextQueryRetriever()
query_item = TextQueryItem(text="Sample query")
text = retriever._convert_query(query_item)
identifier = retriever._get_id(query_item)
```

---

## Documentation for `TextQueryRetriever.get_model_class`

### Functionality
This method returns the concrete model class for text-based queries. It specifically returns the TextQueryItem class.

### Parameters
None.

### Usage
Call this method on a TextQueryRetriever instance to obtain the model class used for handling text queries in the system.

#### Example

```python
from embedding_studio.clickstream_storage.text_query_retriever import TextQueryRetriever

retriever = TextQueryRetriever()
model_cls = retriever.get_model_class()
print(model_cls)  # Should print TextQueryItem
```

---

## Documentation for `TextQueryRetriever._convert_query`

### Functionality
This method converts a query object into its text form. It extracts the text from a TextQueryItem for retrieval operations.

### Parameters
- `query`: A TextQueryItem instance that must have a `dict` attribute.

### Returns
- The text content from the query, which is used as the searchable text.

### Raises
- `ValueError`: If the query object does not have a `dict` attribute.

### Usage
Use this method to transform a query into a text string for further processing in retrieval operations.

#### Example
Assuming `query` is a valid TextQueryItem:

```python
result = retriever._convert_query(query)
```

---

## Documentation for `TextQueryRetriever._get_id`

### Functionality
Extracts a unique identifier from a TextQueryItem by returning its text. It checks that the text is a valid string and raises a ValueError if not.

### Parameters
- `query`: A TextQueryItem instance that should have a string attribute `text`.

### Usage
- **Purpose** - To extract a string identifier from a TextQueryItem and validate the text.

#### Example

```python
# Assuming 'query' is a valid TextQueryItem and 'retriever' is an instance
# of TextQueryRetriever:
identifier = retriever._get_id(query)
```

---

## Documentation for `TextQueryRetriever._get_storage_metadata`

### Functionality
Extracts all storage-related metadata from a text query. It converts the query object into a dictionary using its `model_dump` method. The resulting dictionary holds all the fields required for storage or indexing operations.

### Parameters
- `query`: A TextQueryItem instance from which metadata is extracted. It should contain all relevant fields of the query.

### Return Value
- A dictionary representing all attributes of the input query.

### Usage
**Purpose**: To retrieve a full dictionary of a text query's properties in a format suitable for storage.

#### Example

```python
query = TextQueryItem(text='example text')
retriever = TextQueryRetriever()
metadata = retriever._get_storage_metadata(query)
print(metadata)
```

---

## Documentation for `TextQueryRetriever._get_payload`

### Functionality
No additional payload information is provided for text queries. The method always returns None.

### Parameters
- `query`: A TextQueryItem instance containing text content and optional metadata.

### Usage
- **Purpose**: Retrieve payload information for text queries.

#### Example

```python
# Create an example TextQueryItem
text_query_item = TextQueryItem(text="sample query")

# Retrieve payload, which is expected to be None
payload = text_query_retriever._get_payload(text_query_item)
assert payload is None
```