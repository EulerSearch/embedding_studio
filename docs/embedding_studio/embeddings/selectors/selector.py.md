## Documentation for `AbstractSelector`

### Functionality
`AbstractSelector` is an abstract base class for selector algorithms. It defines a framework for filtering embedding search results based on distance metrics and vector representations. It provides a helper method to convert object parts into padded tensors.

### Motivation
This class enforces a common interface for selector implementations, ensuring that any subclass provides a concrete selection method. This standard approach facilitates interchangeable selection strategies across the project.

### Inheritance
`AbstractSelector` inherits from Python's ABC, requiring subclasses to implement two key abstract methods:

- `select`: Determines the indices of objects to be selected based on criteria such as distance and optional query vector.
- `vectors_are_needed`: Indicates whether the selector requires direct access to embedding vectors.

---

## Documentation for `AbstractSelector._get_categories_tensor`

### Functionality

This method converts a list of `ObjectWithDistance` objects into a padded tensor. It stacks each object's part vectors and pads with zeros so that all objects have a uniform number of parts.

### Parameters

- **items**: A list of `ObjectWithDistance` instances, each containing embedding vectors in their parts.

### Return Value

- Returns a tensor of shape [N, D, M] where:
  - N is the number of objects,
  - D is the embedding dimension, and
  - M is the maximum number of parts across all objects.

### Usage

- **Purpose**: To prepare embedding data for selection by ensuring all objects have consistently padded part vectors.

#### Example

```python
# Example usage for a subclass of AbstractSelector
selector = MySelector()
tensor = selector._get_categories_tensor(items)
print(tensor.shape)
```

---

## Documentation for `AbstractSelector.select`

### Functionality

This method implements distance metrics and optional query embeddings. Different selection strategies can be implemented by overriding this method.

### Parameters

- `categories`: A list of `ObjectWithDistance` instances. Each object holds an embedding vector and its associated distance metric.
- `query_vector`: An optional `torch.Tensor` representing the query embedding. If provided, it is used to enhance the selection process.

### Usage

- **Purpose**: To filter and select objects based on distance scores and embedding comparisons. This method defines the selection strategy.

#### Example

```python
def select(self, categories: List[ObjectWithDistance],
            query_vector: Optional[torch.Tensor] = None) -> List[int]:
    scores = torch.tensor([obj.distance for obj in categories])
    threshold = 0.5
    selected_indices = torch.where(scores < threshold)[0].tolist()
    return selected_indices
```

---

## Documentation for `AbstractSelector.vectors_are_needed`

### Functionality

This property indicates whether the selector requires direct access to the actual embedding vectors. If `True`, the selector needs to compute or inspect the embeddings during selection. If `False`, only pre-computed distances are used.

### Parameters

This property does not take any parameters. It returns a boolean value.

### Usage

Use this property to determine whether the selector implementation requires embedding vectors for its selection logic or can operate with just pre-computed distances.

#### Example

Below is an example of a selector that needs embedding vectors:

```python
@property
def vectors_are_needed(self) -> bool:
    return True  # The selector requires the actual vectors
```

And here is an example for a distance-based selector:

```python
@property
def vectors_are_needed(self) -> bool:
    return False  # The selector uses pre-computed distances only
```