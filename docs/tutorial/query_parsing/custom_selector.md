# Implementing Custom Category Selectors in Embedding Studio

This tutorial will guide you through implementing custom category selectors for Embedding Studio's query parsing system. Category selectors determine which categories are relevant for a given search query based on various selection algorithms and distance metrics.

## Understanding the Selector Architecture

Embedding Studio uses a hierarchical architecture for category selectors:

1. **AbstractSelector** - The base interface for all selectors
2. **DistBasedSelector** - Selectors that work with pre-calculated distances
3. **VectorsBasedSelector** - Selectors that work directly with embedding vectors

Let's explore each level and how to implement your own custom selectors.

## Base Selector Interface

All selectors implement the `AbstractSelector` interface:

```python
class AbstractSelector(ABC):
    @property
    @abstractmethod
    def vectors_are_needed(self) -> bool:
        """Indicates whether this selector requires access to actual embedding vectors."""
        raise NotImplementedError
        
    @abstractmethod
    def select(
        self,
        categories: List[ObjectWithDistance],
        query_vector: Optional[torch.Tensor] = None,
    ) -> List[int]:
        """Selects indices of objects that meet selection criteria."""
        raise NotImplementedError
```

Key methods:
- `vectors_are_needed`: Indicates if the selector needs the raw vectors (vs. just distances)
- `select`: Returns indices of categories that meet the selection criteria

## Distance-Based Selectors

For many applications, you can work with pre-calculated distances using the `DistBasedSelector`:

```python
class DistBasedSelector(AbstractSelector):
    def __init__(
        self,
        search_index_info: SearchIndexInfo,
        is_similarity: bool = False,
        margin: float = 0.2,
        softmin_temperature: float = 1.0,
        scale_to_one: bool = False,
    ):
        self._search_index_info = search_index_info
        self._is_similarity = is_similarity
        self._margin = margin
        self._softmin_temperature = softmin_temperature
        self._scale_to_one = scale_to_one
    
    @property
    def vectors_are_needed(self) -> bool:
        return False
        
    @abstractmethod
    def _calculate_binary_labels(
        self, corrected_values: torch.Tensor
    ) -> torch.Tensor:
        """Calculates binary selection labels (0 or 1) from corrected distance values."""
        raise NotImplementedError
        
    def select(
        self,
        categories: List[ObjectWithDistance],
        query_vector: Optional[torch.Tensor] = None,
    ) -> List[int]:
        values = self._convert_values(categories)
        positive_threshold_min = 1 - self._margin if self._is_similarity else self._margin
        corrected_values = values - positive_threshold_min
        bin_labels = self._calculate_binary_labels(corrected_values)
        return torch.nonzero(bin_labels).T[0].tolist()
```

Key features:
- Works with pre-calculated distance values (doesn't need vectors)
- Handles different types of distance metrics and normalization
- Requires subclasses to implement the actual selection logic

## Example: Probability-Based Selector

The default `ProbsDistBasedSelector` uses a sigmoid function to convert distances to probabilities:

```python
class ProbsDistBasedSelector(DistBasedSelector):
    def __init__(
        self,
        search_index_info: SearchIndexInfo,
        is_similarity: bool = False,
        margin: float = 0.2,
        softmin_temperature: float = 1.0,
        scale: float = 10.0,
        prob_threshold: float = 0.5,
        scale_to_one: bool = False,
    ):
        super().__init__(
            search_index_info=search_index_info,
            is_similarity=is_similarity,
            margin=margin,
            softmin_temperature=softmin_temperature,
            scale_to_one=scale_to_one,
        )
        self._scale = scale
        self._prob_threshold = prob_threshold

    def _calculate_binary_labels(
        self, corrected_values: torch.Tensor
    ) -> torch.Tensor:
        return (
            torch.sigmoid(corrected_values * self._scale)
            > self._prob_threshold
        )
```

This selector:
1. Converts adjusted distance values to probabilities using sigmoid
2. Compares probabilities against a threshold
3. Returns a binary tensor (1 for selected, 0 for not selected)

## Implementing Your Own Selector

### 1. Implementing a Threshold-Based Selector

Here's a simple threshold-based selector that's less complex than the probability-based approach:

```python
class SimpleThresholdSelector(DistBasedSelector):
    def _calculate_binary_labels(
        self, corrected_values: torch.Tensor
    ) -> torch.Tensor:
        # Simply checks if corrected values are positive
        # (which means they passed the margin threshold)
        return corrected_values > 0
```

This selector selects categories where:
- For similarity metrics: similarity > (1 - margin)
- For distance metrics: distance < margin

### 2. Implementing a Top-K Selector

This selector always returns the top K categories regardless of absolute distance:

```python
class TopKSelector(DistBasedSelector):
    def __init__(
        self,
        search_index_info: SearchIndexInfo,
        k: int = 3,
        **kwargs
    ):
        super().__init__(search_index_info=search_index_info, **kwargs)
        self.k = k
        
    def _calculate_binary_labels(
        self, corrected_values: torch.Tensor
    ) -> torch.Tensor:
        if len(corrected_values) <= self.k:
            # If we have fewer values than k, select all
            return torch.ones_like(corrected_values, dtype=torch.bool)
            
        # Get indices of top-k values
        _, indices = torch.topk(corrected_values, self.k)
        
        # Create a binary mask with 1s at top-k indices
        mask = torch.zeros_like(corrected_values, dtype=torch.bool)
        mask[indices] = True
        
        return mask
```

This selector always selects exactly K categories (or all if there are fewer than K).

### 3. Implementing a Dynamic Threshold Selector

This selector adapts its threshold based on the distribution of values:

```python
class DynamicThresholdSelector(DistBasedSelector):
    def __init__(
        self,
        search_index_info: SearchIndexInfo,
        percentile: float = 75.0,
        min_threshold: float = 0.1,
        **kwargs
    ):
        super().__init__(search_index_info=search_index_info, **kwargs)
        self.percentile = percentile
        self.min_threshold = min_threshold
        
    def _calculate_binary_labels(
        self, corrected_values: torch.Tensor
    ) -> torch.Tensor:
        if len(corrected_values) == 0:
            return torch.tensor([], dtype=torch.bool)
            
        # Calculate a threshold at the specified percentile
        threshold = max(
            torch.quantile(corrected_values, self.percentile / 100.0).item(),
            self.min_threshold
        )
        
        # Select values above the threshold
        return corrected_values > threshold
```

This selector dynamically adjusts based on the distribution of distances, selecting categories above a percentile threshold.

## Vector-Based Selectors

For more advanced scenarios, you can work directly with embedding vectors using `VectorsBasedSelector`:

```python
class CustomVectorSelector(VectorsBasedSelector):
    def __init__(
        self,
        search_index_info: SearchIndexInfo,
        reference_vectors: List[torch.Tensor],
        **kwargs
    ):
        super().__init__(search_index_info=search_index_info, **kwargs)
        self.reference_vectors = reference_vectors
        
    def _calculate_binary_labels(
        self, corrected_values: torch.Tensor
    ) -> torch.Tensor:
        # Your custom selection logic
        return corrected_values > 0
        
    def select(
        self,
        categories: List[ObjectWithDistance],
        query_vector: Optional[torch.Tensor] = None,
    ) -> List[int]:
        if query_vector is None:
            return []
            
        # Get tensor of category vectors
        category_vectors = self._get_categories_tensor(categories)
        
        # Compare with reference vectors for additional context
        reference_similarities = torch.stack([
            F.cosine_similarity(query_vector, ref.unsqueeze(0))
            for ref in self.reference_vectors
        ])
        
        # Use reference vector information to inform selection
        # (This is just an example - implement your own logic)
        if torch.max(reference_similarities) > 0.8:
            # If query is very similar to a reference vector,
            # be more selective in category matching
            margin = self._margin * 0.8
        else:
            # Otherwise use standard margin
            margin = self._margin
            
        # Calculate distances using adjusted margin
        values = self._calculate_distance(
            query_vector,
            category_vectors,
            self._softmin_temperature,
            self._is_similarity,
        )
        
        positive_threshold_min = 1 - margin if self._is_similarity else margin
        corrected_values = values - positive_threshold_min
        bin_labels = self._calculate_binary_labels(corrected_values)
        
        return torch.nonzero(bin_labels).T[1].tolist()
```

This selector:
1. Takes reference vectors for additional context
2. Dynamically adjusts its selection threshold based on query similarity to references
3. Works directly with vector embeddings for more sophisticated matching

## Registering Your Custom Selector

To use your custom selector, you need to register it with the category selection system:

```python
# In your plugin initialization code
from embedding_studio.embeddings.selectors.dist_based_selector import DistBasedSelector
from my_custom_selectors import SimpleThresholdSelector

class MyPlugin:
    def get_category_selector(self) -> AbstractSelector:
        # Create and return your custom selector
        return SimpleThresholdSelector(
            search_index_info=self.search_index_info,
            is_similarity=False,  # Using distance metrics (lower is better)
            margin=0.25,  # Select categories with distance < 0.25
        )
```

## Working with Different Distance Metrics

Your selector implementation needs to consider the distance metric being used:

### Cosine Distance

```python
class CosineSelector(DistBasedSelector):
    def __init__(self, **kwargs):
        super().__init__(
            is_similarity=False,  # We're working with distance, not similarity
            **kwargs
        )
        
    def _convert_values(self, categories: List[ObjectWithDistance]) -> torch.Tensor:
        values = []
        for category in categories:
            # For cosine, smaller distance means more similar
            value = category.distance
            values.append(value)
        return torch.tensor(values)
```

### Dot Product

```python
class DotProductSelector(DistBasedSelector):
    def __init__(self, **kwargs):
        super().__init__(
            is_similarity=True,  # For dot product, higher is more similar
            **kwargs
        )
        
    def _convert_values(self, categories: List[ObjectWithDistance]) -> torch.Tensor:
        values = []
        for category in categories:
            # For dot product, we negate the value since our system expects distances
            value = -category.distance
            values.append(value)
        return torch.tensor(values)
```

## Advanced Techniques

### 1. Combining Multiple Selectors

You can create meta-selectors that combine the results from multiple selection strategies:

```python
class CombinedSelector(AbstractSelector):
    def __init__(self, selectors: List[AbstractSelector]):
        self.selectors = selectors
        
    @property
    def vectors_are_needed(self) -> bool:
        # We need vectors if any selector needs them
        return any(selector.vectors_are_needed for selector in self.selectors)
        
    def select(
        self,
        categories: List[ObjectWithDistance],
        query_vector: Optional[torch.Tensor] = None,
    ) -> List[int]:
        # Get selections from each selector
        all_selections = []
        for selector in self.selectors:
            selections = selector.select(categories, query_vector)
            all_selections.extend(selections)
            
        # Remove duplicates and sort
        return sorted(list(set(all_selections)))
```

### 2. Context-Aware Selectors

This selector adjusts its behavior based on the query context:

```python
class ContextAwareSelector(DistBasedSelector):
    def __init__(
        self,
        search_index_info: SearchIndexInfo,
        strict_keywords: List[str],
        relaxed_keywords: List[str],
        strict_margin: float = 0.15,
        relaxed_margin: float = 0.3,
        **kwargs
    ):
        super().__init__(search_index_info=search_index_info, **kwargs)
        self.strict_keywords = [k.lower() for k in strict_keywords]
        self.relaxed_keywords = [k.lower() for k in relaxed_keywords]
        self.strict_margin = strict_margin
        self.relaxed_margin = relaxed_margin
        self.query_text = ""
        
    def select(
        self,
        categories: List[ObjectWithDistance],
        query_vector: Optional[torch.Tensor] = None,
    ) -> List[int]:
        # Determine which margin to use based on query
        effective_margin = self._margin  # Default
        
        query_lower = self.query_text.lower()
        
        # Use strict matching for certain queries
        if any(keyword in query_lower for keyword in self.strict_keywords):
            effective_margin = self.strict_margin
            
        # Use relaxed matching for other queries
        elif any(keyword in query_lower for keyword in self.relaxed_keywords):
            effective_margin = self.relaxed_margin
            
        # Then proceed with selection using the context-appropriate margin
        values = self._convert_values(categories)
        positive_threshold_min = 1 - effective_margin if self._is_similarity else effective_margin
        corrected_values = values - positive_threshold_min
        bin_labels = self._calculate_binary_labels(corrected_values)
        
        return torch.nonzero(bin_labels).T[0].tolist()
```

## Testing Your Selector

Before deploying a custom selector, test it thoroughly:

```python
def test_selector(selector, categories, query_vector=None):
    """Test a selector with sample data."""
    selected_indices = selector.select(categories, query_vector)
    
    print(f"Selected {len(selected_indices)} out of {len(categories)} categories")
    
    for idx in selected_indices:
        category = categories[idx]
        print(f"- {category.object_id}: distance={category.distance:.4f}")
        
    return selected_indices

# Create test data
test_categories = [
    ObjectWithDistance(object_id="category1", distance=0.1, payload={"name": "Category 1"}),
    ObjectWithDistance(object_id="category2", distance=0.2, payload={"name": "Category 2"}),
    ObjectWithDistance(object_id="category3", distance=0.3, payload={"name": "Category 3"}),
    ObjectWithDistance(object_id="category4", distance=0.4, payload={"name": "Category 4"}),
    ObjectWithDistance(object_id="category5", distance=0.5, payload={"name": "Category 5"}),
]

# Test your selector
test_selector(SimpleThresholdSelector(
    search_index_info=mock_search_info,
    margin=0.35
), test_categories)
```

## Debugging and Troubleshooting

When implementing custom selectors, these debugging tips can help:

### Visualizing Selection Boundaries

```python
import matplotlib.pyplot as plt

def visualize_selector_boundary(selector, max_distance=1.0, points=100):
    """Visualize the selection boundary for a distance-based selector."""
    distances = torch.linspace(0, max_distance, points)
    
    # Convert distances to the format expected by the selector
    categories = [
        ObjectWithDistance(object_id=f"test_{i}", distance=float(d), payload={})
        for i, d in enumerate(distances)
    ]
    
    # Get binary selections
    selected = selector.select(categories)
    selected_mask = torch.zeros(points, dtype=torch.bool)
    for idx in selected:
        selected_mask[idx] = True
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.scatter(distances[selected_mask], torch.ones_like(distances[selected_mask]), 
                label='Selected', color='green')
    plt.scatter(distances[~selected_mask], torch.ones_like(distances[~selected_mask]), 
                label='Not Selected', color='red')
    plt.axvline(x=selector._margin, linestyle='--', color='blue', 
                label=f'Margin = {selector._margin}')
    plt.xlabel('Distance')
    plt.yticks([])
    plt.legend()
    plt.title('Selector Decision Boundary')
    plt.show()
```

### Tracing Selector Logic

Add logging statements to your selector to trace its decision process:

```python
import logging
logger = logging.getLogger(__name__)

class TracedSelector(DistBasedSelector):
    def select(self, categories, query_vector=None):
        logger.debug(f"Selecting from {len(categories)} categories")
        
        values = self._convert_values(categories)
        logger.debug(f"Converted values: {values}")
        
        threshold = 1 - self._margin if self._is_similarity else self._margin
        logger.debug(f"Using threshold: {threshold}")
        
        corrected_values = values - threshold
        logger.debug(f"Corrected values: {corrected_values}")
        
        bin_labels = self._calculate_binary_labels(corrected_values)
        logger.debug(f"Binary labels: {bin_labels}")
        
        indices = torch.nonzero(bin_labels).T[0].tolist()
        logger.debug(f"Selected indices: {indices}")
        
        return indices
```

## Best Practices

When implementing your own selectors, follow these guidelines:

1. **Start Simple**: Begin with a simple selector and add complexity only as needed
2. **Test Thoroughly**: Test with a variety of input data and boundary cases
3. **Consider Performance**: Optimize computation-heavy operations for production use
4. **Document Behavior**: Clearly document how your selector works and its parameters
5. **Handle Edge Cases**: Properly handle empty inputs, single categories, etc.
6. **Use Appropriate Metrics**: Ensure your selector works correctly with the chosen distance metric

## Conclusion

Custom category selectors give you fine-grained control over how Embedding Studio matches queries to categories. By implementing your own selectors, you can:

- Adjust selection thresholds based on your specific use case
- Implement domain-specific logic for category matching
- Combine different selection strategies for optimal results
- Create context-aware selectors that adapt to different query types
