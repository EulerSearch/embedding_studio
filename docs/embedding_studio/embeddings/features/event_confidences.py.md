# Documentation for Confidence Score Functions

## Method: `dummy_confidences`

This function returns a tensor of ones, which serves as a baseline confidence score for events. It uses the provided ranks and event flags to output a uniform confidence value.

### Parameters
- `ranks`: A FloatTensor containing search result ranks.
- `events`: A Tensor indicating event presence; use 1 for event and 0 for non-event.

### Usage
- **Purpose:** To generate a constant confidence score for each event, useful for preliminary testing or baseline comparisons.

#### Example
```python
import torch

ranks = torch.tensor([1.0, 2.0, 3.0])
events = torch.tensor([1, 0, 1])

# Returns tensor([1., 1., 1.])
confidence = dummy_confidences(ranks, events)
```

---

## Method: `calculate_confidences`

This function computes confidence scores for events (clicks) and non-events (non-clicks) based on ranking similarity and local context. It uses a sliding window to derive an average rank and click proportion, then combines these to produce a normalized score.

### Parameters
- `ranks`: FloatTensor, list of ranks from search results.
- `results`: Tensor, list of 0 (non-event) or 1 (event).
- `window_size`: int, size of the context window (must be > 1, default: 3).

### Usage
- **Purpose**: Generate confidence scores from search results by analyzing local ranking and click patterns.

#### Example
```python
import torch

ranks = torch.tensor([1, 2, 3])
results = torch.tensor([0, 1, 0])
conf_scores = calculate_confidences(ranks, results, window_size=3)
print(conf_scores)
```