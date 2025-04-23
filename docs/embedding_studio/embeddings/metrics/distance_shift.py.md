# Documentation for DistanceShift and Method _calc_dist_shift

## Class: DistanceShift

### Functionality
DistanceShift computes metrics that capture how the ranks of inputs change after model processing. It calculates average rank shifts for relevant and irrelevant inputs using feature extraction and ranking functions. This metric helps quantify changes in ranking, aiding in fine-tuning the model for better accuracy and relevance. 

### Motivation
In many embedding models, the ranking of items may change after processing. Understanding these shifts is critical for evaluating the performance of ranking algorithms and enhancing the model's optimization.

### Inheritance
DistanceShift is a subclass of MetricCalculator, extending base functionality to perform custom calculations for rank shifts.

### Usage
- **Purpose**: Measure the change in rank of items after model processing, providing insights into the efficacy of ranking algorithms in fine-tuning tasks.

### Example
A simple usage example for the class:

```python
dist_shift = DistanceShift()
metric_values = dist_shift(batch, extractor, items_set, query_retriever)
```

## Method: DistanceShift._calc_dist_shift

### Functionality
Calculates the shift in rank for inputs after model processing. It computes the average change in rank based on whether the input is relevant (positive) or irrelevant (negative). The computation is adjusted based on whether the metric is similarity (higher ranks are better) or distance (lower ranks are better).

### Parameters
- `fine_tuning_input`: Contains the query, events, and previous rank information for comparison.
- `extractor`: Provides model inference, ranking, and indicates if the metric is similarity or distance.
- `items_set`: Supplies the dataset of item embeddings used in ranking.
- `query_retriever`: Retrieves the query embedding from the input.

### Example
A straightforward usage example for the method:

```python
shift = distance_shift._calc_dist_shift(
    fine_tuning_input, extractor, items_set, query_retriever
)
```