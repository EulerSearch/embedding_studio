## Documentation for `FineTuningFeatures`

### Functionality

The `FineTuningFeatures` class is responsible for holding and validating the extracted features for fine-tuning embedding-based models. It ensures that the provided inputs are correctly typed and have matching lengths, preparing data for subsequent processing.

### Parameters

- `positive_ranks`: Tensor containing ranks of positive results.
- `negative_ranks`: Tensor containing ranks of negative results.
- `target`: Tensor that indicates if ranks represent similarities (when target = 1) or distances (when target = -1).
- `positive_confidences`: Tensor representing confidences for positive outcomes (e.g. clicks).
- `negative_confidences`: Tensor representing confidences for negative outcomes.

### Usage

- **Purpose**: Encapsulate fine-tuning input features with built-in validation checks to reduce runtime errors during model training.
- **Motivation**: Provide a structured approach to manage and validate inputs for fine-tuning models.

#### Example

```python
from embedding_studio.embeddings.features.fine_tuning_features import FineTuningFeatures

# Example of creating a FineTuningFeatures object
features = FineTuningFeatures(
    positive_ranks=some_positive_tensor,
    negative_ranks=some_negative_tensor,
    target=some_target_tensor,
    positive_confidences=some_positive_confidences,
    negative_confidences=some_negative_confidences
)
```

### Methods

#### `FineTuningFeatures._check_types`

##### Functionality
This method checks if the attributes of an instance are either None or a valid torch.Tensor. It verifies that `positive_ranks`, `negative_ranks`, `target`, `positive_confidences`, and `negative_confidences` have the expected type. If any attribute is not a torch.Tensor (and not None), it raises a TypeError with a descriptive message.

##### Parameters
None. This method operates on the instance attributes. It does not accept arguments.

##### Usage
Automatically invoked during initialization or via property setters. It ensures features have correct types before further processing.

###### Example

```python
instance = FineTuningFeatures(
    positive_ranks=torch.tensor([0.1, 0.2]),
    negative_ranks=torch.tensor([0.3, 0.4]),
    target=torch.tensor([1, 1])
)

# If an incorrect type is assigned:
instance.positive_ranks = [0.1, 0.2]
# This will raise a TypeError notifying the type mismatch.
```

#### `FineTuningFeatures._check_lengths`

##### Functionality
Verifies that all non-None parameters provided to the constructor have the same length. This check ensures that fine-tuning inputs are consistent across all provided tensors.

##### Parameters
This method does not take any additional parameters. The tensors are passed during object initialization.

##### Usage
- **Purpose**: Validate that input tensors for fine-tuning have uniform lengths to prevent dimension mismatches.

###### Example

Assuming you have initialized a `FineTuningFeatures` object:

```python
features = FineTuningFeatures(
    positive_ranks=torch.tensor([1, 2, 3]),
    negative_ranks=torch.tensor([4, 5, 6]),
    target=torch.tensor([1, 1, 1])
)
```

The `_check_lengths` method is automatically called to verify that all provided tensors have the same length.

#### `FineTuningFeatures.positive_ranks`

##### Functionality
This property returns the positive_ranks value. It is either a torch.Tensor or None that represents the ranks assigned to positive results in fine-tuning.

##### Parameters
None: This property does not take any parameters. It returns the value provided during initialization.

##### Usage
- **Purpose**: Retrieve the ranks for positive results in fine-tuning.

###### Example

```python
features = FineTuningFeatures(positive_ranks=tensor)
print(features.positive_ranks)
```

#### `FineTuningFeatures.negative_ranks`

##### Functionality
This property method is part of the FineTuningFeatures class. When accessed, it returns the negative ranking tensor used in fine-tuning. As a setter, it updates the internal `_negative_ranks` attribute and calls `_check_types()` to ensure the value's validity.

##### Parameters
- `value`: An optional FloatTensor representing the negative ranking values. Setting this property updates the feature and validates it.

##### Usage
- **Purpose**: Retrieve or update the negative ranking tensor in a fine-tuning feature set, ensuring consistency within the feature group.

###### Example

Assume `ft_features` is an instance of `FineTuningFeatures`:

```python
# Retrieve negative ranks
current_neg = ft_features.negative_ranks

# Update negative ranks
new_neg = some_float_tensor
ft_features.negative_ranks = new_neg
```

#### `FineTuningFeatures.target`

##### Functionality
This property method manages the target feature used in fine-tuning. The target tensor holds either similarity scores (when target is 1) or distance measures (when target is -1), as specified in the class constructor.

##### Parameters
For the getter, no parameters are required. For the setter, it accepts:
- `target` (Optional[Tensor]): A tensor representing the fine-tuning target. Use 1 for similarity scores and -1 for distance measures.

##### Usage
- **Purpose**: Retrieve or update the target feature used in fine-tuning tasks.

###### Example

Retrieving the target feature:

```python
current_target = fine_tuning_instance.target
```

Updating the target feature:

```python
fine_tuning_instance.target = new_target_tensor
```

#### `FineTuningFeatures.positive_confidences`

##### Functionality
This property returns the confidence scores for positive results used in fine-tuning. These scores may indicate the likelihood of positive outcomes, such as clicks or likes, and they are integral in adjusting model parameters.

##### Parameters
- Getter: No parameters are required.
- Setter: Accepts a torch.Tensor or None.

##### Usage
- **Purpose**: Retrieve or update positive confidence scores for fine-tuning tasks.

###### Example

```python
# Retrieve the confidence scores
pos_conf = instance.positive_confidences

# Update the confidence scores
instance.positive_confidences = new_values
```

#### `FineTuningFeatures.negative_confidences`

##### Functionality
This property returns the negative confidence scores used in fine-tuning features. These scores indicate the confidence for non-positive results. They are stored as an optional FloatTensor.

##### Parameters
- None.

##### Usage
- **Purpose**: Retrieve or update the confidence scores for negative results.

###### Example

Retrieve negative confidences:

```python
ft_features = FineTuningFeatures(...)
neg_conf = ft_features.negative_confidences
```

Update negative confidences:

```python
ft_features.negative_confidences = new_tensor
```

#### `FineTuningFeatures._accumulate`

##### Functionality
Accumulates two tensors by concatenating them if both are provided. If only the second tensor is available, it returns that tensor. If both are None, it returns None.

##### Parameters
- `self_var`: The initial tensor to be accumulated, or None if not provided.
- `other_var`: The tensor to add to `self_var`. Must be a torch.Tensor if provided.

##### Returns
- A concatenated tensor if both inputs are provided, or the non-None tensor if only one is provided. Returns None if both are None.

##### Usage
- **Purpose**: Merges feature data from different sources.

###### Example

```python
import torch

a = torch.tensor([1, 2])
b = torch.tensor([3, 4])
result = FineTuningFeatures._accumulate(a, b)
# result: tensor([1, 2, 3, 4])
```

#### `FineTuningFeatures.clamp_diff_in`

##### Functionality
Filters examples based on the absolute difference between positive and negative ranks. Only examples with |positive_ranks - negative_ranks| between the given `min` and `max` values are kept.

##### Parameters
- `min`: Minimal allowed difference between positive and negative ranks.
- `max`: Maximal allowed difference between positive and negative ranks.

##### Usage
- **Purpose**: Selects only those examples that have a rank difference within specific bounds to improve fine-tuning.

###### Example

Consider a `FineTuningFeatures` instance:

```python
f = FineTuningFeatures(
    positive_ranks=torch.tensor([0.9, 0.8, 0.2]),
    negative_ranks=torch.tensor([0.2, 0.8, 0.1]),
    target=torch.tensor([1, 1, -1]),
    positive_confidences=torch.tensor([1.0, 0.8, 0.3]),
    negative_confidences=torch.tensor([0.2, 0.8, 0.1])
)

f.clamp_diff_in(min=0.1, max=0.7)
```

After the call, only the examples where:

```
0.1 < |positive_ranks - negative_ranks| < 0.7
```

are retained.

#### `FineTuningFeatures.use_positive_from`

##### Functionality
Replaces the positive examples in an irrelevant input with positive evidences from a relevant input. This changes the loss from triple loss to contrastive loss.

##### Parameters
- `other`: A `FineTuningFeatures` instance with valid positive evidences. Its positive ranks are used to update the current object's positive pairs.

##### Usage
Call this method on an input with irrelevant positive examples, using a relevant input as the source. The method adjusts tensor shapes if needed.

###### Example

```python
input_irrelevant.use_positive_from(input_relevant)
# Afterwards, positive pairs are replaced and loss becomes contrastive.
```