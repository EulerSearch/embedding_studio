# Documentation for `TrainTestSplitter`

## Functionality
The `TrainTestSplitter` class is designed to split fine-tuning clickstream data into training and test sets using a specified ratio. It ensures that related fine-tuning inputs (sharing similar result IDs) remain together during the splitting process. The class also has the option to apply augmentation to the inputs.

### Parameters
- `test_size_ratio`: A float in (0.0, 1.0) that denotes the proportion of data reserved for testing (default: 0.2).
- `shuffle`: A boolean indicating if the inputs should be shuffled before splitting (default: True).
- `random_state`: An optional integer to ensure reproducibility.
- `augmenter`: An optional augmenter which may be a `ClickstreamQueryAugmentationApplier` or a callable that applies augmentation to the inputs.
- `do_augment_test`: A boolean that decides if the test split should also be augmented (default: False).

## Methods

### `TrainTestSplitter._augment_clickstream`

#### Functionality
This method applies augmentation to clickstream inputs if an augmenter is provided. If no augmenter is set, it returns the original inputs unchanged.

#### Parameters
- `inputs`: List of `FineTuningInput` objects to be augmented.

#### Usage
**Purpose** - Enhance fine-tuning inputs by applying query augmentations. If an augmenter is set, it uses its `apply_augmentation` method to modify the inputs; otherwise, it returns the original list.

#### Example
Assume `splitter` is an instance of `TrainTestSplitter` with an augmenter:
```python
augmented_inputs = splitter._augment_clickstream(inputs)
```

### `TrainTestSplitter.shuffle`

#### Functionality
This property returns the shuffle setting used by the splitter. It indicates whether the fine-tuning inputs will be randomly rearranged before being split into train and test sets.

#### Return Value
- Boolean flag indicating if shuffling is enabled.

### `TrainTestSplitter.split`

#### Functionality
Splits a list of fine-tuning inputs into training and testing sets. It divides inputs based on result IDs to keep related inputs together. If overlapping result IDs occur, the majority determines the split.

#### Parameters
- `inputs`: List of `FineTuningInput` instances that represent the fine-tuning input data to be split.

#### Return
Returns a `DatasetDict` with two keys:
- `train`: A `PairedFineTuningInputsDataset` instance for training data, potentially augmented.
- `test`: A `PairedFineTuningInputsDataset` instance for testing data, potentially augmented.

#### Usage
Use this method to maintain consistency among inputs sharing result IDs. It ensures that related inputs are not split across training and testing sets. This approach is critical for tasks with overlapping result contexts.

#### Example
```python
splitter = TrainTestSplitter(test_size_ratio=0.2, random_state=42)
result = splitter.split(inputs)
```