## Documentation for FeaturesExtractor Class

### Functionality
The FeaturesExtractor class extracts fine-tuning features from data inputs. It computes positive and negative example ranks, calculates event confidences, and aggregates clicks and ranks. This results in robust feature representations for training models.

### Motivation
This class leverages multiple signals like ranking, clicks, and event confidences to create a unified feature set that aids in fine-tuning and improves model performance.

### Inheritance
FeaturesExtractor inherits from `pytorch_lightning.LightningModule`, which integrates it with the PyTorch Lightning framework and offers a standardized training workflow.

---

## Documentation for FeaturesExtractor._confidences

### Functionality
Calculates confidences for a given fine-tuning input by aggregating ranks and clicks, and then computing a confidence score using the provided confidence_calculator. It separates the scores into positive and negative confidences based on group participation in events.

### Parameters
- `fine_tuning_input`: An instance of FineTuningInput containing the items, their ranks, and event information.
- `not_events`: A list of IDs representing negative examples used for rank prediction.

### Usage
- **Purpose**: To compute and return the positive and negative confidence scores for the provided fine-tuning input.

#### Example
```python
pos_conf, neg_conf = extractor._confidences(fine_tuning_input, not_events)
```

---

## Documentation for FeaturesExtractor._downsample_not_events

### Functionality
This method downsamples non-event items from the input data. It groups non-event identifiers by their respective group IDs using `fine_tuning_input.get_object_id` and then randomly selects a subset of groups based on the `negative_downsampling_factor`. All not-event items from the selected groups are returned.

### Parameters
- `fine_tuning_input` (FineTuningInput): Input data object that contains non-event identifiers in `not_events` and provides a method to get group IDs via `get_object_id`.

### Returns
- `List[str]`: A list of downsampled non-event identifiers collected from the selected groups.

### Usage
- **Purpose**: Balance non-event items by reducing their count while maintaining group structure.

#### Example
```python
downsampled = features_extractor._downsample_not_events(fine_tuning_input)
# downsampled holds a list of non-event item IDs from randomly selected groups
```

---

## Documentation for FeaturesExtractor._get_fine_tuning_features

### Functionality
This method computes features for a fine-tuning input. It processes positive and negative events while preserving group boundaries. It downsamples negative examples, computes ranking scores and confidences, and prepares tensors for training.

### Parameters
- `fine_tuning_input`: An object containing the query and event IDs.
- `dataset`: An ItemsSet with items corresponding to the events.

### Returns
- `FineTuningFeatures`: An object holding positive and negative ranks along with their confidence values.

### Usage
Use this method to generate feature tensors for training fine-tuned models. It aggregates vectors, computes similarity or distance measures, and assigns ranking scores with corresponding confidences.

#### Example
```python
features = extractor._get_fine_tuning_features(input, dataset)
```

---

## Documentation for FeaturesExtractor._get_paired_inputs_features

### Functionality
Computes fine-tuning features for a pair of inputs: a not-irrelevant and an irrelevant input. It calculates features for each input, merges them, and applies adjustments for fine-tuning.

### Parameters
- `not_irrelevant_input`: FineTuningInput with relevant events.
- `irrelevant_input`: FineTuningInput representing an irrelevant input.
- `dataset`: ItemsSet containing items related to the inputs.

### Usage
- **Purpose**: Merges features from two inputs for fine-tuning tasks.

#### Example
Assume `input1` and `input2` are FineTuningInput objects and `dataset` is an ItemsSet:
```python
features = extractor._get_paired_inputs_features(input1, input2, dataset)
```

---

## Documentation for FeaturesExtractor.forward

### Functionality
Calculates fine-tuning features for a batch of fine-tuning input pairs. It iterates over not-irrelevant and irrelevant inputs and aggregates the features based on events and confidences. It applies filters and returns a FineTuningFeatures object.

### Parameters
- `batch`: A list of tuples, each containing two FineTuningInput objects.
- `dataset`: ItemsSet of items corresponding to clickstream inputs.

### Usage
- **Purpose**: Computes features for fine-tuning models by processing input pairs and aggregating results using auxiliary methods.

#### Example
```python
features = extractor.forward(batch, dataset)
```