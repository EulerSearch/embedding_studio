# Documentation for FineTuningInput

### Functionality
FinetuningInput is a data model that represents a clickstream session. It validates the consistency between results and ranks and provides methods for extracting non-clicked results, mapping part IDs to object IDs, and removing specified results. It is used as input for feature extraction in fine-tuning tasks.

### Parameters
- `query`: The user query or search term initiating the session.
- `events`: List of item IDs that received user interactions.
- `results`: List of result item IDs shown to the user.
- `ranks`: Dictionary mapping each result ID to its rank.
- `event_types`: Optional list of event type indicators.
- `timestamp`: Optional session initialization timestamp.
- `is_irrelevant`: Optional boolean indicating session relevance.
- `part_to_object_dict`: Optional mapping from part IDs to object IDs.

### Inheritance
FineTuningInput inherits from pydantic.BaseModel, which automatically provides data validation and parsing.

### Motivation
This class standardizes clickstream data for feature extraction. It enforces data consistency and simplifies the further processing of results, which is crucial for effective fine-tuning.

### Usage
- **Purpose**: Structure input data for fine-tuning in machine learning workflows that use clickstream and search result data.

#### Example
```python
data = {
    "query": "example query",
    "events": ["item1", "item2"],
    "results": ["item1", "item2", "item3"],
    "ranks": {"item1": 1.0, "item2": 2.0, "item3": 3.0}
}
ft_input = FineTuningInput(**data)
print(len(ft_input))
```

---

## Documentation for `FineTuningInput.not_events`

### Functionality
Returns a list of result IDs that did not receive any user interaction. It iterates through the results list and includes any result not found in the events attribute, providing a list of non-event item IDs.

### Parameters
This property does not take any parameters.

### Usage
- **Purpose**: Identify results that did not record any user events.

#### Example
```python
non_events = fine_tuning_input.not_events
```

---

## Documentation for `FineTuningInput.get_object_id`

### Functionality
Maps a part ID to its parent object ID. If a mapping is defined in part_to_object_dict, the corresponding object ID is returned. Otherwise, the original ID is returned.

### Parameters
- `id`: (str) The part ID to look up.

### Usage
Returns the object ID for a given part ID. If a mapping exists, the parent object is used; otherwise, the part ID is returned.

#### Example
Given part_to_object_dict = {'part1': 'obj1'}:
```python
input_obj.get_object_id('part1')  # returns 'obj1'
input_obj.get_object_id('test')   # returns 'test'
```

---

## Documentation for `FineTuningInput.remove_results`

### Functionality
Removes specified result IDs from the input and updates related structures such as results, events, ranks, and part-to-object mapping. 

### Parameters
- `ids`: A list or set of IDs to remove from the results.

### Usage
- **Purpose**: Remove unwanted result IDs and update all data structures within a FineTuningInput instance.

#### Example
```python
# Create a sample input
input_data = FineTuningInput(
    query="search term",
    events=["id1", "id2"],
    results=["id1", "id2", "id3"],
    ranks={"id1": 1.0, "id2": 2.0, "id3": 3.0}
)

# Remove a result
input_data.remove_results(["id2"])
```

---

## Documentation for `FineTuningInput.preprocess_ids`

### Functionality
This method converts a list of input IDs into strings. If an item is a tuple, its first element is converted to string. If it is a PyTorch Tensor, its value is extracted and converted to string. For all other types, the item is converted directly to string.

### Parameters
- `cls`: The class being validated (provided automatically by Pydantic).
- `value`: A list of items to be processed into string IDs.

### Usage
Use this method as a Pydantic validator for the "results" field to ensure that all IDs are properly formatted as strings.

#### Example
For an input list like: `[(1, "data"), tensor(3), 4]`, this method returns: `["1", "3", "4"]`.