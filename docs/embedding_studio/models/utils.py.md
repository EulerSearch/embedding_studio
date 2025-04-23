# Merged Documentation

## Documentation for `create_failed_deletion_data_item`

### Functionality
Creates a `FailedItemIdWithDetail` instance to represent a failed deletion of a data item. It takes the unique object identifier and a detail message, wraps them in a `FailedItemIdWithDetail` object, and returns it.

### Parameters
- `object_id` (str): Unique identifier of the object.
- `detail` (str): Message describing the failure or error.

### Usage
Invoke this function when a deletion operation fails. It helps encapsulate error information for further processing.

#### Example
```python
result = create_failed_deletion_data_item("id123", "Deletion failed due to missing record")
print(result)
```

---

## Documentation for `create_failed_data_item`

### Functionality
This function creates a failed data item using a `DataItem` instance and error details. It returns a `FailedDataItem` that contains key attributes like `object_id`, `payload`, and `item_info`. The detail is truncated to the last 1500 characters, and a failure stage is set.

### Parameters
- `item`: A `DataItem` object representing the original data.
- `detail`: A string containing error details for the failure.
- `failure_stage`: An enum value indicating the failure stage.

### Usage
- **Purpose:** Convert a valid data item into a failed data item for failure logging and further processing.

#### Example
Assuming you have a data item and a failure stage, the call would be:
```python
failed_item = create_failed_data_item(data_item, "Error message", failure_stage)
```