# Documentation for `handle_deletion`

## Functionality
The `handle_deletion` function is responsible for managing the deletion of a deployed embedding model from the Triton Inference Server. This function performs several key tasks, including validating the deletion request, checking for the model iteration, and ensuring that the model utilizes a supported plugin. Additionally, it prevents concurrent deletion attempts by implementing a file lock. The function also handles the cleanup process by deleting the query model directory and the items model directory, provided that the latter is not shared.

## Parameters
- **task_id**: This parameter represents the ID of the model deletion task that needs to be processed.

## Usage
- **Purpose**: The function is designed to safely remove a deployed embedding model while executing necessary validations, managing file locks, and performing the cleanup of model files.

### Example
To use the `handle_deletion` function, one would call it as follows:
```python
handle_deletion("task123")
```