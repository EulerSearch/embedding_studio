# Documentation for `create_deletion_task`

## Functionality
The `create_deletion_task` method is designed to create a deletion task using a request body of type `DeletionTaskRunRequest`. If a task with the provided `task_id` already exists, the method returns the existing task. If not, a new task is created and sent to the deletion worker for processing.

## Parameters
- **body**: The request body of type `DeletionTaskRunRequest`, which contains task details. It includes an optional `task_id` that can be used to identify existing tasks.

## Usage
- **Purpose**: This method is intended for initializing and processing deletion tasks within the system.

### Example
```python
from embedding_studio.api.api_v1.internal_endpoints.delete import create_deletion_task

# Build request body with required parameters
request_body = DeletionTaskRunRequest(task_id='123', ...)

# Create and run deletion task
response = create_deletion_task(request_body)
```