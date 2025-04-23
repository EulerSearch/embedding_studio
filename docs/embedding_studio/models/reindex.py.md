## Documentation for `ReindexTask`

### Functionality
ReindexTask manages the process of transferring data between embedding models. It coordinates subtasks, tracks progress, and finalizes the deployment of the new model. This task is motivated by the need to efficiently reindex data across models while monitoring individual batches of work.

### Inheritance
ReindexTask inherits from BaseTaskInfo, integrating common task properties and methods for tracking task status in the system.

### Parameters
- `source`: ModelParams object specifying the source model.
- `dest`: ModelParams object specifying the destination model.
- `progress`: Float value indicating the task progress (0.0 to 1.0).
- `count`: Integer count of processed items.
- `total`: Integer total number of items to process.
- `children`: List of subtask identifiers (PyObjectId).
- `failed_items`: List of FailedDataItem instances for error logging.
- `deploy_as_blue`: Optional boolean flag for a special deployment strategy.
- `wait_on_conflict`: Optional boolean flag to manage conflict resolution.

### Usage
- **Purpose**: Organize and manage the complete reindexing process by dividing the work into manageable subtasks. This ensures robust data handling and efficient switching between embedding models.

#### Example
Below is a simple example on how to instantiate a ReindexTask:

```python
task = ReindexTask(
    source=source_params,
    dest=dest_params,
    progress=0.0,
    count=0,
    total=100,
    children=[],
    failed_items=[],
    deploy_as_blue=True,
    wait_on_conflict=False
)
```

This setup provides a clear starting point for performing a reindexing job.

---

## Documentation for `ReindexTask.add_count`

### Functionality
This method updates the processed count by adding a given number of items and recalculates progress as a fraction of the total items.

### Parameters

- `additional_count`: An integer representing the number of items to add to the current count before computing progress.

### Usage

- **Purpose** - Increment the count of processed items and update the progress. Ensure that the total number of items is defined to avoid division errors.

#### Example
For a task with a current count of 10 and a total of 100, calling `add_count(5)` will update the count to 15 and progress to 0.15.