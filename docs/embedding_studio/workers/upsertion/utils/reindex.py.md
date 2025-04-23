# Documentation for Reindexing Methods

## `process_reindex`

### Functionality

Processes a reindex task by dividing the work into batches and scheduling sub-tasks for reindexing data within a given collection. It monitors sub-task progress, handles retries, and updates failure counts to set the final task status.

### Parameters

- `task`: A `ReindexTaskInDb` object containing task details and failure tracking information.
- `collection`: A `Collection` instance that holds data items to be reindexed.
- `reindex_subworker`: A dramatiq Actor used for scheduling sub-task processing.

### Usage

Splits the total number of items into batches based on a configured batch size. Sub-tasks are created for each batch, and their statuses are monitored. If the failure count exceeds a defined limit, the main task is marked as failed.

#### Example

```python
process_reindex(task, collection, reindex_subworker)
```

---

## `update_processing_tasks`

### Functionality

This function checks the status of active reindex subtasks and updates the main reindex task accordingly. It examines each subtask via an external context and processes them based on their status. If a subtask has finished, its results are consolidated into the main task, including failed items and processed counts.

### Parameters

- `task`: A `ReindexTaskInDb` object representing the main reindex task.
- `processing_task_ids`: A list of strings (List[str]) holding IDs of the current reindex subtasks.

### Returns

- A list of strings (List[str]) of subtask IDs that remain pending or processing.

### Usage

This helper function is part of the reindex logic and is called repeatedly to monitor subtask progress and update the main task status accordingly.

#### Example

```python
update_processing_tasks(task, current_processing_ids)
```

---

## `create_additional_tasks`

### Functionality

This function creates additional upsertion tasks when needed. It uses a list of offsets to generate new subtasks by calling the `create_subtask` function. Each new task is sent via the `reindex_subworker` actor using the `create_and_send_task` call. If the subtask creation fails, it calls the `handle_failed_subtask` function.

### Parameters

- `task`: A `ReindexTaskInDb` instance representing the current reindex task and its properties.
- `offsets`: A list of integers representing the batch offsets.
- `limit`: An integer indicating the maximum number of items per subtask.
- `additional_tasks_count`: An integer specifying how many new tasks to create from the offsets list.
- `processing_task_ids`: A list of strings where each element is the ID of a processing subtask.
- `reindex_subworker`: An actor responsible for dispatching the new subtasks.

### Usage

This function is used to dynamically generate additional subtasks when an initial reindex task requires more granular processing. It manages task creation, handles failures, and returns unused offsets for potential later use.

#### Example

Suppose you have a reindex task with offsets `[0, 10, 20, 30]`, a limit of batch size per task, and you need to create 2 additional tasks. The function call:

```python
unused_offsets = create_additional_tasks(
    task, offsets, limit, 2, processing_task_ids, reindex_subworker
)
```

will create tasks for offsets `0` and `10`, update the task's child list, and return `[20, 30]` as unused offsets.

---

## `create_subtask`

### Functionality

Creates a new upsertion task from a main task. This function prepares a subtask for processing a batch of items based on provided limit and offset.

### Parameters

- `task`: A `ReindexTaskInDb` object with task details (source, dest, id).
- `limit`: An integer for the number of items in the subtask batch.
- `offset`: Optional integer specifying the starting index for the batch.

### Returns

- A `ReindexSubtaskInDb` object representing the created subtask.

### Usage

**Purpose**: Divide a large reindex task into smaller, manageable subtasks.

#### Example

```python
task = get_reindex_task()  # Retrieve main task
subtask = create_subtask(task, 100, 0)
process_subtask(subtask)
```

---

## `handle_failed_subtask`

### Functionality

Handles items from a failed upsertion subtask by validating each failed item and appending it to the main task's failed items list.

### Parameters

- `task`: The main reindex task (`ReindexTaskInDb`) holding the overall reindex status.
- `failed_task`: The failed subtask (`ReindexSubtaskInDb`) containing the items that could not be processed.

### Usage

**Purpose**: To process items from a failed subtask, mark them with a preset detail message, and add them to the main task for later handling.

#### Example

```python
handle_failed_subtask(task, failed_task)
```

---

## `handle_reindex_error`

### Functionality

This function handles errors occurring during the reindex process. It logs the error, marks the task as failed, and updates the task's detail with the last 1500 characters of the traceback.

### Parameters

- `task`: `ReindexTaskInDb`  
  The reindex task object which will be marked as failed when an error occurs.

- `error`: Exception  
  The exception that triggered the error handling. Its details are used for logging and updating the task's detail.

### Usage

**Purpose**: To capture and handle unexpected exceptions during reindexing by logging errors and updating task status.

#### Example

```python
try:
    reindex_items(task)
except Exception as e:
    handle_reindex_error(task, e)
```