# Unified Documentation for Worker Methods

This document consolidates the functionalities, parameters, and usage of four worker methods: `deletion_worker`, `upsertion_worker`, `reindex_subworker`, and `reindex_worker`.

## 1. `deletion_worker`

### Functionality
The `deletion_worker` processes a deletion task for an embedding model. It retrieves a deletion task using the provided `task_id` and checks for an active reindex lock. If a lock is found and the configuration permits, the task is passed to a reindexing model utilizing the `create_and_send_task` function. In case of errors, appropriate exceptions are raised.

### Parameters
- **task_id (str)**: The unique identifier for a deletion task.

### Usage
- **Purpose**: To manage and process deletion tasks in a distributed system. It may delegate the deletion process to a reindexing model if necessary.

#### Example
```python
deletion_worker("task_id_123")
```

---

## 2. `upsertion_worker`

### Functionality
The `upsertion_worker` is a Dramatiq actor that handles tasks for upserting embedding items. It retrieves a task using its `task_id` and checks for a reindex lock. If a lock is present and the settings allow forwarding, the task is sent to the reindexing workflow. Otherwise, the worker processes the upsert using `handle_upsert`.

### Parameters
- **task_id (str)**: A string identifier for the upsertion task.

### Usage
- **Purpose**: Processes upsertion tasks as part of the application workflow.
- Checks if the task exists and whether it should be forwarded.
- Forwards the task if a reindex lock exists and settings permit it; otherwise, it processes the task normally via `handle_upsert`.

#### Example
```python
upsertion_worker("task-1234")
```

---

## 3. `reindex_subworker`

### Functionality
The `reindex_subworker` retrieves a reindex subtask using a provided task ID, processes it via a subtask handler, and performs garbage collection. This worker operates as a Dramatiq actor and ensures that the task is executed if it exists.

### Parameters
- **task_id (str)**: Identifier for the reindex subtask to be processed.

### Usage
- **Purpose**: To handle reindex subtasks for updating indexing procedures in the system.

#### Example
```python
reindex_subworker("your_task_id")
```

---

## 4. `reindex_worker`

### Functionality
The `reindex_worker` performs reindexing tasks by checking if the source and destination embedding models are locked. It reschedules the task if a conflict is detected and waiting is allowed; otherwise, it marks the task as refused and raises a `ReindexException`.

### Parameters
- **task_id (str)**: Unique identifier for the reindex task. Used to fetch the task and associated embedding models.

### Usage
- **Purpose**: Execute a reindexing process with built-in conflict resolution and rescheduling logic.

#### Example
```python
reindex_worker("some_task_id")
```