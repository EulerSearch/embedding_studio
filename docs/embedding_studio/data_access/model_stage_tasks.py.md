# Documentation for `CRUDModelStageTasks`

## Functionality
CRUDModelStageTasks provides basic CRUD operations for model stage tasks in a MongoDB database. It extends the generic CRUDBase class to apply common domain logic and enforce filtering by embedding model ID. This specialization simplifies data handling and ensures consistent usage when dealing with tasks related to embedding models.

## Inheritance
CRUDModelStageTasks inherits from CRUDBase[SchemaInDbType, CreateSchemaType, UpdateSchemaType]. It leverages generic CRUD methods while adding functionality pertinent to model stage tasks.

## Parameters (Constructor)
- `collection`: A pymongo Collection object that stores the model stage tasks.
- `model`: A type representing the data model. It is used to validate and parse database records.

## Usage
Instantiate CRUDModelStageTasks with a specific MongoDB collection and data model. Call the `get_by_model_id` method to retrieve tasks associated with a particular embedding model ID.

### Method: `get_by_model_id`

#### Functionality
Retrieves a document from the task collection by its embedding model ID. Accepts an ID as a string or ObjectId and returns a validated model instance if found, or None otherwise.

#### Parameters
- `embedding_model_id`: The identifier for the embedding model. Can be a string or an ObjectId instance.

#### Example
```python
from pymongo import MongoClient
from embedding_studio.data_access.model_stage_tasks import CRUDModelStageTasks
from my_models import ModelStageTask

client = MongoClient()
collection = client.mydb.tasks
crud_tasks = CRUDModelStageTasks(collection, ModelStageTask)

task = crud_tasks.get_by_model_id("60ad0efd1c4ae2b")
if task is not None:
    print("Task found!")
else:
    print("Task not found.")
```

### Purpose
To fetch a task record associated with a specific embedding model.