from typing import Any, Dict, Optional

from embedding_studio.models.task import (
    BaseModelOperationTask,
    BaseTaskCreateSchema,
    BaseTaskInDb,
)


class ModelManagementTaskCreateSchema(BaseTaskCreateSchema):
    """
    A blueprint for creating tasks that manage embedding models in the system.
    This is the base schema for operations like deploying models to the inference
    server or removing them when they're no longer needed.
    """

    ...


class ModelDeploymentTask(BaseModelOperationTask):
    """
    Represents a task that handles deploying an embedding model to make it
    available for inference. It tracks the process of taking a trained model
    and setting it up so it can be used for generating embeddings in real-time.
    """

    metadata: Optional[Dict[str, Any]] = None


class ModelDeploymentTaskInDb(ModelDeploymentTask, BaseTaskInDb):
    """
    The database-friendly version of a model deployment task. It stores
    information about deployment operations, including their current status
    and any additional metadata needed for the deployment process.
    """

    ...


class ModelDeletionTask(BaseModelOperationTask):
    """
    Represents a task that handles removing an embedding model from the inference
    system. It tracks the process of cleanly removing models that are no longer
    needed, freeing up resources for other models.
    """

    metadata: Optional[Dict[str, Any]] = None


class ModelDeletionTaskInDb(ModelDeletionTask, BaseTaskInDb):
    """
    The database-friendly version of a model deletion task. It stores information
    about model removal operations, including their current status and any
    metadata needed for the deletion process.
    """

    ...
