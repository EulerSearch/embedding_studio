from embedding_studio.data_access.model_stage_tasks import CRUDModelStageTasks
from embedding_studio.models.inference_deployment_tasks import (
    ModelDeletionTask,
    ModelDeletionTaskInDb,
    ModelDeploymentTask,
    ModelDeploymentTaskInDb,
    ModelManagementTaskCreateSchema,
)


class CRUDModelDeploymentTasks(
    CRUDModelStageTasks[
        ModelDeploymentTaskInDb,
        ModelManagementTaskCreateSchema,
        ModelDeploymentTask,
    ]
):
    ...


class CRUDModelDeletionTasks(
    CRUDModelStageTasks[
        ModelDeletionTaskInDb,
        ModelManagementTaskCreateSchema,
        ModelDeletionTask,
    ]
):
    ...
