from typing import Any, Dict, Optional

from embedding_studio.models.task import (
    BaseModelOperationTask,
    BaseTaskCreateSchema,
    BaseTaskInDb,
)


class ModelManagementTaskCreateSchema(BaseTaskCreateSchema):
    ...


class ModelDeploymentTask(BaseModelOperationTask):
    metadata: Optional[Dict[str, Any]] = None


class ModelDeploymentTaskInDb(ModelDeploymentTask, BaseTaskInDb):
    ...


class ModelDeletionTask(BaseModelOperationTask):
    metadata: Optional[Dict[str, Any]] = None


class ModelDeletionTaskInDb(ModelDeletionTask, BaseTaskInDb):
    ...
