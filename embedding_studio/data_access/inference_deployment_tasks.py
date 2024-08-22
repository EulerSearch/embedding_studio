from typing import Optional, Type, Union

from bson import ObjectId
from pymongo.collection import Collection

from embedding_studio.data_access.mongo.crud_base import (
    CreateSchemaType,
    CRUDBase,
    SchemaInDbType,
    UpdateSchemaType,
)
from embedding_studio.models.inference_deployment_tasks import (
    ModelDeletionTask,
    ModelDeletionTaskInDb,
    ModelDeploymentTask,
    ModelDeploymentTaskInDb,
    ModelManagementTaskCreateSchema,
)


class CRUDModelStageTasks(
    CRUDBase[SchemaInDbType, CreateSchemaType, UpdateSchemaType]
):
    _EMBEDDING_MODEL_ID: str = "embedding_model_id"

    def __init__(
        self,
        collection: Collection,
        model: Type[SchemaInDbType],
    ):
        super(CRUDModelStageTasks, self).__init__(
            collection,
            model,
            [
                CRUDModelStageTasks._EMBEDDING_MODEL_ID,
            ],
        )

    def get_by_model_id(
        self, embedding_model_id: Union[str, ObjectId]
    ) -> Optional[SchemaInDbType]:
        """Get an object by embedding model ID.

        :param embedding_model_id: ID as string or ObjectId.
        :return: Retrieved object or None if not found.
        """

        obj = self.collection.find_one(
            {self._EMBEDDING_MODEL_ID: embedding_model_id}
        )
        if not obj:
            return None

        return self.model.model_validate(obj)


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
