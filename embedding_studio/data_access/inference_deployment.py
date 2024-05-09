from embedding_studio.api.api_v1.schemas.inference_deployment import (
    DeploymentInfo,
)
from embedding_studio.data_access.mongo.crud_base import CRUDBase
from embedding_studio.models.inference_deployment import (
    DeploymentTask,
    DeploymentTaskInDb,
)


class CRUDDeployment(
    CRUDBase[DeploymentTaskInDb, DeploymentInfo, DeploymentTask]
):
    ...
