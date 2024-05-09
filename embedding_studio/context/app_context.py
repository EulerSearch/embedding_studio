import dataclasses

import pymongo

from embedding_studio.data_access.clickstream import ClickstreamDao
from embedding_studio.data_access.fine_tuning import CRUDFineTuning
from embedding_studio.data_access.inference_deployment import CRUDDeployment
from embedding_studio.data_access.mongo.clickstream import MongoClickstreamDao
from embedding_studio.db import mongo
from embedding_studio.models.fine_tuning import FineTuningTaskInDb
from embedding_studio.models.inference_deployment import DeploymentTaskInDb


@dataclasses.dataclass
class AppContext:
    clickstream_dao: ClickstreamDao
    fine_tuning_task: CRUDFineTuning
    deployment_task: CRUDDeployment


context = AppContext(
    clickstream_dao=MongoClickstreamDao(
        mongo_database=mongo.clckstream_mongo_database
    ),
    fine_tuning_task=CRUDFineTuning(
        collection=mongo.finetuning_mongo_database["fine_tuning"],
        model=FineTuningTaskInDb,
        indexes=[("idempotency_key", pymongo.ASCENDING)],
    ),
    deployment_task=CRUDDeployment(
        collection=mongo.finetuning_mongo_database["deployment"],
        model=DeploymentTaskInDb,
        indexes=[("idempotency_key", pymongo.ASCENDING)],
    ),
)
