import dataclasses

import pymongo

from embedding_studio.data_access.clickstream import ClickstreamDao
from embedding_studio.data_access.fine_tuning import CRUDFineTuning
from embedding_studio.data_access.inference_deployment_tasks import (
    CRUDModelDeletionTasks,
    CRUDModelDeploymentTasks,
)
from embedding_studio.data_access.mongo.clickstream import MongoClickstreamDao
from embedding_studio.db import mongo, postgres
from embedding_studio.models.fine_tuning import FineTuningTaskInDb
from embedding_studio.models.inference_deployment_tasks import (
    ModelDeletionTaskInDb,
    ModelDeploymentTaskInDb,
)
from embedding_studio.vectordb.pgvector.vectordb import PgvectorDb
from embedding_studio.vectordb.vectordb import VectorDb


@dataclasses.dataclass
class AppContext:
    clickstream_dao: ClickstreamDao
    fine_tuning_task: CRUDFineTuning
    deployment_task: CRUDModelDeploymentTasks
    deletion_task: CRUDModelDeletionTasks
    vectordb: VectorDb


context = AppContext(
    clickstream_dao=MongoClickstreamDao(
        mongo_database=mongo.clckstream_mongo_database
    ),
    fine_tuning_task=CRUDFineTuning(
        collection=mongo.finetuning_mongo_database["fine_tuning"],
        model=FineTuningTaskInDb,
        indexes=[("idempotency_key", pymongo.ASCENDING)],
    ),
    deployment_task=CRUDModelDeploymentTasks(
        collection=mongo.finetuning_mongo_database["deployment"],
        model=ModelDeploymentTaskInDb,
        indexes=[("idempotency_key", pymongo.ASCENDING)],
    ),
    deletion_task=CRUDModelDeletionTasks(
        collection=mongo.finetuning_mongo_database["deletion"],
        model=ModelDeletionTaskInDb,
        indexes=[("idempotency_key", pymongo.ASCENDING)],
    ),
    vectordb=PgvectorDb(
        pg_database=postgres.pg_database,
        embeddings_mongo_database=mongo.embeddings_mongo_database,
    ),
)
