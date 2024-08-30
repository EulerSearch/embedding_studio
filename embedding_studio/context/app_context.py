import dataclasses

import pymongo

from embedding_studio.core.plugin import PluginManager
from embedding_studio.data_access.clickstream import ClickstreamDao
from embedding_studio.data_access.deletion_tasks import CRUDDeletion
from embedding_studio.data_access.fine_tuning import CRUDFineTuning
from embedding_studio.data_access.inference_deployment_tasks import (
    CRUDModelDeletionTasks,
    CRUDModelDeploymentTasks,
)
from embedding_studio.data_access.mongo.clickstream import MongoClickstreamDao
from embedding_studio.data_access.upsertion_tasks import CRUDUpsertion
from embedding_studio.db import mongo, postgres
from embedding_studio.models.delete import DeletionTaskInDb
from embedding_studio.models.fine_tuning import FineTuningTaskInDb
from embedding_studio.models.inference_deployment_tasks import (
    ModelDeletionTaskInDb,
    ModelDeploymentTaskInDb,
)
from embedding_studio.models.upsert import UpsertionTaskInDb
from embedding_studio.utils.model_download import ModelDownloader
from embedding_studio.vectordb.pgvector.vectordb import PgvectorDb
from embedding_studio.vectordb.vectordb import VectorDb


@dataclasses.dataclass
class AppContext:
    clickstream_dao: ClickstreamDao
    fine_tuning_task: CRUDFineTuning
    deletion_task: CRUDDeletion
    upsertion_task: CRUDUpsertion
    model_deployment_task: CRUDModelDeploymentTasks
    model_deletion_task: CRUDModelDeletionTasks
    vectordb: VectorDb
    plugin_manager: PluginManager
    model_downloader: ModelDownloader


context = AppContext(
    clickstream_dao=MongoClickstreamDao(
        mongo_database=mongo.clickstream_mongo_database
    ),
    fine_tuning_task=CRUDFineTuning(
        collection=mongo.finetuning_mongo_database["fine_tuning"],
        model=FineTuningTaskInDb,
        indexes=[("idempotency_key", pymongo.ASCENDING)],
    ),
    deletion_task=CRUDDeletion(
        collection=mongo.upsertion_mongo_database["upsertion"],
        model=DeletionTaskInDb,
    ),
    upsertion_task=CRUDUpsertion(
        collection=mongo.upsertion_mongo_database["deletion"],
        model=UpsertionTaskInDb,
    ),
    model_deployment_task=CRUDModelDeploymentTasks(
        collection=mongo.inference_deployment_mongo_database[
            "model_deployment"
        ],
        model=ModelDeploymentTaskInDb,
    ),
    model_deletion_task=CRUDModelDeletionTasks(
        collection=mongo.inference_deployment_mongo_database["model_deletion"],
        model=ModelDeletionTaskInDb,
    ),
    vectordb=PgvectorDb(
        pg_database=postgres.pg_database,
        embeddings_mongo_database=mongo.embeddings_mongo_database,
    ),
    plugin_manager=PluginManager(),
    model_downloader=ModelDownloader(),
)
