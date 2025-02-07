import dataclasses
from typing import Optional

import pymongo
from apscheduler.schedulers.background import BackgroundScheduler

from embedding_studio.core.config import settings
from embedding_studio.core.plugin import PluginManager
from embedding_studio.data_access.clickstream import ClickstreamDao
from embedding_studio.data_access.deletion_tasks import CRUDDeletion
from embedding_studio.data_access.fine_tuning import CRUDFineTuning
from embedding_studio.data_access.improvement_sessions import (
    CRUDSessionsForImprovement,
)
from embedding_studio.data_access.inference_deployment_tasks import (
    CRUDModelDeletionTasks,
    CRUDModelDeploymentTasks,
)
from embedding_studio.data_access.mongo.clickstream import MongoClickstreamDao
from embedding_studio.data_access.reindex_locks import CRUDReindexLocks
from embedding_studio.data_access.reindex_tasks import (
    CRUDReindexSubtasks,
    CRUDReindexTasks,
)
from embedding_studio.data_access.upsertion_tasks import CRUDUpsertion
from embedding_studio.db import mongo, postgres
from embedding_studio.experiments.mlflow_client_wrapper import (
    MLflowClientWrapper,
)
from embedding_studio.models.delete import DeletionTaskInDb
from embedding_studio.models.fine_tuning import FineTuningTaskInDb
from embedding_studio.models.improvement import SessionForImprovementInDb
from embedding_studio.models.inference_deployment_tasks import (
    ModelDeletionTaskInDb,
    ModelDeploymentTaskInDb,
)
from embedding_studio.models.reindex import ReindexSubtaskInDb, ReindexTaskInDb
from embedding_studio.models.reindex_lock import ReindexLockInDb
from embedding_studio.models.upsert import UpsertionTaskInDb
from embedding_studio.utils.model_download import ModelDownloader
from embedding_studio.vectordb.pgvector.vectordb import PgvectorDb
from embedding_studio.vectordb.vectordb import VectorDb


@dataclasses.dataclass
class AppContext:
    clickstream_dao: ClickstreamDao
    fine_tuning_task: CRUDFineTuning
    sessions_for_improvement: CRUDSessionsForImprovement
    deletion_task: CRUDDeletion
    upsertion_task: CRUDUpsertion
    reindex_task: CRUDReindexTasks
    reindex_subtask: CRUDReindexSubtasks
    reindex_locks: CRUDReindexLocks
    model_deployment_task: CRUDModelDeploymentTasks
    model_deletion_task: CRUDModelDeletionTasks
    vectordb: VectorDb
    categories_vectordb: VectorDb
    plugin_manager: PluginManager
    model_downloader: ModelDownloader
    mlflow_client: MLflowClientWrapper
    task_scheduler: Optional[BackgroundScheduler] = None


context = AppContext(
    clickstream_dao=MongoClickstreamDao(
        mongo_database=mongo.clickstream_mongo_database
    ),
    fine_tuning_task=CRUDFineTuning(
        collection=mongo.finetuning_mongo_database["fine_tuning"],
        model=FineTuningTaskInDb,
        indexes=[("idempotency_key", pymongo.ASCENDING)],
    ),
    sessions_for_improvement=CRUDSessionsForImprovement(
        collection=mongo.sessions_for_improvement_mongo_database[
            "sessions_for_improvement"
        ],
        model=SessionForImprovementInDb,
    ),
    deletion_task=CRUDDeletion(
        collection=mongo.upsertion_mongo_database["deletion"],
        model=DeletionTaskInDb,
    ),
    upsertion_task=CRUDUpsertion(
        collection=mongo.upsertion_mongo_database["upsertion"],
        model=UpsertionTaskInDb,
    ),
    reindex_task=CRUDReindexTasks(
        collection=mongo.upsertion_mongo_database["reindex"],
        model=ReindexTaskInDb,
    ),
    reindex_subtask=CRUDReindexSubtasks(
        collection=mongo.upsertion_mongo_database["reindex_subtasks"],
        model=ReindexSubtaskInDb,
    ),
    reindex_locks=CRUDReindexLocks(
        collection=mongo.upsertion_mongo_database["reindex_locks"],
        model=ReindexLockInDb,
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
        prefix="basic",
    ),
    categories_vectordb=PgvectorDb(
        pg_database=postgres.pg_database,
        embeddings_mongo_database=mongo.embeddings_mongo_database,
        prefix="categories",
    ),
    plugin_manager=PluginManager(),
    model_downloader=ModelDownloader(),
    mlflow_client=MLflowClientWrapper(
        tracking_uri=settings.MLFLOW_TRACKING_URI,
    ),
)
