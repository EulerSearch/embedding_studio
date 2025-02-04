import gc
import logging
import os
import tempfile

import torch.cuda
from dramatiq import Middleware

from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.utils.plugin_utils import get_vectordb
from embedding_studio.workers.inference.utils.file_locks import (
    acquire_lock,
    release_lock,
)
from embedding_studio.workers.inference.utils.prepare_for_triton import (
    convert_for_triton,
)

logger = logging.getLogger(__name__)


def init_model_repo_for_plugin(model_repo: str, plugin_name: str):
    """
    Initialize the model repository for a specific plugin.
    Uploads initial models or converts best models for Triton if necessary.
    """
    plugin = context.plugin_manager.get_plugin(plugin_name)
    experiments_manager = plugin.get_manager()
    if not experiments_manager.has_initial_model():
        logger.info("No initial model found, uploading")
        plugin.upload_initial_model()

    run_id = experiments_manager.get_initial_model_run_id()
    logger.info(f"Initial model run_id: {run_id}")

    try:
        vector_db = get_vectordb(plugin)
        embedding_model_info = plugin.get_embedding_model_info(run_id)
        search_index_info = plugin.get_search_index_info()
        logger.info(f"Creating or retrieving Vector DB collection")
        if not vector_db.collection_exists(embedding_model_info):
            logger.warning(
                f"Collection with name: {embedding_model_info.full_name} "
                f"does not exist, creating"
            )
            vector_db.create_collection(
                embedding_model_info, search_index_info
            )
        else:
            logger.info(f"Initial collection exists")
            vector_db.get_or_create_collection(
                embedding_model_info, search_index_info
            )

        logger.info(f"Creating or retrieving Vector DB query collection")
        if not vector_db.query_collection_exists(embedding_model_info):
            logger.warning(
                f"Query collection with name: {embedding_model_info.full_name} "
                f"does not exist, creating"
            )
            vector_db.create_query_collection(
                embedding_model_info, search_index_info
            )
        else:
            logger.info(f"Initial collection exists")
            vector_db.get_or_create_query_collection(
                embedding_model_info, search_index_info
            )

        vector_db.set_blue_collection(embedding_model_info)
        logger.info(f"Initial collection setup finished")

    except Exception:
        logger.exception(f"Something went wrong during collection setup")

        return

    model = experiments_manager.download_initial_model()
    convert_for_triton(model, plugin_name, model_repo, 1, run_id)
    del model
    gc.collect()
    torch.cuda.empty_cache()


class OnStartMiddleware(Middleware):
    """
    Middleware to handle worker start events.
    Ensures all plugins have their models initialized in the model repository.
    """

    def after_worker_boot(self, broker, worker):
        super().after_worker_boot(broker, worker)
        # Specify the worker name that should execute the initialization

        context.plugin_manager.discover_plugins(settings.ES_PLUGINS_PATH)

        model_repo = os.getenv("MODEL_REPOSITORY")
        if not model_repo:
            return

        model_repo = settings.INFERENCE_MODEL_REPO
        for plugin_name in settings.INFERENCE_USED_PLUGINS:
            temp_dir = tempfile.gettempdir()
            lock_file_path = os.path.join(
                temp_dir, f"deployment_lock_{plugin_name}.lock"
            )
            lock_file = acquire_lock(lock_file_path)
            try:
                init_model_repo_for_plugin(model_repo, plugin_name)
            finally:
                release_lock(lock_file)

        if not os.path.exists(model_repo):
            os.makedirs(model_repo)

        with open(
            os.path.join(model_repo, "initialization_complete.flag"), "w"
        ) as f:
            f.write("TRUE")
