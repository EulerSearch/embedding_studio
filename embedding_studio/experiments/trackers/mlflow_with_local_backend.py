import logging
from typing import List, Optional

import mlflow
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository

from embedding_studio.experiments.experiments_tracker import ExperimentsManager
from embedding_studio.experiments.metrics_accumulator import MetricsAccumulator
from embedding_studio.utils.retry import retry_method
from embedding_studio.workers.fine_tuning.utils.config import RetryConfig

logger = logging.getLogger(__name__)


class ExperimentsManagerWithLocalFileSystem(ExperimentsManager):
    def __init__(
        self,
        tracking_uri: str,
        main_metric: str,
        plugin_name: str,
        accumulators: List[MetricsAccumulator],
        is_loss: bool = False,
        n_top_runs: int = 10,
        requirements: Optional[str] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        """Wrapper over mlflow package to manage certain fine-tuning experiments with local path backend.

        :param tracking_uri: url of MLFlow server
        :param main_metric: name of main metric that will be used to find best model
        :param plugin_name: name of fine-tuning method being used
        :param accumulators: accumulators of metrics to be logged
        :param is_loss: is main metric loss (if True, then best quality is minimal) (default: False)
        :param n_top_runs: how many hyper params group consider to be used in following tuning steps (default: 10)
        :param requirements: extra requirements to be passed to mlflow.pytorch.log_model (default: None)
        :param retry_config: retry policy (default: None)
        """
        super(ExperimentsManagerWithLocalFileSystem, self).__init__(
            tracking_uri,
            main_metric,
            plugin_name,
            accumulators,
            is_loss,
            n_top_runs,
            requirements,
            retry_config,
        )

    @retry_method
    def _delete_model(self, run_id: str, experiment_id: str) -> bool:
        run_info = mlflow.get_run(run_id=run_id)
        # Get MLflow artifact URI
        artifact_uri = run_info.info.artifact_uri

        # Initialize LocalArtifactRepository
        local_repo = LocalArtifactRepository(artifact_uri)

        # Delete files from Local File System
        try:
            local_repo.delete_artifacts()
            logger.info(
                f"Model files for run {run_id} successfully deleted from Local File System."
            )
            return True

        except Exception as e:
            logger.exception(
                f"Error deleting model files from Local File System: {e}"
            )
            return False
