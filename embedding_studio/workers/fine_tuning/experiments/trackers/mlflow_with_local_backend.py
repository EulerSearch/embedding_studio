import logging
from typing import List, Optional

import mlflow
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository

from embedding_studio.workers.fine_tuning.experiments.experiments_tracker import (
    ExperimentsManager,
)
from embedding_studio.workers.fine_tuning.experiments.metrics_accumulator import (
    MetricsAccumulator,
)
from embedding_studio.workers.fine_tuning.utils.config import RetryConfig
from embedding_studio.workers.fine_tuning.utils.retry import retry_method

logger = logging.getLogger(__name__)


class ExperimentsManagerWithLocalFileSystem(ExperimentsManager):
    def __init__(
        self,
        tracking_uri: str,
        main_metric: str,
        accumulators: List[MetricsAccumulator],
        is_loss: bool = False,
        n_top_runs: int = 10,
        requirements: Optional[str] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        """Wrapper over mlflow package to manage certain fine-tuning experiments with local path backend.

        :param tracking_uri: url of MLFlow server
        :type tracking_uri: str
        :param main_metric: name of main metric that will be used to find best model
        :type main_metric: str
        :param accumulators: accumulators of metrics to be logged
        :type accumulators: List[MetricsAccumulator]
        :param is_loss: is main metric loss (if True, then best quality is minimal) (default: False)
        :type is_loss:  bool
        :param n_top_runs: how many hyper params group consider to be used in following tuning steps (default: 10)
        :type n_top_runs: int
        :param requirements: extra requirements to be passed to mlflow.pytorch.log_model (default: None)
        :type requirements: Optional[str]
        :param retry_config: retry policy (default: None)
        :type retry_config: Optional[RetryConfig]
        :return: Decorated function
        """
        super(ExperimentsManagerWithLocalFileSystem, self).__init__(
            tracking_uri,
            main_metric,
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
