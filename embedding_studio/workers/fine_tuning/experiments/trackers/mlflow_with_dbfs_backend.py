import logging
from typing import List, Optional

import mlflow
from databricks_cli.dbfs.api import DbfsApi
from databricks_cli.sdk.api_client import ApiClient
from pydantic import BaseModel

from embedding_studio.workers.fine_tuning.experiments.experiments_tracker import (
    ExperimentsManager,
)
from embedding_studio.workers.fine_tuning.experiments.metrics_accumulator import (
    MetricsAccumulator,
)
from embedding_studio.workers.fine_tuning.utils.config import RetryConfig
from embedding_studio.workers.fine_tuning.utils.retry import retry_method

logger = logging.getLogger(__name__)


# WARNING: Experimental. TODO: Testing


class DatabricksCredentials(BaseModel):
    """
    Pydantic model for Databricks credentials.
    """

    host: str
    token: str
    # Add other necessary fields


class ExperimentsManagerWithDBFSBackend(ExperimentsManager):
    def __init__(
        self,
        tracking_uri: str,
        databricks_credentials: DatabricksCredentials,
        main_metric: str,
        accumulators: List[MetricsAccumulator],
        is_loss: bool = False,
        n_top_runs: int = 10,
        requirements: Optional[str] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        """Wrapper over mlflow package to manage certain fine-tuning experiments with DBFS backend.

        :param tracking_uri: url of MLFlow server
        :param databricks_credentials: credentials to connect to DBFS Client
        :param main_metric: name of the main metric that will be used to find models
        :param accumulators: accumulators of metrics to be logged
        :param is_loss: is the main metric loss (if True, then the best quality is minimal) (default: False)
        :param n_top_runs: how many hyper params groups to consider for following tuning steps (default: 10)
        :param requirements: extra requirements to be passed to mlflow.pytorch.log_model (default: None)
        :param retry_config: retry policy (default: None)
        """
        super(ExperimentsManagerWithDBFSBackend, self).__init__(
            tracking_uri,
            main_metric,
            accumulators,
            is_loss,
            n_top_runs,
            requirements,
            retry_config,
        )
        self._dbfs_credentials = databricks_credentials
        self._dbfs_client = DbfsApi(
            ApiClient(
                databricks_credentials.host, databricks_credentials.token
            )
        )
        raise NotImplementedError()

    @retry_method
    def _delete_model(self, run_id: str, experiment_id: str) -> bool:
        run_info = mlflow.get_run(run_id=run_id)
        # Get MLflow artifact URI
        artifact_uri = run_info.info.artifact_uri

        # Extract DBFS path from the artifact URI
        dbfs_path = artifact_uri.replace(self._dbfs_credentials.host, "")

        # Delete files from DBFS
        response = self._dbfs_client.delete(dbfs_path, recursive=True)
        if response.status_code == 200:
            logger.info(
                f"Model files for run {run_id} successfully deleted from DBFS."
            )
            return True
        elif response.status_code == 404:
            logger.error(f"File {dbfs_path} does not exist in DBFS")
            # Handle the case where the file doesn't exist
            return False
