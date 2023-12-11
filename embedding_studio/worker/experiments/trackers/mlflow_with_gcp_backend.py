import logging
from typing import List, Optional

import mlflow
from google.api_core.exceptions import GoogleAPIError, NotFound
from google.cloud import storage
from pydantic import BaseModel

from embedding_studio.worker.experiments.experiments_tracker import (
    ExperimentsManager,
)
from embedding_studio.worker.experiments.metrics_accumulator import (
    MetricsAccumulator,
)
from embedding_studio.worker.utils.config import RetryConfig
from embedding_studio.worker.utils.retry import retry_method

logger = logging.getLogger(__name__)


# WARNING: Experimental. TODO: Testing


class GCPCredentials(BaseModel):
    """
    Pydantic model for GCP credentials.
    """

    project_id: str
    bucket_name: str
    client_email: str
    private_key: str
    private_key_id: str


class ExperimentsManagerWithGCPBackend(ExperimentsManager):
    def __init__(
        self,
        tracking_uri: str,
        gcp_credentials: GCPCredentials,
        main_metric: str,
        accumulators: List[MetricsAccumulator],
        is_loss: bool = False,
        n_top_runs: int = 10,
        requirements: Optional[str] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        """Wrapper over mlflow package to manage certain fine-tuning experiments with GCP backend.

        :param tracking_uri: url of MLFlow server
        :type tracking_uri: str
        :param gcp_credentials: credentials to connect to GCP
        :type gcp_credentials: GCPCredentials
        :param main_metric: name of the main metric that will be used to find models
        :type main_metric: str
        :param accumulators: accumulators of metrics to be logged
        :type accumulators: List[MetricsAccumulator]
        :param is_loss: is the main metric loss (if True, then the best quality is minimal) (default: False)
        :type is_loss: bool
        :param n_top_runs: how many hyper params groups to consider for following tuning steps (default: 10)
        :type n_top_runs: int
        :param requirements: extra requirements to be passed to mlflow.pytorch.log_model (default: None)
        :type requirements: Optional[str]
        :param retry_config: retry policy (default: None)
        :type retry_config: Optional[RetryConfig]
        :return: Decorated function
        """
        super(ExperimentsManagerWithGCPBackend, self).__init__(
            tracking_uri,
            main_metric,
            accumulators,
            is_loss,
            n_top_runs,
            requirements,
            retry_config,
        )
        self._gcp_credentials = gcp_credentials
        raise NotImplementedError()

    def is_retryable_error(self, e: Exception) -> bool:
        if type(e) == GoogleAPIError and e.code >= 500:
            logger.error(f"Google Cloud Storage Server Error (5xx): {e}")
            return True
        return False

    @retry_method
    def _delete_model(self, run_id: str, experiment_id: str) -> bool:
        # Initialize GCP client
        run_info = mlflow.get_run(run_id=run_id)
        gcp_client = storage.Client(project=self._gcp_credentials.project_id)
        # Get MLflow artifact URI
        artifact_uri = run_info.info.artifact_uri

        # Extract GCS object path from the artifact URI
        object_path = artifact_uri.split(self._gcp_credentials.bucket_name)[
            1
        ].lstrip("/")

        # Delete files from GCS
        try:
            bucket = gcp_client.get_bucket(self._gcp_credentials.bucket_name)
            blob = bucket.blob(object_path)
            blob.delete()

            logger.info(
                f"Model files for run {run_id} successfully deleted from GCS."
            )
            return True

        except NotFound as e:

            logger.exception(f"Error deleting model files from GCP: {e}")
            return False
