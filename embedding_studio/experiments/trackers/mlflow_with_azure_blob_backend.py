import logging
from typing import List, Optional

import mlflow
from azure.core.exceptions import HttpResponseError
from azure.storage.blob import (  # Import for Azure Blob Storage
    BlobServiceClient,
)
from pydantic import BaseModel

from embedding_studio.experiments.experiments_tracker import ExperimentsManager
from embedding_studio.experiments.metrics_accumulator import MetricsAccumulator
from embedding_studio.workers.fine_tuning.utils.config import RetryConfig
from embedding_studio.workers.fine_tuning.utils.retry import retry_method

logger = logging.getLogger(__name__)

# WARNING: Experimental. TODO: Testing


class AzureBlobCredentials(BaseModel):
    """
    Pydantic model for Azure Blob Storage credentials.
    """

    account_name: str
    account_key: str
    container_name: str


class ExperimentsManagerWithAzureBlobBackend(ExperimentsManager):
    def __init__(
        self,
        tracking_uri: str,
        azure_blob_credentials: AzureBlobCredentials,
        main_metric: str,
        plugin_name: str,
        accumulators: List[MetricsAccumulator],
        is_loss: bool = False,
        n_top_runs: int = 10,
        requirements: Optional[str] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        """Wrapper over mlflow package to manage certain fine-tuning experiments with Azure Blob Storage backend.

        :param tracking_uri: url of MLFlow server
        :param azure_blob_credentials: credentials to connect to Azure Blob Storage
        :param main_metric: name of the main metric that will be used to find models
        :param plugin_name: name of fine-tuning method being used
        :param accumulators: accumulators of metrics to be logged
        :param is_loss: is the main metric loss (if True, then the best quality is minimal) (default: False)
        :param n_top_runs: how many hyper param groups to consider for following tuning steps (default: 10)
        :param requirements: extra requirements to be passed to mlflow.pytorch.log_model (default: None)
        :param retry_config: retry policy (default: None)
        """
        super(ExperimentsManagerWithAzureBlobBackend, self).__init__(
            tracking_uri,
            main_metric,
            plugin_name,
            accumulators,
            is_loss,
            n_top_runs,
            requirements,
            retry_config,
        )
        self._azure_blob_credentials = azure_blob_credentials
        raise NotImplementedError()

    def is_retryable_error(self, e: Exception) -> bool:
        if isinstance(e, HttpResponseError) and 500 <= e.status_code < 600:
            return True

        return False

    @retry_method
    def _delete_model(self, run_id: str, experiment_id: str) -> bool:
        run_info = mlflow.get_run(run_id=run_id)
        # Initialize Azure Blob Service Client
        connection_string = (
            f"DefaultEndpointsProtocol=https;AccountName={self._azure_blob_credentials.account_name};"
            f"AccountKey={self._azure_blob_credentials.account_key};EndpointSuffix=core.windows.net"
        )
        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )
        # Get MLflow artifact URI
        artifact_uri = run_info.info.artifact_uri

        # Extract blob path from the artifact URI
        blob_path = artifact_uri.split(
            self._azure_blob_credentials.container_name
        )[1].lstrip("/")

        # Delete blob from Azure Blob Storage
        try:
            container_client = blob_service_client.get_container_client(
                self._azure_blob_credentials.container_name
            )
            blob_client = container_client.get_blob_client(blob=blob_path)
            blob_client.delete_blob()

            logger.info(
                f"Model files for run {run_id} successfully deleted from Azure Blob Storage."
            )
            return True

        except HttpResponseError as e:
            if e.status_code == 404:
                logger.exception(
                    f"Blob {blob_path} does not exist in container {self._azure_blob_credentials.container_name}"
                )
                return False
            else:
                logger.exception(
                    f"Error deleting model files from Azure Blob Storage: {e}"
                )
                raise e
