import logging
from typing import List, Optional

import mlflow
from minio import Minio
from minio.error import S3Error, ServerError
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


class MinIOCredentials(BaseModel):
    """
    Pydantic model for MinIO credentials.
    """

    endpoint: str
    access_key: str
    secret_key: str
    bucket_name: str
    secure: bool = True

    class Config:
        env_prefix = "MINIO_"


class ExperimentsManagerWithMinIOBackend(ExperimentsManager):
    def __init__(
        self,
        tracking_uri: str,
        minio_credentials: MinIOCredentials,
        main_metric: str,
        accumulators: List[MetricsAccumulator],
        is_loss: bool = False,
        n_top_runs: int = 10,
        requirements: Optional[str] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        """Wrapper over mlflow package to manage certain fine-tuning experiments with MinIO backend.

        :param tracking_uri: url of MLFlow server
        :param minio_credentials: credentials to connect to MinIO
        :param main_metric: name of main metric that will be used to find best model
        :param accumulators: accumulators of metrics to be logged
        :param is_loss: is main metric loss (if True, then best quality is minimal) (default: False)
        :param n_top_runs: how many hyper params group consider to be used in following tuning steps (default: 10)
        :param requirements: extra requirements to be passed to mlflow.pytorch.log_model (default: None)
        :param retry_config: retry policy (default: None)
        """
        super(ExperimentsManagerWithMinIOBackend, self).__init__(
            tracking_uri,
            main_metric,
            accumulators,
            is_loss,
            n_top_runs,
            requirements,
            retry_config,
        )
        self._minio_credentials = minio_credentials

    def is_retryable_error(self, e: Exception) -> bool:
        if type(e) == ServerError:
            if e.status_code >= 500 and e.status_code < 600:
                return True

        return False

    @retry_method
    def _delete_model(self, run_id: str, experiment_id: str) -> bool:
        # Initialize Minio client
        minio_client = Minio(
            self._minio_credentials.endpoint,
            access_key=self._minio_credentials.access_key,
            secret_key=self._minio_credentials.secret_key,
            secure=self._minio_credentials.secure,
        )
        run_info = mlflow.get_run(run_id=run_id)
        # Get MLflow artifact URI
        artifact_uri = run_info.info.artifact_uri

        # Extract Minio object path from the artifact URI
        object_path = artifact_uri.split(self._minio_credentials.bucket_name)[
            1
        ].lstrip("/")

        try:
            # Delete files from Minio
            minio_client.remove_object(
                self._minio_credentials.bucket_name, object_path
            )
            logger.info(
                f"Model files for run {run_id} successfully deleted from Minio."
            )
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
                logger.error(
                    f"Object {object_path} does not exist in bucket {self._minio_credentials.bucket_name}"
                )
                return False
            else:
                raise e
