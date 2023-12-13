import logging
from typing import List, Optional

import boto3
import mlflow
from botocore.exceptions import EndpointConnectionError
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


class S3Credentials(BaseModel):
    """
    Pydantic model for S3 credentials.
    """

    endpoint: str
    access_key: str
    secret_key: str
    bucket_name: str
    secure: bool = True

    class Config:
        env_prefix = "S3_"


class ExperimentsManagerWithAmazonS3Backend(ExperimentsManager):
    def __init__(
        self,
        tracking_uri: str,
        s3_credentials: S3Credentials,
        main_metric: str,
        accumulators: List[MetricsAccumulator],
        is_loss: bool = False,
        n_top_runs: int = 10,
        requirements: Optional[str] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        """Wrapper over mlflow package to manage certain fine-tuning experiments with Amazon S3 backend.

        :param tracking_uri: url of MLFlow server
        :type tracking_uri: str
        :param s3_credentials: credentials to connect to Amazon S3
        :type s3_credentials: S3Credentials
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
        super(ExperimentsManagerWithAmazonS3Backend, self).init(
            tracking_uri,
            main_metric,
            accumulators,
            is_loss,
            n_top_runs,
            requirements,
            retry_config,
        )
        self._s3_credentials = s3_credentials
        self.attempt_exception_types += [EndpointConnectionError]

    @retry_method
    def _delete_model(self, run_id: str, experiment_id: str) -> bool:
        # Initialize S3 client
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=self._s3_credentials.aws_access_key_id,
            aws_secret_access_key=self._s3_credentials.aws_secret_access_key,
            region_name=self._s3_credentials.region_name,
        )

        run_info = mlflow.get_run(run_id=run_id)
        # Get MLflow artifact URI
        artifact_uri = run_info.info.artifact_uri

        # Extract S3 object path from the artifact URI
        object_path = artifact_uri.split(self._s3_credentials.bucket_name)[
            1
        ].lstrip("/")

        # Delete files from S3
        response = s3_client.delete_object(
            Bucket=self._s3_credentials.bucket_name, Key=object_path
        )
        if response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 204:
            logger.info(
                f"Model files for run {run_id} successfully deleted from S3."
            )
            return True
        else:
            logger.info(
                f"Object {object_path} does not exist in bucket {self._s3_credentials.bucket_name}"
            )
            return False
