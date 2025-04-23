import logging
import os
import subprocess
import urllib.parse
from socket import setdefaulttimeout
from typing import Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import RestException

from embedding_studio.core.config import settings
from embedding_studio.embeddings.models.interface import (
    EmbeddingsModelInterface,
)
from embedding_studio.experiments.finetuning_iteration import (
    FineTuningIteration,
)
from embedding_studio.experiments.finetuning_params import FineTuningParams
from embedding_studio.experiments.status import MLflowStatus
from embedding_studio.utils.mlflow_utils import (
    get_experiment_id_by_name,
    get_run_id_by_name,
)
from embedding_studio.utils.retry import retry_method
from embedding_studio.workers.fine_tuning.utils.config import (
    RetryConfig,
    RetryParams,
)

MODEL_ARTIFACT_PATH = "model/data/model.pth"
DEFAULT_TIMEOUT: int = 120000

# MLFlow upload models using urllib3, if model is heavy enough provided default timeout is not enough
# That's why increase it here. TODO: check from time to time whether this issue is resolved by MLFlow
setdefaulttimeout(DEFAULT_TIMEOUT)

logger = logging.getLogger(__name__)


class MLflowClientWrapper:
    """
    A wrapper class for MLflow client that provides retry functionality for MLflow operations
    and additional utility methods for managing models, experiments, and runs.

    :param tracking_uri: URL of the MLflow tracking server
    :param requirements: List of requirements to be used for model logging (optional)
    :param retry_config: Configuration for retry functionality (optional)
    :return: An instance of MLflowClientWrapper
    """

    def __init__(
        self,
        tracking_uri: str,
        requirements: Optional[str] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        if not isinstance(tracking_uri, str) or len(tracking_uri) == 0:
            raise ValueError(
                f"MLFlow tracking URI value should be a not empty string"
            )
        mlflow.set_tracking_uri(tracking_uri)
        self._tracking_uri = tracking_uri
        if self._tracking_uri.endswith("/"):
            self._tracking_uri = self._tracking_uri[:-1]

        self.retry_config = (
            retry_config
            if retry_config
            else MLflowClientWrapper._get_default_retry_config()
        )
        self.attempt_exception_types = [RestException]

        self.client = mlflow.tracking.MlflowClient()

        self._requirements: List[str] = (
            self._get_base_requirements()
            if requirements is None
            else requirements
        )

    @property
    def tracking_uri(self) -> str:
        """
        Get the tracking URI for the MLflow server.

        :return: The tracking URI as a string
        """
        return self._tracking_uri

    @property
    def requirements(self) -> List[str]:
        """
        Get the list of requirements used for model logging.

        :return: List of requirement strings
        """
        return self._requirements

    @staticmethod
    def _get_default_retry_config() -> RetryConfig:
        """
        Create default retry configuration for MLflow operations.

        :return: RetryConfig object with default settings for different MLflow operations
        """
        default_retry_params = RetryParams(
            max_attempts=settings.DEFAULT_MAX_ATTEMPTS,
            wait_time_seconds=settings.DEFAULT_WAIT_TIME_SECONDS,
        )

        config = RetryConfig(default_params=default_retry_params)
        config["log_metric"] = RetryParams(
            max_attempts=settings.MLFLOW_LOG_METRIC_ATTEMPTS,
            wait_time_seconds=settings.MLFLOW_LOG_METRIC_WAIT_TIME_SECONDS,
        )
        config["log_param"] = RetryParams(
            max_attempts=settings.MLFLOW_LOG_PARAM_ATTEMPTS,
            wait_time_seconds=settings.MLFLOW_LOG_PARAM_WAIT_TIME_SECONDS,
        )
        config["log_model"] = RetryParams(
            max_attempts=settings.MLFLOW_LOG_MODEL_ATTEMPTS,
            wait_time_seconds=settings.MLFLOW_LOG_MODEL_WAIT_TIME_SECONDS,
        )
        config["load_model"] = RetryParams(
            max_attempts=settings.MLFLOW_LOAD_MODEL_ATTEMPTS,
            wait_time_seconds=settings.MLFLOW_LOAD_MODEL_WAIT_TIME_SECONDS,
        )
        config["delete_model"] = RetryParams(
            max_attempts=settings.MLFLOW_DELETE_MODEL_ATTEMPTS,
            wait_time_seconds=settings.MLFLOW_DELETE_MODEL_WAIT_TIME_SECONDS,
        )
        config["search_runs"] = RetryParams(
            max_attempts=settings.MLFLOW_SEARCH_RUNS_ATTEMPTS,
            wait_time_seconds=settings.MLFLOW_SEARCH_RUNS_WAIT_TIME_SECONDS,
        )
        config["list_artifacts"] = RetryParams(
            max_attempts=settings.MLFLOW_LIST_ARTIFACTS_ATTEMPTS,
            wait_time_seconds=settings.MLFLOW_LIST_ARTIFACTS_WAIT_TIME_SECONDS,
        )
        config["end_run"] = RetryParams(
            max_attempts=settings.MLFLOW_END_RUN_ATTEMPTS,
            wait_time_seconds=settings.MLFLOW_END_RUN_WAIT_TIME_SECONDS,
        )
        config["get_run"] = RetryParams(
            max_attempts=settings.MLFLOW_GET_RUN_ATTEMPTS,
            wait_time_seconds=settings.MLFLOW_GET_RUN_WAIT_TIME_SECONDS,
        )
        config["search_experiments"] = RetryParams(
            max_attempts=settings.MLFLOW_SEARCH_EXPERIMENTS_ATTEMPTS,
            wait_time_seconds=settings.MLFLOW_SEARCH_EXPERIMENTS_WAIT_TIME_SECONDS,
        )
        config["delete_experiment"] = RetryParams(
            max_attempts=settings.MLFLOW_DELETE_EXPERIMENT_ATTEMPTS,
            wait_time_seconds=settings.MLFLOW_DELETE_EXPERIMENT_WAIT_TIME_SECONDS,
        )
        config["rename_experiment"] = RetryParams(
            max_attempts=settings.MLFLOW_RENAME_EXPERIMENT_ATTEMPTS,
            wait_time_seconds=settings.MLFLOW_RENAME_EXPERIMENT_WAIT_TIME_SECONDS,
        )
        config["create_experiment"] = RetryParams(
            max_attempts=settings.MLFLOW_CREATE_EXPERIMENT_ATTEMPTS,
            wait_time_seconds=settings.MLFLOW_CREATE_EXPERIMENT_WAIT_TIME_SECONDS,
        )
        config["get_experiment"] = RetryParams(
            max_attempts=settings.MLFLOW_GET_EXPERIMENT_ATTEMPTS,
            wait_time_seconds=settings.MLFLOW_GET_EXPERIMENT_WAIT_TIME_SECONDS,
        )

        return config

    def _get_base_requirements(self):
        """
        Generate base requirements using poetry export.

        :return: List of requirement strings exported from poetry
        """
        try:
            logger.info("Generate requirements with poetry")
            # Run the poetry export command
            result = subprocess.run(
                [
                    "poetry",
                    "export",
                    f"--directory={os.path.dirname(__file__)}",
                    "--with",
                    "ml",
                    "-f",
                    "requirements.txt",
                    "--without-hashes",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            # Get the requirements from the standard output
            requirements = result.stdout.strip().split("\n")

            return requirements
        except subprocess.CalledProcessError as e:
            print(f"Error running poetry export: {e}")
            return []

    def _get_model_exists_filter(self) -> str:
        """
        Get filter string to find runs with uploaded models.

        :return: Filter string for searching runs with uploaded models
        """
        return "metrics.model_uploaded = 1"

    def _get_artifact_url(self, run_id: str, artifact_path: str) -> str:
        """
        Create URL for accessing artifacts stored in MLflow.

        :param run_id: ID of the run containing the artifact
        :param artifact_path: Path to the artifact within the run
        :return: URL for accessing the artifact
        """
        return (
            f"{self._tracking_uri}/get-artifact?path="
            f'{urllib.parse.quote(artifact_path, safe="")}&run_uuid={run_id}'
        )

    @retry_method(name="list_artifacts")
    def _check_artifact_exists(self, run_id, artifact_path):
        """
        Check if an artifact exists in the specified run.

        :param run_id: ID of the run to check
        :param artifact_path: Path to the artifact within the run
        :return: Boolean indicating whether the artifact exists
        """
        client = mlflow.MlflowClient()
        artifacts = client.list_artifacts(run_id, path=artifact_path)

        return any(
            artifact.path.startswith(artifact_path) for artifact in artifacts
        )

    @retry_method(name="search_experiments")
    def _get_experiment_id_by_name(
        self, experiment_name: str
    ) -> Optional[str]:
        """
        Get experiment ID by its name.

        :param experiment_name: Name of the experiment
        :return: Experiment ID if found, None otherwise
        """
        return get_experiment_id_by_name(experiment_name)

    @retry_method(name="search_runs")
    def _get_run_id_by_name(self, experiment_id: str, run_name: str):
        """
        Get run ID by its name within an experiment.

        :param experiment_id: ID of the experiment containing the run
        :param run_name: Name of the run
        :return: Run ID if found, None otherwise
        """
        return get_run_id_by_name(experiment_id, run_name)

    def is_retryable_error(self, e: Exception) -> bool:
        """
        Check if an exception should trigger a retry.

        :param e: Exception to check
        :return: Boolean indicating whether to retry on this exception
        """
        return False

    @retry_method(name="search_runs")
    def get_runs(
        self, experiment_id: str, models_only: bool = False
    ) -> pd.DataFrame:
        """
        Get runs from an experiment as a pandas DataFrame.

        :param experiment_id: ID of the experiment to get runs from
        :param models_only: If True, return only runs with uploaded models
        :return: DataFrame containing run information
        """
        if models_only:
            return mlflow.search_runs(
                experiment_ids=[experiment_id],
                filter_string=self._get_model_exists_filter(),
            )

        else:
            return mlflow.search_runs(experiment_ids=[experiment_id])

    @retry_method(name="load_model")
    def _download_model_by_run_id(
        self, run_id: str
    ) -> EmbeddingsModelInterface:
        """
        Download a model by its run ID.

        :param run_id: ID of the run containing the model
        :return: The downloaded embedding model
        """
        model_uri: str = f"runs:/{run_id}/model"
        logger.info(f"Download the model from {model_uri}")
        model = mlflow.pytorch.load_model(model_uri)
        logger.info("Downloading is finished")
        return model

    @retry_method(name="delete_model")
    def _delete_model(self, run_id: str, experiment_id: str) -> bool:
        """
        Delete a model for a specific run.

        :param run_id: ID of the run containing the model to delete
        :param experiment_id: ID of the experiment containing the run
        :return: Boolean indicating success of deletion
        """
        logger.warning(
            f"Unable to delete a model for run {run_id}, "
            f"MLFlow has no such functionality, please implement on your own."
        )
        return False

    @retry_method(name="get_run")
    def _get_run(self, run_id: str) -> mlflow.entities.Run:
        """
        Get run information by run ID.

        :param run_id: ID of the run to get
        :return: Run object containing run information
        """
        run_info = None
        try:
            run_info = mlflow.get_run(run_id=run_id)
        except RestException as e:
            if e.get_http_status_code() == 404:
                logger.exception(f"Run with ID {run_id} doesn't exist.")
            else:
                raise e

        return run_info

    @retry_method(name="log_param")
    def _set_model_as_deleted(self, run_id: str, experiment_id: str):
        """
        Mark a model as deleted in MLflow by setting appropriate metrics.

        :param run_id: ID of the run containing the model
        :param experiment_id: ID of the experiment containing the run
        :return: None
        """
        with mlflow.start_run(
            run_id=run_id, experiment_id=experiment_id
        ) as run:
            mlflow.log_metric("model_deleted", 1)
            mlflow.log_metric("model_uploaded", 0)

    @retry_method(name="search_runs")
    def get_model_url_by_run_id(self, run_id: str) -> Optional[str]:
        """
        Get URL for downloading a model by its run ID.

        :param run_id: ID of the run containing the model
        :return: URL for downloading the model, or None if not found
        """
        runs: pd.DataFrame = mlflow.search_runs(
            filter_string=self._get_model_exists_filter(),
        )
        runs = runs[
            runs.status == MLflowStatus.FINISHED
        ]  # and only finished ones
        specific_run = runs[runs["run_id"] == run_id]

        if specific_run.empty:
            logger.error(
                f"No run with ID {run_id} which contains a model was found."
            )
            return None

        return self._get_artifact_url(run_id, MODEL_ARTIFACT_PATH)

    @retry_method(name="rename_experiment")
    def _archive_experiment(self, experiment_id: str):
        """
        Archive an experiment by renaming it with "_archive" suffix.

        :param experiment_id: ID of the experiment to archive
        :return: None
        """
        mlflow.tracking.MlflowClient().rename_experiment(
            experiment_id, str(experiment_id) + "_archive"
        )

    @retry_method(name="delete_experiment")
    def _delete_experiment(self, experiment_id: str):
        """
        Delete an experiment from MLflow.

        :param experiment_id: ID of the experiment to delete
        :return: None
        """
        mlflow.delete_experiment(experiment_id)

    @retry_method(name="search_runs")
    def get_params_by_run_id(self, run_id: str) -> Optional[FineTuningParams]:
        """Get a fine-tuning params from a run by ID.

        :param run_id: ID of a run.
        :return: fine-tuning params.
        """
        runs: pd.DataFrame = mlflow.search_runs()
        specific_run = runs[runs["run_id"] == run_id]

        if specific_run.empty:
            logger.error(f"No run with ID {run_id} was found.")
            return None

        # Define a mapping dictionary to remove the "params." prefix
        column_mapping: Dict[str, str] = {
            col: col.replace("params.", "") for col in specific_run.columns
        }

        # Rename the columns
        rows: np.ndarray = specific_run.rename(columns=column_mapping).to_dict(
            orient="records"
        )

        return FineTuningParams(**rows[0])

    @retry_method(name="search_runs")
    def get_iteration_by_id(self, run_id: str) -> FineTuningIteration:
        """Get a fine-tuning iteration info from a run by ID.

        :param run_id: ID of a run.
        :return: fine-tuning iteration info.
        """
        # Get all experiments
        experiments = mlflow.search_experiments()

        # Collect all experiment IDs
        experiment_ids = [exp.experiment_id for exp in experiments]

        # Get all runs from all experiments
        runs: pd.DataFrame = mlflow.search_runs(experiment_ids=experiment_ids)
        specific_run = runs[runs["run_id"] == run_id]

        if specific_run.empty:
            logger.error(f"No run with ID {run_id} was found.")
            return None

        experiment_id = specific_run["experiment_id"].iloc[0]
        experiment = mlflow.get_experiment(experiment_id)
        return FineTuningIteration.parse(experiment.name)

    @retry_method(name="search_runs")
    def get_experiment_id(self, run_id: str) -> Optional[str]:
        """Get experiment ID of a run by ID.

        :param run_id: ID of a run.
        :return: experiment ID.
        """
        runs: pd.DataFrame = mlflow.search_runs()
        specific_run = runs[runs["run_id"] == run_id]

        if specific_run.empty:
            logger.error(f"No run with ID {run_id} was found.")
            return None

        return specific_run["experiment_id"].iloc[0]
