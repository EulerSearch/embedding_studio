import logging
import os
import subprocess
from socket import setdefaulttimeout
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from mlflow.entities import Experiment
from mlflow.exceptions import RestException, MlflowException

from embedding_studio.core.config import settings
from embedding_studio.embeddings.models.interface import (
    EmbeddingsModelInterface,
)
from embedding_studio.workers.fine_tuning.experiments.finetuning_iteration import (
    EXPERIMENT_PREFIX,
    FineTuningIteration,
)
from embedding_studio.workers.fine_tuning.experiments.finetuning_params import (
    FineTuningParams,
)
from embedding_studio.workers.fine_tuning.experiments.metrics_accumulator import (
    MetricsAccumulator,
    MetricValue,
)
from embedding_studio.workers.fine_tuning.mlflow.utils import (
    get_experiment_id_by_name,
    get_run_id_by_name,
)
from embedding_studio.workers.fine_tuning.utils.config import (
    RetryConfig,
    RetryParams,
)
from embedding_studio.workers.fine_tuning.utils.exceptions import (
    MaxAttemptsReachedException,
)
from embedding_studio.workers.fine_tuning.utils.retry import retry_method

INITIAL_EXPERIMENT_NAME: str = f"{EXPERIMENT_PREFIX} / initial"
INITIAL_RUN_NAME: str = "initial_model"
DEFAULT_TIMEOUT: int = 120000

# MLFlow upload models using urllib3, if model is heavy enough provided default timeout is not enough
# That's why increase it here. TODO: check from time to time whether this issue is resolved by MLFlow
setdefaulttimeout(DEFAULT_TIMEOUT)
logger = logging.getLogger(__name__)


def _get_base_requirements():
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


class ExperimentsManager:
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
        """Wrapper over mlflow package to manage certain fine-tuning experiments.

        :param tracking_uri: url of MLFlow server
        :param main_metric: name of main metric that will be used to find best model
        :param accumulators: accumulators of metrics to be logged
        :param is_loss: is main metric loss (if True, then best quality is minimal) (default: False)
        :param n_top_runs: how many hyper params group consider to be used in following tuning steps (default: 10)
        :param requirements: extra requirements to be passed to mlflow.pytorch.log_model (default: None)
        :param retry_config: retry policy (default: None)
        """
        if not isinstance(tracking_uri, str) or len(tracking_uri) == 0:
            raise ValueError(
                f"MLFlow tracking URI value should be a not empty string"
            )
        mlflow.set_tracking_uri(tracking_uri)

        self.retry_config = (
            retry_config
            if retry_config
            else ExperimentsManager._get_default_retry_config()
        )
        self.attempt_exception_types = [RestException]

        if not isinstance(main_metric, str) or len(main_metric) == 0:
            raise ValueError(f"main_metric value should be a not empty string")
        self.main_metric = main_metric
        self._metric_field = f"metrics.{self.main_metric}"

        self._n_top_runs = n_top_runs
        self._is_loss = is_loss

        if len(accumulators) == 0:
            logger.warning(
                "No accumulators were provided, there will be no metrics logged except loss"
            )
        self._accumulators = accumulators

        self._requirements: List[str] = (
            _get_base_requirements() if requirements is None else requirements
        )

        self._iteration_experiment = None
        self._tuning_iteration = None
        self._tuning_iteration_id = None

        self._run = None
        self._run_params = None
        self._run_id = None

    @staticmethod
    def _get_default_retry_config() -> RetryConfig:
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
        config["create_experiment"] = RetryParams(
            max_attempts=settings.MLFLOW_CREATE_EXPERIMENT_ATTEMPTS,
            wait_time_seconds=settings.MLFLOW_CREATE_EXPERIMENT_WAIT_TIME_SECONDS,
        )
        config["get_experiment"] = RetryParams(
            max_attempts=settings.MLFLOW_GET_EXPERIMENT_ATTEMPTS,
            wait_time_seconds=settings.MLFLOW_GET_EXPERIMENT_WAIT_TIME_SECONDS,
        )

        return config

    @property
    def is_loss(self) -> bool:
        return self._is_loss

    def __del__(self):
        self.finish_run()
        self.finish_iteration()

    def is_retryable_error(self, e: Exception) -> bool:
        return False

    def _get_model_exists_filter(self) -> str:
        return "params.model_uploaded = '1' AND params.model_deleted != '1'"

    @retry_method(name="log_model")
    def upload_initial_model(self, model: EmbeddingsModelInterface):
        """Upload the very first, initial model to the mlflow server

        :param model: model to be uploaded
        """
        self.finish_iteration()
        experiment_id = get_experiment_id_by_name(INITIAL_EXPERIMENT_NAME)
        if experiment_id is None:
            logger.info(f"Can\'t find any active iteration with name: {INITIAL_EXPERIMENT_NAME}")
            try:
                logger.info("Create initial experiment")
                mlflow.create_experiment(INITIAL_EXPERIMENT_NAME)
            except MlflowException as e:
                if 'Cannot set a deleted experiment' in str(e):
                    logger.error(f"Creation of initial experiment is failed: experiment with the same name {INITIAL_EXPERIMENT_NAME} is deleted, but not archived")
                    experiments = mlflow.search_experiments(view_type=mlflow.entities.ViewType.ALL)
                    deleted_experiment_id = None

                    for exp in experiments:
                        if exp.name == INITIAL_EXPERIMENT_NAME:
                            deleted_experiment_id = exp.experiment_id
                            break

                    logger.info(f'Restore deleted experiment with the same name: {INITIAL_EXPERIMENT_NAME}')
                    mlflow.tracking.MlflowClient().restore_experiment(deleted_experiment_id)
                    logger.info(f'Archive deleted experiment with the same name: {INITIAL_EXPERIMENT_NAME}')
                    mlflow.tracking.MlflowClient().rename_experiment(deleted_experiment_id, INITIAL_EXPERIMENT_NAME + '_archive')
                    logger.info(f'Delete archived experiment with the same name: {INITIAL_EXPERIMENT_NAME}')
                    mlflow.delete_experiment(deleted_experiment_id)
                    logger.info(f'Create initial experiment')
                    mlflow.create_experiment(INITIAL_EXPERIMENT_NAME)
                else:
                    raise e

        with mlflow.start_run(
            experiment_id=get_experiment_id_by_name(INITIAL_EXPERIMENT_NAME),
            run_name=INITIAL_RUN_NAME,
        ) as run:
            logger.info(
                f"Upload initial model to {INITIAL_EXPERIMENT_NAME} / {INITIAL_RUN_NAME}"
            )
            mlflow.pytorch.log_model(
                model, "model", pip_requirements=self._requirements
            )
            logger.info("Uploading is finished")

    @retry_method(name="load_model")
    def download_initial_model(self) -> EmbeddingsModelInterface:
        """Download initial model.

        :return: initial embeddings model
        """
        model_uri: str = f"runs:/{get_run_id_by_name(get_experiment_id_by_name(INITIAL_EXPERIMENT_NAME), INITIAL_RUN_NAME)}/model"
        logger.info(f"Download the model from {model_uri}")
        model = mlflow.pytorch.load_model(model_uri)
        logger.info("Downloading is finished")
        return model

    @retry_method(name="search_runs")
    def get_top_params(self) -> Optional[List[FineTuningParams]]:
        """Get top N previous fine-tuning iteration best params

        :return: fine-tuning iteration params
        """
        initial_id: Optional[str] = get_experiment_id_by_name(
            INITIAL_EXPERIMENT_NAME
        )
        last_session_id: Optional[str] = self.get_previous_iteration_id()
        if initial_id == last_session_id:
            logger.warning(
                "Can't retrieve top params, no previous iteration in history"
            )
            return None

        else:
            runs: pd.DataFrame = mlflow.search_runs(
                experiment_ids=[last_session_id],
                filter_string=self._get_model_exists_filter(),
            )
            runs = runs[runs.status == "FINISHED"]  # and only finished ones
            if runs.shape[0] == 0:
                logger.warning(
                    "Can't retrieve top params, no previous iteration's finished runs with uploaded model in history"
                )
                return None

            # Get the indices that would sort the DataFrame based on the specified parameter
            sorted_indices: np.ndarray = np.argsort(
                runs[self._metric_field].values
            )
            if not self.is_loss:
                sorted_indices = sorted_indices[
                    ::-1
                ]  # Use [::-1] to sort in descending order

            # Extract the top N rows based on the sorted indices
            top_n_rows: np.ndarray = runs.iloc[
                sorted_indices[: self._n_top_runs]
            ]

            # Define a mapping dictionary to remove the "params." prefix
            column_mapping: Dict[str, str] = {
                col: col.replace("params.", "") for col in top_n_rows.columns
            }

            # Rename the columns
            top_n_rows: np.ndarray = top_n_rows.rename(
                columns=column_mapping
            ).to_dict(orient="records")

            return [FineTuningParams(**row) for row in top_n_rows]

    @retry_method(name="load_model")
    def get_last_model(self) -> EmbeddingsModelInterface:
        """Get previous iteration best embedding model.

        :return: best embedding model
        """
        initial_id: Optional[str] = get_experiment_id_by_name(
            INITIAL_EXPERIMENT_NAME
        )
        last_session_id: Optional[str] = self.get_previous_iteration_id()
        if initial_id == last_session_id or last_session_id is None:
            logger.warning(
                "Download initial model, no previous iteration in history"
            )
            return self.download_initial_model()
        else:
            run_id, _ = self._get_best_quality(last_session_id)
            if run_id is None:
                logger.warning(
                    "Download initial model, no previous iteration's finished runs with uploaded model in history"
                )
                return self.download_initial_model()
            else:
                model_uri: str = f"runs:/{run_id}/model"
                logger.info(f"Download the model from {model_uri}")
                model = mlflow.pytorch.load_model(model_uri)
                logger.info("Downloading is finished")
                return model

    @retry_method(name="search_experiments")
    def get_previous_iteration_id(self) -> Optional[str]:
        plugin_name = f"{self._tuning_iteration.plugin_name}"
        experiments: List[Experiment] = [
            e
            for e in mlflow.search_experiments()
            if (
                e.name.startswith(EXPERIMENT_PREFIX)
                and e.name.find(plugin_name) != -1
            )
        ]
        if len(experiments) == 0:
            logger.warning("No iteration found")
            return None
        else:
            return max(
                experiments, key=lambda exp: exp.creation_time
            ).experiment_id

    @retry_method(name="delete_experiment")
    def delete_previous_iteration(self):
        experiment_id: Optional[str] = self.get_previous_iteration_id()

        logger.info("Delete models of previous iteration.")
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=self._get_model_exists_filter(),
        )
        runs = runs[runs.status == "FINISHED"]
        run_ids = runs["run_id"].tolist()

        for run_id in run_ids:
            self.delete_model(run_id, experiment_id)

        if experiment_id is not None:
            logger.info(
                f"Iteration with ID {experiment_id} is going to be deleted"
            )
            mlflow.tracking.MlflowClient().rename_experiment(experiment_id, INITIAL_EXPERIMENT_NAME + '_archive')
            mlflow.delete_experiment(experiment_id)
        else:
            logger.warning(
                "Can't delete a previous iteration, no previous iteration in history"
            )

    @retry_method(name="create_experiment")
    def set_iteration(self, iteration: FineTuningIteration):
        """Start a new fine-tuning session.

        :param iteration: fine-tuning iteration info
        """
        if self._tuning_iteration == INITIAL_EXPERIMENT_NAME:
            self.finish_iteration()

        logger.info("Start a new fine-tuning iterations")

        self._tuning_iteration = iteration
        self._tuning_iteration_id = get_experiment_id_by_name(str(iteration))
        if self._tuning_iteration_id is None:
            self._tuning_iteration_id = mlflow.create_experiment(
                str(iteration)
            )

        self._iteration_experiment = mlflow.set_experiment(
            experiment_id=self._tuning_iteration_id
        )

    @retry_method(name="start_run")
    def set_run(self, params: FineTuningParams) -> bool:
        """Start a new run with provided fine-tuning params

        :param params: provided fine-tuning params
        :return: True if it's a finished run (otherwise False)
        """
        convert_value = (
            lambda value: ", ".join(map(str, value))
            if isinstance(value, list)
            else value
        )

        if self._tuning_iteration == INITIAL_EXPERIMENT_NAME:
            # TODO: implement exception
            raise ValueError("You can't start run for initial iteration")

        if self._run is not None:
            self.finish_run()

        logger.info(
            f"Start a new run for iteration {self._tuning_iteration_id} with params:\n\t{str(params)}"
        )

        self._run_params = params
        run_name: str = self._run_params.id
        self._run_id = get_run_id_by_name(self._tuning_iteration_id, run_name)

        self._run = mlflow.start_run(
            self._run_id, self._tuning_iteration_id, run_name
        )
        if self._run_id is None:
            self._run_id = self._run.info.run_id
            for key, value in dict(self._tuning_iteration).items():
                mlflow.log_param(key, convert_value(value))

            for key, value in dict(self._run_params).items():
                mlflow.log_param(key, convert_value(value))

            return False
        else:
            return self._run.info.status == "FINISHED"

    @retry_method(name="search_runs")
    def model_is_uploaded(self) -> bool:
        runs: pd.DataFrame = mlflow.search_runs(
            experiment_ids=[self._tuning_iteration_id],
            filter_string=self._get_model_exists_filter(),
        )
        runs = runs[runs["run_id"] == self._run_id]
        return runs.shape[0] > 0

    @retry_method(name="get_experiment")
    def finish_iteration(self):
        logger.info(f"Finish current iteration {self._tuning_iteration_id}")
        self._tuning_iteration = INITIAL_EXPERIMENT_NAME
        self._tuning_iteration_id = get_experiment_id_by_name(
            INITIAL_EXPERIMENT_NAME
        )

        if self._tuning_iteration_id is None:
            self._iteration_experiment = mlflow.set_experiment(
                experiment_name=INITIAL_EXPERIMENT_NAME
            )
            self._tuning_iteration_id = (
                self._iteration_experiment.experiment_id
            )
        else:
            self._iteration_experiment = mlflow.set_experiment(
                experiment_id=self._tuning_iteration_id
            )

        logger.info(f"Current iteration is finished")

    @retry_method(name="end_run")
    def finish_run(self):
        logger.info(
            f"Finish current run {self._tuning_iteration_id} / {self._run_id}"
        )
        for accumulator in self._accumulators:
            accumulator.clear()

        mlflow.end_run()

        # Set params to default None
        self._run = None
        self._run_params = None
        self._run_id = None

        logger.info(f"Current run is finished")

    @retry_method(name="log_param")
    def _set_model_as_deleted(self, run_id: str, experiment_id: str):
        with mlflow.start_run(
            run_id=run_id, experiment_id=experiment_id
        ) as run:
            mlflow.log_param("model_deleted", "1")

    @retry_method(name="delete_model")
    def _delete_model(self, run_id: str, experiment_id: str) -> bool:
        logger.warning(
            f"Unable to delete a model for run {run_id}, MLFlow has no such functionality, please implement on your own."
        )
        return False

    @retry_method(name="get_run")
    def delete_model(self, run_id: str, experiment_id: Optional[str] = None):
        experiment_id = (
            self._tuning_iteration_id
            if experiment_id is None
            else experiment_id
        )
        if experiment_id is None:
            raise ValueError(
                f"No iteration was initialized, unable to delete model."
            )

        if experiment_id == INITIAL_EXPERIMENT_NAME:
            raise ValueError(f"Initial model can't be deleted.")

        run_info = None
        try:
            run_info = mlflow.get_run(run_id=run_id)
        except RestException as e:
            if e.get_http_status_code() == 404:
                logger.exception(f"Run with ID {run_id} doesn't exist.")
            else:
                raise e

        if run_info is not None:
            runs: pd.DataFrame = mlflow.search_runs(
                filter_string=self._get_model_exists_filter()
            )
            runs = runs[runs["run_id"] == run_id]
            if runs.shape[0] == 0:
                logger.warning(
                    f"Run {run_id} has no model being uploaded. Nothing to delete"
                )

            else:
                deleted = None
                try:
                    deleted = self._delete_model(run_id, experiment_id)
                except MaxAttemptsReachedException:
                    pass

                if deleted:
                    self._set_model_as_deleted(run_id, experiment_id)

    @retry_method(name="log_model")
    def save_model(
        self, model: EmbeddingsModelInterface, best_only: bool = True
    ):
        """Save fine-tuned embedding model

        :param model: model to be saved
        :param best_only: save only if it's the best (default: True)
        """
        if self._tuning_iteration == INITIAL_EXPERIMENT_NAME:
            raise ValueError(
                f"Can't save not initial model for {INITIAL_EXPERIMENT_NAME} experiment"
            )

        if self._run_id is None:
            raise ValueError("There is no current Run")

        logger.info(
            f"Save model for {self._tuning_iteration_id} / {self._run_id}"
        )
        if not best_only:
            mlflow.pytorch.log_model(
                model, "model", pip_requirements=self._requirements
            )
            mlflow.log_param("model_uploaded", 1)
            logger.info("Upload is finished")
        else:
            current_quality = self.get_quality()
            best_run_id, best_quality = self.get_best_quality()

            if best_run_id is None or (
                current_quality <= best_quality
                if self.is_loss
                else current_quality >= best_quality
            ):
                mlflow.pytorch.log_model(
                    model, "model", pip_requirements=self._requirements
                )
                mlflow.log_param("model_uploaded", 1)
                logger.info("Upload is finished")

                if best_run_id is not None:
                    self.delete_model(best_run_id)
            else:
                logger.info("Not the best run - ignore saving")

    @retry_method(name="log_metric")
    def save_metric(self, metric_value: MetricValue):
        """Accumulate and save metric value

        :param metric_value: value to be logged
        """
        for accumulator in self._accumulators:
            for name, value in accumulator.accumulate(metric_value):
                mlflow.log_metric(name, value)

    @retry_method(name="search_runs")
    def get_quality(self) -> float:
        """Current run quality value

        :return: quality value
        """
        if self._tuning_iteration == INITIAL_EXPERIMENT_NAME:
            raise ValueError(
                f"No metrics for {INITIAL_EXPERIMENT_NAME} experiment"
            )

        if self._run_id is None:
            raise ValueError("There is no current Run")

        runs: pd.DataFrame = mlflow.search_runs(
            experiment_ids=[self._tuning_iteration_id]
        )
        quality: np.ndarray = runs[runs.run_id == self._run_id][
            self._metric_field
        ]
        return float(quality) if quality.shape[0] == 1 else float(quality[0])

    @retry_method(name="search_runs")
    def _get_best_quality(
        self, experiment_id: str
    ) -> Tuple[Optional[str], float]:
        runs: pd.DataFrame = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=self._get_model_exists_filter(),
        )
        runs = runs[runs.status == "FINISHED"]  # and not finished ones
        if runs.shape[0] == 0:
            logger.warning(
                "No finished experiments found with model uploaded, except initial"
            )
            return None, 0.0

        else:
            value: float = (
                runs[self._metric_field].min()
                if self.is_loss
                else runs[self._metric_field].max()
            )
            best: pd.DataFrame = runs[runs[self._metric_field] == value][
                ["run_id", self._metric_field]
            ]
            return list(best.itertuples(index=False, name=None))[0]

    def get_best_quality(self) -> Tuple[str, float]:
        """Get current fine-tuning iteration best quality

        :return: run_id and best metric value
        """
        if self._tuning_iteration == INITIAL_EXPERIMENT_NAME:
            raise ValueError(
                f"No metrics for {INITIAL_EXPERIMENT_NAME} experiment"
            )

        return self._get_best_quality(self._tuning_iteration_id)
