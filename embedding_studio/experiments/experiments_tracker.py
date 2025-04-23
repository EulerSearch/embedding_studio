import logging
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from mlflow.entities import Experiment
from mlflow.exceptions import MlflowException

from embedding_studio.embeddings.models.interface import (
    EmbeddingsModelInterface,
)
from embedding_studio.experiments.finetuning_iteration import (
    EXPERIMENT_PREFIX,
    FineTuningIteration,
)
from embedding_studio.experiments.finetuning_params import FineTuningParams
from embedding_studio.experiments.metrics_accumulator import (
    MetricsAccumulator,
    MetricValue,
)
from embedding_studio.experiments.mlflow_client_wrapper import (
    MODEL_ARTIFACT_PATH,
    MLflowClientWrapper,
)
from embedding_studio.experiments.status import MLflowStatus
from embedding_studio.utils.mlflow_utils import get_experiment_id_by_name
from embedding_studio.utils.retry import retry_method
from embedding_studio.workers.fine_tuning.utils.config import RetryConfig
from embedding_studio.workers.fine_tuning.utils.exceptions import (
    MaxAttemptsReachedException,
)

INITIAL_EXPERIMENT_NAME: str = f"{EXPERIMENT_PREFIX} / initial"
INITIAL_RUN_NAME: str = "initial_model"


logger = logging.getLogger(__name__)


class ExperimentsManager(MLflowClientWrapper):
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
        """Wrapper over mlflow package to manage certain fine-tuning experiments.

        :param tracking_uri: url of MLFlow server
        :param main_metric: name of main metric that will be used to find best model
        :param plugin_name: name of fine-tuning method being used
        :param accumulators: accumulators of metrics to be logged
        :param is_loss: is main metric loss (if True, then best quality is minimal) (default: False)
        :param n_top_runs: how many hyper params group consider to be used in following tuning steps (default: 10)
        :param requirements: extra requirements to be passed to mlflow.pytorch.log_model (default: None)
        :param retry_config: retry policy (default: None)
        """

        super(ExperimentsManager, self).__init__(
            tracking_uri, requirements, retry_config
        )

        if not isinstance(main_metric, str) or len(main_metric) == 0:
            raise ValueError(f"main_metric value should be a not empty string")
        self.main_metric = main_metric
        self._metric_field = f"metrics.{self.main_metric}"

        self._plugin_name = plugin_name
        self.initial_experiment_name = self._fix_name(INITIAL_EXPERIMENT_NAME)
        self._n_top_runs = n_top_runs
        self._is_loss = is_loss

        if len(accumulators) == 0:
            logger.warning(
                "No accumulators were provided, there will be no metrics logged except loss"
            )
        self._accumulators = accumulators

        self._iteration_experiment = None
        self._tuning_iteration = None
        self._tuning_iteration_id = None

        self._run = None
        self._run_params = None
        self._run_id = None

    @staticmethod
    def from_wrapper(
        wrapper: MLflowClientWrapper,
        main_metric: str,
        plugin_name: str,
        accumulators: List[MetricsAccumulator],
        is_loss: bool = False,
        n_top_runs: int = 10,
    ) -> "ExperimentsManager":
        """
        Create an ExperimentsManager from an existing MLflowClientWrapper.

        :param wrapper: Existing MLflowClientWrapper instance
        :param main_metric: Name of main metric that will be used to find best model
        :param plugin_name: Name of fine-tuning method being used
        :param accumulators: Accumulators of metrics to be logged
        :param is_loss: Is main metric loss (if True, then best quality is minimal)
        :param n_top_runs: How many hyper params group consider to be used in following tuning steps
        :return: A new ExperimentsManager instance
        """
        return ExperimentsManager(
            wrapper.tracking_uri,
            main_metric,
            plugin_name,
            accumulators,
            is_loss,
            n_top_runs,
            wrapper.requirements,
            wrapper.retry_config,
        )

    @property
    def is_loss(self) -> bool:
        """
        Check if the main metric is a loss metric.

        :return: True if the main metric is a loss metric (lower is better), False otherwise
        """
        return self._is_loss

    def __del__(self):
        self.finish_run()
        self.finish_iteration()

    def _fix_name(self, name: str) -> str:
        """
        Prefix a name with the plugin name.

        :param name: Base name to fix
        :return: Name prefixed with plugin name
        """
        return f"{self._plugin_name} / {name}"

    # START: INITIAL MODEL MANAGEMENT

    def is_initial_run(self, run_id: str) -> bool:
        """Check whether passed run_id is actually initial_run.

        :param run_id: ID of a run to check.
        :return: True or False.
        """
        return run_id == self.get_initial_run_id()

    def get_initial_run_id(self) -> str:
        """
        Get the ID of the initial run.

        :return: ID of the initial run
        """
        initial_run_id: str = self._get_run_id_by_name(
            self._get_experiment_id_by_name(self.initial_experiment_name),
            INITIAL_RUN_NAME,
        )
        return initial_run_id

    def is_initial_run(self, run_id: str) -> bool:
        """Check whether passed run_id is actually initial_run.

        :param run_id: ID of a run to check.
        :return: True or False.
        """
        initial_run_id: str = self._get_run_id_by_name(
            get_experiment_id_by_name(self.initial_experiment_name),
            INITIAL_RUN_NAME,
        )
        return run_id == initial_run_id

    def has_initial_model(self) -> bool:
        """
        Check if an initial model exists.

        :return: True if initial model exists, False otherwise
        """
        experiment_id = self._get_experiment_id_by_name(
            self.initial_experiment_name
        )
        if experiment_id is None:
            return False
        else:
            run_id = self._get_run_id_by_name(
                self._get_experiment_id_by_name(self.initial_experiment_name),
                INITIAL_RUN_NAME,
            )
            if run_id is None:
                return False
            else:
                return self._check_artifact_exists(
                    run_id,
                    "model",
                )

    @retry_method(name="log_model")
    def upload_initial_model(self, model: EmbeddingsModelInterface):
        """Upload the very first, initial model to the mlflow server

        :param model: model to be uploaded
        """
        self.finish_iteration()
        experiment_id = self._get_experiment_id_by_name(
            self.initial_experiment_name
        )
        if experiment_id is None:
            logger.info(
                f"Can't find any active iteration with name: {self.initial_experiment_name}"
            )
            try:
                logger.info("Create initial experiment")
                mlflow.create_experiment(self.initial_experiment_name)
            except MlflowException as e:
                if "Cannot set a deleted experiment" in str(e):
                    logger.error(
                        f"Creation of initial experiment is failed: experiment with the same name {self.initial_experiment_name} is deleted, but not archived"
                    )
                    experiments = mlflow.search_experiments(
                        view_type=mlflow.entities.ViewType.ALL
                    )
                    deleted_experiment_id = None

                    for exp in experiments:
                        if exp.name == self.initial_experiment_name:
                            deleted_experiment_id = exp.experiment_id
                            break

                    logger.info(
                        f"Restore deleted experiment with the same name: {self.initial_experiment_name}"
                    )
                    mlflow.tracking.MlflowClient().restore_experiment(
                        deleted_experiment_id
                    )
                    logger.info(
                        f"Archive deleted experiment with the same name: {self.initial_experiment_name}"
                    )
                    mlflow.tracking.MlflowClient().rename_experiment(
                        deleted_experiment_id,
                        self.initial_experiment_name + "_archive",
                    )
                    logger.info(
                        f"Delete archived experiment with the same name: {self.initial_experiment_name}"
                    )
                    mlflow.delete_experiment(deleted_experiment_id)
                    logger.info(f"Create initial experiment")
                    mlflow.create_experiment(self.initial_experiment_name)
                else:
                    raise e

        with mlflow.start_run(
            experiment_id=get_experiment_id_by_name(
                self.initial_experiment_name
            ),
            run_name=INITIAL_RUN_NAME,
        ) as run:
            logger.info(
                f"Upload initial model to {self.initial_experiment_name} / {INITIAL_RUN_NAME}"
            )
            if self._check_artifact_exists(
                self._get_run_id_by_name(
                    self._get_experiment_id_by_name(
                        self.initial_experiment_name
                    ),
                    INITIAL_RUN_NAME,
                ),
                "model",
            ):
                logger.info("Model is already uploaded")
                return

            mlflow.pytorch.log_model(
                model, "model", pip_requirements=self._requirements
            )
            logger.info("Uploading is finished")

    def get_initial_model_run_id(self) -> Optional[str]:
        """Get run ID related to the initial model

        :return: Run ID, or None.
        """
        if not self.has_initial_model():
            logger.error("No initial model was found.")
            return None

        return self._get_run_id_by_name(
            self._get_experiment_id_by_name(self.initial_experiment_name),
            INITIAL_RUN_NAME,
        )

    # END: INITIAL MODEL MANAGEMENT

    # START: EXPERIMENTS MANAGEMENT

    @retry_method(name="search_experiments")
    def get_experiments(self) -> List[Experiment]:
        """
        Get all experiments related to the current plugin.

        :return: List of experiments
        """
        experiments: List[Experiment] = [
            e
            for e in mlflow.search_experiments()
            if (
                e.name.startswith(f"{self._plugin_name} / {EXPERIMENT_PREFIX}")
            )
        ]
        return experiments

    def get_previous_iteration_id(self) -> Optional[str]:
        """
        Get the ID of the previous iteration.

        :return: ID of the previous iteration, or None if not found
        """
        if (
            self._tuning_iteration == self.initial_experiment_name
            or self._tuning_iteration is None
        ):
            logger.warning(
                f"Can't find previous iteration - no current iteration was setup"
            )
            return None

        experiments: List[Experiment] = self.get_experiments()
        if len(experiments) == 0:
            logger.warning("No iteration found")
            return None

        return max(
            experiments, key=lambda exp: exp.creation_time
        ).experiment_id

    def get_last_iteration_id(self) -> Optional[str]:
        """
        Get the ID of the last iteration (excluding current).

        :return: ID of the last iteration, or None if not found
        """
        if (
            self._tuning_iteration == self.initial_experiment_name
            or self._tuning_iteration is None
        ):
            logger.warning(
                f"Can't find previous iteration - no current iteration was setup"
            )
            return None

        experiments: List[Experiment] = [
            e
            for e in self.get_experiments()
            if e.name != str(self._tuning_iteration)
        ]
        if len(experiments) == 0:
            logger.warning("No iteration found")
            return None

        return max(
            experiments, key=lambda exp: exp.creation_time
        ).experiment_id

    def get_last_finished_iteration_id(self) -> Optional[str]:
        """
        Get the ID of the last finished iteration.

        :return: ID of the last finished iteration, or None if not found
        """
        if (
            self._tuning_iteration == self.initial_experiment_name
            or self._tuning_iteration is None
        ):
            logger.warning(
                f"Can't find previous iteration - no current iteration was setup"
            )
            return None

        experiments: List[Experiment] = self.get_experiments()
        if len(experiments) == 0:
            logger.warning("No iteration found")
            return None

        finished_experiments = []
        for experiment in experiments:
            # Get all runs for the experiment
            runs = self.get_runs(experiment.experiment_id)

            all_finished = True
            for _, run in runs.iterrows():
                if (
                    run["status"] != MLflowStatus.FINISHED
                    and run["status"] != MLflowStatus.FAILED
                ):
                    all_finished = False

            if all_finished:
                finished_experiments.append(experiment)

        if len(finished_experiments) == 0:
            logger.warning("No finished iteration found")
            return None

        return max(
            finished_experiments, key=lambda exp: exp.creation_time
        ).experiment_id

    def delete_previous_iteration(self):
        """
        Delete models of the previous iteration and archive/delete the experiment.

        :return: None
        """
        experiment_id: Optional[str] = self.get_previous_iteration_id()
        logger.info("Delete models of previous iteration.")
        runs = self.get_runs(experiment_id, models_only=True)

        runs = runs[runs.status == MLflowStatus.FINISHED.name]
        run_ids = runs["run_id"].tolist()

        for run_id in run_ids:
            self.delete_model(run_id, experiment_id)

        if experiment_id is not None:
            logger.info(
                f"Iteration with ID {experiment_id} is going to be deleted"
            )
            self._archive_experiment(experiment_id)
            self._delete_experiment(experiment_id)
        else:
            logger.warning(
                "Can't delete a previous iteration, no previous iteration in history"
            )

    @retry_method(name="create_experiment")
    def _set_experiment_with_name(self, experiment_name: str):
        """
        Set current experiment by name.

        :param experiment_name: Name of the experiment
        :return: None
        """
        self._iteration_experiment = mlflow.set_experiment(
            experiment_name=experiment_name
        )
        self._tuning_iteration_id = self._iteration_experiment.experiment_id

    @retry_method(name="create_experiment")
    def _set_experiment_with_id(self, experiment_id: str):
        """
        Set current experiment by ID.

        :param experiment_id: ID of the experiment
        :return: None
        """
        self._tuning_iteration_id = experiment_id
        self._iteration_experiment = mlflow.set_experiment(
            experiment_id=self._tuning_iteration_id
        )

    def set_iteration(self, iteration: FineTuningIteration):
        """Start a new fine-tuning session.

        :param iteration: fine-tuning iteration info
        """
        if iteration.plugin_name != self._plugin_name:
            logger.error(
                f"Can't set iteration with different plugin name: {iteration.plugin_name} != {self._plugin_name}"
            )
            return

        if self._tuning_iteration == self.initial_experiment_name:
            self.finish_iteration()

        logger.info("Start a new fine-tuning iterations")

        self._tuning_iteration = iteration
        tuning_iteration_id = self._get_experiment_id_by_name(str(iteration))
        if tuning_iteration_id is None:
            self._set_experiment_with_name(str(iteration))

        else:
            self._set_experiment_with_id(tuning_iteration_id)

    def finish_iteration(self):
        """
        Finish current iteration and reset to initial experiment.

        :return: None
        """
        logger.info(f"Finish current iteration {self._tuning_iteration_id}")
        self._tuning_iteration = self.initial_experiment_name
        tuning_iteration_id = self._get_experiment_id_by_name(
            self.initial_experiment_name
        )

        if tuning_iteration_id is None:
            self._set_experiment_with_name(self.initial_experiment_name)
        else:
            self._set_experiment_with_id(tuning_iteration_id)

        logger.info(f"Current iteration is finished")

    # END: EXPERIMENTS MANAGEMENT

    # START: RUN IDS MANAGEMENT

    @retry_method(name="search_runs")
    def get_run_by_id(
        self, run_id: str, models_only: bool = False
    ) -> pd.DataFrame:
        """
        Get run information by ID.

        :param run_id: ID of the run to get
        :param models_only: If True, only return runs with models
        :return: DataFrame containing run information
        """
        if models_only:
            runs: pd.DataFrame = mlflow.search_runs(
                filter_string=self._get_model_exists_filter()
            )

        else:
            runs: pd.DataFrame = mlflow.search_runs()

        runs = runs[runs["run_id"] == run_id]
        return runs

    def _get_best_previous_run_id(self) -> Tuple[Optional[str], bool]:
        """
        Get the ID of the best run from the previous iteration.

        :return: Tuple containing (run_id, is_initial),
        where is_initial is True if no previous iteration exists
        """
        initial_id: Optional[str] = self._get_experiment_id_by_name(
            self.initial_experiment_name
        )
        last_session_id: Optional[str] = self.get_previous_iteration_id()
        if initial_id == last_session_id or last_session_id is None:
            return None, True

        run_id, _ = self._get_best_quality(last_session_id)
        return run_id, False

    def get_best_current_run_id(self) -> Tuple[Optional[str], bool]:
        """Current experiment's best run ID.

        :return: run ID.
        """
        initial_id: Optional[str] = self._get_experiment_id_by_name(
            self.initial_experiment_name
        )

        if (
            initial_id == self._tuning_iteration_id
            or self._tuning_iteration_id is None
        ):
            return None, True

        run_id, _ = self._get_best_quality(self._tuning_iteration_id)
        return run_id, False

    # END: RUN IDS MANAGEMENT

    # START: RUN MANAGEMENT

    @retry_method(name="start_run")
    def _start_run(self, params: FineTuningParams):
        """
        Start a new MLflow run with the given parameters.

        :param params: Fine-tuning parameters for the run
        :return: None
        """
        self._run_params = params
        run_name: str = self._run_params.id
        self._run_id = self._get_run_id_by_name(
            self._tuning_iteration_id, run_name
        )

        self._run = mlflow.start_run(
            self._run_id, self._tuning_iteration_id, run_name
        )

    @retry_method(name="log_params")
    def _save_params(self):
        """
        Save all parameters to MLflow.

        :return: None
        """
        convert_value = lambda value: (
            ", ".join(map(str, value)) if isinstance(value, list) else value
        )
        for key, value in dict(self._tuning_iteration).items():
            mlflow.log_param(key, convert_value(value))

        for key, value in dict(self._run_params).items():
            mlflow.log_param(key, convert_value(value))

    def _set_run(self, params: FineTuningParams):
        """
        Set up a run with provided parameters.

        :param params: Fine-tuning parameters for the run
        :return: False if run already exists, True if new run
        """
        self._start_run(params)
        if self._run_id is None:
            self._run_id = self._run.info.run_id
            self._save_params()
            mlflow.log_metric("model_uploaded", 0)
            return True

        return False

    def set_run(self, params: FineTuningParams) -> bool:
        """Start a new run with provided fine-tuning params

        :param params: provided fine-tuning params
        :return: True if it's a finished run (otherwise False)
        """

        if self._tuning_iteration == self.initial_experiment_name:
            # TODO: implement exception
            raise ValueError("You can't start run for initial iteration")

        if self._run is not None:
            self.finish_run()

        logger.info(
            f"Start a new run for iteration {self._tuning_iteration_id} with params:\n\t{str(params)}"
        )

        if self._set_run(params):
            return False

        return self._run.info.status == MLflowStatus.FINISHED.name

    @retry_method(name="end_run")
    def finish_run(self, as_failed: bool = False):
        """
        Finish current run and clear accumulators.

        :param as_failed: If True, end run with failed status
        :return: None
        """
        logger.info(
            f"Finish current run {self._tuning_iteration_id} / {self._run_id}"
        )
        for accumulator in self._accumulators:
            accumulator.clear()

        if as_failed:
            mlflow.end_run(status="FAILED")
        else:
            mlflow.end_run()

        # Set params to default None
        self._run = None
        self._run_params = None
        self._run_id = None

        logger.info(f"Current run is finished")

    # END: RUN MANAGEMENT

    # START: DOWNLOAD MODELS FUNCTIONS

    def download_initial_model(self) -> EmbeddingsModelInterface:
        """Download initial model.

        :return: initial embeddings model
        """
        return self._download_model_by_run_id(
            self._get_run_id_by_name(
                self._get_experiment_id_by_name(self.initial_experiment_name),
                INITIAL_RUN_NAME,
            )
        )

    def download_model_by_run_id(
        self, run_id: str
    ) -> Optional[EmbeddingsModelInterface]:
        """Get model by an experiment_name and a run_name.

        :param run_id: ID of a run
        :return: an embedding model related to a run, or None
        """
        model = None
        try:
            model = self._download_model_by_run_id(run_id)
        except Exception:
            logger.exception("Something went wrong while downloading")

        return model

    def download_model(
        self, experiment_name: str, run_name: str
    ) -> Optional[EmbeddingsModelInterface]:
        """Get model by an experiment_name and a run_name.

        :param experiment_name: name of an experiment
        :param run_name: name of a run
        :return: an embedding model related to a run, or None
        """
        experiment_id = self._get_experiment_id_by_name(experiment_name)
        if experiment_id is None:
            logger.info(
                f"Can't find any active iteration with name: {experiment_name}"
            )
            return None

        run_id = self._get_run_id_by_name(experiment_id, run_name)
        if run_id is None:
            logger.info(
                f"Can't find a run with name {run_name} for iteration with name: {experiment_name}"
            )
            return None

        return self._download_model_by_run_id(run_id)

    def download_last_model(self) -> EmbeddingsModelInterface:
        """Get previous iteration best embedding model.

        :return: best embedding model
        """
        run_id, is_initial = self._get_best_previous_run_id()
        if is_initial:
            logger.warning(
                "Download initial model, no previous iteration in history"
            )
            return self.download_initial_model()

        if run_id is None:
            logger.warning(
                "Download initial model, no previous iteration's "
                "finished runs with uploaded model in history"
            )
            return self.download_initial_model()

        return self._download_model_by_run_id(run_id)

    def download_best_model(
        self, experiment_id: str
    ) -> EmbeddingsModelInterface:
        """Get previous iteration best embedding model.

        :param experiment_id: ID of interesting experiment
        :return: best embedding model
        """
        run_id, _ = self._get_best_quality(experiment_id)
        if run_id is None:
            logger.warning(
                "Download initial model, no previous iteration's "
                "finished runs with uploaded model in history"
            )
            return self.download_initial_model()

        return self._download_model_by_run_id(run_id)

    @retry_method(name="log_param")
    def _set_model_as_deleted(self, run_id: str, experiment_id: str):
        """
        Mark model as deleted by setting appropriate metrics.

        :param run_id: ID of the run containing the model
        :param experiment_id: ID of the experiment containing the run
        :return: None
        """
        with mlflow.start_run(
            run_id=run_id, experiment_id=experiment_id
        ) as run:
            mlflow.log_metric("model_deleted", 1)
            mlflow.log_metric("model_uploaded", 0)

        if self._tuning_iteration == self.initial_experiment_name:
            logger.info("Download initial model")
            return self.download_initial_model()

        run_id, is_initial = self.get_best_current_run_id()
        return self._download_model_by_run_id(run_id)

    # END: DOWNLOAD MODELS FUNCTIONS

    # START: GET MODEL URL

    def get_last_model_url(self) -> Optional[str]:
        """
        Get URL for the best model from previous iteration.

        :return: URL for the model, or None if not found
        """
        run_id, is_initial = self._get_best_previous_run_id()
        if is_initial:
            logger.warning(
                "Can't get the best model URL, no previous iteration in history"
            )
            return None

        if run_id is None:
            logger.warning(
                "Can't get the best model URL, no previous iterations "
                "finished runs with uploaded model in history"
            )
            return None

        return self._get_artifact_url(run_id, MODEL_ARTIFACT_PATH)

    def get_current_model_url(self) -> Optional[str]:
        """
        Get URL for the best model from current iteration.

        :return: URL for the model, or None if not found
        """
        run_id, is_initial = self.get_best_current_run_id()
        if is_initial:
            logger.warning(
                "Can't get the best model URL, current run is initial"
            )
            return None

        if run_id is None:
            logger.warning(
                "Can't get the best model URL, no iterations "
                "finished runs with uploaded model in history"
            )
            return None

        return self._get_artifact_url(run_id, MODEL_ARTIFACT_PATH)

    # END: GET MODEL URL

    # START: MODEL MANAGEMENT

    @retry_method(name="search_runs")
    def model_is_uploaded(self) -> bool:
        """
        Check if a model is uploaded for the current run.

        :return: True if model is uploaded, False otherwise
        """
        runs: pd.DataFrame = mlflow.search_runs(
            experiment_ids=[self._tuning_iteration_id],
            filter_string=self._get_model_exists_filter(),
        )
        runs = runs[runs["run_id"] == self._run_id]
        return runs.shape[0] > 0

    def delete_model(self, run_id: str, experiment_id: Optional[str] = None):
        """
        Delete a model for specified run.

        :param run_id: ID of the run containing the model
        :param experiment_id: ID of the experiment containing the run (uses current if None)
        :return: None
        """
        experiment_id = (
            self._tuning_iteration_id
            if experiment_id is None
            else experiment_id
        )
        if experiment_id is None:
            raise ValueError(
                f"No iteration was initialized, unable to delete model."
            )

        if experiment_id == self.initial_experiment_name:
            raise ValueError(f"Initial model can't be deleted.")

        run_info = self._get_run(run_id)

        if run_info is not None:
            runs: pd.DataFrame = self.get_runs(run_id=run_id, models_only=True)
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
    def _save_model(self, model: EmbeddingsModelInterface):
        """
        Upload a model to MLflow.

        :param model: Model to be uploaded
        :return: None
        """
        mlflow.pytorch.log_model(
            model, "model", pip_requirements=self._requirements
        )
        mlflow.log_metric("model_uploaded", 1)
        logger.info("Upload is finished")

    def save_model(
        self, model: EmbeddingsModelInterface, best_only: bool = True
    ):
        """Save fine-tuned embedding model

        :param model: model to be saved
        :param best_only: save only if it's the best (default: True)
        """
        if self._tuning_iteration == self.initial_experiment_name:
            raise ValueError(
                f"Can't save not initial model for {self.initial_experiment_name} experiment"
            )

        if self._run_id is None:
            raise ValueError("There is no current Run")

        logger.info(
            f"Save model for {self._tuning_iteration_id} / {self._run_id}"
        )
        if not best_only:
            self._save_model(model)
        else:
            current_quality = self.get_quality()
            best_run_id, best_quality = self.get_best_quality()

            if best_run_id is None or (
                current_quality <= best_quality
                if self.is_loss
                else current_quality >= best_quality
            ):
                self._save_model(model)

                if best_run_id is not None:
                    self.delete_model(best_run_id)
            else:
                logger.info("Not the best run - ignore saving")

    # END: MODEL MANAGEMENT

    # START: FINE-TUNING PARAMS

    def get_top_params_by_experiment_id(
        self, experiment_id: str
    ) -> Optional[List[FineTuningParams]]:
        """Get top N previous fine-tuning iteration best params from experiment by ID.

        :param experiment_id: ID of an experiment.
        :return: fine-tuning iteration params
        """
        initial_id: Optional[str] = self._get_experiment_id_by_name(
            self.initial_experiment_name
        )
        if initial_id == experiment_id:
            logger.warning(
                "Can't retrieve top params, no previous iteration in history"
            )
            return None

        runs: pd.DataFrame = self.get_runs(experiment_id, models_only=True)
        runs = runs[
            runs.status == MLflowStatus.FINISHED.name
        ]  # and only finished ones
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
        top_n_rows: np.ndarray = runs.iloc[sorted_indices[: self._n_top_runs]]

        # Define a mapping dictionary to remove the "params." prefix
        column_mapping: Dict[str, str] = {
            col: col.replace("params.", "") for col in top_n_rows.columns
        }

        # Rename the columns
        top_n_rows: np.ndarray = top_n_rows.rename(
            columns=column_mapping
        ).to_dict(orient="records")

        return [FineTuningParams(**row) for row in top_n_rows]

    def get_top_params(self) -> Optional[List[FineTuningParams]]:
        """Get top N previous fine-tuning iteration best params

        :return: fine-tuning iteration params
        """
        last_session_id: Optional[str] = self.get_previous_iteration_id()
        return self.get_top_params_by_experiment_id(last_session_id)

    # END: FINE-TUNING PARAMS

    # START: METRICS MANAGEMENT

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
        if self._tuning_iteration == self.initial_experiment_name:
            raise ValueError(
                f"No metrics for {self.initial_experiment_name} experiment"
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

    def _get_best_quality(
        self, experiment_id: str
    ) -> Tuple[Optional[str], float]:
        """
        Get best quality run from an experiment.

        :param experiment_id: ID of the experiment
        :return: Tuple containing (run_id, quality_value), run_id may be None if no runs found
        """
        runs: pd.DataFrame = self.get_runs(experiment_id, models_only=True)
        runs = runs[
            runs.status == MLflowStatus.FINISHED.name
        ]  # and not finished ones
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
        if self._tuning_iteration == self.initial_experiment_name:
            raise ValueError(
                f"No metrics for {self.initial_experiment_name} experiment"
            )

        return self._get_best_quality(self._tuning_iteration_id)

    # END: METRICS MANAGEMENT
