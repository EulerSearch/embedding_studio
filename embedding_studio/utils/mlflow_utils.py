import logging
from typing import Optional

import mlflow

logger = logging.getLogger(__name__)

DEFAULT_FINE_TUNING_METHOD_NAME = "Default Fine Tuning Method"


def get_experiment_id_by_name(experiment_name: str) -> str:
    """
    Given an experiment name, this function returns the experiment ID.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    return experiment.experiment_id if experiment else None


def get_run_id_by_name(experiment_id: str, run_name: str) -> str:
    """
    Given an experiment ID and run name, this function returns the run ID
    associated with that run name within the specified experiment.
    """
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    if runs.shape[0] == 0:
        return None

    # Filter runs by the given run name
    matching_runs = runs[runs.get("tags.mlflow.runName") == run_name]

    # Return embedding_model_id if found, else return None
    return (
        matching_runs.iloc[0]["embedding_model_id"]
        if not matching_runs.empty
        else None
    )


def get_mlflow_results_url(
    mlflow_url: str, batch_id: str, model_id: str
) -> Optional[str]:
    """Generate URL where to check results.

    :param mlflow_url: MLFlow connection URL
    :param batch_id: released batch ID
    :param model_id: starting embedding_model_id (embedding_model_id which contains a model)
    :return:
    """
    mlflow.set_tracking_uri(mlflow_url)
    iteration_name = f"iteration / {DEFAULT_FINE_TUNING_METHOD_NAME} / {model_id} / {batch_id}"
    experiment_ids = [
        experiment.id
        for experiment in mlflow.search_experiments()
        if experiment.name.startswith(iteration_name)
    ]
    if len(experiment_ids) == 0:
        logger.error(f"Can't find any experiments with name {iteration_name}")
        return None

    return experiment_ids[0]
