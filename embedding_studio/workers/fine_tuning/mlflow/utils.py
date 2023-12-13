import mlflow


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

    # Return run_id if found, else return None
    return matching_runs.iloc[0]["run_id"] if not matching_runs.empty else None
